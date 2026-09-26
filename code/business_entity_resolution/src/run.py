"""Run the whole candidate-pipeline search, compare the pipelines and submit the best one.

    python run.py all      --data-dir <student_resource/dataset> --work-dir <work>
    python run.py prepare  ...   clean, block (wide), sample, featurise           → artifacts/
    python run.py tune     ...   Optuna: blocking (multi-objective) + matchers     → experiments/hpo, blocking/
    python run.py neural   ...   optional: frozen E5 similarity feature (GPU)      → artifacts/dense_*.parquet
    python run.py evaluate ...   5-fold OOF evaluation of every candidate pipeline → experiments/runs/<name>/
    python run.py compare  ...   leaderboard, paired deltas, all analysis figures  → reports/index.html
    python run.py submit   ...   refit best, holdout score, test inference, validate, zip → output/

Every stage is resumable: finished outputs are skipped unless --force is given.
See ../README.md for SageMaker instructions and runtime expectations.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(HERE), os.environ.get("PYTHONPATH")]))

import pandas as pd  # noqa: E402

from data import Paths, log  # noqa: E402
from er_eval import Registry, write_json  # noqa: E402


def find_data_dir(start: Path) -> Path | None:
    for base in [start, *start.parents]:
        hits = [p for p in base.glob("**/student_resource/dataset") if "__MACOSX" not in p.parts]
        if hits:
            return hits[0]
    return None


def make_paths(a) -> Paths:
    data = Path(a.data_dir) if a.data_dir else find_data_dir(HERE)
    if data is None or not (data / "train").exists():
        sys.exit("--data-dir must point at student_resource/dataset (with train/ and test/)")
    return Paths(data, Path(a.work_dir), a.clean_dir, a.cache_dir)


# ---- stages ------------------------------------------------------------------------------------------
def stage_prepare(a, paths):
    import prepare
    prepare.run(paths, a.n_dev, a.n_hpo, a.n_holdout, jobs=a.jobs, seed=a.seed, feature_chunk=a.feature_chunk)


def stage_tune(a, paths):
    import models as M
    from er_tune import study
    from pipelines import Workspace
    ws = Workspace(paths)
    what = a.tune.split(",")
    if "blocking" in what and (a.force or not (paths.exp / "blocking" / "blocking_configs.json").exists()):
        study.tune_blocking(ws, n_trials=a.blocking_trials, seed=a.seed, random_baseline=not a.no_random_baseline)
    for lib in ("lgbm", "xgb", "cat"):
        if lib not in what:
            continue
        if not M.available(lib):
            log(f"skip HPO for {lib}: library not installed")
            continue
        if a.force or not (paths.exp / "hpo" / lib / "best_params.json").exists():
            study.tune_matcher(ws, lib, n_trials=a.trials, seed=a.seed,
                               random_baseline=(lib == "lgbm" and not a.no_random_baseline),
                               timeout=a.tune_timeout)


def stage_neural(a, paths):
    import neural
    neural.dense_for_roles(paths)


def stage_evaluate(a, paths):
    from pipelines import Workspace, default_pipelines, evaluate
    ws = Workspace(paths)
    reg = Registry(paths.exp)
    cfgs = default_pipelines(ws, neural=a.neural)
    if a.pipelines != "all":
        want = set(a.pipelines.split(","))
        cfgs = [c for c in cfgs if c.name in want]
    log(f"evaluating {len(cfgs)} pipelines: {[c.name for c in cfgs]}")
    for c in cfgs:
        try:
            evaluate(c, ws, reg, n_folds=a.folds, seed=a.seed, force=a.force)
        except Exception as ex:   # one broken pipeline must not stop the search
            log(f"[{c.name}] FAILED: {type(ex).__name__}: {ex}")
            if a.strict:
                raise


def stage_compare(a, paths):
    import numpy as np

    from er_viz import blocking_plots as BP
    from er_viz import compare_plots as CP
    from er_viz import feature_space as FS
    from er_viz import hpo_plots as HP
    from er_viz import matcher_plots as MP
    from er_viz import report
    from er_viz.common import FigureSink
    from pipelines import PipelineConfig, Workspace, cross_country

    ws = Workspace(paths)
    reg = Registry(paths.exp)
    idx = reg.index()
    if idx.empty:
        sys.exit("no evaluated pipelines yet: run `evaluate` first")
    idx.to_csv(paths.reports / "leaderboard.csv", index=False)
    cands = idx[~idx.name.isin(["P-oracle"])]
    best = cands.sort_values(["macro_f05", "wall_s"], ascending=[False, True]).iloc[0]["name"]
    ref = "P0" if "P0" in set(idx.name) else best
    sink = FigureSink(paths.reports)
    metrics = {n: reg.load(n, "metrics.json") for n in idx.name}
    entities = {n: reg.load(n, "entity.parquet") for n in idx.name}
    ranked = cands.sort_values("macro_f05", ascending=False).name.tolist()
    top = ranked[:6]

    def safe(fn, *args, **kw):
        try:
            fn(*args, **kw)
        except Exception as ex:   # a failing figure should not stop the report
            log(f"figure {fn.__module__}.{fn.__name__} failed: {type(ex).__name__}: {ex}")

    # comparison + errors
    safe(CP.leaderboard, sink, idx, best, ref)
    fr = pd.DataFrame()
    try:
        fr = CP.forest(sink, idx, entities)
    except Exception as ex:
        log(f"forest failed: {ex}")
    safe(CP.ablation_waterfall, sink, idx, best)
    safe(CP.pareto_time, sink, idx, best)
    safe(CP.slice_heatmap, sink, entities, ref)
    safe(CP.fold_lines, sink, metrics, top)
    if ref in entities and ref != best:
        safe(CP.entity_scatter, sink, best, entities[best], ref, entities[ref])
    if a.xcountry:
        xc = {}
        for n in top[:3]:
            f = reg.dir(n) / "xcountry.json"
            if f.exists() and not a.force:
                xc[n] = json.loads(f.read_text())
            else:
                xc[n] = cross_country(PipelineConfig.from_dict(reg.load(n, "config.json")), ws, seed=a.seed)
                write_json(f, xc[n])
        safe(CP.xcountry, sink, xc, metrics)
    safe(CP.loss_bars, sink, metrics)
    safe(CP.loss_waterfall, sink, best, metrics[best])
    safe(CP.error_heatmap, sink, best, entities[best])
    best_pairs = reg.load(best, "pairs.parquet")
    try:
        from data import load_records
        ids = pd.concat([best_pairs.s1_entity_id, best_pairs.cand_entity_id]).unique()
        sub = best_pairs[(best_pairs.pred & (best_pairs.label == 0)) | (~best_pairs.pred & (best_pairs.label == 1))]
        ids = pd.concat([sub.s1_entity_id, sub.cand_entity_id]).unique()[:20000]
        recs = load_records(paths, "train", ["S1", "S2", "S3"], ids=ids)
        safe(CP.error_gallery, sink, best, best_pairs, recs)
    except Exception as ex:
        log(f"error gallery failed: {ex}")

    # blocking
    try:
        wide, ent = ws.wide_pairs("dev"), ws.entities("dev")
        from er_tune.study import load_study
        blocking_cfgs = {k: v for k, v in __import__("data").blocking_configs(paths).items() if k != "B0"}
        safe(BP.recall_at_k, sink, wide, ent)
        safe(BP.pareto, sink, paths.exp / "blocking" / "blocking_front.json", blocking_cfgs, wide, ent,
             load_study(ws, "blocking-tpe"))
        safe(BP.pass_combinations, sink, wide)
        safe(BP.leave_one_out, sink, wide, ent)
        safe(BP.rank_hist, sink, wide)
        safe(BP.candidate_counts, sink, wide, ent)
        mf = paths.art / "blocking_misses.parquet"
        safe(BP.miss_taxonomy, sink, pd.read_parquet(mf) if mf.exists() else None)
        gs = paths.art / "gold_by_source.parquet"
        if gs.exists():
            g = pd.read_parquet(gs)
            safe(BP.recall_by_slice, sink, wide, ent, g[g.s1_entity_id.isin(ent.s1_entity_id)])
    except Exception as ex:
        log(f"blocking figures failed: {ex}")

    # matcher + decision
    pairs_top = {n: reg.load(n, "pairs.parquet") for n in top[:6] if metrics[n]["overall"].get("pred_per_s1_mean") is not None}
    safe(MP.pr_curves, sink, pairs_top)
    safe(MP.score_density, sink, best, best_pairs, entities[best])
    model_runs = [n for n in ranked if (reg.dir(n) / "importance.csv").exists()]
    if model_runs:
        imp = reg.load(model_runs[0], "importance.csv")
        safe(MP.importance, sink, model_runs[0], imp)
        safe(MP.family_importance, sink, model_runs[0], imp)
    xs_runs = [n for n in ranked if (reg.dir(n) / "xsample.parquet").exists()]
    if xs_runs:
        xs = reg.load(xs_runs[0], "xsample.parquet")
        safe(MP.shap_summary, sink, xs_runs[0], xs)
        safe(MP.shap_dependence, sink, xs_runs[0], xs)
        safe(FS.shap_space, sink, xs_runs[0], xs)
        safe(FS.fp_reasons, sink, xs_runs[0], xs)
    safe(MP.slice_bars, sink, {n: metrics[n] for n in top[:4]}, "country")
    safe(MP.slice_bars, sink, {n: metrics[n] for n in top[:4]}, "name_freq_bin")
    safe(MP.reliability, sink, best, metrics[best])
    safe(MP.threshold_curves, sink, {n: metrics[n] for n in top})
    gated = [n for n in ranked if metrics[n].get("curves", {}).get("singleton_gate")]
    if gated:
        safe(MP.singleton_gate, sink, gated[0], metrics[gated[0]])
    safe(MP.entity_f_hist, sink, best, entities[best])
    safe(MP.f_by_multiplicity, sink, best, entities[best])

    # tuning
    safe(HP.all_studies, sink, ws)

    # SAE on neural embeddings (only if the neural stage produced them and torch is available)
    if a.sae and (paths.art / "emb_train.npy").exists():
        try:
            from er_viz import sae
            ids = pd.read_parquet(paths.art / "emb_train_ids.parquet")
            emb = np.load(paths.art / "emb_train.npy", mmap_mode="r")
            dev_pairs = ws.wide_pairs("dev")[["s1_entity_id", "cand_entity_id", "label"]]
            # bounded sample: a few thousand dev S1 and all their candidates (dense SAE codes are n x 8d floats)
            rng = np.random.default_rng(a.seed)
            s1s = dev_pairs.s1_entity_id.unique()
            keep = set(rng.choice(s1s, min(a.sae_entities, len(s1s)), replace=False))
            dev_pairs = dev_pairs[dev_pairs.s1_entity_id.isin(keep)]
            want = pd.Index(pd.unique(np.r_[dev_pairs.s1_entity_id.to_numpy(), dev_pairs.cand_entity_id.to_numpy()]))
            rows = np.sort(pd.Index(ids.entity_id).get_indexer(want))
            rows = rows[rows >= 0]
            sub = ids.iloc[rows]
            log(f"SAE on {len(rows):,} records from {len(keep):,} dev S1")
            sae.run(sink, np.asarray(emb[rows], np.float32), sub.entity_id.to_numpy(), sub.set_index("entity_id").text,
                    dev_pairs, log=log)
        except Exception as ex:
            log(f"SAE analysis failed: {type(ex).__name__}: {ex}")

    extra = ""
    if not fr.empty:
        adm = fr[fr.admitted]
        extra = (f"<p>{len(adm)} of {len(fr)} changes pass the admission gate (paired 95% CI lower bound &gt; 0 and "
                 f"Δ ≥ +0.001): {', '.join(adm.pipeline)}.</p>")
    hold = reg.dir(best) / "holdout.json"
    if hold.exists():
        h = json.loads(hold.read_text())
        extra += f"<p>Holdout (scored once): macro F0.5 <b>{h['macro_f05']:.4f}</b>.</p>"
    out = report.write(sink, idx, best, extra)
    write_json(paths.reports / "selection.json", {"best": best, "reference": ref, "top": top})
    log(f"report: {out}  (best = {best})")


def stage_submit(a, paths):
    import submit
    from pipelines import Workspace
    ws = Workspace(paths)
    reg = Registry(paths.exp)
    repo = next((p for p in [HERE, *HERE.parents] if (p / "Documentation_template.md").exists()), paths.work)
    res = submit.run(paths, ws, reg, name=a.run, batch_chunks=a.batch_chunks, team=a.team, repo_root=repo,
                     feature_chunk=a.feature_chunk)
    if not res["validator_pass"]:
        sys.exit("validator FAILED: see output/submission_report.json")


STAGES = {"prepare": stage_prepare, "tune": stage_tune, "neural": stage_neural, "evaluate": stage_evaluate,
          "compare": stage_compare, "submit": stage_submit}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=[*STAGES, "all"])
    ap.add_argument("--data-dir", help="student_resource/dataset (auto-detected when inside the repo)")
    ap.add_argument("--work-dir", default="work", help="all artifacts, runs, reports and outputs go here")
    ap.add_argument("--clean-dir", help="reuse an existing cleaned/ directory")
    ap.add_argument("--cache-dir", help="reuse an existing blocking_cache/ directory")
    ap.add_argument("--jobs", type=int, default=4, help="processes for cleaning / key building (~1 GB RAM each)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--n-dev", type=int, default=100_000)
    ap.add_argument("--n-hpo", type=int, default=50_000)
    ap.add_argument("--n-holdout", type=int, default=50_000)
    ap.add_argument("--feature-chunk", type=int, default=500_000, help="pairs per feature batch (memory knob)")
    ap.add_argument("--tune", default="blocking,lgbm,xgb,cat", help="what to tune")
    ap.add_argument("--trials", type=int, default=30, help="Optuna trials per matcher study")
    ap.add_argument("--blocking-trials", type=int, default=80)
    ap.add_argument("--tune-timeout", type=float, default=None, help="seconds per matcher study")
    ap.add_argument("--no-random-baseline", action="store_true", help="skip equal-budget random-search studies")
    ap.add_argument("--neural", action="store_true", help="include the dense-similarity pipeline (needs `neural` stage)")
    ap.add_argument("--neural-cpu", action="store_true", help="allow the neural stage without a CUDA GPU (very slow)")
    ap.add_argument("--gpu", action="store_true", help="train XGBoost on the CUDA GPU (tuning, evaluation, final refit)")
    ap.add_argument("--sae-entities", type=int, default=3000, help="dev S1 entities whose records feed the SAE analysis")
    ap.add_argument("--pipelines", default="all", help="comma list of pipeline names to evaluate")
    ap.add_argument("--folds", type=int, default=None, help="evaluate only the first N outer folds (quick runs)")
    ap.add_argument("--strict", action="store_true", help="stop on the first failing pipeline")
    ap.add_argument("--xcountry", action="store_true", default=True)
    ap.add_argument("--no-xcountry", dest="xcountry", action="store_false")
    ap.add_argument("--sae", action="store_true", default=True)
    ap.add_argument("--no-sae", dest="sae", action="store_false")
    ap.add_argument("--run", default="best", help="pipeline to submit (default: best OOF macro F0.5)")
    ap.add_argument("--batch-chunks", type=int, default=2, help="S1 key chunks per test retrieval batch")
    ap.add_argument("--team", default=None, help="team name → also build <team>_submission.zip")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    paths = make_paths(a)
    if a.gpu:                                       # checked up front, like --neural below
        import models as M
        msg = M.xgb_gpu_check()
        if msg:
            sys.exit(msg)
        M.set_xgb_device("cuda")
        log("XGBoost will train on the GPU (device=cuda)")
    if a.neural or a.stage == "neural":           # fail now, not hours later when the stage is reached
        import neural
        msg = neural.check(require_cuda=not a.neural_cpu)
        if msg:
            sys.exit(msg)
    t0 = time.time()
    order = ["prepare", "tune"] + (["neural"] if a.neural else []) + ["evaluate", "compare", "submit"] \
        if a.stage == "all" else [a.stage]
    for s in order:
        log(f"===== stage {s} =====")
        t1 = time.time()
        STAGES[s](a, paths)
        log(f"===== stage {s} done in {(time.time() - t1) / 60:.1f} min =====")
    log(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

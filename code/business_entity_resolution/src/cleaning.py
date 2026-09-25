"""Text cleaning for business names and addresses.

Extracted from notebooks/01_nlp_cleaning_eda.ipynb (section 6) so the blocking and matching stages
can import one implementation, plus fixes found by running that notebook on the full data:
  * noisy Hindi transliterations of "private limited" ("piraivet limitet", "praivrr limirrd", "pra li")
  * French departments (Gironde, Nord, ...) mapped to their region
  * "N°35" no longer becomes "ndeg35"

Run as a script to clean every source file into parquet:
    python -m cleaning --data-dir <student_resource/dataset> --out-dir <cleaned> [--jobs 3]
"""
from __future__ import annotations

import argparse
import csv
import re
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from anyascii import anyascii
from ftfy import fix_text

READ_KW = dict(sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False, encoding="utf-8")

# ---- reference vocabularies -------------------------------------------------------------------
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "district of columbia": "dc", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "puerto rico": "pr",
}
IN_STATE_CODES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar", "cg": "chhattisgarh",
    "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana", "hp": "himachal pradesh",
    "jh": "jharkhand", "jk": "jammu and kashmir", "ka": "karnataka", "kl": "kerala", "mp": "madhya pradesh",
    "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland",
    "od": "odisha", "or": "odisha", "pb": "punjab", "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu",
    "tg": "telangana", "ts": "telangana", "tr": "tripura", "up": "uttar pradesh", "uk": "uttarakhand",
    "ut": "uttarakhand", "wb": "west bengal", "dl": "delhi", "ch": "chandigarh", "py": "puducherry",
    "an": "andaman and nicobar islands", "la": "ladakh", "dn": "dadra and nagar haveli",
}
IN_STATES = set(IN_STATE_CODES.values()) | {"orissa", "pondicherry", "new delhi", "nct of delhi"}
IN_STATE_ALIASES = {"orissa": "odisha", "pondicherry": "puducherry", "nct of delhi": "delhi", "new delhi": "delhi"}
# State names written in native scripts, as observed in S2/S3 Indian addresses.
NATIVE_REGIONS = {
    "महाराष्ट्र": "maharashtra", "दिल्ली": "delhi", "ಕರ್ನಾಟಕ": "karnataka", "পশ্চিমবঙ্গ": "west bengal",
    "தமிழ்நாடு": "tamil nadu", "उत्तर प्रदेश": "uttar pradesh", "ગુજરાત": "gujarat", "తెలంగాణ": "telangana",
    "हरियाणा": "haryana", "കേരളം": "kerala", "राजस्थान": "rajasthan", "बिहार": "bihar",
    "मध्य प्रदेश": "madhya pradesh", "ఆంధ్రప్రదేశ్": "andhra pradesh", "ఆంధ్ర ప్రదేశ్": "andhra pradesh",
    "ଓଡ଼ିଶା": "odisha", "ਪੰਜਾਬ": "punjab", "झारखंड": "jharkhand", "छत्तीसगढ़": "chhattisgarh",
    "उत्तराखंड": "uttarakhand", "हिमाचल प्रदेश": "himachal pradesh", "गोवा": "goa", "অসম": "assam",
}
# French regions, and departments → region (S2/S3 often give the department instead of the region).
FR_DEPARTMENTS = {
    "auvergne-rhone-alpes": ["ain", "allier", "ardeche", "cantal", "drome", "isere", "loire", "haute-loire",
                             "puy-de-dome", "rhone", "savoie", "haute-savoie", "metropole de lyon"],
    "bourgogne-franche-comte": ["cote-dor", "doubs", "jura", "nievre", "haute-saone", "saone-et-loire", "yonne",
                                "territoire de belfort"],
    "bretagne": ["cotes-darmor", "finistere", "ille-et-vilaine", "morbihan"],
    "centre-val de loire": ["cher", "eure-et-loir", "indre", "indre-et-loire", "loir-et-cher", "loiret"],
    "corse": ["corse-du-sud", "haute-corse"],
    "grand est": ["ardennes", "aube", "marne", "haute-marne", "meurthe-et-moselle", "meuse", "moselle",
                  "bas-rhin", "haut-rhin", "vosges"],
    "hauts-de-france": ["aisne", "nord", "oise", "pas-de-calais", "somme"],
    "ile-de-france": ["paris", "seine-et-marne", "yvelines", "essonne", "hauts-de-seine", "seine-saint-denis",
                      "val-de-marne", "val-doise"],
    "normandie": ["calvados", "eure", "manche", "orne", "seine-maritime"],
    "nouvelle-aquitaine": ["charente", "charente-maritime", "correze", "creuse", "dordogne", "gironde", "landes",
                           "lot-et-garonne", "pyrenees-atlantiques", "deux-sevres", "vienne", "haute-vienne"],
    "occitanie": ["ariege", "aude", "aveyron", "gard", "haute-garonne", "gers", "herault", "lot", "lozere",
                  "hautes-pyrenees", "pyrenees-orientales", "tarn", "tarn-et-garonne"],
    "pays de la loire": ["loire-atlantique", "maine-et-loire", "mayenne", "sarthe", "vendee"],
    "provence-alpes-cote dazur": ["alpes-de-haute-provence", "hautes-alpes", "alpes-maritimes", "bouches-du-rhone",
                                  "var", "vaucluse"],
}

# Address abbreviations. Generic (English) map, used for US, India and any unseen country ...
ADDR_ABBR_EN = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place", "hwy": "highway", "pkwy": "parkway",
    "cir": "circle", "ter": "terrace", "trl": "trail", "sq": "square", "mt": "mount", "ft": "fort",
    "apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor",
    "bldg": "building", "rm": "room", "po": "po", "no": "number", "nr": "near", "opp": "opposite",
    "nagr": "nagar", "ngr": "nagar", "sec": "sector", "extn": "extension", "ext": "extension",
    "mkt": "market", "cplx": "complex", "soc": "society", "socy": "society", "hsg": "housing",
    "dist": "district", "tq": "taluka", "tal": "taluka", "vill": "village", "vpo": "village post office",
    "hno": "house number", "h": "house", "plt": "plot", "gf": "ground floor", "ff": "first floor",
    "blk": "block", "jn": "junction", "stn": "station", "clny": "colony",
}
# ... and French overrides: st = saint, ste = sainte, bd = boulevard, etc.
ADDR_ABBR_FR = {
    **{k: v for k, v in ADDR_ABBR_EN.items() if k in {"bldg", "apt"}},
    "st": "saint", "ste": "sainte", "bd": "boulevard", "bld": "boulevard", "boul": "boulevard",
    "av": "avenue", "ave": "avenue", "r": "rue", "ch": "chemin", "che": "chemin", "imp": "impasse",
    "all": "allee", "pl": "place", "sq": "square", "rte": "route", "fg": "faubourg", "fbg": "faubourg",
    "crs": "cours", "qu": "quai", "res": "residence", "lot": "lotissement", "zi": "zone industrielle",
    "za": "zone artisanale", "zac": "zone amenagement concerte", "cedex": "",
}
# Single-letter compass points are only safe in US addresses ("S.S Girl School" in India is not "south south").
ADDR_ABBR_US = {**ADDR_ABBR_EN, "n": "north", "s": "south", "e": "east", "w": "west",
                "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest"}
ADDR_ABBR = {"US": ADDR_ABBR_US, "India": ADDR_ABBR_EN, "France": ADDR_ABBR_FR}

# Legal forms: variant → canonical. Longest variants are matched first (so "pvt ltd" beats "ltd").
LEGAL_FORMS = {
    "private limited": ["private limited", "pvt ltd", "pvt limited", "private ltd", "p ltd", "pvt"],
    "limited": ["limited", "ltd", "limitedd", "limited company"],
    "llp": ["llp", "l l p", "elelpi"],
    "llc": ["llc", "l l c", "limited liability company"],
    "inc": ["inc", "incorporated", "incorporation"],
    "corp": ["corp", "corporation"],
    "company": ["company", "co"],
    "lp": ["lp", "l p", "limited partnership"],
    "plc": ["plc"],
    "pllc": ["pllc"],
    "pc": ["pc", "p c", "professional corporation"],
    "sarl": ["sarl", "s a r l", "eurl"],
    "sas": ["sas", "sasu", "s a s"],
    "sa": ["sa", "s a"],
    "sci": ["sci"],
    "gmbh": ["gmbh"],
    "oyj": ["oyj"],
    "ag": ["ag"],
}
LEGAL_VARIANTS = sorted(((v, k) for k, vs in LEGAL_FORMS.items() for v in vs), key=lambda x: -len(x[0]))
LEGAL_RE = re.compile(r"\b(" + "|".join(re.escape(v) for v, _ in LEGAL_VARIANTS) + r")\b")
LEGAL_CANON = dict(LEGAL_VARIANTS)
LEGAL_TOKEN = {k: k.replace(" ", "_") for k in LEGAL_FORMS}
# Transliterated "प्राइवेट लिमिटेड" comes out in many spellings (praivet limited, piraivet limitet,
# praivrr limirrd, ...). Match the shape: a p…v… word followed by an l…m… word; plus "pra li" (प्रा. लि.).
TRANSLIT_PVT_LTD_RE = r"\bp[a-z]{1,5}v[a-z]{1,4}\s+l[a-z]{0,2}m[a-z]{2,6}\b|\bpra\s+li\b"

# Generic words that carry little identity (removed only in name_core, kept in name_norm).
NAME_STOPWORDS = {"the", "and", "of", "a", "an", "de", "la", "le", "les", "du", "des", "et", "dba", "d", "l",
                  "m", "s", "shri", "sri", "shree", "mr", "mrs", "ms"}
ADDR_STOPWORDS = {"null", "none", "na", "n/a", "unknown", "nil"}

# ---- shared text primitives ---------------------------------------------------------------------
NATIVE_RE = re.compile("|".join(sorted(map(re.escape, NATIVE_REGIONS), key=len, reverse=True)))
NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")
# Double-encoded punctuation that ftfy cannot recover (the lead byte was lost): Â\x80\x99 → '
MOJIBAKE_RE = re.compile(r"[ÂÃâ]?[\x80-\x9f]+")
MOJIBAKE_MAP = {"\x99": "'", "\x98": "'", "\x93": "-", "\x94": "-", "\x9c": '"', "\x9d": '"'}


def to_ascii(s: pd.Series, native_map: bool = False) -> pd.Series:
    """ftfy → NFKC → (optional) native-script region names → anyascii. Only touches non-ASCII rows."""
    s = s.fillna("").str.replace(r"(?i)\bn\s?[°º]\s*", "no ", regex=True)   # N°35 → no 35, not ndeg35
    mask = s.str.contains(NON_ASCII_RE)
    if not mask.any():
        return s

    def fix(x: str) -> str:
        x = MOJIBAKE_RE.sub(lambda m: MOJIBAKE_MAP.get(m.group(0)[-1], " "), x)  # before ftfy reads it as cp1252
        x = unicodedata.normalize("NFKC", fix_text(x))
        if native_map:
            x = NATIVE_RE.sub(lambda m: NATIVE_REGIONS[m.group(0)], x)
        return anyascii(x)

    s = s.copy()
    s[mask] = s[mask].map(fix)
    return s


def squash(s: pd.Series) -> pd.Series:
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


# ---- business_name ------------------------------------------------------------------------------
PHONE_RE = r"(?:\+?\d{1,3}[\s-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-]?)\d{3}[\s.-]?\d{4}\b|\b\d{10,12}\b"
DOMAIN_RE = r"^(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:com|net|org|co\.in|in|co|fr|io|biz|us|info)\b.*$"


def _legal_set(found: list[str]) -> str:
    forms = {LEGAL_CANON[x] for x in found}
    if "private limited" in forms:   # "Pvt. X Ltd." → one form, not {private limited, limited}
        forms.discard("limited")
    return " ".join(sorted(forms))


def clean_name(name: pd.Series) -> pd.DataFrame:
    s = to_ascii(name).str.lower()
    s = s.str.replace(r"^\s*null\s*$", "", regex=True)
    s = s.str.replace(DOMAIN_RE, r"\1", regex=True)                  # qureontinto.com → qureontinto
    s = s.str.replace(PHONE_RE, " ", regex=True)                     # "heartland - 9240300650"
    s = s.str.replace(r"\s-\s*$|^\s*-+\s", " ", regex=True)          # dangling separators left behind
    s = s.str.replace(r"\((?:india|usa?|france|uk)\)", " ", regex=True)  # "mohan (india) technologies"
    s = s.str.replace(r"&", " and ", regex=False)
    s = s.str.replace(r"\+", " plus ", regex=True)
    s = s.str.replace(r"'s\b", "s", regex=True)                      # possessives: moyna's → moynas
    s = s.str.replace(r"(?<=\b[a-z])\.(?=[a-z]\b)", "", regex=True)  # l.l.c → llc, p.c. → pc
    s = s.str.replace(r"[^a-z0-9 ]+", " ", regex=True)               # all remaining punctuation & junk
    s = squash(s)
    s = s.str.replace(TRANSLIT_PVT_LTD_RE, "private limited", regex=True)

    legal = s.str.findall(LEGAL_RE).map(_legal_set)
    norm = squash(s.str.replace(LEGAL_RE, lambda m: " " + LEGAL_TOKEN[LEGAL_CANON[m.group(1)]] + " ", regex=True))
    core = squash(s.str.replace(LEGAL_RE, " ", regex=True)).map(
        lambda x: " ".join(t for t in x.split() if t not in NAME_STOPWORDS))
    return pd.DataFrame({"name_norm": norm, "name_core": core, "legal_form": legal}, index=name.index)


# ---- business_address ---------------------------------------------------------------------------
def _region_key(comp: str) -> str:
    return comp.replace("-", " ")


REGION_CANON = {
    "US": {**US_STATES, **{v: v for v in US_STATES.values()}},
    "India": {**{s: IN_STATE_ALIASES.get(s, s) for s in IN_STATES}, **IN_STATE_CODES},
    "France": {**{r: r for r in FR_DEPARTMENTS}, **{d: r for r, ds in FR_DEPARTMENTS.items() for d in ds}},
}
REGION_CANON = {c: {_region_key(k): v for k, v in m.items()} for c, m in REGION_CANON.items()}
POSTCODE_RE = {"US": r"\b(\d{5})(?:-\d{4})?\b", "India": r"\b(\d{3}\s?\d{3})\b", "France": r"\b(\d{5})\b"}


def _normalise_components(addr: str, abbr: dict, regions: dict) -> tuple[str, str]:
    """Per comma-separated component: expand abbreviations, canonicalise region, drop null/dupes."""
    comps, region = [], ""
    for comp in addr.split(","):
        comp = re.sub(r"[^a-z0-9 /-]+", " ", comp).strip()
        if not comp or comp in ADDR_STOPWORDS:
            continue
        key = _region_key(comp)
        if key in regions:           # a whole component that is a region (full name, code or department)
            region = region or regions[key]
            comp = regions[key]
        else:
            comp = " ".join(abbr.get(t, t) for t in comp.split())
        if comp and comp not in comps:
            comps.append(comp)
    return ", ".join(comps), region


def clean_address(addr: pd.Series, country: pd.Series) -> pd.DataFrame:
    s = to_ascii(addr, native_map=True).str.lower()
    s = s.str.replace(r"(?<=\d)(st|nd|rd|th)\b", r"\1", regex=True)     # keep ordinals attached: 3rd
    s = s.str.replace(r"\b(h|s|f|p|g)\.\s*no\b\.?", r"\1no", regex=True)  # h. no. → hno, s.no → sno
    s = s.str.replace(r"\bp\.?\s?o\.?\s?box\b", "pobox", regex=True)
    s = s.str.replace(r"\.", " ", regex=True).str.replace("'", "", regex=False)
    out = []
    for c, idx in s.groupby(country.fillna("")).groups.items():
        abbr, regions = ADDR_ABBR.get(c, ADDR_ABBR_EN), REGION_CANON.get(c, {})
        comps = [_normalise_components(a, abbr, regions) for a in s[idx]]
        part = pd.DataFrame(comps, index=idx, columns=["addr_norm", "region"])
        part["postcode"] = s[idx].str.extract(POSTCODE_RE.get(c, r"\b(\d{4,6})\b"), expand=False) \
            .str.replace(" ", "", regex=False).fillna("")
        out.append(part)
    res = pd.concat(out).reindex(addr.index)
    res["house_no"] = res.addr_norm.str.extract(r"^(\d+[a-z]?(?:[/-]\d+[a-z]?)*)\b", expand=False).fillna("")
    return res[["addr_norm", "region", "postcode", "house_no"]]


def clean_records(df: pd.DataFrame) -> pd.DataFrame:
    country = squash(to_ascii(df.country))   # open set: keep whatever label arrives, just tidy it
    return pd.concat([df.assign(country=country), clean_name(df.business_name),
                      clean_address(df.business_address, country)], axis=1)


# ---- full-file cleaning -------------------------------------------------------------------------
def clean_file(path: Path, out: Path, chunksize: int = 200_000) -> str:
    """Stream one TSV through clean_records into a zstd parquet file (raw + cleaned columns)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    t0, n, writer = time.time(), 0, None
    tmp = out.with_suffix(".tmp")
    for chunk in pd.read_csv(path, chunksize=chunksize, **READ_KW):
        table = pa.Table.from_pandas(clean_records(chunk), preserve_index=False)
        writer = writer or pq.ParquetWriter(tmp, table.schema, compression="zstd")
        writer.write_table(table)
        n += len(chunk)
    if writer:
        writer.close()
        tmp.replace(out)
    return f"{path.name}: {n:,} rows in {time.time() - t0:.0f}s"


SOURCE_FILES = [f"{split}/{split}_source{i}.tsv" for split in ("train", "test") for i in (1, 2, 3)]


def clean_all(data_dir: Path, out_dir: Path, jobs: int = 2, overwrite: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(data_dir / f, out_dir / (Path(f).stem + ".parquet")) for f in SOURCE_FILES]
    todo = [(src, dst) for src, dst in todo if overwrite or not dst.exists()]
    if not todo:
        print("all files already cleaned")
        return
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for msg in ex.map(clean_file, *zip(*todo)):
            print(msg, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    clean_all(args.data_dir, args.out_dir, args.jobs, args.overwrite)

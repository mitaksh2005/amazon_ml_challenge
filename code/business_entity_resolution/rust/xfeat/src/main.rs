//! XTRA2 pair features (P10), computed in parallel over all cores.
//!
//!   xfeat <left_store_dir> <right_store_dir> <pairs_dir>
//!
//! A store dir holds two files per record column (written by xfeatures2.XStore2):
//!   <col>.off   u64 offsets[n + 1] (little endian), <col>.dat UTF-8 bytes
//! columns: joined, initials, skel, dtok, house, nums, street, raw
//! The pairs dir holds li.u32 / ri.u32 (record positions, little endian); the result is written to
//! out.f32 there, column-major (N_FEAT columns of n_pairs values), NaN where a feature is undefined.
//!
//! String similarities use the rapidfuzz definitions: ratio = 2·LCS / (|a| + |b|) (normalised Indel
//! similarity), partial_ratio = best ratio of the shorter string against any window of the longer,
//! token_set_ratio = best ratio among (intersection, intersection + rest_a, intersection + rest_b).

use rayon::prelude::*;
use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

const COLS: [&str; 14] = [
    "x2_join_eq", "x2_join_ratio", "x2_join_partial", "x2_acr", "x2_skel_tset", "x2_dtok_tset",
    "x2_house_eq", "x2_house_trunc", "x2_addrnum_eq", "x2_addrnum_jacc", "x2_addrnum_ordered",
    "x2_script_conflict", "x2_street_ratio", "x2_street_tset",
];
const N_FEAT: usize = COLS.len();
const NAN: f32 = f32::NAN;

// ---- column store ----------------------------------------------------------------------------------
struct StrCol {
    off: Vec<u64>,
    data: Vec<u8>,
}

impl StrCol {
    fn load(dir: &Path, name: &str) -> StrCol {
        let read = |ext: &str| {
            let p = dir.join(format!("{name}.{ext}"));
            fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        };
        let off: Vec<u64> = read("off").chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        let data = read("dat");
        assert!(!off.is_empty() && *off.last().unwrap() as usize == data.len(), "{name}: bad offsets");
        assert!(std::str::from_utf8(&data).is_ok(), "{name}: not UTF-8");
        StrCol { off, data }
    }
    fn len(&self) -> usize {
        self.off.len() - 1
    }
    fn get(&self, i: usize) -> &str {
        let s = &self.data[self.off[i] as usize..self.off[i + 1] as usize];
        // whole buffer validated as UTF-8 in load, and offsets fall on string boundaries
        unsafe { std::str::from_utf8_unchecked(s) }
    }
}

struct Store {
    joined: StrCol,
    initials: StrCol,
    skel: StrCol,
    dtok: StrCol,
    house: StrCol,
    nums: StrCol,
    street: StrCol,
    script: Vec<u8>,
}

impl Store {
    fn load(dir: &Path) -> Store {
        let c = |name: &str| StrCol::load(dir, name);
        let raw = c("raw");
        let script = (0..raw.len()).into_par_iter().map(|i| script_of(raw.get(i))).collect();
        let s = Store {
            joined: c("joined"), initials: c("initials"), skel: c("skel"), dtok: c("dtok"),
            house: c("house"), nums: c("nums"), street: c("street"), script,
        };
        let n = s.joined.len();
        for col in [&s.initials, &s.skel, &s.dtok, &s.house, &s.nums, &s.street] {
            assert_eq!(col.len(), n, "{}: columns of different length", dir.display());
        }
        s
    }
}

fn read_u32(path: &Path) -> Vec<u32> {
    let b = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

// ---- scripts ---------------------------------------------------------------------------------------
/// Dominant Unicode script of a raw name: 0 none, 1 Latin (incl. accented), 2.. other blocks.
fn script_of(s: &str) -> u8 {
    let mut counts = [0u32; 16];
    for ch in s.chars() {
        let c = ch as u32;
        let k = match c {
            0x41..=0x5A | 0x61..=0x7A | 0xC0..=0x24F | 0x1E00..=0x1EFF => 1,
            0x370..=0x3FF => 2,
            0x400..=0x52F => 3,
            0x590..=0x5FF => 4,
            0x600..=0x6FF | 0x750..=0x77F => 5,
            0x900..=0x97F => 6,
            0x980..=0x9FF => 7,
            0xA00..=0xA7F => 8,
            0xA80..=0xAFF => 9,
            0xB00..=0xB7F => 10,
            0xB80..=0xBFF => 11,
            0xC00..=0xC7F => 12,
            0xC80..=0xCFF => 13,
            0xD00..=0xD7F => 14,
            0x3040..=0x30FF | 0x4E00..=0x9FFF | 0xAC00..=0xD7AF => 15,
            _ => 0,
        };
        if k > 0 {
            counts[k] += 1;
        }
    }
    let (best, n) = counts.iter().enumerate().skip(1).max_by_key(|(_, n)| **n).unwrap();
    if *n == 0 { 0 } else { best as u8 }
}

// ---- string similarity -----------------------------------------------------------------------------
/// Bit-parallel LCS pattern (Hyyrö) for a string of at most 64 chars.
struct Pattern {
    ascii: [u64; 128],
    other: Vec<(char, u64)>,
    len: usize,
}

impl Pattern {
    fn new(a: &[char]) -> Pattern {
        debug_assert!(a.len() <= 64);
        let mut p = Pattern { ascii: [0; 128], other: Vec::new(), len: a.len() };
        for (i, &c) in a.iter().enumerate() {
            let bit = 1u64 << i;
            if (c as u32) < 128 {
                p.ascii[c as usize] |= bit;
            } else if let Some(e) = p.other.iter_mut().find(|e| e.0 == c) {
                e.1 |= bit;
            } else {
                p.other.push((c, bit));
            }
        }
        p
    }
    #[inline]
    fn mask(&self, c: char) -> u64 {
        if (c as u32) < 128 {
            self.ascii[c as usize]
        } else {
            self.other.iter().find(|e| e.0 == c).map_or(0, |e| e.1)
        }
    }
    fn lcs(&self, b: &[char]) -> usize {
        if self.len == 0 {
            return 0;
        }
        let mut v: u64 = !0;
        for &c in b {
            let u = v & self.mask(c);
            v = v.wrapping_add(u) | v.wrapping_sub(u);
        }
        let m = if self.len == 64 { !0u64 } else { (1u64 << self.len) - 1 };
        (!v & m).count_ones() as usize
    }
}

fn lcs_dp(a: &[char], b: &[char]) -> usize {
    let mut prev = vec![0u32; b.len() + 1];
    let mut cur = vec![0u32; b.len() + 1];
    for &x in a {
        for (j, &y) in b.iter().enumerate() {
            cur[j + 1] = if x == y { prev[j] + 1 } else { cur[j].max(prev[j + 1]) };
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()] as usize
}

fn lcs(a: &[char], b: &[char]) -> usize {
    let (s, l) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if s.len() <= 64 { Pattern::new(s).lcs(l) } else { lcs_dp(s, l) }
}

fn ratio_c(a: &[char], b: &[char]) -> f32 {
    let n = a.len() + b.len();
    if n == 0 {
        return 1.0;
    }
    2.0 * lcs(a, b) as f32 / n as f32
}

fn chars(s: &str) -> Vec<char> {
    s.chars().collect()
}

fn ratio(a: &str, b: &str) -> f32 {
    ratio_c(&chars(a), &chars(b))
}

fn partial_ratio(a: &str, b: &str) -> f32 {
    let (a, b) = (chars(a), chars(b));
    let (s, l) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let (m, n) = (s.len(), l.len());
    if m == 0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    if m == n || m > 64 {
        return ratio_c(&s, &l);
    }
    let p = Pattern::new(&s);
    let r = |w: &[char]| 2.0 * p.lcs(w) as f32 / (m + w.len()) as f32;
    let mut best = 0f32;
    for st in 0..=(n - m) {
        best = best.max(r(&l[st..st + m]));
        if best >= 1.0 {
            return 1.0;
        }
    }
    for k in 1..m {
        best = best.max(r(&l[..k])).max(r(&l[n - k..]));
    }
    best
}

fn token_set_ratio(a: &str, b: &str) -> f32 {
    let ta: BTreeSet<&str> = a.split_whitespace().collect();
    let tb: BTreeSet<&str> = b.split_whitespace().collect();
    let inter: Vec<&str> = ta.intersection(&tb).copied().collect();
    let da: Vec<&str> = ta.difference(&tb).copied().collect();
    let db: Vec<&str> = tb.difference(&ta).copied().collect();
    if !inter.is_empty() && (da.is_empty() || db.is_empty()) {
        return 1.0;
    }
    let sect = inter.join(" ");
    let join = |rest: &Vec<&str>| {
        if sect.is_empty() { rest.join(" ") } else { format!("{} {}", sect, rest.join(" ")) }
    };
    let (t2, t3) = (join(&da), join(&db));
    let mut r = ratio(&t2, &t3);
    if !sect.is_empty() {
        r = r.max(ratio(&sect, &t2)).max(ratio(&sect, &t3));
    }
    r
}

// ---- numeric tokens --------------------------------------------------------------------------------
fn sorted_set(s: &str) -> Vec<&str> {
    let mut v: Vec<&str> = s.split_whitespace().collect();
    v.sort_unstable();
    v.dedup();
    v
}

fn inter_count(a: &[&str], b: &[&str]) -> usize {
    let (mut i, mut j, mut k) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                k += 1;
                i += 1;
                j += 1;
            }
        }
    }
    k
}

// ---- one pair --------------------------------------------------------------------------------------
fn pair(l: &Store, r: &Store, li: usize, ri: usize, out: &mut [f32]) {
    out.fill(NAN);
    let b = |x: bool| if x { 1.0 } else { 0.0 };

    let (jl, jr) = (l.joined.get(li), r.joined.get(ri));
    let both = !jl.is_empty() && !jr.is_empty();
    if both {
        out[0] = b(jl == jr);
        out[1] = ratio(jl, jr);
        out[2] = partial_ratio(jl, jr);
        let (il, ir) = (l.initials.get(li), r.initials.get(ri));
        if !il.is_empty() || !ir.is_empty() {
            out[3] = b((!il.is_empty() && jr == il) || (!ir.is_empty() && jl == ir));
        }
    }
    let (sl, sr) = (l.skel.get(li), r.skel.get(ri));
    if !sl.is_empty() && !sr.is_empty() {
        out[4] = token_set_ratio(sl, sr);
    }
    let (dl, dr) = (l.dtok.get(li), r.dtok.get(ri));
    if !dl.is_empty() && !dr.is_empty() {
        out[5] = token_set_ratio(dl, dr);
    }
    let (hl, hr) = (l.house.get(li), r.house.get(ri));
    if !hl.is_empty() && !hr.is_empty() {
        out[6] = b(hl == hr);
        out[7] = b(hl != hr && (hl.ends_with(hr) || hr.ends_with(hl) || hl.starts_with(hr) || hr.starts_with(hl)));
    }
    let (nl, nr) = (l.nums.get(li), r.nums.get(ri));
    if !nl.is_empty() && !nr.is_empty() {
        let (a, c) = (sorted_set(nl), sorted_set(nr));
        let k = inter_count(&a, &c);
        out[8] = b(a == c);
        out[9] = k as f32 / (a.len() + c.len() - k) as f32;
        out[10] = b(nl.split_whitespace().eq(nr.split_whitespace()));
    }
    let (cl, cr) = (l.script[li], r.script[ri]);
    if cl != 0 && cr != 0 {
        out[11] = b(cl != cr);
    }
    let (tl, tr) = (l.street.get(li), r.street.get(ri));
    if !tl.is_empty() && !tr.is_empty() {
        out[12] = ratio(tl, tr);
        out[13] = token_set_ratio(tl, tr);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 2 && args[1] == "--columns" {
        println!("{}", COLS.join(","));
        return;
    }
    if args.len() != 4 {
        eprintln!("usage: xfeat <left_store_dir> <right_store_dir> <pairs_dir> | xfeat --columns");
        std::process::exit(2);
    }
    let t0 = Instant::now();
    let (l, r) = rayon::join(|| Store::load(Path::new(&args[1])), || Store::load(Path::new(&args[2])));
    let pdir = Path::new(&args[3]);
    let li = read_u32(&pdir.join("li.u32"));
    let ri = read_u32(&pdir.join("ri.u32"));
    assert_eq!(li.len(), ri.len(), "li/ri length");
    let n = li.len();
    let (nl, nr) = (l.joined.len() as u32, r.joined.len() as u32);
    assert!(li.iter().all(|&i| i < nl) && ri.iter().all(|&i| i < nr), "pair position out of range");
    let t_load = t0.elapsed().as_secs_f64();

    let mut rows = vec![0f32; n * N_FEAT];
    rows.par_chunks_mut(N_FEAT).enumerate().for_each(|(k, out)| {
        pair(&l, &r, li[k] as usize, ri[k] as usize, out);
    });
    let t_feat = t0.elapsed().as_secs_f64() - t_load;

    // row-major → column-major, then one write
    let mut colmajor = vec![0f32; n * N_FEAT];
    colmajor.par_chunks_mut(n.max(1)).enumerate().for_each(|(c, col)| {
        for (k, v) in col.iter_mut().enumerate() {
            *v = rows[k * N_FEAT + c];
        }
    });
    drop(rows);
    let bytes: Vec<u8> = colmajor.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut f = fs::File::create(pdir.join("out.f32")).expect("create out.f32");
    f.write_all(&bytes).expect("write out.f32");
    eprintln!(
        "xfeat: {n} pairs, {} threads, load {t_load:.1}s, features {t_feat:.1}s ({:.0} pairs/s)",
        rayon::current_num_threads(),
        n as f64 / t_feat.max(1e-9)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratios() {
        assert!((ratio("abc", "abc") - 1.0).abs() < 1e-6);
        assert!((ratio("abcd", "abce") - 0.75).abs() < 1e-6);
        assert!((partial_ratio("bronaughseries", "bronaughseriescom") - 1.0).abs() < 1e-6);
        assert!((token_set_ratio("foot", "aide") - 0.0).abs() < 1e-6);
        assert!((token_set_ratio("a b c", "c b a d") - 1.0).abs() < 1e-6);
        let long: String = "x".repeat(80);
        assert!((ratio(&long, &long) - 1.0).abs() < 1e-6);
        assert_eq!(lcs(&chars("kitten"), &chars("sitting")), lcs_dp(&chars("kitten"), &chars("sitting")));
    }

    #[test]
    fn scripts() {
        assert_eq!(script_of("Café Léon"), 1);
        assert_eq!(script_of("राम ट्रेडर्स"), 6);
        assert_eq!(script_of("123"), 0);
    }
}

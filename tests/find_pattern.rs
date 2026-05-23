//! correctness tests for every public entry point
//!
//! each test runs against:
//!   find_pattern         runtime-dispatched (AVX2/SSE2/NEON depending on host)
//!   get_offset_neon      128-bit legacy entry point existing plugins use
//!   find_pattern_scalar  portable scalar fallback
//!
//! wherever the three should agree, we assert all three at once

use lazysimd::{find_pattern, find_pattern_scalar, get_offset_neon};

fn assert_all(data: &[u8], pattern: &str, expected: Option<usize>) {
    assert_eq!(find_pattern(data, pattern), expected, "find_pattern mismatch");
    assert_eq!(get_offset_neon(data, pattern), expected, "get_offset_neon mismatch");
    assert_eq!(find_pattern_scalar(data, pattern), expected, "find_pattern_scalar mismatch");
}

fn random_buf(seed: u64, len: usize) -> Vec<u8> {
    // xorshift, dep-free, just for filling noise
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s as u8
        })
        .collect()
}

#[test]
fn plain_match_in_middle() {
    let mut buf = random_buf(1, 1024);
    let pat_bytes = [0xCA, 0xFE, 0xBA, 0xBE, 0xDE, 0xAD];
    buf[500..506].copy_from_slice(&pat_bytes);
    assert_all(&buf, "CA FE BA BE DE AD", Some(500));
}

#[test]
fn match_with_internal_wildcards() {
    let mut buf = random_buf(2, 1024);
    let pat_bytes = [0xCA, 0xFE, 0xBA, 0xBE, 0xDE, 0xAD];
    buf[300..306].copy_from_slice(&pat_bytes);
    // wildcards on positions 2 and 4, mid-pattern bytes shouldn't matter
    assert_all(&buf, "CA FE ?? BE ?? AD", Some(300));
}

#[test]
fn match_with_leading_wildcards() {
    let mut buf = random_buf(3, 1024);
    let pat_bytes = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
    buf[800..806].copy_from_slice(&pat_bytes);
    // leading wildcards are part of the pattern, the returned offset is the start of
    // the padded pattern in the buffer, i.e. 800 where the first wildcard slot is
    assert_all(&buf, "?? ?? CC DD EE FF", Some(800));
}

#[test]
fn no_match_returns_none() {
    let buf = random_buf(4, 4096);
    // pattern unlikely to occur, two literal anchors to prevent false positives
    assert_all(&buf, "FE ED FA CE F0 0D F1 ED", None);
}

#[test]
fn match_in_last_16_bytes_regression() {
    // regression for the buffer-tail gap, the original SIMD bailed before the last
    // max(pattern_len, 16) bytes, scalar tail fallback now picks those up
    const LEN: usize = 256;
    let mut buf = random_buf(5, LEN);
    let pat_bytes = [0x12, 0x34, 0x56, 0x78];
    buf[LEN - 4..].copy_from_slice(&pat_bytes);
    assert_all(&buf, "12 34 56 78", Some(LEN - 4));
}

#[test]
fn buffer_smaller_than_simd_register() {
    let buf = [0u8, 0xAB, 0xCD, 0xEF, 0xAA];
    assert_all(&buf, "AB CD EF", Some(1));
}

#[test]
fn pattern_spans_16_byte_boundary() {
    let mut buf = random_buf(6, 4096);
    let pat_bytes: [u8; 19] = [
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22,
    ];
    buf[1000..1019].copy_from_slice(&pat_bytes);
    assert_all(
        &buf,
        "10 11 12 13 14 15 ?? 17 18 19 1A 1B 1C 1D 1E 1F 20 21 22",
        Some(1000),
    );
}

#[test]
fn buffer_shorter_than_pattern_is_none() {
    let buf = [0x12u8, 0x34, 0x56];
    assert_all(&buf, "12 34 56 78", None);
}

#[test]
fn worst_case_first_byte_recurring_no_full_match() {
    // 3 MiB with first byte at every 16-byte boundary but no full match, used to
    // SIGSEGV before the safe-bound fix
    const LEN: usize = 3 * 1024 * 1024;
    let mut buf = vec![0u8; LEN];
    for chunk in buf.chunks_mut(16) {
        chunk[0] = 0xAB;
    }
    let pat = "AB CD EF 12 34 56 78 9A";
    assert_eq!(find_pattern(&buf, pat), None);
    assert_eq!(get_offset_neon(&buf, pat), None);
}

#[test]
fn match_at_start() {
    let mut buf = vec![0u8; 4096];
    buf[..4].copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
    assert_all(&buf, "DE AD BE EF", Some(0));
}

#[test]
fn large_buffer_with_match_at_end() {
    const LEN: usize = 3 * 1024 * 1024;
    let mut buf = random_buf(7, LEN);
    let pat_bytes: [u8; 12] = [0xDE, 0xAD, 0xBE, 0xEF, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    let plant_at = LEN - 128;
    buf[plant_at..plant_at + 12].copy_from_slice(&pat_bytes);
    assert_all(&buf, "DE AD BE EF 11 22 ?? 44 55 66 77 88", Some(plant_at));
}

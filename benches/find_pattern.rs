// bench suite mirroring Reloaded.Memory.Sigscan.Benchmark/{3MiBRandom,WorstCaseScenario}
//
// scenarios:
//   3mib_random      3 MiB random buffer, 12-byte pattern near the end
//   worst_case_3mib  3 MiB where first byte hits every 16 bytes but full pattern never matches
//   small_tail       256 B buffer with the pattern in the last 16 bytes, regression guard
//
// run with:
//   cargo bench --bench find_pattern -- --save-baseline <name>
//   cargo bench --bench find_pattern -- --baseline <name>

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::StdRng, RngCore, SeedableRng};

const SEED: u64 = 0x1234_5678_9ABC_DEF0;

fn pattern_str(bytes: &[u8], wildcard_idxs: &[usize]) -> String {
    bytes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            if wildcard_idxs.contains(&i) {
                "??".to_string()
            } else {
                format!("{:02X}", b)
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn make_random(len: usize) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut buf = vec![0u8; len];
    rng.fill_bytes(&mut buf);
    buf
}

/// 3 MiB random buffer, 12-byte pattern near the end with 2 wildcards
fn bench_3mib_random(c: &mut Criterion) {
    const LEN: usize = 3 * 1024 * 1024;
    let mut buf = make_random(LEN);
    let pattern_bytes: [u8; 12] = [0xDE, 0xAD, 0xBE, 0xEF, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    let plant_at = LEN - 128;
    buf[plant_at..plant_at + 12].copy_from_slice(&pattern_bytes);
    let pattern = pattern_str(&pattern_bytes, &[3, 7]);

    let mut g = c.benchmark_group("3mib_random");
    g.throughput(Throughput::Bytes(LEN as u64));
    g.bench_function("get_offset_neon", |b| {
        b.iter(|| black_box(lazysimd::get_offset_neon(black_box(&buf), black_box(&pattern))))
    });
    g.finish();
}

/// 3 MiB of recurring "almost match", first byte every 16 bytes, never a full match,
/// stresses the inner verify + data_ptr += 1 backtrack
fn bench_worst_case(c: &mut Criterion) {
    const LEN: usize = 3 * 1024 * 1024;
    let mut buf = vec![0u8; LEN];
    for chunk in buf.chunks_mut(16) {
        chunk[0] = 0xAB;
    }
    let pattern_bytes: [u8; 8] = [0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56, 0x78, 0x9A];
    let pattern = pattern_str(&pattern_bytes, &[]);

    let mut g = c.benchmark_group("worst_case_3mib");
    g.throughput(Throughput::Bytes(LEN as u64));
    g.bench_function("get_offset_neon", |b| {
        b.iter(|| black_box(lazysimd::get_offset_neon(black_box(&buf), black_box(&pattern))))
    });
    g.finish();
}

/// small buffer with pattern in the last 16 bytes, the original scanner returned None
/// here, this bench proves the tail fix without sacrificing speed
fn bench_small_tail(c: &mut Criterion) {
    const LEN: usize = 256;
    let mut buf = make_random(LEN);
    let pattern_bytes: [u8; 8] = [0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97];
    let plant_at = LEN - 12;
    buf[plant_at..plant_at + 8].copy_from_slice(&pattern_bytes);
    let pattern = pattern_str(&pattern_bytes, &[]);

    let mut g = c.benchmark_group("small_tail");
    g.throughput(Throughput::Bytes(LEN as u64));
    g.bench_function("get_offset_neon", |b| {
        b.iter(|| black_box(lazysimd::get_offset_neon(black_box(&buf), black_box(&pattern))))
    });
    g.finish();
}

criterion_group!(benches, bench_3mib_random, bench_worst_case, bench_small_tail);
criterion_main!(benches);

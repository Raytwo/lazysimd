# lazysimd

[![crates.io](https://img.shields.io/crates/v/lazysimd.svg)](https://crates.io/crates/lazysimd)

Fast portable SIMD byte-pattern (signature) scanner for Rust. Originally a port of [uberhalit's LazySIMD](https://github.com/uberhalit) (with changes by [Sewer56](https://github.com/Sewer56)) used in [Reloaded.Memory.SigScan](https://github.com/Reloaded-Project/Reloaded.Memory.SigScan), now stable-Rust and zero-dep.

- **AVX2 / SSE2 / NEON / scalar** with automatic runtime dispatch on x86_64. NEON on aarch64. Scalar everywhere else.
- **No nightly required.**
- **Zero runtime dependencies.** No platform coupling, no proc-macros, just the scanner.

## Usage

```rust
let data: &[u8] = /* ... */;
match lazysimd::find_pattern(data, "DE AD BE EF ?? 11 22") {
    Some(offset) => println!("found at {offset}"),
    None => println!("not found"),
}
```

Patterns are space-separated hex bytes; `??` is a wildcard. An optional `0x` prefix per byte is accepted (`0xDE 0xAD ...`).

### Available entry points

| Function | Path it takes |
|---|---|
| `find_pattern(&[u8], &str) -> Option<usize>` | Runtime-dispatched (AVX2 > SSE2/NEON > scalar). Use this for new code. |
| `find_pattern_scalar(&[u8], &str) -> Option<usize>` | Forces the portable scalar implementation. |
| `get_offset_neon(&[u8], &str) -> Option<usize>` | Legacy name for the 128-bit SIMD path (NEON on aarch64, SSE2 on x86_64). Kept for backwards-compatibility. |
| `get_offset(&[u8], &str) -> Option<usize>` | Alias for `find_pattern`. |

## What's new compared to the original port

- Stable-Rust (`int_roundings` feature dropped — `div_ceil` is now stable).
- 32-byte AVX2 path on x86_64 with runtime CPU detection.
- Portable scalar fallback for non-SIMD targets.
- Buffer-tail handling — matches in the last `max(pattern_len, register)` bytes are now found.
- Correct handling of patterns whose second byte is a wildcard (`?? ?? CC DD ...`) — the original algorithm had a latent off-by-one in `build_match_indexes` that produced false rejections; the SIMD callers now trim the match table to its populated count.
- Forward-progress guard on leading-wildcard patterns to avoid infinite loops when the first-byte hit appears before the safe anchor position.
- **4× skim loop** in the no-hit fast path: when the current 16-byte window has no first-byte match, three additional windows are checked in parallel via vector OR-reduce (no `movemask` on the hot path). Per-core throughput on Apple Silicon went from ~14 GB/s to ~25 GB/s on 3 MiB random scans without affecting worst-case performance.

## License

[MPL-2.0](LICENSE).

## Credits

- [uberhalit](https://github.com/uberhalit) — original LazySIMD algorithm.
- [Sewer56](https://github.com/Sewer56) — Reloaded.Memory.SigScan / refinements.
- Raytwo — Rust port and AVX2/scalar/dispatch work.

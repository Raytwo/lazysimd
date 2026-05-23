# lazysimd

[![crates.io](https://img.shields.io/crates/v/lazysimd.svg)](https://crates.io/crates/lazysimd)

Fast SIMD byte-pattern (signature) scanner for Rust. Originally a port of [uberhalit's LazySIMD](https://github.com/uberhalit) (with changes by [Sewer56](https://github.com/Sewer56)) used in [Reloaded.Memory.SigScan](https://github.com/Reloaded-Project/Reloaded.Memory.SigScan), now stable-Rust, cross-platform, and crates.io-ready.

- **AVX2 / SSE2 / NEON / scalar** — automatic runtime dispatch on x86_64; NEON on aarch64; scalar everywhere else.
- **No nightly required.**
- **No mandatory `skyline` dependency** — pulled in only when building for the Skyline Switch target.
- **`#[from_pattern]` proc-macro** — generates lazy offset-resolving shims for Skyline plugins, unchanged from upstream.

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

### Skyline plugin example

When the crate is built for the `aarch64-skyline-switch` target, the `scan` module and the `#[from_pattern]` macro become available:

```rust,no_run
# #[cfg(all(target_arch = "aarch64", target_vendor = "switch"))]
# mod example {
#[skyline::from_offset]
extern "C" fn original_init();

#[lazysimd::from_pattern("FF 83 02 D1 FD 7B 04 A9 FD 03 01 91")]
fn init();   // resolved lazily by scanning the .text region the first time it's called
# }
```

`scan::get_text()` returns the running module's `.text` section as a `&'static [u8]`; on non-Switch targets the module isn't compiled in.

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

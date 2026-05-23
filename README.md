# lazysimd

[![crates.io](https://img.shields.io/crates/v/lazysimd.svg)](https://crates.io/crates/lazysimd)

Fast portable SIMD byte-pattern (signature) scanner for Rust. Originally a port of [uberhalit's LazySIMD](https://github.com/uberhalit) (with changes by [Sewer56](https://github.com/Sewer56)) used in [Reloaded.Memory.SigScan](https://github.com/Reloaded-Project/Reloaded.Memory.SigScan)

## Usage

```rust
let data: &[u8] = /* ... */;
match lazysimd::find_pattern(data, "DE AD BE EF ?? 11 22") {
    Some(offset) => println!("found at {offset}"),
    None => println!("not found"),
}
```

Patterns are space-separated hex bytes; `??` is a wildcard. An optional `0x` prefix per byte is accepted (`0xDE 0xAD ...`).

## License

[MPL-2.0](LICENSE).

## Credits

- [uberhalit](https://github.com/uberhalit) — original LazySIMD algorithm.
- [Sewer56](https://github.com/Sewer56) — Reloaded.Memory.SigScan / refinements.
- [Raytwo](https://github.com/Raytwo) — Rust port

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(switch)");

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let vendor = std::env::var("CARGO_CFG_TARGET_VENDOR").unwrap_or_default();
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    let is_switch = arch == "aarch64"
        && (vendor == "switch"
            || vendor == "nintendo"
            || os == "switch"
            || os == "horizon");

    if is_switch {
        println!("cargo:rustc-cfg=switch");
    }
}

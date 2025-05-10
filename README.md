# Vulkan Tutorial in Rust

This project is a Rust implementation of the Vulkan Tutorial (<https://vulkan-tutorial.com/>).

It aims to provide a clear and modern example of using the Ash crate for Vulkan bindings in Rust.

## Current Example

The main example currently implemented is `26_texture_mapping`, which demonstrates texture mapping in Vulkan.

## How to Run

1. **Ensure Vulkan SDK is installed and configured.**
    You need `glslc` (SPIR-V compiler) in your PATH for compiling shaders. This is typically included with the Vulkan SDK.

2. **Compile and Run:**
    To run the texture mapping example with detailed debug logs, execute the following command from the project root:

    ```bash
    RUST_LOG=debug cargo run --bin 26_texture_mapping
    ```

    You can change `debug` to other log levels like `info`, `warn`, or `error` as needed.
    For example, to see only informational messages:

    ```bash
    RUST_LOG=info cargo run --bin 26_texture_mapping
    ```

    If the `RUST_LOG` environment variable is not set, `pretty_env_logger` might default to a higher log level (e.g., errors only), and you might not see the `info!` or `debug!` messages from the application.

## Building Shaders

Shaders (files with `.vert` and `.frag` extensions in `assets/shaders/`) are automatically compiled into SPIR-V (`.spv`) format during the build process using `glslc` via the `build.rs` script. If you modify these shader source files, Cargo will automatically re-run `build.rs` to recompile them.

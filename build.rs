use std::path::Path;
use std::process::Command;
use walkdir::WalkDir;

fn main() {
    // Define the directory containing GLSL shaders
    let shader_dir = Path::new("assets/shaders");

    // Ensure the shader output directory exists
    if !shader_dir.exists() {
        std::fs::create_dir_all(shader_dir).expect("Failed to create assets/shaders directory");
    }

    // Tell Cargo to rerun this build script if any files in assets/shaders/ change.
    // This ensures shaders are recompiled when their source changes.
    println!("cargo:rerun-if-changed=assets/shaders/");
    // Tell Cargo to rerun this build script if build.rs itself changes.
    println!("cargo:rerun-if-changed=build.rs");

    println!(
        "Searching for shaders in: {:?}",
        shader_dir
            .canonicalize()
            .unwrap_or_else(|_| shader_dir.to_path_buf())
    );

    for entry in WalkDir::new(shader_dir)
        .min_depth(1) // Process files within shader_dir, not the directory itself
        .max_depth(1) // Only process direct children, no recursive subdirectories
        .into_iter()
        .filter_map(Result::ok) // Ignore any errors during directory traversal
        .filter(|e| e.file_type().is_file())
    // Only consider files
    {
        let source_path = entry.path();
        // Check if the file extension is .vert or .frag
        if let Some(extension_str) = source_path.extension().and_then(|s| s.to_str()) {
            if extension_str == "vert" || extension_str == "frag" {
                // Construct the output SPIR-V filename (e.g., shader.vert.spv)
                let source_filename = source_path
                    .file_name()
                    .expect("Failed to get shader source filename")
                    .to_str()
                    .expect("Shader source filename is not valid UTF-8");

                let output_filename = format!("{}.spv", source_filename);
                let output_path = shader_dir.join(&output_filename);

                println!(
                    "cargo:warning=Compiling shader: {:?} -> {:?}",
                    source_path, output_path
                );

                // Execute glslc to compile the shader
                let result = Command::new("glslc")
                    .arg(source_path) // Input GLSL file
                    .arg("-o") // Output file flag
                    .arg(&output_path) // Output SPIR-V file path
                    .status();

                match result {
                    Ok(status) => {
                        if !status.success() {
                            // If compilation fails, panic and stop the build
                            panic!(
                                "Failed to compile shader {:?}. glslc exited with code: {:?}",
                                source_path,
                                status.code()
                            );
                        }
                        println!(
                            "cargo:warning=Successfully compiled shader: {:?} to {:?}",
                            source_path, output_path
                        );
                    }
                    Err(e) => {
                        // If glslc command itself fails to run (e.g., not found)
                        panic!(
                            "Failed to execute glslc for shader {:?}. Error: {}. Ensure glslc is in your PATH.",
                            source_path, e
                        );
                    }
                }
            }
        }
    }
}

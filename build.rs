use std::path::Path;
use std::process::Command;
use walkdir::WalkDir;

fn main() {
    let shader_dir = Path::new("assets/shaders");

    // 确保 assets/shaders 目录存在
    std::fs::create_dir_all(shader_dir).expect("Failed to create assets/shaders directory");

    // 告诉 Cargo 如果 assets/shaders 目录中的任何内容发生更改，则重新运行此脚本
    println!("cargo:rerun-if-changed=assets/shaders/");
    // 告诉 Cargo 如果 build.rs 本身发生更改，则重新运行此脚本
    println!("cargo:rerun-if-changed=build.rs");

    for entry in WalkDir::new(shader_dir)
        .min_depth(1) // 确保我们处理的是 shader_dir 中的文件，而不是 shader_dir 本身
        .max_depth(1) // 限制只处理 assets/shaders 下的直接文件，不递归子目录
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
    // 只处理文件
    {
        let source_path = entry.path();
        // 检查文件扩展名是否为 "vert" 或 "frag"
        if let Some(extension_str) = source_path.extension().and_then(|s| s.to_str()) {
            if extension_str == "vert" || extension_str == "frag" {
                // 获取源文件名，例如 "shader.vert"
                let source_filename = source_path.file_name().unwrap().to_str().unwrap();

                // 输出文件名将是 "shader.vert.spv" 或 "shader.frag.spv"
                let output_filename = format!("{}.spv", source_filename);
                // 输出路径将是 assets/shaders/shader.vert.spv
                let output_path = shader_dir.join(output_filename);

                println!("Compiling shader: {:?} -> {:?}", source_path, output_path);

                let result = Command::new("glslc")
                    .arg(source_path) // 输入文件
                    .arg("-o") // 输出文件标志
                    .arg(&output_path) // 输出文件路径
                    .status();

                match result {
                    Ok(status) => {
                        if !status.success() {
                            panic!(
                                "Failed to compile shader {:?}: exit code {:?}",
                                source_path,
                                status.code()
                            );
                        }
                        println!("Successfully compiled shader to: {:?}", output_path);
                    }
                    Err(e) => {
                        panic!(
                            "Failed to execute glslc for shader {:?}: {}",
                            source_path, e
                        );
                    }
                }
            }
        }
    }
}

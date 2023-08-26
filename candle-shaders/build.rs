use std::fs::{self, OpenOptions};
use std::io::Write;

use shaderc;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let rust_types = ["u8", "u32", "i64", "f16", "f32", "f64"];
    let glsl_types = [
        "uint8_t",
        "uint32_t",
        "int64_t",
        "float16_t",
        "float32_t",
        "float64_t",
    ];
    let ops = ["add"];

    for op in ops {
        for i in 0..rust_types.len() {
            let rust_type = rust_types[i];
            let glsl_type = glsl_types[i];

            let filename = format!("src/{op}.glsl");
            let src = fs::read_to_string(&filename).unwrap();

            let compiler = shaderc::Compiler::new().unwrap();
            let mut options = shaderc::CompileOptions::new().unwrap();
            options.add_macro_definition("TYPE", Some(glsl_type));

            let binary_result = compiler
                .compile_into_spirv(
                    &src,
                    shaderc::ShaderKind::Compute,
                    &filename,
                    "main",
                    Some(&options),
                )
                .unwrap();

            let filename = format!(
                "{out_dir}/{op}_{rust_type}.spv",
                out_dir = std::env::var("OUT_DIR").unwrap()
            );

            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(filename)
                .unwrap();

            file.write_all(binary_result.as_binary_u8()).unwrap();
        }
    }

    let mut f = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open("src/lib.rs")
        .unwrap();

    for op in ops {
        for i in 0..rust_types.len() {
            let rust_type = rust_types[i];
            let glsl_type = glsl_types[i];
            writeln!(f, r#"pub const {OP}_{TY}: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/{op}_{ty}.spv"));"#,
            OP = op.to_uppercase(),
            TY = rust_type.to_uppercase(),
            op = op, ty = rust_type).unwrap();
        }
    }
}

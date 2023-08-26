use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;

use naga::valid::*;
use rayon::prelude::*;
use shaderc;
use text_placeholder::Template;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let wgsl_types = ["u32", "f32"];

    let global_context = HashMap::<&str, &str, _>::from([("", "")]);

    let operations = [("unary", [("cos", "cos(x)")])];

    let mut codes = HashMap::new();

    // Evaluate templates
    for (template_file, ops) in operations {
        for (op, func) in ops {
            let mut unary_context = global_context.clone();
            unary_context.insert("func", func);

            let filename = format!("src/templates/{template_file}.wgsl");
            let template_string = fs::read_to_string(&filename).unwrap();

            let template = Template::new(&template_string);

            for ty in wgsl_types {
                let mut type_context = unary_context.clone();
                type_context.insert("elem", ty);

                let code = template.fill_with_hashmap(&type_context);
                // dbg!(&code);
                codes.insert(format!("{op}_{ty}"), code);
            }
        }
    }

    // Compile shaders in paralell
    let compiled = codes
        .par_iter()
        .map(|(name, code)| {
            let module = naga::front::wgsl::parse_str(&code).unwrap();

            let opts = naga::back::spv::Options {
                lang_version: (1, 2),
                flags: naga::back::spv::WriterFlags::DEBUG,
                ..Default::default()
            };
            let info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            let spv = naga::back::spv::write_vec(&module, &info, &opts, None).unwrap();
            let filename = format!(
                "{out_dir}/{name}.spv",
                out_dir = std::env::var("OUT_DIR").unwrap()
            );

            fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .create(true)
                .open(filename)
                .unwrap()
                .write_all(bytemuck::cast_slice(&spv))
                .unwrap();
            (name.clone(), spv)
        })
        .collect::<Vec<_>>();

    let filename = "src/lib.rs";
    let mut f = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(filename)
        .unwrap();

    for (name, spv) in compiled.iter() {
        write!(
            f,
            "pub const {name}: &[u32] = &{spv:?};",
            name = name.to_uppercase()
        )
        .unwrap();
    }

    // Add in lib.rs
}

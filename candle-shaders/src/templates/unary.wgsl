@group(0)
@binding(0)
var<storage, read> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read_write> output: array<{{ elem }}>;

@compute
@workgroup_size(32, 32, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let id = global_id.x;

    let x = input[id];

    output[id] = {{ func }};
}

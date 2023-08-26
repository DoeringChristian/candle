use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    let vulkan = Device::new_wgpu(0)?;

    let x = Tensor::from_slice(&[1f32, 2., 3., 4.], (2, 2), &vulkan)?;

    dbg!(x.to_vec2::<f32>()?);

    Ok(())
}

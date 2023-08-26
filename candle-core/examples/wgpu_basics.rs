use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    let device = Device::new_wgpu(0)?;
    Ok(())
}

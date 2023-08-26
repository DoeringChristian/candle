#![allow(dead_code)]
use std::ops::Deref;
use std::sync::{Arc, Mutex, MutexGuard};

use futures::executor::block_on;
use wgpu::util::DeviceExt;

use crate::backend::BackendStorage;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};

#[derive(thiserror::Error, Debug)]
pub enum WgpuError {}

impl From<WgpuError> for crate::Error {
    fn from(val: WgpuError) -> Self {
        crate::Error::Cuda(Box::new(val)).bt()
    }
}

#[derive(Debug)]
pub struct Device {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}
impl Deref for Device {
    type Target = wgpu::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}
impl Device {
    fn create() -> Self {
        block_on(async {
            let instance = wgpu::Instance::default();

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await?;

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: None,
                        features: wgpu::Features::empty(),
                        limits: wgpu::Limits::downlevel_defaults(),
                    },
                    None,
                )
                .await
                .unwrap();

            let encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            Some(Self {
                instance,
                adapter,
                device,
                queue,
            })
        })
        .unwrap()
    }
}

#[derive(Debug)]
pub struct Buffer {
    dtype: DType,
    device: WgpuDevice,
    buffer: wgpu::Buffer,
}
impl Buffer {
    fn size(&self) -> usize {
        self.buffer.size() as _
    }
    fn uninit(device: &WgpuDevice, size: usize, dtype: DType) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size as _,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        Buffer {
            dtype: dtype,
            device: device.clone(),
            buffer,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WgpuDevice(Arc<Device>);
impl Deref for WgpuDevice {
    type Target = Arc<Device>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug)]
pub struct WgpuStorage(Arc<Buffer>);
impl Deref for WgpuStorage {
    type Target = Arc<Buffer>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl crate::backend::BackendStorage for WgpuStorage {
    type Device = WgpuDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        let dst = Buffer::uninit(&self.device, self.size(), self.dtype());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(&self.buffer, 0, &dst.buffer, 0, self.buffer.size());
        self.device.queue.submit(Some(encoder.finish()));
        Ok(WgpuStorage(Arc::new(dst)))
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: self.buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, self.buffer.size());

        self.device.queue.submit(Some(encoder.finish()));

        let (sender, mut receiver) = futures::channel::oneshot::channel();

        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::MaintainBase::Wait);

        block_on(async {
            receiver.try_recv().unwrap();
            let slice = buffer_slice.get_mapped_range();
            match self.dtype() {
                DType::U8 => Ok(CpuStorage::U8(Vec::from(bytemuck::cast_slice(&slice)))),
                DType::U32 => Ok(CpuStorage::U32(Vec::from(bytemuck::cast_slice(&slice)))),
                DType::I64 => Ok(CpuStorage::I64(Vec::from(bytemuck::cast_slice(&slice)))),
                DType::BF16 => Ok(CpuStorage::BF16(Vec::from(bytemuck::cast_slice(&slice)))),
                DType::F16 => Ok(CpuStorage::F16(Vec::from(bytemuck::cast_slice(&slice)))),
                DType::F32 => Ok(CpuStorage::F32(Vec::from(bytemuck::cast_slice(&slice)))),
                DType::F64 => Ok(CpuStorage::F64(Vec::from(bytemuck::cast_slice(&slice)))),
            }
        })
    }

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        todo!()
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        todo!()
    }

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        todo!()
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        todo!()
    }

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        todo!()
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        todo!()
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        todo!()
    }
    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        todo!()
    }

    fn scatter_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        todo!()
    }

    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        todo!()
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        todo!()
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        todo!()
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        todo!()
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        todo!()
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        todo!()
    }
}

impl crate::backend::BackendDevice for WgpuDevice {
    type Storage = WgpuStorage;
    fn new(_: usize) -> Result<Self> {
        Ok(Self(Arc::new(Device::create())))
    }

    fn location(&self) -> crate::DeviceLocation {
        todo!()
    }

    fn same_device(&self, _: &Self) -> bool {
        todo!()
    }

    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        todo!()
    }

    fn ones_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        todo!()
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        let slice: &[u8] = match storage {
            CpuStorage::U8(storage) => bytemuck::cast_slice(storage),
            CpuStorage::U32(storage) => bytemuck::cast_slice(storage),
            CpuStorage::I64(storage) => bytemuck::cast_slice(storage),
            CpuStorage::BF16(storage) => bytemuck::cast_slice(storage),
            CpuStorage::F16(storage) => bytemuck::cast_slice(storage),
            CpuStorage::F32(storage) => bytemuck::cast_slice(storage),
            CpuStorage::F64(storage) => bytemuck::cast_slice(storage),
        };

        let buffer = self.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: slice,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        Ok(WgpuStorage(Arc::new(Buffer {
            dtype: storage.dtype(),
            device: self.clone(),
            buffer,
        })))
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        todo!()
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        todo!()
    }
}

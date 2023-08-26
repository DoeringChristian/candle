#![allow(dead_code)]
use std::sync::Arc;

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
struct Device {}

#[derive(Debug)]
struct Buffer {}

#[derive(Debug, Clone)]
pub struct WgpuDevice(Arc<Device>);

#[derive(Debug)]
pub struct WgpuStorage(Arc<Buffer>);

impl crate::backend::BackendStorage for WgpuStorage {
    type Device = WgpuDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn dtype(&self) -> DType {
        todo!()
    }

    fn device(&self) -> &Self::Device {
        todo!()
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        todo!()
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
        todo!()
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

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage> {
        todo!()
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        todo!()
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        todo!()
    }
}

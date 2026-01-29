use candle_core::{CpuStorage, Layout, Shape, Tensor};
use std::ops::Deref;

pub struct Layer {
    pub edge2vtx: Vec<usize>,
}

impl candle_core::CustomOp1 for Layer {
    fn name(&self) -> &'static str {
        "vtx2xyz_to_edgevector"
    }

    #[allow(clippy::identity_op)]
    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let (_num_vtx, num_dim) = layout.shape().dims2()?;
        let vtx2xy = storage.as_slice::<f32>()?;
        let num_edge = self.edge2vtx.len() / 2;
        let mut edge2xy = vec![0f32; num_edge * num_dim];
        for i_edge in 0..num_edge {
            let i0_vtx = self.edge2vtx[i_edge * 2 + 0];
            let i1_vtx = self.edge2vtx[i_edge * 2 + 1];
            for i_dim in 0..num_dim {
                edge2xy[i_edge * num_dim + i_dim] += vtx2xy[i1_vtx * num_dim + i_dim];
                edge2xy[i_edge * num_dim + i_dim] -= vtx2xy[i0_vtx * num_dim + i_dim];
            }
        }
        let shape = candle_core::Shape::from_dims(&[num_edge, num_dim]);
        let storage = candle_core::WithDType::to_cpu_storage_owned(edge2xy);
        Ok((storage, shape))
    }

    /// This function takes as argument the argument `arg` used in the forward pass, the result
    /// produced by the forward operation `res` and the gradient of the result `grad_res`.
    /// The function should return the gradient of the argument.
    #[allow(clippy::identity_op)]
    fn bwd(
        &self,
        vtx2xy: &Tensor,
        _edge2xy: &Tensor,
        dw_edge2xy: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let (num_edge, num_dim) = dw_edge2xy.shape().dims2()?;
        let (num_vtx, num_dim0) = vtx2xy.shape().dims2()?;
        assert_eq!(num_dim, num_dim0);
        assert_eq!(num_edge, self.edge2vtx.len() / 2);
        // dbg!(num_edge, num_vtx);
        let dw_edge2xy = dw_edge2xy.storage_and_layout().0;
        let dw_edge2xy = match dw_edge2xy.deref() {
            candle_core::Storage::Cpu(cpu_tri2vtx) => cpu_tri2vtx.as_slice::<f32>()?,
            _ => panic!(),
        };
        let mut dw_vtx2xy = vec![0f32; num_vtx * num_dim];
        for i_edge in 0..num_edge {
            let i0_vtx = self.edge2vtx[i_edge * 2 + 0];
            let i1_vtx = self.edge2vtx[i_edge * 2 + 1];
            for i_dim in 0..num_dim {
                dw_vtx2xy[i1_vtx * num_dim + i_dim] += dw_edge2xy[i_edge * num_dim + i_dim];
                dw_vtx2xy[i0_vtx * num_dim + i_dim] -= dw_edge2xy[i_edge * num_dim + i_dim];
            }
        }
        let dw_vtx2xy = candle_core::Tensor::from_vec(
            dw_vtx2xy,
            candle_core::Shape::from_dims(&[num_vtx, num_dim]),
            &candle_core::Device::Cpu,
        )?;
        Ok(Some(dw_vtx2xy))
    }
}

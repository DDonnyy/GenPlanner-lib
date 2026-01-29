use candle_core::{CpuStorage, Layout, Shape, Tensor};
use std::ops::Deref;

pub struct Layer {
    pub elem2idx: Vec<usize>,
    pub idx2vtx: Vec<usize>,
}

impl candle_core::CustomOp1 for Layer {
    fn name(&self) -> &'static str {
        "polyloop2_to_area"
    }

    #[allow(clippy::identity_op)]
    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        assert_eq!(layout.shape().dims2()?.1, 2);
        let vtx2xy = storage.as_slice::<f32>()?;
        let elem2area =
            del_msh_core::polygon_mesh::elem2area(&self.elem2idx, &self.idx2vtx, vtx2xy);
        let shape = candle_core::Shape::from(elem2area.len());
        let storage = candle_core::WithDType::to_cpu_storage_owned(elem2area);
        Ok((storage, shape))
    }

    /// This function takes as argument the argument `arg` used in the forward pass, the result
    /// produced by the forward operation `res` and the gradient of the result `grad_res`.
    /// The function should return the gradient of the argument.
    #[allow(clippy::identity_op, clippy::needless_range_loop)]
    fn bwd(
        &self,
        vtx2xy: &Tensor,
        _area: &Tensor,
        dw_area: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let dw_area = dw_area.storage_and_layout().0;
        let dw_area = match dw_area.deref() {
            candle_core::Storage::Cpu(cpu_dw_area) => cpu_dw_area.as_slice::<f32>()?,
            _ => panic!(),
        };
        //
        let (num_vtx, two) = vtx2xy.shape().dims2()?;
        assert_eq!(two, 2);
        let vtx2xy = vtx2xy.storage_and_layout().0;
        let vtx2xy = match vtx2xy.deref() {
            candle_core::Storage::Cpu(cpu_vtx2xy) => cpu_vtx2xy.as_slice::<f32>()?,
            _ => panic!(),
        };
        //
        let mut dw_vtx2xy = vec![0f32; num_vtx * 2];
        for i_elem in 0..self.elem2idx.len() - 1 {
            let num_vtx_in_elem = self.elem2idx[i_elem + 1] - self.elem2idx[i_elem];
            for i_edge in 0..num_vtx_in_elem {
                let i0_vtx = self.idx2vtx[self.elem2idx[i_elem] + i_edge];
                let i1_vtx = self.idx2vtx[self.elem2idx[i_elem] + (i_edge + 1) % num_vtx_in_elem];
                dw_vtx2xy[i0_vtx * 2 + 0] += 0.5f32 * vtx2xy[i1_vtx * 2 + 1] * dw_area[i_elem];
                dw_vtx2xy[i1_vtx * 2 + 1] += 0.5f32 * vtx2xy[i0_vtx * 2 + 0] * dw_area[i_elem];
                dw_vtx2xy[i0_vtx * 2 + 1] -= 0.5f32 * vtx2xy[i1_vtx * 2 + 0] * dw_area[i_elem];
                dw_vtx2xy[i1_vtx * 2 + 0] -= 0.5f32 * vtx2xy[i0_vtx * 2 + 1] * dw_area[i_elem];
            }
        }
        let dw_vtx2xy = Tensor::from_vec(
            dw_vtx2xy,
            candle_core::Shape::from((num_vtx, 2)),
            &candle_core::Device::Cpu,
        )?;
        Ok(Some(dw_vtx2xy))
    }
}

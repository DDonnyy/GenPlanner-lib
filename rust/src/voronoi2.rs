use candle_core::{CpuStorage, Layout, Shape, Tensor};

pub struct Layer {
    pub vtxl2xy: Vec<f32>,
    pub vtxv2info: Vec<[usize; 4]>,
}

use nalgebra::Vector2;
use crate::polygonmesh2_to_cogs;

#[inline]
fn to_navec2(xy: &[f32], i: usize) -> Vector2<f32> {
    Vector2::new(xy[i * 2], xy[i * 2 + 1])
}


impl candle_core::CustomOp1 for Layer {
    fn name(&self) -> &'static str {
        "site2_to_volonoi2"
    }

    #[allow(clippy::identity_op)]
    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let (_num_site, num_dim) = layout.shape().dims2()?;
        assert_eq!(num_dim, 2);
        let site2xy = storage.as_slice::<f32>()?;
        let num_vtxv = self.vtxv2info.len();
        let mut vtxv2xy = vec![0f32; num_vtxv * 2];
        for i_vtxv in 0..num_vtxv {
            let cc = del_msh_core::voronoi2::position_of_voronoi_vertex(
                &self.vtxv2info[i_vtxv],
                &self.vtxl2xy[..],
                site2xy,
            );
            vtxv2xy[i_vtxv * 2 + 0] = cc[0];
            vtxv2xy[i_vtxv * 2 + 1] = cc[1];
        }
        let shape = Shape::from_dims(&[num_vtxv, 2]);
        let storage = candle_core::WithDType::to_cpu_storage_owned(vtxv2xy);
        Ok((storage, shape))
    }

    /// This function takes as argument the argument `arg` used in the forward pass, the result
    /// produced by the forward operation `res` and the gradient of the result `grad_res`.
    /// The function should return the gradient of the argument.
    #[allow(clippy::identity_op)]
    fn bwd(
        &self,
        site2xy: &Tensor,
        _vtxv2xy: &Tensor,
        dw_vtxv2xy: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let (num_site, two) = site2xy.shape().dims2()?;
        assert_eq!(two, 2);
        let (num_vtxv, two) = _vtxv2xy.shape().dims2()?;
        assert_eq!(two, 2);

        let site2xy = site2xy.storage_and_layout().0;
        let site2xy = match &*site2xy {
            candle_core::Storage::Cpu(site2xy) => site2xy.as_slice::<f32>()?,
            _ => panic!("site2xy must be on CPU"),
        };
        //
        let dw_vtxv2xy = dw_vtxv2xy.storage_and_layout().0;
        let dw_vtxv2xy = match &*dw_vtxv2xy {
            candle_core::Storage::Cpu(dw_vtxv2xy) => dw_vtxv2xy.as_slice::<f32>()?,
            _ => panic!("dw_vtxv2xy must be on CPU"),
        };
        //
        let mut dw_site2xy = vec![0f32; num_site * 2];
        for i_vtxv in 0..num_vtxv {
            let info = self.vtxv2info[i_vtxv];
            if info[1] == usize::MAX {
                // this vtxv is one of vtxl
                assert!(info[0] < self.vtxl2xy.len() / 2);
            } else if info[3] == usize::MAX {
                // intersection of loop edge and two voronoi
                let num_vtxl = self.vtxl2xy.len() / 2;
                assert!(info[0] < num_vtxl);
                let i1_loop = info[0];
                let i2_loop = (i1_loop + 1) % num_vtxl;
                let l1 = to_navec2(&self.vtxl2xy, i1_loop);
                let l2 = to_navec2(&self.vtxl2xy, i2_loop);
                let i0_site = info[1];
                let i1_site = info[2];
                let s0 = to_navec2(site2xy, i0_site);
                let s1 = to_navec2(site2xy, i1_site);
                let (_r, drds0, drds1) = del_geo_nalgebra::line2::dw_intersection_against_bisector(
                    &l1,
                    &(l2 - l1),
                    &s0,
                    &s1,
                );
                let dv = to_navec2(dw_vtxv2xy, i_vtxv);
                {
                    let ds0 = drds0.transpose() * dv;
                    dw_site2xy[i0_site * 2 + 0] += ds0.x;
                    dw_site2xy[i0_site * 2 + 1] += ds0.y;
                }
                {
                    let ds1 = drds1.transpose() * dv;
                    dw_site2xy[i1_site * 2 + 0] += ds1.x;
                    dw_site2xy[i1_site * 2 + 1] += ds1.y;
                }
            } else {
                // circumference of three voronoi vtx
                let idx_site = [info[1], info[2], info[3]];
                let s0 = to_navec2(site2xy, idx_site[0]);
                let s1 = to_navec2(site2xy, idx_site[1]);
                let s2 = to_navec2(site2xy, idx_site[2]);
                let (_v, dvds) = del_geo_nalgebra::tri2::wdw_circumcenter(&s0, &s1, &s2);
                let dv = to_navec2(dw_vtxv2xy, i_vtxv);
                for i_node in 0..3 {
                    let ds0 = dvds[i_node].transpose() * dv;
                    let is0 = idx_site[i_node];
                    dw_site2xy[is0 * 2 + 0] += ds0.x;
                    dw_site2xy[is0 * 2 + 1] += ds0.y;
                }
            }
        }
        let dw_site2xy = Tensor::from_vec(
            dw_site2xy,
            Shape::from_dims(&[num_site, 2usize]),
            &candle_core::Device::Cpu,
        )?;
        Ok(Some(dw_site2xy))
    }
}

pub struct VoronoiInfo {
    pub site2idx: Vec<usize>,
    pub idx2vtxv: Vec<usize>,
    pub idx2site: Vec<usize>,
    pub vtxv2info: Vec<[usize; 4]>,
}

impl VoronoiInfo {
    pub fn new() -> Self {
        Self {
            site2idx: Vec::new(),
            idx2vtxv: Vec::new(),
            idx2site: Vec::new(),
            vtxv2info: Vec::new(),
        }
    }
}

pub fn voronoi<F>(
    vtxl2xy: &[f32],
    site2xy: &Tensor,
    site2isalive: F,
) -> (Tensor, VoronoiInfo)
where
    F: Fn(usize) -> bool,
{
    let site2cell = del_msh_core::voronoi2::voronoi_cells(
        vtxl2xy,
        &site2xy.flatten_all().unwrap().to_vec1::<f32>().unwrap()[..],
        &site2isalive,
    );
    let voronoi_mesh = del_msh_core::voronoi2::indexing(&site2cell[..]);
    let site2_to_voronoi2 = Layer {
        vtxl2xy: vtxl2xy.to_vec(),
        vtxv2info: voronoi_mesh.vtxv2info.clone(),
    };
    let vtxv2xy = site2xy.apply_op1(site2_to_voronoi2).unwrap();
    let idx2site = del_msh_core::elem2elem::from_polygon_mesh(
        &voronoi_mesh.site2idx,
        &voronoi_mesh.idx2vtxv,
        vtxv2xy.dims2().unwrap().0,
    );
    let voronoi_info = VoronoiInfo {
        site2idx: voronoi_mesh.site2idx,
        idx2vtxv: voronoi_mesh.idx2vtxv,
        vtxv2info: voronoi_mesh.vtxv2info,
        idx2site,
    };
    (vtxv2xy, voronoi_info)
}

pub fn loss_lloyd(
    elem2idx: &[usize],
    idx2vtx: &[usize],
    site2xy: &Tensor,
    vtxv2xy: &Tensor,
) -> candle_core::Result<Tensor> {
    let polygonmesh2_to_cogs = polygonmesh2_to_cogs::Layer {
        elem2idx: elem2idx.to_vec(),
        idx2vtx: idx2vtx.to_vec(),
    };
    let site2cogs = vtxv2xy.apply_op1(polygonmesh2_to_cogs)?;
    site2xy.sub(&site2cogs)?.sqr().unwrap().sum_all()
}

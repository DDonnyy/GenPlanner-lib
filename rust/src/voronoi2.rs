use candle_core::{CpuStorage, Layout, Shape, Tensor};

pub struct Layer {
    pub boundary_xy: Vec<f32>,
    pub vtxv2info: Vec<[usize; 4]>,
}

use crate::polygonmesh2_to_cogs;
use nalgebra::Vector2;

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
        let (_num_generator_points, num_dim) = layout.shape().dims2()?;
        assert_eq!(num_dim, 2);
        let generator_points_xy = storage.as_slice::<f32>()?;
        let num_cell_vertices = self.vtxv2info.len();
        let mut voronoi_vertices_xy = vec![0f32; num_cell_vertices * 2];
        for i_vtxv in 0..num_cell_vertices {
            let cc = del_msh_core::voronoi2::position_of_voronoi_vertex(
                &self.vtxv2info[i_vtxv],
                &self.boundary_xy[..],
                generator_points_xy,
            );
            voronoi_vertices_xy[i_vtxv * 2 + 0] = cc[0];
            voronoi_vertices_xy[i_vtxv * 2 + 1] = cc[1];
        }
        let shape = Shape::from_dims(&[num_cell_vertices, 2]);
        let storage = candle_core::WithDType::to_cpu_storage_owned(voronoi_vertices_xy);
        Ok((storage, shape))
    }

    /// This function takes as argument the argument `arg` used in the forward pass, the result
    /// produced by the forward operation `res` and the gradient of the result `grad_res`.
    /// The function should return the gradient of the argument.
    #[allow(clippy::identity_op)]
    fn bwd(
        &self,
        generator_points_xy: &Tensor,
        _voronoi_vertices_xy: &Tensor,
        dw_voronoi_vertices_xy: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let (num_generator_points, two) = generator_points_xy.shape().dims2()?;
        assert_eq!(two, 2);
        let (num_cell_vertices, two) = _voronoi_vertices_xy.shape().dims2()?;
        assert_eq!(two, 2);

        let generator_points_xy = generator_points_xy.storage_and_layout().0;
        let generator_points_xy = match &*generator_points_xy {
            candle_core::Storage::Cpu(generator_points_xy) => {
                generator_points_xy.as_slice::<f32>()?
            }
            _ => panic!("generator_points_xy must be on CPU"),
        };
        //
        let dw_voronoi_vertices_xy = dw_voronoi_vertices_xy.storage_and_layout().0;
        let dw_voronoi_vertices_xy = match &*dw_voronoi_vertices_xy {
            candle_core::Storage::Cpu(dw_voronoi_vertices_xy) => {
                dw_voronoi_vertices_xy.as_slice::<f32>()?
            }
            _ => panic!("dw_voronoi_vertices_xy must be on CPU"),
        };
        //
        let mut dw_generator_points_xy = vec![0f32; num_generator_points * 2];
        for i_vtxv in 0..num_cell_vertices {
            let info = self.vtxv2info[i_vtxv];
            if info[1] == usize::MAX {
                // this vtxv is one of vtxl
                assert!(info[0] < self.boundary_xy.len() / 2);
            } else if info[3] == usize::MAX {
                // intersection of loop edge and two voronoi
                let num_boundary_vertices = self.boundary_xy.len() / 2;
                assert!(info[0] < num_boundary_vertices);
                let i1_loop = info[0];
                let i2_loop = (i1_loop + 1) % num_boundary_vertices;
                let l1 = to_navec2(&self.boundary_xy, i1_loop);
                let l2 = to_navec2(&self.boundary_xy, i2_loop);
                let i0_site = info[1];
                let i1_site = info[2];
                let s0 = to_navec2(generator_points_xy, i0_site);
                let s1 = to_navec2(generator_points_xy, i1_site);
                let (_r, drds0, drds1) = del_geo_nalgebra::line2::dw_intersection_against_bisector(
                    &l1,
                    &(l2 - l1),
                    &s0,
                    &s1,
                );
                let dv = to_navec2(dw_voronoi_vertices_xy, i_vtxv);
                {
                    let ds0 = drds0.transpose() * dv;
                    dw_generator_points_xy[i0_site * 2 + 0] += ds0.x;
                    dw_generator_points_xy[i0_site * 2 + 1] += ds0.y;
                }
                {
                    let ds1 = drds1.transpose() * dv;
                    dw_generator_points_xy[i1_site * 2 + 0] += ds1.x;
                    dw_generator_points_xy[i1_site * 2 + 1] += ds1.y;
                }
            } else {
                // circumference of three voronoi vtx
                let idx_site = [info[1], info[2], info[3]];
                let s0 = to_navec2(generator_points_xy, idx_site[0]);
                let s1 = to_navec2(generator_points_xy, idx_site[1]);
                let s2 = to_navec2(generator_points_xy, idx_site[2]);
                let (_v, dvds) = del_geo_nalgebra::tri2::wdw_circumcenter(&s0, &s1, &s2);
                let dv = to_navec2(dw_voronoi_vertices_xy, i_vtxv);
                for i_node in 0..3 {
                    let ds0 = dvds[i_node].transpose() * dv;
                    let is0 = idx_site[i_node];
                    dw_generator_points_xy[is0 * 2 + 0] += ds0.x;
                    dw_generator_points_xy[is0 * 2 + 1] += ds0.y;
                }
            }
        }
        let dw_generator_points_xy = Tensor::from_vec(
            dw_generator_points_xy,
            Shape::from_dims(&[num_generator_points, 2usize]),
            &candle_core::Device::Cpu,
        )?;
        Ok(Some(dw_generator_points_xy))
    }
}

pub struct VoronoiInfo {
    pub point2cell_idx: Vec<usize>,
    pub cell_idx2vertex_idx: Vec<usize>,
    pub idx2site: Vec<usize>,
    pub vtxv2info: Vec<[usize; 4]>,
}

impl VoronoiInfo {
    pub fn new() -> Self {
        Self {
            point2cell_idx: Vec::new(),
            cell_idx2vertex_idx: Vec::new(),
            idx2site: Vec::new(),
            vtxv2info: Vec::new(),
        }
    }
}

pub fn voronoi<F>(
    boundary_xy: &[f32],
    generator_points_xy: &Tensor,
    site2isalive: F,
) -> (Tensor, VoronoiInfo)
where
    F: Fn(usize) -> bool,
{
    let site2cell = del_msh_core::voronoi2::voronoi_cells(
        boundary_xy,
        &generator_points_xy
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[..],
        &site2isalive,
    );
    let voronoi_mesh = del_msh_core::voronoi2::indexing(&site2cell[..]);
    let site2_to_voronoi2 = Layer {
        boundary_xy: boundary_xy.to_vec(),
        vtxv2info: voronoi_mesh.vtxv2info.clone(),
    };
    let voronoi_vertices_xy = generator_points_xy.apply_op1(site2_to_voronoi2).unwrap();
    let idx2site = del_msh_core::elem2elem::from_polygon_mesh(
        &voronoi_mesh.site2idx,
        &voronoi_mesh.idx2vtxv,
        voronoi_vertices_xy.dims2().unwrap().0,
    );
    let voronoi_info = VoronoiInfo {
        point2cell_idx: voronoi_mesh.site2idx,
        cell_idx2vertex_idx: voronoi_mesh.idx2vtxv,
        vtxv2info: voronoi_mesh.vtxv2info,
        idx2site,
    };
    (voronoi_vertices_xy, voronoi_info)
}

pub fn loss_lloyd(
    cell_elem2polygon_idx: &[usize],
    idx2vtx: &[usize],
    generator_points_xy: &Tensor,
    voronoi_vertices_xy: &Tensor,
) -> candle_core::Result<Tensor> {
    let polygonmesh2_to_cogs = polygonmesh2_to_cogs::Layer {
        cell_elem2polygon_idx: cell_elem2polygon_idx.to_vec(),
        idx2vtx: idx2vtx.to_vec(),
    };
    let site2cogs = voronoi_vertices_xy.apply_op1(polygonmesh2_to_cogs)?;
    generator_points_xy
        .sub(&site2cogs)?
        .sqr()
        .unwrap()
        .sum_all()
}

mod loss_topo;
mod polygonmesh2_to_areas;
mod polygonmesh2_to_cogs;
mod voronoi2;
mod vtx2xyz_to_edgevector;

use crate::voronoi2::VoronoiInfo;

use pyo3::prelude::*;
use std::panic;

use crate::loss_topo::edge2vtvx_forbidden_wall;
use candle_core::{DType, Tensor};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};

#[pyfunction]
#[pyo3(signature = (
    boundary_xy,
    generator_points_xy,
    point2zone,
    point_fixed_mask,
    zone_target_area,
    zone_neighbors,
    zone_forbidden,
    write_logs,
    run_name
))]
fn optimize_territory_zoning(
    boundary_xy: Vec<f32>,
    generator_points_xy: Vec<f32>,
    point2zone: Vec<usize>,
    point_fixed_mask: Vec<f32>,
    zone_target_area: Vec<f32>,
    zone_neighbors: Vec<(usize, usize)>,
    zone_forbidden: Vec<(usize, usize)>,
    write_logs: bool,
    run_name: String,
) -> PyResult<Vec<f32>> {
    let result = panic::catch_unwind(|| {
        optimize_zoning(
            boundary_xy,
            generator_points_xy,
            point2zone,
            point_fixed_mask,
            zone_target_area,
            zone_neighbors,
            zone_forbidden,
            write_logs,
            run_name,
        )
    });
    match result {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(e)) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            e.to_string(),
        )),
        Err(panic_info) => {
            let panic_message = if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic occurred".to_string()
            };
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                panic_message,
            ))
        }
    }
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize_territory_zoning, m)?)?;
    Ok(())
}

pub fn edge2vtvx_wall(voronoi_info: &VoronoiInfo, point2zone: &[usize]) -> Vec<usize> {
    let point2cell_idx = &voronoi_info.point2cell_idx;
    let idx2vtxv = &voronoi_info.cell_idx2vertex_idx;
    let mut edge2vtxv = vec![0usize; 0];
    // get wall between rooms
    for i_site in 0..point2cell_idx.len() - 1 {
        let i_zone = point2zone[i_site];
        if i_zone == usize::MAX {
            continue;
        }
        let num_vtx_in_site = point2cell_idx[i_site + 1] - point2cell_idx[i_site];
        for i0_vtx in 0..num_vtx_in_site {
            let i1_vtx = (i0_vtx + 1) % num_vtx_in_site;
            let idx = point2cell_idx[i_site] + i0_vtx;
            let i0_vtxv = idx2vtxv[idx];
            let i1_vtxv = idx2vtxv[point2cell_idx[i_site] + i1_vtx];
            let j_site = voronoi_info.idx2site[idx];
            if j_site == usize::MAX {
                continue;
            }
            if i_site >= j_site {
                continue;
            }
            let j_room = point2zone[j_site];
            if i_zone == j_room {
                continue;
            }
            edge2vtxv.push(i0_vtxv);
            edge2vtxv.push(i1_vtxv);
        }
    }
    edge2vtxv
}

pub fn zone2area(
    point2zone: &[usize],
    num_zones: usize,
    point2cell_idx: &[usize],
    idx2vtxv: &[usize],
    voronoi_vertices_xy: &candle_core::Tensor,
) -> candle_core::Result<candle_core::Tensor> {
    let polygonmesh2_to_areas = polygonmesh2_to_areas::Layer {
        elem2idx: Vec::<usize>::from(point2cell_idx),
        idx2vtx: Vec::<usize>::from(idx2vtxv),
    };
    let site2areas = voronoi_vertices_xy.apply_op1(polygonmesh2_to_areas)?;
    let site2areas = site2areas.reshape((site2areas.dim(0).unwrap(), 1))?;

    let num_site = point2zone.len();
    let sum_sites_for_rooms = {
        let mut sum_sites_for_rooms = vec![0f32; num_site * num_zones];
        for i_site in 0..num_site {
            let i_zone = point2zone[i_site];
            if i_zone == usize::MAX {
                continue;
            }
            assert!(i_zone < num_zones);
            sum_sites_for_rooms[i_zone * num_site + i_site] = 1f32;
        }
        candle_core::Tensor::from_slice(
            &sum_sites_for_rooms,
            candle_core::Shape::from_dims(&[num_zones, num_site]),
            &candle_core::Device::Cpu,
        )?
    };
    sum_sites_for_rooms.matmul(&site2areas)
}

fn open_csv_with_header(path: &str, header: &str) -> anyhow::Result<BufWriter<std::fs::File>> {
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true) // <-- ключевая строка
        .open(path)?;

    let mut w = BufWriter::new(file);
    writeln!(&mut w, "{}", header)?;
    w.flush()?;
    Ok(w)
}

pub fn optimize_zoning(
    boundary_xy: Vec<f32>,         // Границы полигона
    generator_points_xy: Vec<f32>, // Движимые и зафиксированные точки
    point2zone: Vec<usize>,        // Принадлежность точки к зоне
    point_fixed_mask: Vec<f32>,
    zone_target_area: Vec<f32>,
    zone_neighbors: Vec<(usize, usize)>,
    zone_forbidden: Vec<(usize, usize)>,
    write_logs: bool,
    run_name: String,
) -> anyhow::Result<Vec<f32>> {
    let fixed_flags = point_fixed_mask.iter().filter(|&&x| x != 0.0).count();

    let num_zones = zone_target_area.len();
    let num_sites = point2zone.len();
    let mut zone_fixed_points: Vec<Vec<(f32, f32)>> = vec![vec![]; num_zones];

    for i_site in 0..num_sites {
        let zone_idx = point2zone[i_site];
        let x_flag = point_fixed_mask[i_site * 2];
        let y_flag = point_fixed_mask[i_site * 2 + 1];

        if x_flag != 0.0 || y_flag != 0.0 {
            let x = generator_points_xy[i_site * 2];
            let y = generator_points_xy[i_site * 2 + 1];
            zone_fixed_points[zone_idx].push((x, y));
        }
    }

    // +++ список “двигающихся” сайтов: у кого обе компоненты флага == 0
    let movable_sites: Vec<usize> = (0..num_sites)
        .filter(|&i| point_fixed_mask[i * 2] == 0.0 && point_fixed_mask[i * 2 + 1] == 0.0)
        .collect();

    let mut loss_writer: Option<BufWriter<std::fs::File>> = None;
    let mut sites_writer: Option<BufWriter<std::fs::File>> = None;

    if write_logs {
        let loss_path = format!("{}.csv", run_name);
        let sites_path = format!("{}_sites.csv", run_name);

        loss_writer = Some(open_csv_with_header(
            &loss_path,
            "iter,loss_each_area,loss_total_area,loss_walllen,loss_topo,loss_fix,loss_lloyd,loss_group_fix,loss_forbidden,lr",
        )?);

        sites_writer = Some(open_csv_with_header(&sites_path, "iter,site,zone,x,y")?);
    }

    let generator_points_xy = candle_core::Var::from_slice(
        &generator_points_xy,
        candle_core::Shape::from_dims(&[generator_points_xy.len() / 2, 2]),
        &candle_core::Device::Cpu,
    )
    .unwrap();

    let point_fixed_mask = candle_core::Var::from_slice(
        &point_fixed_mask,
        candle_core::Shape::from_dims(&[point_fixed_mask.len() / 2, 2]),
        &candle_core::Device::Cpu,
    )
    .unwrap();

    let generator_points_xy_ini = candle_core::Tensor::from_vec(
        generator_points_xy
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()?,
        candle_core::Shape::from_dims(&[generator_points_xy.dims2()?.0, 2usize]),
        &candle_core::Device::Cpu,
    )
    .unwrap();

    assert_eq!(point2zone.len(), generator_points_xy.dims2()?.0);

    let zone_target_area = {
        let num_zones = zone_target_area.len();
        candle_core::Tensor::from_vec(
            zone_target_area,
            candle_core::Shape::from_dims(&[num_zones, 1]),
            &candle_core::Device::Cpu,
        )
        .unwrap()
    };

    let adamw_params = candle_nn::ParamsAdamW {
        lr: 0.2,
        ..Default::default()
    };
    use candle_nn::Optimizer;
    let mut zoning_optimizer =
        candle_nn::AdamW::new(vec![generator_points_xy.clone()], adamw_params)?;

    let n_sites = point2zone.len();
    let base_iterations = 1000;
    let max_iterations = 2000;
    let mut num_iterations =
        base_iterations + ((n_sites.saturating_sub(10) * (max_iterations - base_iterations)) / 50);

    if fixed_flags > 0 {
        num_iterations = (num_iterations as f32 * 1.1).round() as usize;
    }

    let max_lr = 0.09;
    let min_lr = 0.002;
    let lr_decay_start_iter = num_iterations / 2;

    for iter_idx in 0..num_iterations {
        let learning_rate = if iter_idx < lr_decay_start_iter {
            max_lr
        } else {
            let decay_progress = (iter_idx - lr_decay_start_iter) as f32
                / (num_iterations - lr_decay_start_iter) as f32;
            min_lr
                + 0.5 * (max_lr - min_lr) * (1.0 + f32::cos(std::f32::consts::PI * decay_progress))
        };

        zoning_optimizer.set_params(candle_nn::ParamsAdamW {
            lr: learning_rate as f64,
            beta2: 0.95,
            ..Default::default()
        });

        let (voronoi_vertices_xy, voronoi_info) =
            voronoi2::voronoi(&boundary_xy, &generator_points_xy, |i_site| {
                point2zone[i_site] != usize::MAX
            });
        let edge2vtxv_wall = edge2vtvx_wall(&voronoi_info, &point2zone);

        let loss_walllen = {
            let vtx2xyz_to_edgevector = vtx2xyz_to_edgevector::Layer {
                edge2vtx: Vec::<usize>::from(edge2vtxv_wall.clone()),
            };
            let edge2xy = voronoi_vertices_xy.apply_op1(vtx2xyz_to_edgevector)?;
            edge2xy.abs()?.sum_all()?
        };

        let edge2vtxv_forbidden =
            edge2vtvx_forbidden_wall(&voronoi_info, &point2zone, &zone_forbidden);

        let loss_forbidden_walllen = {
            if edge2vtxv_forbidden.is_empty() {
                Tensor::zeros((), DType::F32, generator_points_xy.device())?
            } else {
                let op = vtx2xyz_to_edgevector::Layer {
                    edge2vtx: edge2vtxv_forbidden.clone(),
                };
                let edge2xy = voronoi_vertices_xy.apply_op1(op)?;
                // edge2xy.abs()?.sum_all()?
                edge2xy.sqr()?.sum_all()?
            }
        };

        let (loss_each_area, loss_total_area) = {
            let zone2area = zone2area(
                &point2zone,
                zone_target_area.dims2()?.0,
                &voronoi_info.point2cell_idx,
                &voronoi_info.cell_idx2vertex_idx,
                &voronoi_vertices_xy,
            )?;
            /*
            {
                let zone2area = zone2area.flatten_all()?.to_vec1::<f32>()?;
                let total_area = del_msh::polyloop2::area_(&boundary_xy);
                for i_zone in 0..zone2area.len() {
                    println!("    room:{} area:{}", i_zone, zone2area[i_zone]/total_area);
                }
            }
             */
            let loss_each_area = zone2area.sub(&zone_target_area)?.sqr()?.sum_all()?;
            let total_area_trg = del_msh_core::polyloop2::area(&boundary_xy);
            let total_area_trg = candle_core::Tensor::from_vec(
                vec![total_area_trg],
                candle_core::Shape::from_dims(&[]),
                &candle_core::Device::Cpu,
            )?;
            let loss_total_area = (zone2area.sum_all()? - total_area_trg)?.abs()?;
            (loss_each_area, loss_total_area)
        };
        // println!("  loss each_area {}", loss_each_area.to_vec0::<f32>()?);
        // println!("  loss total_area {}", loss_total_area.to_vec0::<f32>()?);

        let loss_topo = loss_topo::compute_topo_loss(
            &generator_points_xy,
            &point2zone,
            zone_target_area.dims2()?.0,
            &voronoi_info,
            &zone_neighbors,
        )?;

        let loss_group_fix = loss_topo::compute_group_fix_loss(
            &generator_points_xy,
            &zone_fixed_points,
            &point2zone,
        )?;
        // println!("  loss topo: {}", loss_topo.to_vec0::<f32>()?);
        // let loss_fix = generator_points_xy.sub(&generator_points_xy_ini)?.mul(&point_fixed_mask)?.sum_all()?;
        // let loss_fix = generator_points_xy.sub(&generator_points_xy_ini)?.mul(&point_fixed_mask)?.sum_all()?;

        let loss_fix = generator_points_xy
            .sub(&generator_points_xy_ini)?
            .mul(&point_fixed_mask)?
            .sqr()?
            .sqr()?
            .sum_all()?;

        let loss_lloyd = voronoi2::loss_lloyd(
            &voronoi_info.point2cell_idx,
            &voronoi_info.cell_idx2vertex_idx,
            &generator_points_xy,
            &voronoi_vertices_xy,
        )?;

        let loss_each_area = loss_each_area.affine(50000.0, 0.0)?.clone();
        let loss_total_area = loss_total_area.affine(100000.0, 0.0)?.clone();
        let loss_walllen = loss_walllen.affine(50.0, 0.0)?;
        let loss_topo = loss_topo.affine(150.0, 0.0)?;
        let loss_fix = loss_fix.affine(10000000., 0.0)?;
        let loss_lloyd = loss_lloyd.affine(120.0, 0.0)?;
        let loss_group_fix = loss_group_fix.affine(80.0, 0.0)?;
        let loss_forbidden_walllen = loss_forbidden_walllen.affine(2000.0, 0.0)?;

        if write_logs {
            if let Some(w) = loss_writer.as_mut() {
                writeln!(
                    w,
                    "{},{},{},{},{},{},{},{},{},{}",
                    iter_idx,
                    loss_each_area.to_vec0::<f32>()?,
                    loss_total_area.to_vec0::<f32>()?,
                    loss_walllen.to_vec0::<f32>()?,
                    loss_topo.to_vec0::<f32>()?,
                    loss_fix.to_vec0::<f32>()?,
                    loss_lloyd.to_vec0::<f32>()?,
                    loss_group_fix.to_vec0::<f32>()?,
                    loss_forbidden_walllen.to_vec0::<f32>()?,
                    learning_rate,
                )?;
            }

            if let Some(w) = sites_writer.as_mut() {
                let xy = generator_points_xy.flatten_all()?.to_vec1::<f32>()?;
                for &i_site in &movable_sites {
                    let x = xy[i_site * 2];
                    let y = xy[i_site * 2 + 1];
                    let room = point2zone[i_site];
                    writeln!(w, "{},{},{},{},{}", iter_idx, i_site, room, x, y)?;
                }
            }
        }

        let loss = (loss_each_area
            + loss_total_area
            + loss_walllen
            + loss_topo
            + loss_fix
            + loss_lloyd
            + loss_group_fix
            + loss_forbidden_walllen)?;

        zoning_optimizer.backward_step(&loss)?;
    }

    if let Some(mut w) = loss_writer {
        w.flush()?;
    }
    if let Some(mut w) = sites_writer {
        w.flush()?;
    }

    let xy = generator_points_xy.flatten_all()?.to_vec1::<f32>()?;
    let mut movable_xy = Vec::<f32>::with_capacity(movable_sites.len() * 2);
    for &i_site in &movable_sites {
        movable_xy.push(xy[i_site * 2]);
        movable_xy.push(xy[i_site * 2 + 1]);
    }
    Ok(movable_xy)
}

use std::collections::HashSet;
use crate::voronoi2::VoronoiInfo;
use candle_core::{Result, Tensor};
use nalgebra::Vector2;

fn topology(
    voronoi_info: &VoronoiInfo,
    num_room: usize,
    site2room: &[usize],
) -> (usize, Vec<usize>, Vec<Vec<usize>>) {
    let (num_group, site2group) = {
        // return j_site if it is a same room
        let siteface2adjsitesameroom = |i_site, i_face| {
            let i_room = site2room[i_site];
            if i_room == usize::MAX {
                return usize::MAX;
            }
            let j_site = voronoi_info.idx2site[voronoi_info.point2cell_idx[i_site] + i_face];
            if j_site == usize::MAX {
                return usize::MAX;
            }
            let j_room = site2room[j_site];
            assert_ne!(j_room, usize::MAX);
            if i_room != j_room {
                return usize::MAX;
            }
            return j_site;
        };
        del_msh_core::elem2group::from_polygon_mesh(
            &voronoi_info.point2cell_idx[..],
            siteface2adjsitesameroom,
        )
    };
    assert_eq!(site2group.len(), site2room.len());
    //
    let room2group = {
        let mut room2group = vec![std::collections::BTreeSet::<usize>::new(); num_room];
        for i_site in 0..site2room.len() {
            let i_room = site2room[i_site];
            if i_room == usize::MAX {
                continue;
            }
            let i_group = site2group[i_site];
            room2group[i_room].insert(i_group);
        }
        room2group
    };
    let room2group: Vec<Vec<usize>> = room2group
        .iter()
        .map(|v| v.iter().cloned().collect())
        .collect();
    (num_group, site2group, room2group)
}

pub fn inverse_map(num_group: usize, site2group: &[usize]) -> Vec<Vec<usize>> {
    let mut group2site = vec![std::collections::BTreeSet::<usize>::new(); num_group];
    for i_site in 0..site2group.len() {
        let i_group = site2group[i_site];
        if i_group == usize::MAX {
            continue;
        }
        group2site[i_group].insert(i_site);
    }
    group2site
        .iter()
        .map(|v| v.iter().cloned().collect())
        .collect()
}

/*
pub fn room2site(
    num_room: usize,
    site2room: &[usize]) -> Vec<std::collections::BTreeSet<usize>>
{
    let mut room2site = vec![std::collections::BTreeSet::<usize>::new(); num_room];
    for i_site in 0..site2room.len() {
        let i_room = site2room[i_site];
        if i_room == usize::MAX {
            continue;
        }
        room2site[i_room].insert(i_site);
    }
    room2site
}
 */

fn is_two_room_connected(
    i0_room: usize,
    i1_room: usize,
    site2room: &[usize],
    room2site: &Vec<Vec<usize>>,
    voronoi_info: &VoronoiInfo,
) -> bool {
    let mut is_connected = false;
    for &i_site in room2site[i0_room].iter() {
        for &j_site in &voronoi_info.idx2site
            [voronoi_info.point2cell_idx[i_site]..voronoi_info.point2cell_idx[i_site + 1]]
        {
            if j_site == usize::MAX {
                continue;
            }
            if site2room[j_site] != i1_room {
                continue;
            }
            is_connected = true;
            break;
        }
        if is_connected {
            break;
        }
    }
    is_connected
}

fn find_nearest_site(
    i0_room: usize,
    i1_room: usize,
    room2site: &Vec<Vec<usize>>,
    site2xy: &[f32],
) -> (usize, usize) {
    let mut pair = (0usize, 0usize);
    let mut min_dist = f32::INFINITY;
    for &i_site in room2site[i0_room].iter() {
        let pi = del_msh_core::vtx2xy::to_vec2(site2xy, i_site);
        for &j_site in room2site[i1_room].iter() {
            let pj = del_msh_core::vtx2xy::to_vec2(site2xy, j_site);
            let dist = (v2(pi) - v2(pj)).norm();
            if dist < min_dist {
                min_dist = dist;
                pair = (i_site, j_site);
            }
        }
    }
    pair
}

#[inline]
fn v2(p: &[f32; 2]) -> Vector2<f32> {
    Vector2::new(p[0], p[1])
}

pub fn compute_topo_loss(
    site2xy: &Tensor,
    site2room: &[usize],
    num_room: usize,
    voronoi_info: &VoronoiInfo,
    room_connections: &Vec<(usize, usize)>,
) -> Result<Tensor> {
    let num_site = site2xy.dims2()?.0;
    let (num_group, site2group, room2group) = topology(voronoi_info, num_room, site2room);
    let room2site = inverse_map(num_room, site2room);
    let group2site = inverse_map(num_group, &site2group);

    let site2xy0 = site2xy.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(site2xy0.len(), num_site * 2);
    let mut site2xytrg = site2xy0.clone();

    // existing logic: split-fix + required connections
    for i_room in 0..num_room {
        assert!(!room2group[i_room].is_empty());
        if room2group[i_room].len() == 1 {
            // the room is in one piece
            let rooms_to_connect: Vec<usize> = {
                let mut rooms_to_connect = vec![];
                for &(i0_room, i1_room) in room_connections.iter() {
                    if i0_room == i_room {
                        rooms_to_connect.push(i1_room);
                    } else if i1_room == i_room {
                        rooms_to_connect.push(i0_room);
                    }
                }
                rooms_to_connect
            };
            for &j_room in rooms_to_connect.iter() {
                let is_connected =
                    is_two_room_connected(i_room, j_room, site2room, &room2site, voronoi_info);
                if is_connected {
                    continue;
                }
                // println!("{} {}", i_room, j_room);
                let (i_site, j_site) = find_nearest_site(i_room, j_room, &room2site, &site2xy0);
                site2xytrg[i_site * 2 + 0] = site2xy0[j_site * 2 + 0];
                site2xytrg[i_site * 2 + 1] = site2xy0[j_site * 2 + 1];
            }
        } else {
            // the room is split
            let i_group = {
                // group to attract other groups
                let mut i_group = usize::MAX;
                for &j_group in room2group[i_room].iter() {
                    for &j_site in &group2site[j_group] {
                        // this site has cell
                        if voronoi_info.point2cell_idx[j_site + 1]
                            > voronoi_info.point2cell_idx[j_site]
                        {
                            i_group = j_group;
                            break;
                        }
                    }
                    if i_group != usize::MAX {
                        break;
                    }
                }
                i_group
            };
            if i_group == usize::MAX {
                // no cell for this room
                for ij_group in 0..room2group[i_room].len() {
                    let j_group = room2group[i_room][ij_group];
                    for &j_site in &group2site[j_group] {
                        site2xytrg[j_site * 2 + 0] = 0.5;
                        site2xytrg[j_site * 2 + 1] = 0.5;
                    }
                }
                continue;
            }
            for ij_group in 0..room2group[i_room].len() {
                let j_group = room2group[i_room][ij_group];
                if i_group == j_group {
                    continue;
                }
                for &j_site in &group2site[j_group] {
                    let pj = del_msh_core::vtx2xy::to_vec2(&site2xy0, j_site);
                    let mut dist_min = f32::INFINITY;
                    let mut pi_min = nalgebra::Vector2::<f32>::new(0., 0.);
                    for &i_site in &group2site[i_group] {
                        assert_ne!(i_site, j_site);
                        let pi = del_msh_core::vtx2xy::to_vec2(&site2xy0, i_site);
                        let dist = (v2(pi) - v2(pj)).norm();
                        if dist < dist_min {
                            pi_min = v2(pi);
                            dist_min = dist;
                        }
                    }
                    site2xytrg[j_site * 2 + 0] = pi_min[0];
                    site2xytrg[j_site * 2 + 1] = pi_min[1];
                }
            }
        }
    }

    let site2xytrg = Tensor::from_vec(
        site2xytrg,
        candle_core::Shape::from_dims(&[num_site, 2usize]),
        &candle_core::Device::Cpu,
    )?;
    (site2xy - site2xytrg).unwrap().sqr()?.sum_all()
}

pub fn compute_group_fix_loss(
    site2xy: &Tensor,
    room2fixed_sites: &Vec<Vec<(f32, f32)>>,
    site2room: &[usize],
) -> Result<Tensor> {
    let num_site = site2xy.dims2()?.0;
    let site2xy0 = site2xy.flatten_all()?.to_vec1::<f32>()?;
    let mut site2xytrg = site2xy0.clone();

    for i_site in 0..num_site {
        let room_idx = site2room[i_site];
        let fixed_sites = &room2fixed_sites[room_idx];

        if fixed_sites.is_empty() {
            continue;
        }

        let px = site2xy0[i_site * 2];
        let py = site2xy0[i_site * 2 + 1];
        let p = Vector2::new(px, py);

        let mut min_dist = f32::INFINITY;
        let mut closest = (px, py);

        for &(fx, fy) in fixed_sites {
            let fp = Vector2::new(fx, fy);
            let dist = (fp - p).norm_squared();
            if dist < min_dist {
                min_dist = dist;
                closest = (fx, fy);
            }
        }

        site2xytrg[i_site * 2] = closest.0;
        site2xytrg[i_site * 2 + 1] = closest.1;
    }

    let site2xytrg = Tensor::from_vec(
        site2xytrg,
        candle_core::Shape::from_dims(&[num_site, 2usize]),
        &candle_core::Device::Cpu,
    )?;
    (site2xy - site2xytrg).unwrap().sqr()?.sum_all()
}

pub fn edge2vtvx_forbidden_wall(
    voronoi_info: &VoronoiInfo,
    point2zone: &[usize],
    zone_forbidden: &Vec<(usize, usize)>,
) -> Vec<usize> {
    // forbidden pairs as (min,max)
    let mut forb = HashSet::<(usize, usize)>::new();
    for &(a, b) in zone_forbidden.iter() {
        if a == usize::MAX || b == usize::MAX || a == b {
            continue;
        }
        let (x, y) = if a < b { (a, b) } else { (b, a) };
        forb.insert((x, y));
    }

    let point2cell_idx = &voronoi_info.point2cell_idx;
    let idx2vtxv = &voronoi_info.cell_idx2vertex_idx;

    let mut edge2vtxv = Vec::<usize>::new();

    for i_site in 0..point2cell_idx.len() - 1 {
        let zi = point2zone[i_site];
        if zi == usize::MAX {
            continue;
        }

        let nv = point2cell_idx[i_site + 1] - point2cell_idx[i_site];
        for i0 in 0..nv {
            let i1 = (i0 + 1) % nv;
            let idx = point2cell_idx[i_site] + i0;

            let v0 = idx2vtxv[idx];
            let v1 = idx2vtxv[point2cell_idx[i_site] + i1];

            let j_site = voronoi_info.idx2site[idx];
            if j_site == usize::MAX {
                continue;
            }
            if i_site >= j_site {
                continue; // avoid duplicates
            }

            let zj = point2zone[j_site];
            if zj == usize::MAX || zi == zj {
                continue;
            }

            let (a, b) = if zi < zj { (zi, zj) } else { (zj, zi) };
            if !forb.contains(&(a, b)) {
                continue;
            }

            edge2vtxv.push(v0);
            edge2vtxv.push(v1);
        }
    }

    edge2vtxv
}
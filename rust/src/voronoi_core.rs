// Adapted from del-msh-core 0.1.39 src/voronoi2.rs (MIT license).
// Copyright (c) Nobuyuki Umetani. See rust/third_party/del-msh-core-LICENSE.

//! methods for 2D Voronoi diagram

use anyhow::{ensure, Result};

#[derive(Clone)]
pub struct Cell {
    pub vtx2xy: Vec<f32>,
    pub vtx2info: Vec<[usize; 4]>,
}

type Intersection = (f32, usize, [f32; 2], [usize; 4]);

impl Cell {
    fn is_inside(&self, p: &[f32; 2]) -> bool {
        let wn = del_msh_core::polyloop2::winding_number(&self.vtx2xy, p);
        (wn - 1.0).abs() < 0.1
    }

    fn area(&self) -> f32 {
        del_msh_core::polyloop2::area(&self.vtx2xy)
    }

    fn new_from_polyloop2(vtx2xy_in: &[f32]) -> Self {
        let vtx2info = (0..vtx2xy_in.len() / 2)
            .map(|v| [v, usize::MAX, usize::MAX, usize::MAX])
            .collect();
        Cell {
            vtx2xy: vtx2xy_in.to_owned(),
            vtx2info,
        }
    }

    fn new_empty() -> Self {
        let vtx2xy: Vec<f32> = vec![];
        let vtx2info = vec![[usize::MAX; 4]; 0];
        Cell { vtx2xy, vtx2info }
    }
}

#[inline]
fn shared_site(info0: &[usize; 4], info1: &[usize; 4]) -> Result<Option<usize>> {
    let mut shared = None;
    for k in [info0[2], info0[3]] {
        if k == usize::MAX || (k != info1[2] && k != info1[3]) || shared == Some(k) {
            continue;
        }
        ensure!(shared.is_none(), "Ambiguous Voronoi intersection topology");
        shared = Some(k);
    }
    Ok(shared)
}

fn hoge(
    vtx2xy: &[f32],
    vtx2info: &[[usize; 4]],
    vtxnews: &[Intersection],
    vtx2vtxnew: &[usize],
    vtxnew2isvisisted: &mut [bool],
) -> Result<Option<Cell>> {
    let num_vtx = vtx2xy.len() / 2;
    let mut vtx2xy_new: Vec<f32> = vec![];
    let mut vtx2info_new = vec![[usize::MAX; 4]; 0];
    let Some(i_vtx0) = vtxnew2isvisisted.iter().position(|is_visited| !is_visited) else {
        return Ok(None);
    };
    let is_new0 = true;
    let (mut i_vtx, mut is_new) = (i_vtx0, is_new0);
    let mut is_entry = true;
    let mut steps = 0;
    loop {
        steps += 1;
        ensure!(
            steps <= (num_vtx + vtxnews.len()) * 4,
            "Voronoi cell traversal did not close"
        );
        // dbg!(i_vtx, is_new, is_entry, i_vtx0, is_new0);
        if is_new {
            ensure!(
                i_vtx < vtxnews.len(),
                "Voronoi intersection index out of range"
            );
            vtx2xy_new.push(vtxnews[i_vtx].2[0]);
            vtx2xy_new.push(vtxnews[i_vtx].2[1]);
            vtx2info_new.push(vtxnews[i_vtx].3);
            vtxnew2isvisisted[i_vtx] = true;
            if is_entry {
                i_vtx = vtxnews[i_vtx].1;
                i_vtx = (i_vtx + 1) % num_vtx;
                // assert!(depth(&vtx2xy[i_vtx]) < 0., "{}", depth(&vtx2xy[i_vtx]));
                is_new = false;
                is_entry = false;
            } else {
                // Intersections are cyclic. The upstream decrement underflows
                // at zero and then indexes usize::MAX on the next pass.
                i_vtx = (i_vtx + vtxnews.len() - 1) % vtxnews.len();
                is_new = true;
                is_entry = true;
            }
        } else {
            vtx2xy_new.push(vtx2xy[i_vtx * 2]);
            vtx2xy_new.push(vtx2xy[i_vtx * 2 + 1]);
            vtx2info_new.push(vtx2info[i_vtx]);
            if vtx2vtxnew[i_vtx] == usize::MAX {
                i_vtx = (i_vtx + 1) % num_vtx;
                is_new = false;
            } else {
                i_vtx = vtx2vtxnew[i_vtx];
                is_new = true;
                is_entry = false;
            }
        }
        if i_vtx == i_vtx0 && is_new == is_new0 {
            break;
        }
    }
    Ok(Some(Cell {
        vtx2xy: vtx2xy_new,
        vtx2info: vtx2info_new,
    }))
}

/// vtx2xy should be counter-clockwise
pub fn cut_polygon_by_line(
    cell: Cell,
    line_s: &[f32; 2],
    line_n: &[f32; 2],
    i_vtx: usize,
    j_vtx: usize,
    output: &mut Vec<Cell>,
    vtxnews: &mut Vec<Intersection>,
) -> Result<()> {
    use del_geo_core::vec2::Vec2;
    // negative->inside
    let depth = |p: &[f32; 2]| p.sub(line_s).dot(line_n);
    let num_vtx = cell.vtx2xy.len() / 2;
    vtxnews.clear();
    let is_inside = {
        let line_t = del_geo_core::vec2::rotate90(line_n);
        let mut is_inside = false;
        for i0_vtx in 0..num_vtx {
            let i1_vtx = (i0_vtx + 1) % num_vtx;
            let p0 = del_msh_core::vtx2xy::to_vec2(&cell.vtx2xy, i0_vtx);
            let p1 = del_msh_core::vtx2xy::to_vec2(&cell.vtx2xy, i1_vtx);
            let d0 = depth(p0);
            if d0 < 0. {
                is_inside = true;
            }
            let d1 = depth(p1);
            ensure!(
                d0.is_finite() && d1.is_finite(),
                "Non-finite Voronoi line distance"
            );
            ensure!(
                d0 != 0. && d1 != 0.,
                "Voronoi bisector passes through a polygon vertex"
            );
            if (d0 > 0. && d1 > 0.) || (d0 < 0. && d1 < 0.) {
                continue;
            }
            let pm = p0.scale(d1 / (d1 - d0)).add(&p1.scale(d0 / (d0 - d1)));
            let t0 = line_t.dot(&pm);
            ensure!(t0.is_finite(), "Non-finite Voronoi intersection");
            //
            let info0 = cell.vtx2info[i0_vtx];
            let info1 = cell.vtx2info[i1_vtx];
            let info = if let Some(k_vtx) = shared_site(&info0, &info1)? {
                [usize::MAX, i_vtx, k_vtx, j_vtx]
            } else {
                [info0[0], i_vtx, j_vtx, usize::MAX]
            };
            //
            vtxnews.push((-t0, i0_vtx, pm, info));
        }
        vtxnews.sort_by(|a, b| a.0.total_cmp(&b.0));
        is_inside
    };
    if vtxnews.is_empty() {
        // no intersection
        if is_inside {
            output.push(cell);
        }
        return Ok(());
    }
    ensure!(
        vtxnews.len() % 2 == 0,
        "Odd number of Voronoi intersections"
    );
    let vtx2vtxnew = {
        let mut vtx2vtxnew = vec![usize::MAX; num_vtx];
        for (i_vtxnew, vtxnew) in vtxnews.iter().enumerate() {
            ensure!(
                vtx2vtxnew[vtxnew.1] == usize::MAX,
                "Repeated Voronoi intersection on one edge"
            );
            vtx2vtxnew[vtxnew.1] = i_vtxnew;
        }
        vtx2vtxnew
    };
    let mut vtxnew2isvisisted = vec![false; vtxnews.len()];
    loop {
        let c0 = hoge(
            &cell.vtx2xy,
            &cell.vtx2info,
            &vtxnews,
            &vtx2vtxnew,
            &mut vtxnew2isvisisted,
        )?;
        let Some(cell) = c0 else {
            break;
        };
        output.push(cell);
    }
    Ok(())
}

pub fn voronoi_cells<F>(vtxl2xy: &[f32], site2xy: &[f32], site2isalive: F) -> Result<Vec<Cell>>
where
    F: Fn(usize) -> bool,
{
    use del_geo_core::vec2::Vec2;
    ensure!(
        vtxl2xy.len() >= 6 && vtxl2xy.len() % 2 == 0,
        "Invalid Voronoi boundary"
    );
    ensure!(site2xy.len() % 2 == 0, "Invalid Voronoi site coordinates");
    ensure!(
        vtxl2xy.iter().chain(site2xy).all(|v| v.is_finite()),
        "Non-finite Voronoi input"
    );
    let num_site = site2xy.len() / 2;
    let mut site2cell = vec![Cell::new_empty(); num_site];
    for (i_site, pos_i) in site2xy.chunks(2).enumerate() {
        let pos_i = arrayref::array_ref![pos_i, 0, 2];
        if !site2isalive(i_site) {
            continue;
        }
        let mut cell_stack = vec![Cell::new_from_polyloop2(vtxl2xy)];
        let mut cell_stack_new = Vec::new();
        let mut intersections = Vec::new();
        for (j_site, pos_j) in site2xy.chunks(2).enumerate() {
            let pos_j = arrayref::array_ref![pos_j, 0, 2];
            if !site2isalive(j_site) {
                continue;
            }
            if j_site == i_site {
                continue;
            }
            let delta = pos_j.sub(pos_i);
            ensure!(
                delta.dot(&delta) > 1.0e-14,
                "Coincident Voronoi sites {i_site} and {j_site}"
            );
            let line_s = pos_i.add(pos_j).scale(0.5);
            let line_n = delta.normalize();
            for cell_in in cell_stack.drain(..) {
                cut_polygon_by_line(
                    cell_in,
                    &line_s,
                    &line_n,
                    i_site,
                    j_site,
                    &mut cell_stack_new,
                    &mut intersections,
                )?;
            }
            std::mem::swap(&mut cell_stack, &mut cell_stack_new);
        }
        if cell_stack.is_empty() {
            site2cell[i_site] = Cell::new_empty();
            continue;
        }
        if cell_stack.len() == 1 {
            site2cell[i_site] = cell_stack.pop().unwrap();
            continue;
        }
        let mut depthcell: Vec<(f32, usize)> = vec![];
        for (i_cell, cell) in cell_stack.iter().enumerate() {
            let is_inside = cell.is_inside(del_msh_core::vtx2xy::to_vec2(site2xy, i_site));
            let dist = if is_inside { 0. } else { 1.0 / cell.area() };
            ensure!(!dist.is_nan(), "Invalid Voronoi cell area");
            depthcell.push((dist, i_cell));
        }
        depthcell.sort_by(|a, b| a.0.total_cmp(&b.0));
        let i_cell = depthcell[0].1;
        assert!(!cell_stack[i_cell].vtx2xy.is_empty());
        site2cell[i_site] = cell_stack.swap_remove(i_cell);
    }
    Ok(site2cell)
}

pub struct VoronoiMesh {
    pub site2idx: Vec<usize>,
    pub idx2vtxv: Vec<usize>,
    pub vtxv2info: Vec<[usize; 4]>,
}

pub fn indexing(site2cell: &[Cell]) -> VoronoiMesh {
    let num_site = site2cell.len();
    let sort_info = |info: &[usize; 4]| {
        let mut tmp = [info[1], info[2], info[3]];
        tmp.sort();
        [info[0], tmp[0], tmp[1], tmp[2]]
    };
    let mut info2vtxv = std::collections::HashMap::<[usize; 4], usize>::new();
    let mut vtxv2info: Vec<[usize; 4]> = vec![];
    for cell in site2cell.iter() {
        for info in &cell.vtx2info {
            let info0 = sort_info(info);
            let i_vtxc = info2vtxv.len();
            if let std::collections::hash_map::Entry::Vacant(entry) = info2vtxv.entry(info0) {
                entry.insert(i_vtxc);
                vtxv2info.push(info0);
            }
        }
    }
    let mut site2idx = vec![0; 1];
    let mut idx2vtxc = vec![0usize; 0];
    for cell in site2cell.iter() {
        for info in &cell.vtx2info {
            let info0 = sort_info(info);
            let i_vtxv = info2vtxv.get(&info0).unwrap();
            idx2vtxc.push(*i_vtxv);
        }
        site2idx.push(idx2vtxc.len());
    }
    assert_eq!(site2idx.len(), num_site + 1);
    VoronoiMesh {
        site2idx,
        idx2vtxv: idx2vtxc,
        vtxv2info,
    }
}

pub fn position_of_voronoi_vertex(info: &[usize; 4], vtxl2xy: &[f32], site2xy: &[f32]) -> [f32; 2] {
    use del_geo_core::vec2::Vec2;
    if info[1] == usize::MAX {
        // original point
        *del_msh_core::vtx2xy::to_vec2(vtxl2xy, info[0])
    } else if info[3] == usize::MAX {
        // two points against edge
        let num_vtxl = vtxl2xy.len() / 2;
        assert!(info[0] < num_vtxl);
        let i1_loop = info[0];
        let i2_loop = (i1_loop + 1) % num_vtxl;
        let l1 = del_msh_core::vtx2xy::to_vec2(vtxl2xy, i1_loop);
        let l2 = del_msh_core::vtx2xy::to_vec2(vtxl2xy, i2_loop);
        let s1 = &del_msh_core::vtx2xy::to_vec2(site2xy, info[1]);
        let s2 = &del_msh_core::vtx2xy::to_vec2(site2xy, info[2]);
        return del_geo_core::line2::intersection(
            l1,
            &l2.sub(l1),
            &s1.add(s2).scale(0.5),
            &del_geo_core::vec2::rotate90(&s2.sub(s1)),
        );
    } else {
        // three points
        assert_eq!(info[0], usize::MAX);
        return del_geo_core::tri2::circumcenter(
            del_msh_core::vtx2xy::to_vec2(site2xy, info[1]),
            del_msh_core::vtx2xy::to_vec2(site2xy, info[2]),
            del_msh_core::vtx2xy::to_vec2(site2xy, info[3]),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{hoge, shared_site, voronoi_cells};

    #[test]
    fn shared_site_matches_set_intersection_for_duplicate_and_missing_indices() {
        use std::collections::BTreeSet;

        for a in [0, 1, 2, usize::MAX] {
            for b in [0, 1, 2, usize::MAX] {
                for c in [0, 1, 2, usize::MAX] {
                    for d in [0, 1, 2, usize::MAX] {
                        let set_a = BTreeSet::from([a, b]);
                        let set_b = BTreeSet::from([c, d]);
                        let expected: Vec<_> = set_a
                            .intersection(&set_b)
                            .copied()
                            .filter(|&index| index != usize::MAX)
                            .collect();
                        let result = shared_site(&[0, 0, a, b], &[0, 0, c, d]);
                        if expected.len() > 1 {
                            assert!(result.is_err());
                        } else {
                            assert_eq!(result.unwrap(), expected.first().copied());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn ordinary_two_site_split_succeeds() {
        let boundary = [0., 0., 1., 0., 1., 1., 0., 1.];
        let sites = [0.25, 0.5, 0.75, 0.5];
        let cells = voronoi_cells(&boundary, &sites, |_| true).unwrap();
        assert_eq!(cells.len(), 2);
        assert!(cells.iter().all(|cell| !cell.vtx2xy.is_empty()));
    }

    #[test]
    fn coincident_sites_return_error_instead_of_sort_panic() {
        let boundary = [0., 0., 1., 0., 1., 1., 0., 1.];
        let sites = [0.25, 0.5, 0.25, 0.5];
        let error = voronoi_cells(&boundary, &sites, |_| true).err().unwrap();
        assert!(error.to_string().contains("Coincident Voronoi sites"));
    }

    #[test]
    fn bisector_through_vertex_returns_error_instead_of_assertion() {
        let boundary = [0., 0., 0.5, 0., 1., 0., 1., 1., 0., 1.];
        let sites = [0.25, 0.5, 0.75, 0.5];
        let error = voronoi_cells(&boundary, &sites, |_| true).err().unwrap();
        assert!(error.to_string().contains("bisector passes through"));
    }

    #[test]
    fn traversal_at_zero_wraps_to_last_intersection() {
        let boundary = [0., 0., 1., 0., 1., 1., 0., 1.];
        let info = [[0, usize::MAX, usize::MAX, usize::MAX]; 4];
        let crossings = [(0., 0, [0.5, 0.], info[0]), (1., 2, [0.5, 1.], info[2])];
        let mapping = [0, usize::MAX, 1, usize::MAX];
        let mut visited = [true, false];
        let cell = hoge(&boundary, &info, &crossings, &mapping, &mut visited)
            .unwrap()
            .unwrap();
        assert!(!cell.vtx2xy.is_empty());
    }
}

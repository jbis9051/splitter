use crate::alg;
use crate::mask::Mask;

pub type BoundingBox = [usize; 4];
pub type Corners = [[usize; 2]; 4];
pub type Point =  [f64; 2];
pub type Hull = Vec<[usize; 2]>;

pub struct MaskOps;

impl MaskOps {

    pub fn bounding_box(mask: &Mask) -> BoundingBox {
        let mut min_x = mask.width();
        let mut min_y = mask.height();
        let mut max_x = 0;
        let mut max_y = 0;

        for y in 0..mask.height() {
            for x in 0..mask.width() {
                if !mask[y][x] {
                    continue;
                }
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
        [min_x, min_y, max_x, max_y]
    }
    
    
    pub fn bound_center(bounding_box: &BoundingBox) -> Point{
        let [min_x, min_y, max_x, max_y] = bounding_box;
        [
            (min_x + max_x) as f64 / 2.0,
            (min_y + max_y) as f64 / 2.0
        ]
    }
    
    pub fn corners(mask: &Mask, bounding_box: &BoundingBox, bound_center: Point) -> Corners {
        let [min_x, min_y, max_x, max_y] = *bounding_box;

        let center_pixel = [bound_center[0] as usize, bound_center[1] as usize];

        let mut north = center_pixel;
        let mut north_distance = 0.0;
        let mut south = center_pixel;
        let mut south_distance = 0.0;
        let mut east = center_pixel;
        let mut east_distance = 0.0;
        let mut west = center_pixel;
        let mut west_distance = 0.0;

        for y in 0..mask.height() {
            for x in 0..mask.width() {
                if !mask[y][x] {
                    continue;
                }
                let distance = alg::pixel_distance([x as f64, y as f64], bound_center);

                if y == min_y && distance > north_distance {
                    north = [x, y];
                    north_distance = distance;
                }

                if y == max_y && distance > south_distance {
                    south = [x, y];
                    south_distance = distance;
                }

                if x == max_x && distance > east_distance {
                    east = [x, y];
                    east_distance = distance;
                }

                if x == min_x && distance > west_distance {
                    west = [x, y];
                    west_distance = distance;
                }
            }
        }
        [north, east, south, west]
    }
    
    pub fn convex_hull(mask: &Mask) -> Hull {
        let mut hull = Vec::new();

        // Collect all points in the mask
        let mut points = Vec::new();
        for y in 0..mask.height() {
            for x in 0..mask.width() {
                if mask[y][x] {
                    points.push((x as i32, y as i32));
                }
            }
        }

        if points.len() < 3 {
            return hull; // Convex hull is not possible
        }

        // Find the bottom-most point (and leftmost if tie)
        let mut bottommost = 0;
        for i in 1..points.len() {
            if points[i].1 > points[bottommost].1 || (points[i].1 == points[bottommost].1 && points[i].0 < points[bottommost].0) {
                bottommost = i;
            }
        }

        let mut p = bottommost;
        loop {
            hull.push([points[p].0 as usize, points[p].1 as usize]);
            let mut q = (p + 1) % points.len();
            for i in 0..points.len() {
                if orientation(points[p], points[i], points[q]) == 2 {
                    q = i;
                }
            }
            p = q;
            if p == bottommost {
                break;
            }
        }
        hull
    }
}

fn orientation(p: (i32, i32), q: (i32, i32), r: (i32, i32)) -> i32 {
    let val = (q.1 - p.1) * (r.0 - q.0) - (q.0 - p.0) * (r.1 - q.1);
    if val == 0 {
        0 // collinear
    } else if val > 0 {
        1 // clockwise
    } else {
        2 // counterclockwise
    }
}
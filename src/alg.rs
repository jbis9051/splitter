use image::{Rgb, RgbImage};
use crate::color_distance;
use crate::mask::Mask;

pub fn find_objects(image: &RgbImage, background_color: &Rgb<u8>, sensitivity: f64) -> Vec<Mask> {
    let mut objects: Vec<Mask> = Vec::new();
    let mut discovered: Mask = Mask::new(image.height() as usize, image.width() as usize);

    for (x, y, pixel) in image.enumerate_pixels() {
        if discovered[x as usize][y as usize] {
            continue;
        }

        let distance = color_distance(pixel, background_color);
        if distance < sensitivity {
            discovered[x as usize][y as usize] = true;
            continue;
        }
        let object = search_object(&image, x, y, &mut discovered, |x, y| {
            let pixel = image.get_pixel(x, y);
            color_distance(pixel, background_color) >= sensitivity
        });

        objects.push(object);
    }
    
    objects
}

// inside is a closure that returns true if the pixel at (x, y) is inside the object
// aka, !inside(..) is the stop condition for the search
pub fn search_object(image: &RgbImage, start_x: u32, start_y: u32, discovered: &mut [Vec<bool>], inside: impl Fn(u32, u32) -> bool) -> Mask {
    let mut object: Mask = Mask::new(image.width() as usize, image.height() as usize);
    let mut stack = Vec::new();
    stack.push([start_x, start_y]);
    while let Some([x, y]) = stack.pop() {
        if x >= image.width() || y >= image.height() {
            continue;
        }
        if discovered[x as usize][y as usize] {
            continue;
        }
        discovered[x as usize][y as usize] = true;

        if !inside(x, y) {
            continue;
        }

        object[y as usize][x as usize] = true;

        stack.push([x + 1, y]);
        if x > 0 {
            stack.push([x - 1, y]);
        }
        stack.push([x, y + 1]);
        if y > 0 {
            stack.push([x, y - 1]);
        }
    }
    object
}


pub fn convex_hull(mask: &Mask) -> Vec<[usize; 2]> {
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

fn orientation(p: (i32, i32), q: (i32, i32), r: (i32, i32)) -> i32 {
    let val = (q.1 - p.1) * (r.0 - q.0) - (q.0 - p.0) * (r.1 - q.1);
    if val == 0 {
        return 0; // collinear
    } else if val > 0 {
        return 1; // clockwise
    } else {
        return 2; // counterclockwise
    }
}
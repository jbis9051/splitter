use std::cmp::max;
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

pub fn pixel_distance<T: Into<f64> + Copy, K: Into<f64> + Copy>(pixel1: [T; 2], pixel2: [K; 2]) -> f64 {
    let dx = pixel1[0].into() - pixel2[0].into();
    let dy = pixel1[1].into() - pixel2[1].into();
    (dx * dx + dy * dy).sqrt()
}

pub fn color_opacity_combine(color1: &Rgb<u8>, color2: &Rgb<u8>, opacity: f64) -> Rgb<u8> {
    let r = color1[0] as f64 * opacity + color2[0] as f64 * (1.0 - opacity);
    let g = color1[1] as f64 * opacity + color2[1] as f64 * (1.0 - opacity);
    let b = color1[2] as f64 * opacity + color2[2] as f64 * (1.0 - opacity);
    Rgb([r as u8, g as u8, b as u8])
}

pub fn draw_line(x1: u32, y1: u32, x2: u32, y2: u32, mut draw: impl FnMut(f64, f64)) {
    let dx = x2 as i32 - x1 as i32;
    let dy = y2 as i32 - y1 as i32;
    let steps = max(dx.abs(), dy.abs());
    for step in 0..steps {
        let x = x1 as f64 + (dx as f64 / steps as f64) * step as f64;
        let y = y1 as f64 + (dy as f64 / steps as f64) * step as f64;
        draw(x, y);
    }
}

pub fn fix_edge_pixels(img: &mut RgbImage, edge_radius: u32, threshold: f64) {
    // 1. we start by contracting the edge by edge_radius pixels, and then expanding until 0
    // 2. for each iteration, we check if the pixel is an edge pixel, if not we skip
    // 3. then we calculate the 3 inner neighbors gradient and see if the edge pixel is within the threshold
    // 4. if it is not, then we repeat the first inner neighbor
    for r in (0..edge_radius).rev() {
        
    }
}
use image::{RgbImage, Rgba, RgbaImage};
use crate::alg;
use crate::constants::{BLUE, GREEN, RED, YELLOW};
use crate::mask::Mask;
use crate::mask_ops::MaskOps;

pub fn debug_out(image: &RgbImage, objects: &[&Mask], background: &Mask){
    let mut overlay = image.clone();
    let mut overlay_t = RgbaImage::new(overlay.width(), overlay.height());

    // overlay red on all object pixels
    
    for object in objects.iter() {
        for y in 0..object.height() {
            for x in 0..object.width() {
                if !object.inside(x, y) {
                    continue
                }
                let (x, y) = (x as u32, y as u32);
                let initial = overlay.get_pixel(x, y);
                let new = alg::color_opacity_combine(initial, &RED, 0.6);
                overlay.put_pixel(x, y, new);
            }
        }
    }

    // overlay green on all background pixels

    for y in 0..background.height() {
        for x in 0..background.width() {
            if !background.inside(x, y) {
                continue
            }
            let (x, y) = (x as u32, y as u32);
            let initial = overlay.get_pixel(x, y);
            let new = alg::color_opacity_combine(initial, &GREEN, 0.6);
            overlay.put_pixel(x, y, new);
        }
    }
    
    let bounds = objects.iter().map(|o| MaskOps::bounding_box(o)).collect::<Vec<_>>();

    // draw bounding box

    for (_, bounds) in objects.iter().zip(&bounds) {
        let [min_x, min_y, max_x, max_y] = *bounds;
        for x in min_x..max_x {
            overlay.put_pixel(x as u32, min_y as u32, RED);
            overlay.put_pixel(x as u32, max_y as u32, RED);
        }
        for y in min_y..max_y {
            overlay.put_pixel(min_x as u32, y as u32, RED);
            overlay.put_pixel(max_x as u32, y as u32, RED);
        }
    }
    
    let centers = bounds.iter().map(|b| MaskOps::bound_center(b)).collect::<Vec<_>>();
    let corners = objects.iter().zip(bounds.iter()).zip(centers.iter()).map(|((o, b), c)| MaskOps::corners(o, b, *c)).collect::<Vec<_>>();
    
    for c in &corners {
        for i in 0..4 {
            let next = (i + 1) % 4;
            let [x1, y1] = c[i];
            let [x2, y2] = c[next];
            alg::draw_line(x1 as u32, y1 as u32, x2 as u32, y2 as u32, |x, y| {
                overlay.put_pixel(x as u32, y as u32, BLUE);
            });
        }
    }

    // draw corners

    for c in corners {
        for [x, y] in c {
            overlay.put_pixel(x as u32, y as u32, YELLOW);
        }
    }

    // draw our transparent overlay

    for object in objects {
        for y in 0..object.height() {
            for x in 0..object.width() {
                if !object.inside(x, y) {
                    continue
                }
                let (x, y) = (x as u32, y as u32);
                let pixel = image.get_pixel(x, y);
                overlay_t.put_pixel(x, y, Rgba([pixel[0], pixel[1], pixel[2], 255]));
            }
        }
    }

    overlay.save("overlay.png").unwrap();
    overlay_t.save("overlay-t.png").unwrap();
}
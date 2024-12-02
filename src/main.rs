mod constants;
mod mask;
mod alg;

use std::cmp::max;
use std::collections::HashSet;
use image::{RgbImage, Rgb, GenericImageView, RgbaImage, Rgba};
use crate::constants::{BLUE, GREEN, RED, WHITE, YELLOW};
use crate::mask::Mask;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!("Usage: {} --debug <image> [sensitivity]", args[0]);
        std::process::exit(1);
    }

    let debug = args[1] == "--debug";
    let path = if debug {
        &args[2]
    } else {
        &args[1]
    };

    let sensitivity = if debug {
        args.get(3).map(|s| s.parse().unwrap()).unwrap_or(50.0)
    } else {
        args.get(2).map(|s| s.parse().unwrap()).unwrap_or(50.0)
    };

    let image = image::open(path).unwrap();
    let rgb = image.into_rgb8();

    let start = std::time::Instant::now();

    // first we find each object using a simple flood fill algorithm
    let objects = alg::find_objects(&rgb, &WHITE, sensitivity);
    
    // we then find the bounding box for each object for two purposes:
    // 1. to filter out small objects
    // 2. to find the corners of the rotated image (context hull finds us many points because the scan isn't perfect, but we just want the four corners)
    let bounding_boxes: Vec<[usize; 4]> = objects.iter().map(|object| {
        let mut min_x = usize::MAX;
        let mut min_y = usize::MAX;
        let mut max_x = 0;
        let mut max_y = 0;

        for y in 0..object.height() {
            for x in 0..object.width() {
                if !object.inside(x, y) {
                    continue;
                }
                if x < min_x {
                    min_x = x;
                }
                if x > max_x {
                    max_x = x;
                }
                if y < min_y {
                    min_y = y;
                }
                if y > max_y {
                    max_y = y;
                }
            }
        }
        [min_x, min_y, max_x, max_y]
    }).collect();
    
    let objects: Vec<(_, _)> = objects.into_iter().zip(bounding_boxes).filter(|(_, [min_x, min_y, max_x, max_y])| {
        let width = max_x - min_x;
        let height = max_y - min_y;
        width > 30 && height > 30
    }).collect();
    
    // now we use the context hull algorithm to get the full object
    // because the original object search algorithm isn't perfect and won't include parts of the objects that are the same color as the background
    let objects: Vec<(_, _, _)> = objects.iter().map(|(object, bounding_box)| {
        let convex = alg::convex_hull(object);
        
        let mut convex_fill = Mask::new(object.width(), object.height());
        
        // first we connect all the points of the convex hull
        for i in 0..convex.len() {
            let next = (i + 1) % convex.len();
            let [x1, y1] = convex[i];
            let [x2, y2] = convex[next];
            draw_line(x1 as u32, y1 as u32, x2 as u32, y2 as u32, |x, y| {
                convex_fill[y.floor() as usize][x.floor() as usize] = true;
            });
        }
        
        // now we want to flood fill, but we need a starting point
        // we can assume that the centroid of the bounding box is inside the object
        let [min_x, min_y, max_x, max_y] = *bounding_box;
        let x = (min_x as f64 + max_x as f64) / 2.0;
        let y = (min_y as f64 + max_y as f64) / 2.0;
        
        // now we flood fill
        let mut discovered: Mask = Mask::new(rgb.height() as usize, rgb.width() as usize);
        let object = alg::search_object(&rgb, x as u32, y as u32, &mut discovered, |x, y| {
            !convex_fill.inside(x as usize, y as usize)
        });
        let object = Mask::combine(&[&object, &convex_fill]); // otherwise, we're shopping off the convex hull
        (object, bounding_box, [x, y])
    }).collect::<Vec<_>>();
    
    // now we need to find the corners of the rotated image
    // this basically finds the pixel furthest from the centroid that touches the bounding box, it's a bit hacky
    let objects = objects.into_iter().map(|(object,bounding_box, center)| {
        let center_pixel = [center[0] as usize, center[1] as usize];
        let [min_x, min_y, max_x, max_y] = *bounding_box;

        let mut north = center_pixel;
        let mut north_distance = 0f64;
        let mut south = center_pixel;
        let mut south_distance = 0f64;
        let mut east = center_pixel;
        let mut east_distance = 0f64;
        let mut west = center_pixel;
        let mut west_distance = 0f64;

        for y in 0..object.height() {
            for x in 0..object.width() {
                if !object.inside(x, y) {
                    continue;
                }

                // north
                if y == min_y {
                    let distance = pixel_distance([x as f64, y as f64], center);
                    if distance > north_distance {
                        north = [x, y];
                        north_distance = distance;
                    }
                }
                // south
                if y == max_y {
                    let distance = pixel_distance([x as f64, y as f64], center);
                    if distance > south_distance {
                        south = [x, y];
                        south_distance = distance;
                    }
                }

                // east
                if x == max_x {
                    let distance = pixel_distance([x as f64, y as f64], center);
                    if distance > east_distance {
                        east = [x, y];
                        east_distance = distance;
                    }
                }

                // west
                if x == min_x {
                    let distance = pixel_distance([x as f64, y as f64], center);
                    if distance > west_distance {
                        west = [x, y];
                        west_distance = distance;
                    }
                }
            }
            
        }

        let corners = [north, east, south, west];
        
        (object, [min_x, min_y, max_x, max_y], corners)
    }).collect::<Vec<_>>();
    
    let mut background = Mask::combine(&objects.iter().map(|(object, _, _)| object).collect::<Vec<_>>());
    background.invert();


    let elapsed = start.elapsed();
    println!("Found {} objects in {:?}", objects.len(), elapsed);

    // overlay red on all object pixels

    let mut overlay = rgb.clone();
    let mut overlay_t = RgbaImage::new(overlay.width(), overlay.height());

    for (object, _, _) in &objects {
        for y in 0..object.height() {
            for x in 0..object.width() {
                if !object.inside(x, y) {
                    continue
                }
                let (x, y) = (x as u32, y as u32);
                let initial = overlay.get_pixel(x, y);
                let new = color_opacity_combine(initial, &RED, 0.6);
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
            let new = color_opacity_combine(initial, &GREEN, 0.6);
            overlay.put_pixel(x, y, new);
        }
    }

    // draw bounding box

    for (_, [min_x, min_y, max_x, max_y], _) in &objects {
        for x in *min_x..*max_x {
            overlay.put_pixel(x as u32, *min_y as u32, RED);
            overlay.put_pixel(x as u32, *max_y as u32, RED);
        }
        for y in *min_y..*max_y {
            overlay.put_pixel(*min_x as u32, y as u32, RED);
            overlay.put_pixel(*max_x as u32, y as u32, RED);
        }
    }

    for (_, _, corners) in &objects {
        for i in 0..4 {
            let next = (i + 1) % 4;
            let [x1, y1] = corners[i];
            let [x2, y2] = corners[next];
            draw_line(x1 as u32, y1 as u32, x2 as u32, y2 as u32, |x, y| {
                overlay.put_pixel(x as u32, y as u32, BLUE);
            });
        }
    }

    // draw corners

    for (_, _, corners) in &objects {
        for [x, y] in corners {
            overlay.put_pixel(*x as u32, *y as u32, YELLOW);
        }
    }

    // draw our transparent overlay

    for (object, _, _) in &objects {
        for y in 0..object.height() {
            for x in 0..object.width() {
                if !object.inside(x, y) {
                    continue
                }
                let (x, y) = (x as u32, y as u32);
                let pixel = rgb.get_pixel(x, y);
                overlay_t.put_pixel(x, y, Rgba([pixel[0], pixel[1], pixel[2], 255]));
            }
        }
    }

    // now we need to rotate each object to be upright

    for (i, (_, _, corners)) in objects.iter().enumerate() {
        let start = std::time::Instant::now();
        let north = [corners[0][0] as f64, corners[0][1] as f64];
        let east = [corners[1][0] as f64, corners[1][1] as f64];
        let south = [corners[2][0] as f64, corners[2][1] as f64];
        let west = [corners[3][0] as f64, corners[3][1] as f64];

        let turning_clockwise = west[1] < east[1];

        let [upper_left, upper_right, lower_right, lower_left] = if turning_clockwise {
            [west, north, east, south]
        } else {
            [north, east, south, west]
        };

        let width = f64_max(pixel_distance(upper_left, upper_right), pixel_distance(lower_left, lower_right));
        let height = f64_max(pixel_distance(upper_left, lower_left), pixel_distance(upper_right, lower_right));
        let width_pixels = width.ceil() as u32;
        let height_pixels = height.ceil() as u32;

        let mut rotated_nn_interpolated = RgbImage::new(width_pixels, height_pixels);
        let mut rotated_bi_interpolated = RgbImage::new(width_pixels, height_pixels);

        let x_angle = pixel_angle_for_rotation(upper_left, upper_right);
        let y_angle = pixel_angle_for_rotation(upper_left, lower_left);


        for x in 0..width_pixels {
            for y in 0..height_pixels {
                let x_percent = x as f64 / width_pixels as f64;
                let y_percent = y as f64 / height_pixels as f64;

                let x_start = calculate_point_with_vector_and(upper_left, x_angle.0, x_angle.1, x_percent * width);
                let [x_orig, y_orig] = calculate_point_with_vector_and(x_start, y_angle.0, y_angle.1, y_percent * height);

                rotated_nn_interpolated.put_pixel(x, y, nearest_neighbor(&rgb, x_orig, y_orig));
                rotated_bi_interpolated.put_pixel(x, y, bilinear_interpolation(&rgb, &background, x_orig, y_orig));
            }
        }

        println!("Finished object {} in {:?}", i, start.elapsed());

        rotated_nn_interpolated.save(format!("{}-nn-interpolated.png", i)).unwrap();
        rotated_bi_interpolated.save(format!("{}-bi-interpolated.png", i)).unwrap();
    }


    overlay.save("overlay.png").unwrap();
    overlay_t.save("overlay-t.png").unwrap();
}

fn color_distance(color1: &Rgb<u8>, color2: &Rgb<u8>) -> f64 {
    // euclidean distance between two colors
    let r = color1[0] as f64 - color2[0] as f64;
    let g = color1[1] as f64 - color2[1] as f64;
    let b = color1[2] as f64 - color2[2] as f64;
    ((r * r) + (g * g) + (b * b)).sqrt()
}

fn pixel_distance<T: Into<f64> + Copy, K: Into<f64> + Copy>(pixel1: [T; 2], pixel2: [K; 2]) -> f64 {
    let dx = pixel1[0].into() - pixel2[0].into();
    let dy = pixel1[1].into() - pixel2[1].into();
    (dx * dx + dy * dy).sqrt()
}

fn pixel_angle<T: Into<f64> + Copy, K: Into<f64> + Copy>(pixel1: [T; 2], pixel2: [K; 2]) -> f64 {
    f64::atan2(pixel2[1].into() - pixel1[1].into(), pixel2[0].into() - pixel1[0].into())
}

fn pixel_angle_for_rotation<T: Into<f64> + Copy, K: Into<f64> + Copy>(pixel1: [T; 2], pixel2: [K; 2]) -> (f64, f64) {
    let dx = pixel2[0].into() - pixel1[0].into();
    let dy = pixel2[1].into() - pixel1[1].into();
    let distance = (dx * dx + dy * dy).sqrt();
    (dx / distance, dy / distance)
}

fn color_opacity_combine(color1: &Rgb<u8>, color2: &Rgb<u8>, opacity: f64) -> Rgb<u8> {
    let r = color1[0] as f64 * opacity + color2[0] as f64 * (1.0 - opacity);
    let g = color1[1] as f64 * opacity + color2[1] as f64 * (1.0 - opacity);
    let b = color1[2] as f64 * opacity + color2[2] as f64 * (1.0 - opacity);
    Rgb([r as u8, g as u8, b as u8])
}

fn draw_line(x1: u32, y1: u32, x2: u32, y2: u32, mut draw: impl FnMut(f64, f64)) {
    let dx = x2 as i32 - x1 as i32;
    let dy = y2 as i32 - y1 as i32;
    let steps = max(dx.abs(), dy.abs());
    for step in 0..steps {
        let x = x1 as f64 + (dx as f64 / steps as f64) * step as f64;
        let y = y1 as f64 + (dy as f64 / steps as f64) * step as f64;
        draw(x, y);
    }
}

fn rotate_point<T: Into<f64> + Copy, K: Into<f64> + Copy>(point: [T; 2], center: [K; 2], angle: f64) -> [f64; 2] {
    let x = point[0].into() - center[0].into();
    let y = point[1].into() - center[1].into();
    let x_rotated = x * angle.cos() - y * angle.sin();
    let y_rotated = x * angle.sin() + y * angle.cos();
    [x_rotated + center[0].into(), y_rotated + center[1].into()]
}

fn rotate_point_with<T: Into<f64> + Copy, K: Into<f64> + Copy>(point: [T; 2], center: [K; 2], cos_theta: f64, sin_theta: f64) -> [f64; 2] {
    let x = point[0].into() - center[0].into();
    let y = point[1].into() - center[1].into();
    let x_rotated = x * cos_theta - y * sin_theta;
    let y_rotated = x * sin_theta + y * cos_theta;
    [x_rotated + center[0].into(), y_rotated + center[1].into()]
}

fn calculate_point_with_vector<T: Into<f64> + Copy>(point: [T; 2], angle: f64, distance: f64) -> [f64; 2] {
    calculate_point_with_vector_and(point, angle.cos(), angle.sin(), distance)
}

fn calculate_point_with_vector_and<T: Into<f64> + Copy>(point: [T; 2], cos_theta: f64, sin_theta: f64, distance: f64) -> [f64; 2] {
    let x = point[0].into();
    let y = point[1].into();
    let x_rotated = x + distance * cos_theta;
    let y_rotated = y + distance * sin_theta;
    [x_rotated, y_rotated]
}

fn nearest_neighbor(image: &RgbImage, x: f64, y: f64) -> Rgb<u8> {
    let x = x.round() as u32;
    let y = y.round() as u32;
    *image.get_pixel(x, y)
}

fn bilinear_interpolation(image: &RgbImage, background: &Mask, x: f64, y: f64) -> Rgb<u8> {
    // for fractional coordinates, we use bilinear interpolation
    // take the nearest pixel to the fractional coordinate
    // then take the neighboring pixel and weight them based on the distance
    // ignore pixels that are not in the object

    let (x_floor, y_floor) = (x.floor() as u32, y.floor() as u32);
    let (x_floor_u, y_floor_u) = (x_floor as usize, y_floor as usize);
    let (x_ceil, y_ceil) = (x.ceil() as u32, y.ceil() as u32);
    let (x_ceil_u, y_ceil_u) = (x_ceil as usize, y_ceil as usize);

    let (x_frac, y_frac) = (x - x_floor as f64, y - y_floor as f64);

    let mut transparent = 0f64;

    let mut r = 0f64;
    let mut g = 0f64;
    let mut b = 0f64;

    let mut sampled = false;

    if !background.inside(x_floor_u, y_floor_u) {
        sampled = true;
        let pixel = image.get_pixel(x_floor, y_floor);
        r += pixel[0] as f64 * (1.0 - x_frac) * (1.0 - y_frac);
        g += pixel[1] as f64 * (1.0 - x_frac) * (1.0 - y_frac);
        b += pixel[2] as f64 * (1.0 - x_frac) * (1.0 - y_frac);
    } else {
        transparent += (1.0 - x_frac) * (1.0 - y_frac);
    }

    if !background.inside(x_ceil_u, y_floor_u) {
        sampled = true;
        let pixel = image.get_pixel(x_ceil, y_floor);
        r += pixel[0] as f64 * x_frac * (1.0 - y_frac);
        g += pixel[1] as f64 * x_frac * (1.0 - y_frac);
        b += pixel[2] as f64 * x_frac * (1.0 - y_frac);
    } else {
        transparent += x_frac * (1.0 - y_frac);
    }

    if !background.inside(x_floor_u, y_ceil_u) {
        sampled = true;
        let pixel = image.get_pixel(x_floor, y_ceil);
        r += pixel[0] as f64 * (1.0 - x_frac) * y_frac;
        g += pixel[1] as f64 * (1.0 - x_frac) * y_frac;
        b += pixel[2] as f64 * (1.0 - x_frac) * y_frac;
    } else {
        transparent += (1.0 - x_frac) * y_frac;
    }

    if !background.inside(x_ceil_u, y_ceil_u) {
        sampled = true;
        let pixel = image.get_pixel(x_ceil, y_ceil);
        r += pixel[0] as f64 * x_frac * y_frac;
        g += pixel[1] as f64 * x_frac * y_frac;
        b += pixel[2] as f64 * x_frac * y_frac;
    } else {
        transparent += x_frac * y_frac;
    }

    if transparent > 0.0 {
        r += r * transparent;
        g += g * transparent;
        b += b * transparent;
    }

    if !sampled {
        return WHITE;
    }

    Rgb([r as u8, g as u8, b as u8])
}

fn f64_max(a: f64, b: f64) -> f64 {
    if a > b {
        a
    } else {
        b
    }
}



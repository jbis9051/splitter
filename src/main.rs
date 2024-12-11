mod constants;
mod mask;
mod alg;
mod debug;
mod mask_ops;

use std::fmt;
use std::fmt::{Display};
use std::path::Path;
use image::{RgbImage, Rgb, GenericImageView};
use crate::constants::{WHITE};
use crate::mask::Mask;
use crate::mask_ops::MaskOps;
use clap::{Parser, ValueEnum};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct CliArgs {
    #[arg(short, long)]
    debug: bool,
    #[arg(index = 1)]
    image: String,
    #[arg(short, long, default_value_t = 50.0)]
    sensitivity: f64,
    #[arg(short, long, default_value = "bilinear")]
    interpolation: Interpolation
}

#[derive(Debug, Clone, ValueEnum)]
#[clap(rename_all = "snake_case")]
pub enum Interpolation {
    #[clap(alias = "nn")]
    NearestNeighbor,
    #[clap(alias = "b")]
    Bilinear
}

impl Display for Interpolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}



fn main() {
    let args: CliArgs = CliArgs::parse();
    let CliArgs { debug, image: path, sensitivity, interpolation } = args;

    let image = image::open(&path).unwrap();
    let rgb = image.into_rgb8();

    let start = std::time::Instant::now();

    // first we find each object using a simple flood fill algorithm
    let objects = alg::find_objects(&rgb, &WHITE, sensitivity);
    let bounds = objects.iter().map(|o| MaskOps::bounding_box(o)).collect::<Vec<_>>();
    
    // we filter out small objects
    let objects: Vec<(_, _)> = objects.into_iter().zip(bounds).filter(|(object,bounds)| {
        let [min_x, min_y, max_x, max_y] = *bounds;
        let width = max_x - min_x;
        let height = max_y - min_y;
        width > 30 && height > 30
    }).collect();
    
    // now we use the convex hull algorithm to get the full object
    // because the original object search algorithm isn't perfect and won't include parts of the objects that are the same color as the background
    let mut objects: Vec<(_, _)> = objects.iter().map(|(object, bounding_box)| {
        let convex = MaskOps::convex_hull(object);
        
        let mut convex_fill = Mask::new(object.width(), object.height());
        
        // first we connect all the points of the convex hull
        for i in 0..convex.len() {
            let next = (i + 1) % convex.len();
            let [x1, y1] = convex[i];
            let [x2, y2] = convex[next];
            alg::draw_line(x1 as u32, y1 as u32, x2 as u32, y2 as u32, |x, y| {
                convex_fill[y.ceil() as usize][x.ceil() as usize] = true;
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
        let object = Mask::combine(&[&object, &convex_fill]); // otherwise, we're chopping off the convex hull
        (object, bounding_box)
    }).collect::<Vec<_>>();
    
    let mut background = Mask::combine(&objects.iter().map(|(object, _)| object).collect::<Vec<_>>());
    background.invert();


    let elapsed = start.elapsed();
    println!("Found {} objects in {:?}", objects.len(), elapsed);
    
    
    if debug {
        println!("Debug Enabled: outputting debug images");
        let objects = objects.iter().map(|(object, _)| object).collect::<Vec<_>>();
        debug::debug_out(&rgb, &objects, &background);
    }
    
    

    // now we need to rotate each object to be upright
    
    let file_name = Path::new(&path).file_stem().unwrap().to_str().unwrap();
    

    for (i, (object, bounding_box)) in objects.iter().enumerate() {
        let corners = MaskOps::corners(object, bounding_box, MaskOps::bound_center(bounding_box));
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

        let width = f64_max(alg::pixel_distance(upper_left, upper_right), alg::pixel_distance(lower_left, lower_right));
        let height = f64_max(alg::pixel_distance(upper_left, lower_left), alg::pixel_distance(upper_right, lower_right));
        let width_pixels = width.ceil() as u32;
        let height_pixels = height.ceil() as u32;
        
        let mut rotated_interpolated = RgbImage::new(width_pixels, height_pixels);
        
        let x_angle = pixel_angle_for_rotation(upper_left, upper_right);
        let y_angle = pixel_angle_for_rotation(upper_left, lower_left);


        for x in 0..width_pixels {
            for y in 0..height_pixels {
                let x_percent = x as f64 / width_pixels as f64;
                let y_percent = y as f64 / height_pixels as f64;

                let x_start = calculate_point_with_vector_and(upper_left, x_angle.0, x_angle.1, x_percent * width);
                let [x_orig, y_orig] = calculate_point_with_vector_and(x_start, y_angle.0, y_angle.1, y_percent * height);

                match interpolation {
                    Interpolation::NearestNeighbor => {
                        rotated_interpolated.put_pixel(x, y, nearest_neighbor(&rgb, x_orig, y_orig));
                    }
                    Interpolation::Bilinear => {
                        rotated_interpolated.put_pixel(x, y, bilinear_interpolation(&rgb, &background, x_orig, y_orig));
                    }
                }
            }
        }

        println!("Finished object {} in {:?}", i, start.elapsed());
        
        rotated_interpolated.save(format!("{}-{}.png",file_name, i)).unwrap();
    }
}

fn color_distance(color1: &Rgb<u8>, color2: &Rgb<u8>) -> f64 {
    // euclidean distance between two colors
    let r = color1[0] as f64 - color2[0] as f64;
    let g = color1[1] as f64 - color2[1] as f64;
    let b = color1[2] as f64 - color2[2] as f64;
    ((r * r) + (g * g) + (b * b)).sqrt()
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



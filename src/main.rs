use std::cmp::max;
use std::collections::HashSet;
use image::{RgbImage, Rgb, GenericImageView};

const WHITE: Rgb<u8> = Rgb([255, 255, 255]);
const RED: Rgb<u8> = Rgb([255, 0, 0]);

const GREEN: Rgb<u8> = Rgb([0, 255, 0]);

const BLUE: Rgb<u8> = Rgb([0, 0, 255]);

const YELLOW: Rgb<u8> = Rgb([255, 255, 0]);

const ORANGE: Rgb<u8> = Rgb([255, 165, 0]);

const PURPLE: Rgb<u8> = Rgb([128, 0, 128]);

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!("Usage: {} <image> [sensitivity]", args[0]);
        std::process::exit(1);
    }

    let path = &args[1];

    let sensitivity = if args.len() > 2 {
        args[2].parse().unwrap()
    } else {
        50.0
    };

    let image = image::open(path).unwrap();

    let rgb = image.into_rgb8();

    let start = std::time::Instant::now();

    let mut objects = Vec::new();
    let mut discovered = vec![vec![false; rgb.height() as usize]; rgb.width() as usize];

    for (x, y, pixel) in rgb.enumerate_pixels() {
        if discovered[x as usize][y as usize] {
            continue;
        }

        let distance = color_distance(pixel, &WHITE);
        if distance < sensitivity {
            discovered[x as usize][y as usize] = true;
            continue;
        }
        let object = search_object(&rgb, x, y, &mut discovered, |x, y| {
            let pixel = rgb.get_pixel(x, y);
            color_distance(pixel, &WHITE) >= sensitivity
        });

        objects.push(object);
    }

    let bounding_boxes: Vec<[u32; 4]> = objects.iter().map(|object| {
        let mut min_x = u32::MAX;
        let mut min_y = u32::MAX;
        let mut max_x = 0;
        let mut max_y = 0;
        for [x, y] in object {
            if *x < min_x {
                min_x = *x;
            }
            if *x > max_x {
                max_x = *x;
            }
            if *y < min_y {
                min_y = *y;
            }
            if *y > max_y {
                max_y = *y;
            }
        }
        [min_x, min_y, max_x, max_y]
    }).collect();

    // filter out small objects

    let objects: Vec<(_, _)> = objects.into_iter().zip(bounding_boxes).filter(|(_, [min_x, min_y, max_x, max_y])| {
        let width = max_x - min_x;
        let height = max_y - min_y;
        width > 30 && height > 30
    }).collect();

    // lets create a set of pixels that are part of the object
    let mut object_pixels = vec![HashSet::new(); objects.len()];

    for (i, (object, _)) in objects.iter().enumerate() {
        for [x, y] in object {
            object_pixels[i].insert([*x, *y]);
        }
    }

    // now we want to find all background pixels
    // we begin by finding a background pixel outside of the bounding boxes

    let background = {
        let mut background = None;
        for x in 0..rgb.width() {
            for y in 0..rgb.height() {
                let mut is_background = true;
                for (_, [min_x, min_y, max_x, max_y]) in &objects {
                    if x >= *min_x && x <= *max_x && y >= *min_y && y <= *max_y {
                        is_background = false;
                        break;
                    }
                }
                if is_background {
                    background = Some([x, y]);
                    break;
                }
            }
        }
        if let Some([x, y]) = background {
            [x, y]
        } else {
            panic!("No background pixel found");
        }
    };

    // now expand out

    let background_color = rgb.get_pixel(background[0], background[1]);

    assert_eq!(color_distance(background_color, &WHITE), 0.0); // for now

    let mut discovered = vec![vec![false; rgb.height() as usize]; rgb.width() as usize];


    let background  = search_object(&rgb, background[0], background[1], &mut discovered, |x, y| {
        for pixels in &object_pixels {
            if pixels.contains(&[x, y]) {
                return false;
            }
        }
        true
    });

    let background_set = HashSet::from_iter(background.clone());

    // find the center of each object

    let objects = objects.into_iter().map(|(object, [min_x, min_y, max_x, max_y])| {
        // first we find the centeroid

        let [x_sum, y_sum] = object.iter().fold([0, 0], |[x_sum, y_sum], [x, y]| {
            [x_sum + *x as u64, y_sum + *y as u64]
        });
        let center = [x_sum as f64 / object.len() as f64, y_sum as f64 / object.len() as f64];

        (object, [min_x, min_y, max_x, max_y], center)
    }).collect::<Vec<_>>();

    // now we need to find the corners of the rotated image

    let objects = objects.into_iter().map(|(object, [min_x, min_y, max_x, max_y], center)| {
        let center_pixel = [center[0] as u32, center[1] as u32];

        let mut north = center_pixel;
        let mut north_distance = 0f64;
        let mut south = center_pixel;
        let mut south_distance = 0f64;
        let mut east = center_pixel;
        let mut east_distance = 0f64;
        let mut west = center_pixel;
        let mut west_distance = 0f64;

        for [x, y] in &object {
            // north
            if *y == min_y {
                let distance = pixel_distance([*x, *y], center);
                if distance > north_distance {
                    north = [*x, *y];
                    north_distance = distance;
                }
            }
            // south
            if *y == max_y {
                let distance = pixel_distance([*x, *y], center);
                if distance > south_distance {
                    south = [*x, *y];
                    south_distance = distance;
                }
            }

            // east
            if *x == max_x {
                let distance = pixel_distance([*x, *y], center);
                if distance > east_distance {
                    east = [*x, *y];
                    east_distance = distance;
                }
            }

            // west
            if *x == min_x {
                let distance = pixel_distance([*x, *y], center);
                if distance > west_distance {
                    west = [*x, *y];
                    west_distance = distance;
                }
            }
        }

        let corners = [north, east, south, west];

        (object, [min_x, min_y, max_x, max_y], corners)
    }).collect::<Vec<_>>();



    let elapsed = start.elapsed();
    println!("Found {} objects in {:?}", objects.len(), elapsed);


    // overlay red on all object pixels

    let mut overlay = rgb.clone();

    for (object, _, _) in &objects {
        for [x, y] in object {
            let initial = overlay.get_pixel(*x, *y);
            let new = color_opacity_combine(initial, &RED, 0.6);
            overlay.put_pixel(*x, *y, new);
        }
    }

    // overlay green on all background pixels

    for [x, y] in background {
        let initial = overlay.get_pixel(x, y);
        let new = color_opacity_combine(initial, &GREEN, 0.8);
        overlay.put_pixel(x, y, new);
    }

    // draw bounding box

    for (_, [min_x, min_y, max_x, max_y], _) in &objects {
        for x in *min_x..*max_x {
            overlay.put_pixel(x, *min_y, RED);
            overlay.put_pixel(x, *max_y, RED);
        }
        for y in *min_y..*max_y {
            overlay.put_pixel(*min_x, y, RED);
            overlay.put_pixel(*max_x, y, RED);
        }
    }

    // highlight object pixels that intersect with bounding box

    for (object, [min_x, min_y, max_x, max_y], _) in &objects {
        for [x, y] in object {
            if x == min_x || x == max_x || y == min_y || y == max_y {
                overlay.put_pixel(*x, *y, GREEN);
            }
        }
    }

    // draw lines between corners

    for (_, _, corners) in &objects {
        for i in 0..4 {
            let next = (i + 1) % 4;
            let [x1, y1] = corners[i];
            let [x2, y2] = corners[next];
            draw_line(&mut overlay, x1, y1, x2, y2, BLUE);
        }
    }

    // draw corners

    for (_, _, corners) in &objects {
        for [x, y] in corners {
            overlay.put_pixel(*x, *y, YELLOW);
        }
    }

    // now we need to rotate each object to be upright

    for (i, (_, _, corners)) in objects.iter().enumerate() {
        let start = std::time::Instant::now();
        let north = corners[0];
        let east = corners[1];
        let south = corners[2];
        let west = corners[3];

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

        let mut rotated = RgbImage::new(width_pixels, height_pixels);
        let mut rotated_interpolated = RgbImage::new(width_pixels, height_pixels);

        let x_angle = pixel_angle_for_rotation(upper_left, upper_right);
        let y_angle = pixel_angle_for_rotation(upper_left, lower_left);


        for x in 0..width_pixels {
            for y in 0..height_pixels {
                let x_percent = x as f64 / width_pixels as f64;
                let y_percent = y as f64 / height_pixels as f64;

                let x_start = calculate_point_with_vector_and(upper_left, x_angle.0, x_angle.1, x_percent * width);
                let [x_orig, y_orig] = calculate_point_with_vector_and(x_start, y_angle.0, y_angle.1, y_percent * height);


                let (x_floor, y_floor) = (x_orig.floor() as u32, y_orig.floor() as u32);
                let (x_ceil, y_ceil) = (x_orig.ceil() as u32, y_orig.ceil() as u32);

                //overlay.put_pixel(x_floor, y_floor, ORANGE);
                //overlay.put_pixel(x_ceil, y_ceil, ORANGE);
                //overlay.put_pixel(x_floor, y_ceil, ORANGE);
                //overlay.put_pixel(x_ceil, y_floor, ORANGE);


                //  if x == 0 || y == 0 || x == width_pixels - 1 || y == height_pixels - 1 {
                //    overlay.put_pixel(x_orig as u32, y_orig as u32, ORANGE);
                //}

                rotated.put_pixel(x, y, *rgb.get_pixel(x_orig as u32, y_orig as u32));

                let interpolated = bilinear_interpolation(&rgb, &background_set, x_orig, y_orig);
                rotated_interpolated.put_pixel(x, y, interpolated);
            }
        }

        println!("Finished object {} in {:?}", i, start.elapsed());

        rotated.save(format!("{}.png", i)).unwrap();
        rotated_interpolated.save(format!("{}-interpolated.png", i)).unwrap();
    }

    overlay.save("overlay.png").unwrap();
}

fn search_object(image: &RgbImage, start_x: u32, start_y: u32, discovered: &mut [Vec<bool>], inside: impl Fn(u32, u32) -> bool) -> Vec<[u32; 2]> {
    let mut object = Vec::new();
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

        object.push([x, y]);

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

fn draw_line(image: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32, color: Rgb<u8>) {
    let dx = x2 as i32 - x1 as i32;
    let dy = y2 as i32 - y1 as i32;
    let steps = max(dx.abs(), dy.abs());
    for step in 0..steps {
        let x = x1 as f64 + (dx as f64 / steps as f64) * step as f64;
        let y = y1 as f64 + (dy as f64 / steps as f64) * step as f64;
        image.put_pixel(x as u32, y as u32, color);
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

fn bilinear_interpolation(image: &RgbImage, background: &HashSet<[u32; 2]>, x: f64, y: f64) -> Rgb<u8> {
    // for fractional coordinates, we use bilinear interpolation
    // take the nearest pixel to the fractional coordinate
    // then take the neighboring pixel and weight them based on the distance
    // ignore pixels that are not in the object

    let (x_floor, y_floor) = (x.floor() as u32, y.floor() as u32);
    let (x_ceil, y_ceil) = (x.ceil() as u32, y.ceil() as u32);

    let (x_frac, y_frac) = (x - x_floor as f64, y - y_floor as f64);

    let mut transparent = 0f64;

    let mut r = 0f64;
    let mut g = 0f64;
    let mut b = 0f64;

    let mut sampled = false;

    if !background.contains(&[x_floor, y_floor]) {
        sampled = true;
        let pixel = image.get_pixel(x_floor, y_floor);
        r += pixel[0] as f64 * (1.0 - x_frac) * (1.0 - y_frac);
        g += pixel[1] as f64 * (1.0 - x_frac) * (1.0 - y_frac);
        b += pixel[2] as f64 * (1.0 - x_frac) * (1.0 - y_frac);
    } else {
        transparent += (1.0 - x_frac) * (1.0 - y_frac);
    }

    if !background.contains(&[x_ceil, y_floor]) {
        sampled = true;
        let pixel = image.get_pixel(x_ceil, y_floor);
        r += pixel[0] as f64 * x_frac * (1.0 - y_frac);
        g += pixel[1] as f64 * x_frac * (1.0 - y_frac);
        b += pixel[2] as f64 * x_frac * (1.0 - y_frac);
    } else {
        transparent += x_frac * (1.0 - y_frac);
    }

    if !background.contains(&[x_floor, y_ceil]) {
        sampled = true;
        let pixel = image.get_pixel(x_floor, y_ceil);
        r += pixel[0] as f64 * (1.0 - x_frac) * y_frac;
        g += pixel[1] as f64 * (1.0 - x_frac) * y_frac;
        b += pixel[2] as f64 * (1.0 - x_frac) * y_frac;
    } else {
        transparent += (1.0 - x_frac) * y_frac;
    }

    if !background.contains(&[x_ceil, y_ceil]) {
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



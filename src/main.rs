use std::cmp::max;
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
    // iterate over the pixels, and print x, y, and the pixel value

    for (x, y, pixel) in rgb.enumerate_pixels() {
        if discovered[x as usize][y as usize] {
            continue;
        }

        let distance = color_distance(pixel, &WHITE);
        if distance < sensitivity {
            discovered[x as usize][y as usize] = true;
            continue;
        }
        let object = search_object(&rgb, x, y, &mut discovered, sensitivity, WHITE);
        objects.push(object);
    }

    let bounding_boxes: Vec<[u32;4]> = objects.iter().map(|object| {
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

    let objects: Vec<(_, _)> = objects.into_iter().zip(bounding_boxes).filter(|(_, [min_x,min_y,max_x, max_y])| {
        let width = max_x - min_x;
        let height = max_y - min_y;
        width > 30 && height > 30
    }).collect();

    // now we need to find the corners of the rotated image

    let objects = objects.into_iter().map(|(object, [min_x, min_y, max_x, max_y])| {
        let mut north = [max_x, 0];
        let mut south = [min_x, 0];
        let mut east = [0, max_y];
        let mut west = [0, min_y];

        for [x, y] in &object {
            // north
            if *y == min_y && x < &north[0] {
                north = [*x, *y];
            }
            // south
            if *y == max_y && x > &south[0] {
                south = [*x, *y];
            }

            // east
            if *x == max_x && y < &east[1] {
                east = [*x, *y];
            }

            // west
            if *x == min_x && y > &west[1] {
                west = [*x, *y];
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
            let new = color_opacity_combine(initial, &RED, 0.8);
            overlay.put_pixel(*x, *y, new);
        }
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

    
    // now we need to rotate each object to be upright
    
    for (i, (object, _, corners)) in objects.iter().enumerate() {
        let north = corners[0];
        let east = corners[1];
        let south = corners[2];
        let west = corners[3];
        
        let turning_clockwise = west[1] < east[1];

        println!("north: {:?}, east: {:?}, south: {:?}, west: {:?}, turning_clockwise: {}", north, east, south, west, turning_clockwise);
        
        let [upper_left, upper_right, lower_right, lower_left] = if turning_clockwise {
            [west, north, east, south]
        } else {
            [north, east, south, west]
        };
        
        let width = pixel_distance(upper_left, upper_right);
        let height =  pixel_distance(upper_right, lower_right);
        
        let center = [(upper_left[0] + upper_right[0] + lower_right[0] + lower_left[0]) as f64 / 4f64, (upper_left[1] + upper_right[1] + lower_right[1] + lower_left[1]) as f64 / 4f64];
        let center_rotated = [width/2f64, height/2f64];
        
        let upper_left_rotated = [0u32, 0];
        
        let upper_left_distance = pixel_distance(upper_left, center);
        let upper_left_rotated_distance = pixel_distance(upper_left_rotated, center_rotated);
        let upper_left_rotated_in_original = [center[0] - center_rotated[0], center[1] - center_rotated[1]];
        let upper_lefts_distance= pixel_distance(upper_left, upper_left_rotated_in_original);
        
        
        // atan2(north_y - west_y, north_x - west_x)
        // now we need to find the rotation angle
        let a = upper_left_rotated_distance;
        let c = upper_left_distance;
        let b = upper_lefts_distance;
        
        let cos_b = (c * c + a * a - b * b) / (2.0 * c * a);
        let rotation_angle = cos_b.acos();
        
        let rotation_angle = if turning_clockwise {
            -rotation_angle
        } else {
            rotation_angle
        };
        
        println!("atan2({}, {}) = {}", upper_right[1] as f64 - upper_left[1] as f64, upper_right[0] as f64 - upper_left[0] as f64, f64::atan2(upper_right[1] as f64 - upper_left[1] as f64,upper_right[0] as f64 - upper_left[0] as f64));
        let rotation_calc_two = f64::atan2(upper_right[1] as f64 - upper_left[1] as f64,upper_right[0] as f64 - upper_left[0] as f64);
        
        
        println!("upper_left: {:?}, center: {:?}, center_rotated: {:?}, upper_left_rotated: {:?}, upper_left_distance: {}, upper_left_rotated_distance: {}, upper_left_rotated_in_original: {:?}, upper_lefts_distance: {}, a: {}, b: {}, c: {}, cos_b: {}, rotation_angle: {}", upper_left, center, center_rotated, upper_left_rotated, upper_left_distance, upper_left_rotated_distance, upper_left_rotated_in_original, upper_lefts_distance, a, b, c, cos_b, rotation_angle.to_degrees());
        println!("width: {}, height: {}", width, height);
        println!("rotation_calc_two: {}", rotation_calc_two.to_degrees());
        
        draw_line(&mut overlay, upper_left_rotated_in_original[0] as u32, upper_left_rotated_in_original[1] as u32, center[0] as u32, center[1] as u32, PURPLE);
        draw_line(&mut overlay, upper_left_rotated_in_original[0] as u32, upper_left_rotated_in_original[1] as u32, upper_left[0], upper_left[1], PURPLE);
        draw_line(&mut overlay, upper_left[0], upper_left[1], center[0] as u32, center[1] as u32, PURPLE);
       
        let rotation_angle = rotation_calc_two;
        
        //let mut rotated = RgbImage::new(width, height);
        
        // now we need to rotate each pixel
        
        let rotated_upper_left = rotate_point([0,0], center_rotated, rotation_angle);
        let rotated_upper_left = [rotated_upper_left[0] + center[0] - center_rotated[0], rotated_upper_left[1] + center[1] - center_rotated[1]];
        
        let rotated_upper_right = rotate_point([width,0f64], center_rotated, rotation_angle);
        let rotated_upper_right = [rotated_upper_right[0] + center[0] - center_rotated[0], rotated_upper_right[1] + center[1] - center_rotated[1]];
        
        let rotated_lower_right = rotate_point([width,height], center_rotated, rotation_angle);
        let rotated_lower_right = [rotated_lower_right[0] + center[0] - center_rotated[0], rotated_lower_right[1] + center[1] - center_rotated[1]];
        
        let rotated_lower_left = rotate_point([0f64,height], center_rotated, rotation_angle);
        let rotated_lower_left = [rotated_lower_left[0] + center[0] - center_rotated[0], rotated_lower_left[1] + center[1] - center_rotated[1]];
        
        println!("rotated_upper_left: {:?}, actual_upper_left: {:?}", rotated_upper_left, upper_left);
        println!("rotated_upper_right: {:?}, actual_upper_right: {:?}", rotated_upper_right, upper_right);
        println!("rotated_lower_right: {:?}, actual_lower_right: {:?}", rotated_lower_right, lower_right);
        println!("rotated_lower_left: {:?}, actual_lower_left: {:?}", rotated_lower_left, lower_left);



        let width = width.ceil() as u32;
        let height = height.ceil() as u32;
        for x in 0..width {
            for y in 0..height {
                
                
                let [x_transformed, y_transformed] = rotate_point([x, y], center_rotated, rotation_angle);
                let x_orig = x_transformed + center[0] - center_rotated[0];
                let y_orig = y_transformed + center[1] - center_rotated[1];
                
                
                if x == 0 || y == 0 || x == width - 1 || y == height - 1 {
                    overlay.put_pixel(x_orig as u32, y_orig as u32, ORANGE);
                }
               // println!("x: {}, y: {}, x_transformed: {}, y_transformed: {}, x_orig: {}, y_orig: {}", x, y, x_transformed, y_transformed, x_orig, y_orig);
                
                
                //let orig_pixel = rgb.get_pixel(x_orig as u32, y_orig as u32);
                //rotated.put_pixel(x, y, *orig_pixel);
            }
        }
        //panic!("dadf");
        
      //rotated.save(format!("{}.png", i)).unwrap()
    }

    // draw corners

    for (_, _, corners) in &objects {
        for [x, y] in corners {
            overlay.put_pixel(*x, *y, YELLOW);
        }
    }

    overlay.save("overlay.png").unwrap();

}

fn search_object(image: &RgbImage, start_x: u32, start_y: u32, discovered: &mut [Vec<bool>], sensitivity: f64, background: Rgb<u8>) -> Vec<[u32; 2]> {
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

        let distance = color_distance(image.get_pixel(x, y), &background);

        if distance < sensitivity {
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



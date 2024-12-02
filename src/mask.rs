use std::ops::{Deref, DerefMut};

#[derive(Clone)]
pub struct Mask {
    pub mask: Vec<Vec<bool>>
}

impl Deref for Mask {
    type Target = Vec<Vec<bool>>;

    fn deref(&self) -> &Self::Target {
        &self.mask
    }
}

impl DerefMut for Mask {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.mask
    }
}

impl Mask {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            mask: vec![vec![false; width]; height]
        }
    }
    
    pub fn inside(&self, x: usize, y: usize) -> bool {
        if y >= self.mask.len() || x >= self.mask[y].len() {
            return false;
        }
        self.mask[y][x]
    }
    
    pub fn width(&self) -> usize {
        self.mask[0].len()
    }

    pub fn height(&self) -> usize {
        self.mask.len()
    }
    
    pub fn invert(&mut self) {
        for y in 0..self.mask.len() {
            for x in 0..self.mask[y].len() {
                self.mask[y][x] = !self.mask[y][x]
            }
        }
    }

    pub fn combine(masks: &[&Mask]) -> Mask {
        let mut out = masks[0].clone();
        for mask in masks.iter().skip(1) {
            for y in 0..mask.height() {
                for x in 0..mask.width() {
                    out[y][x] = out[y][x] || mask[y][x]
                }
            }
        }
        out
    }
}
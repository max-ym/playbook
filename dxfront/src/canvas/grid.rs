use dioxus::html::geometry::{
    euclid::{Length, Point2D, Rect}, Pixels
};

use crate::{canvas::GridUnitConvert, *};

use super::{GridUnit, Shift};

/// Information about all added grid lines to the rendered canvas.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct GridLines {
    hor: u32,
    ver: u32,
    max_hor: f64,
    max_ver: f64,
}

impl GridLines {
    pub const CELL_SIZE: Length<u32, GridUnit> = Length::new(2);

    /// Calculate required ranges of added lines to cover all visible area.
    pub fn calc_grid(size: Rect<f64, Pixels>) -> Self {
        let cell_size: f64 = GridLines::CELL_SIZE.to_pixels();

        let hor = (size.size.height / cell_size).floor() as u32;
        let ver = (size.size.width / cell_size).floor() as u32;

        trace!("Calculated grid ranges: {hor:#?}, {ver:#?}");

        Self {
            hor,
            ver,
            max_hor: size.size.height,
            max_ver: size.size.width,
        }
    }
}

impl IntoIterator for GridLines {
    type Item = (Point2D<f64, Pixels>, Point2D<f64, Pixels>);
    type IntoIter = GridLinesIter;

    fn into_iter(self) -> Self::IntoIter {
        GridLinesIter::new(self)
    }
}

/// Iterator to emit all lines that are required to render a grid.
#[derive(Debug, Clone, PartialEq)]
pub struct GridLinesIter {
    hor: u32,
    ver: u32,
    max_hor: f64,
    max_ver: f64,
    current: i32,
    is_hor: bool,
}

impl GridLinesIter {
    fn new(grid: GridLines) -> Self {
        // In this iterator, we add extra line on top, bottom, left and right
        // to compensate for shifting the canvas.

        Self {
            hor: grid.hor + 1,
            ver: grid.ver + 1,
            max_hor: grid.max_hor,
            max_ver: grid.max_ver,
            current: -1,
            is_hor: true,
        }
    }
}

impl Iterator for GridLinesIter {
    // Starting and ending points of a line.
    type Item = (Point2D<f64, Pixels>, Point2D<f64, Pixels>);

    fn next(&mut self) -> Option<Self::Item> {
        let cell_size: f64 = GridLines::CELL_SIZE.to_pixels();

        if self.is_hor {
            if self.current > self.hor as i32 {
                self.is_hor = false;
                self.current = -1;
            }
        } else if self.current > self.ver as i32 {
            return None;
        }

        let current = self.current as f64 * cell_size;
        let points = if self.is_hor {
            (
                Point2D::new(-cell_size, current),
                Point2D::new(self.max_ver + cell_size, current),
            )
        } else {
            (
                Point2D::new(current, -cell_size),
                Point2D::new(current, self.max_hor + cell_size),
            )
        };
        self.current += 1;

        Some(points)
    }
}

#[component]
pub fn Grid(grid: GridLines, shift: Shift) -> Element {
    let shift = shift.wrap_to_cell();

    rsx! {
        div {
            position: "fixed",
            overflow: "visible",
            width: "100%",
            height: "100%",
            min_height: "100vh",
            div {
                position: "relative",
                transform: "{shift}",
                GridSvg { grid }
            }
        }
    }
}

#[component]
fn GridSvg(grid: GridLines) -> Element {
    rsx! {
        svg {
            class: "gridlines",
            overflow: "visible",

            for line in grid {
                line {
                    x1: line.0.x,
                    y1: line.0.y,
                    x2: line.1.x,
                    y2: line.1.y,
                }
            }
        }
    }
}

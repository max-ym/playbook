use crate::*;
use std::ops;

use dioxus::html::{
    geometry::{
        Pixels,
        euclid::{Point2D, Rect},
    },
    input_data::MouseButton,
};
use enumset::EnumSet;

/// Information about all added grid lines to the rendered canvas.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct GridLines {
    hor: u32,
    ver: u32,
    max_hor: f64,
    max_ver: f64,
}

impl GridLines {
    pub const CELL_SIZE: u32 = 32;

    /// Calculate required ranges of added lines to cover all visible area.
    pub fn calc_grid(size: Rect<f64, Pixels>) -> Self {
        const CELL_SIZE: f64 = GridLines::CELL_SIZE as _;

        let hor = (size.size.height / CELL_SIZE).floor() as u32;
        let ver = (size.size.width / CELL_SIZE).floor() as u32;

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
    type Item = (Point2D<f64, Pixels>, Point2D<f64, Pixels>);

    fn next(&mut self) -> Option<Self::Item> {
        const CELL_SIZE: f64 = GridLines::CELL_SIZE as f64;

        if self.is_hor {
            if self.current > self.hor as i32 {
                self.is_hor = false;
                self.current = -1;
            }
        } else if self.current > self.ver as i32 {
            return None;
        }

        let current = self.current as f64 * CELL_SIZE;

        let start = if self.is_hor {
            Point2D::new(-CELL_SIZE, current)
        } else {
            Point2D::new(current, -CELL_SIZE)
        };
        let end = if self.is_hor {
            Point2D::new(self.max_ver, current)
        } else {
            Point2D::new(current, self.max_hor)
        };

        self.current += 1;
        Some((start, end))
    }
}

/// Amount of shift applied to the canvas.
#[derive(Default, Debug, Copy, Clone, PartialEq)]
pub struct Shift {
    point: Point2D<f64, Pixels>,
}

impl Shift {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_point(point: Point2D<f64, Pixels>) -> Self {
        Self { point }
    }

    /// Ensure that this shift is no more than a single cell, wrapping around
    /// the excess shift.
    pub fn wrap_to_cell(self) -> Self {
        let cell_size = GridLines::CELL_SIZE as f64;
        let x = (self.point.x % cell_size).round();
        let y = (self.point.y % cell_size).round();
        Self {
            point: Point2D::new(x, y),
        }
    }
}

impl ops::Add<Point2D<f64, Pixels>> for Shift {
    type Output = Self;

    fn add(self, rhs: Point2D<f64, Pixels>) -> Self::Output {
        Self {
            point: Point2D::new(self.point.x + rhs.x, self.point.y + rhs.y),
        }
    }
}

impl ops::AddAssign<Point2D<f64, Pixels>> for Shift {
    fn add_assign(&mut self, rhs: Point2D<f64, Pixels>) {
        self.point.x += rhs.x;
        self.point.y += rhs.y;
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
                transform: format!("translate({}px, {}px)", shift.point.x, shift.point.y),
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

#[component]
pub fn Canvas() -> Element {
    // The div element that contains the canvas.
    let mut div = use_signal(|| None as Option<Rc<MountedData>>);

    // The dimensions of the canvas.
    let mut dimensions = use_signal(Rect::zero);

    // How much the canvas has been shifted.
    let mut shift = use_signal(Shift::new);

    // Position of the mouse. Used to calculate the shift when shifting the canvas
    // with middle mouse button.
    let mut last_mouse_pos = use_signal(Point2D::zero);

    // Update the dimensions of the canvas to accomodate for window resizing.
    let update_dims = move |_| async move {
        let read = div.read();
        let client_rect = read.as_ref().map(|el| el.get_client_rect());

        if let Some(client_rect) = client_rect {
            if let Ok(rect) = client_rect.await {
                dimensions.set(rect);
            }
        }
    };

    // Track mouse movements.
    let mouse_move = move |e: Event<MouseData>| {
        let coords = e.client_coordinates().cast_unit();
        let is_middle_trigger = e.held_buttons() == EnumSet::only(MouseButton::Auxiliary);

        if is_middle_trigger {
            let last = Shift::from_point(*last_mouse_pos.read());
            shift.with_mut(|shift| {
                *shift += (coords - last.point).to_point();
            })
        }

        last_mouse_pos.set(coords);
    };

    let dimensions = *dimensions.read();
    let shift = *shift.read();
    rsx! {
        div {
            onmounted: move |cx| div.set(Some(cx.data())),
            onresize: move |e| update_dims(e.get_content_box_size()),
            onmousemove: mouse_move,

            width: "100%",
            height: "100%",
            min_height: "100vh",

            Grid { grid: GridLines::calc_grid(dimensions), shift }
        }
    }
}

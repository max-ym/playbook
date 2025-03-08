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

/// Grid lines that are rendered on the canvas.
mod grid;
use grid::*;

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
        let cur_pos = e.client_coordinates().cast_unit();
        let is_middle_trigger = e.held_buttons() == EnumSet::only(MouseButton::Auxiliary);

        if is_middle_trigger {
            let last_pos = *last_mouse_pos.read();
            shift.with_mut(|shift| {
                *shift += (cur_pos - last_pos).to_point();
            })
        }

        last_mouse_pos.set(cur_pos);
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

/// Amount of shift applied to the canvas.
#[derive(Default, Debug, Copy, Clone, PartialEq)]
pub struct Shift {
    point: Point2D<f64, Pixels>,
}

impl Shift {
    pub fn new() -> Self {
        Self::default()
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

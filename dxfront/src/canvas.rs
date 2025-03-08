use crate::*;
use std::{fmt, ops};

use dioxus::html::{
    geometry::{
        euclid::{Length, Point2D, Rect}, Pixels
    },
    input_data::MouseButton,
};
use enumset::EnumSet;

/// Grid lines that are rendered on the canvas.
mod grid;
use grid::*;

mod node;
use node::*;

/// A unit of the grid. Elements on the grid are positioned in multiples of this unit.
#[derive(Debug, Clone, Copy)]
pub struct GridUnit;

impl GridUnit {
    pub const U: u32 = 16;
    pub const F: f64 = Self::U as f64;
}

trait GridUnitConvert {
    type Output;

    fn to_pixels(self) -> Self::Output;
}

impl GridUnitConvert for Point2D<u32, GridUnit> {
    type Output = Point2D<f64, Pixels>;

    fn to_pixels(self) -> Self::Output {
        Point2D::new(self.x as f64 * GridUnit::F, self.y as f64 * GridUnit::F)
    }
}

impl GridUnitConvert for Length<u32, GridUnit> {
    type Output = f64;

    fn to_pixels(self) -> Self::Output {
        self.get() as f64 * GridUnit::F
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
            Nodes { shift }
        }
    }
}

#[component]
pub fn Nodes(shift: Shift) -> Element {
    let n1 = Cfg {
        inputs: 0,
        outputs: 1,
        icon: Icon::Process,
        class: Class::Input,
        label: "Person Record".into(),
    };
    let n2 = Cfg {
        inputs: 3,
        outputs: 3,
        icon: Icon::Process,
        class: Class::Process,
        label: "Dummy Node".into(),
    };
    let n3 = Cfg {
        inputs: 2,
        outputs: 4,
        icon: Icon::Comment,
        class: Class::Comment,
        label: "Dummy Node".into(),
    };
    let n4 = Cfg {
        inputs: 1,
        outputs: 0,
        icon: Icon::Output,
        class: Class::Output,
        label: "Database".into(),
    };

    let unit = GridUnit::U as f64;
    let off1 = Point2D::new(unit, unit * 2.0);
    let off2 = Point2D::new(unit * 10.0, unit * 10.0);
    let off3 = Point2D::new(unit * 40.0, unit * 2.0);
    let off4 = Point2D::new(unit * 30.0, unit * 15.0);

    let n1 = CfgAndOffset { cfg: n1, offset: off1 };
    let n2 = CfgAndOffset { cfg: n2, offset: off2 };
    let n3 = CfgAndOffset { cfg: n3, offset: off3 };
    let n4 = CfgAndOffset { cfg: n4, offset: off4 };

    let conn1 = Cfg::connect_pins(&n1, &n2, 0, 0);

    rsx! {
        div {
            position: "fixed",
            transform: "{shift}",

            Line { cfg: conn1 }

            Node { cfg: n1 }
            Node { cfg: n2 }
            Node { cfg: n3 }
            Node { cfg: n4 }
        }
    }
}

/// Amount of shift applied to the canvas.
#[derive(Default, Debug, Copy, Clone, PartialEq)]
pub struct Shift {
    point: Point2D<f64, Pixels>,
}

impl fmt::Display for Shift {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "translate({x}px, {y}px)", x = self.point.x, y = self.point.y)
    }
}

impl Shift {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ensure that this shift is no more than a single cell, wrapping around
    /// the excess shift.
    pub fn wrap_to_cell(self) -> Self {
        let cell_size = GridLines::CELL_SIZE.to_pixels();
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

use crate::*;
use std::{
    cell::RefCell,
    fmt, ops,
    sync::{Arc, Mutex, mpsc},
};

use dioxus::{
    html::{
        geometry::{
            Pixels,
            euclid::{Length, Point2D, Rect},
        },
        input_data::MouseButton,
    },
    web::WebEventExt,
};
use enumset::EnumSet;

/// Grid lines that are rendered on the canvas.
mod grid;
use grid::*;

mod node;
use node::*;

/// Global variable to pass the element that was pressed on on the canvas.
static CANVAS_DRAG: Global<Signal<CanvasDrag>, CanvasDrag> = Signal::global(CanvasDrag::zero);

#[derive(Debug, Clone, Copy, PartialEq)]
struct CanvasDrag {
    element_offset: Point2D<f64, Pixels>,
    mouse_pos: Option<Point2D<f64, Pixels>>,
    is_tracked: bool,
}

impl CanvasDrag {
    pub fn zero() -> Self {
        Self {
            element_offset: Point2D::zero(),
            mouse_pos: None,
            is_tracked: false,
        }
    }

    /// Create a new untracked offset for an element.
    pub fn new(element_offset: Point2D<f64, Pixels>) -> Self {
        Self {
            element_offset,
            mouse_pos: None,
            is_tracked: false,
        }
    }

    /// Track element's offset.
    /// The element will be moved when the mouse is dragged.
    pub fn track_new(offset: Signal<Self>) {
        let element_offset = offset.read().element_offset;
        Self::new(element_offset).track(offset)
    }

    /// Begin tracking the element. All future drag events will be applied to this element,
    /// this change will be propagated to the listeners of the provided signal.
    pub fn track(mut self, child_sig: Signal<CanvasDrag>) {
        self.is_tracked = true;
        let mut sig = CANVAS_DRAG.resolve();
        let _ = sig.point_to(child_sig);
        *sig.write() = self;
    }

    /// Register new mouse movement and update the offset of the element accordingly.
    pub fn update(&mut self, mouse_pos: Point2D<f64, Pixels>) {
        let old = self.mouse_pos.unwrap_or(mouse_pos);
        self.mouse_pos = Some(mouse_pos);
        let diff = mouse_pos - old;
        self.element_offset += diff;
    }

    /// Remove all tracking of any element.
    pub fn untrack() {
        let mut sig = CANVAS_DRAG.resolve();
        sig.with_mut(|d| {
            d.mouse_pos = None; // So that the next drag event doesn't jerk the element.
            d.is_tracked = false;
        });
        let _ = sig.point_to(Signal::new(CanvasDrag::zero()));
    }

    /// Whether any element is being tracked.
    pub fn has_tracking() -> bool {
        let sig = CANVAS_DRAG.resolve();
        sig.read().is_tracked
    }
}

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

    let mut cursor = use_signal(|| "default");

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
        let cur_pos = e.screen_coordinates().cast_unit();
        let is_middle_trigger = e.held_buttons() == EnumSet::only(MouseButton::Auxiliary);
        let is_primary_trigger = e.held_buttons() == EnumSet::only(MouseButton::Primary);

        if is_middle_trigger {
            let last_pos = *last_mouse_pos.read();
            shift.with_mut(|shift| {
                *shift += (cur_pos - last_pos).to_point();
            })
        } else if is_primary_trigger {
            let mut resv = CANVAS_DRAG.resolve();
            resv.with_mut(|drag| {
                drag.update(cur_pos);
            })
        }

        last_mouse_pos.set(cur_pos);
    };

    let mouse_up = move |e: Event<MouseData>| {
        let is_primary = e.data().trigger_button() == Some(MouseButton::Primary);

        *cursor.write() = "default";
        if is_primary {
            CanvasDrag::untrack();
        }
    };

    // Check for mouse down events to change the cursor type.
    let mouse_down = move |e: Event<MouseData>| {
        let is_middle_trigger = e.held_buttons() == EnumSet::only(MouseButton::Auxiliary);

        *cursor.write() = if is_middle_trigger {
            "move"
        } else {
            return;
        };
    };

    use_effect(move || {
        if CanvasDrag::has_tracking() {
            *cursor.write() = "grabbing";
        }
    });

    let dimensions = *dimensions.read();
    let shift = *shift.read();
    rsx! {
        div {
            onmounted: move |cx| {
                div.set(Some(cx.data()));

            },
            onresize: move |e| update_dims(e.get_content_box_size()),
            onmousemove: mouse_move,
            onmouseup: mouse_up,
            onmousedown: mouse_down,

            cursor: cursor,

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

    let n1 = CfgAndOffset {
        cfg: n1,
        offset: off1,
    };
    let n2 = CfgAndOffset {
        cfg: n2,
        offset: off2,
    };
    let n3 = CfgAndOffset {
        cfg: n3,
        offset: off3,
    };
    let n4 = CfgAndOffset {
        cfg: n4,
        offset: off4,
    };

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
        write!(
            f,
            "translate({x}px, {y}px)",
            x = self.point.x,
            y = self.point.y
        )
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

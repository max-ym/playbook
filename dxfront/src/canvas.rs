use crate::*;
use std::{
    cell::RefCell,
    fmt, ops,
    sync::{
        OnceLock, RwLock,
        atomic::{self, AtomicU32},
    },
};

use dioxus::html::{
    geometry::{
        Pixels,
        euclid::{Length, Point2D, Rect, Vector2D},
    },
    input_data::MouseButton,
};
use enumset::EnumSet;

/// Grid lines that are rendered on the canvas.
mod grid;
use grid::*;

mod node;
use hashbrown::HashMap;
use node::*;
use tracing::{debug, error};

/// Pool of IDs to uniquely identify new elements being added to the canvas.
/// IDs start from 1.
/// Zero ID indicates uninitialized state.
static NEXT_ID: AtomicId = AtomicId::one();

/// ID of the node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Id(u32);

#[derive(Debug)]
struct AtomicId(AtomicU32);

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl Id {
    /// [NEXT_ID] is incremented by 1 and the new value is returned.
    /// New nodes added to the canvas get their IDs as z-indexes.
    /// This effectively allows for new nodes to be on top of the older ones
    /// after they are added.
    pub fn next() -> Self {
        let id = NEXT_ID.0.fetch_add(1, atomic::Ordering::AcqRel);
        Self(id)
    }

    pub fn latest() -> Self {
        let id = NEXT_ID.0.load(atomic::Ordering::Acquire);
        Self(id)
    }

    pub const fn uninit() -> Self {
        Self::zero()
    }

    pub const fn zero() -> Self {
        Id(0)
    }

    pub const fn is_uninit(self) -> bool {
        self.0 == Self::uninit().0
    }

    /// Return this object if it is a valid ID.
    pub fn only_valid(self) -> Option<Id> {
        (self != Self::uninit()).then_some(self)
    }
}

impl From<Id> for u32 {
    fn from(id: Id) -> u32 {
        id.0
    }
}

impl AtomicId {
    pub const fn one() -> Self {
        Self(AtomicU32::new(1))
    }
}

impl From<AtomicId> for Id {
    fn from(id: AtomicId) -> Id {
        Id(id.0.load(atomic::Ordering::Acquire))
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

    // Currently set cursor type (value for CSS).
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

    let mut nodes = use_signal(dummy_nodes);

    // Tracking of the selected node for drag.
    let mut dragged_node_id = use_signal(Id::uninit);

    // Position of the mouse. Used to calculate the shift when shifting the canvas
    // with middle mouse button. It also is used to calculate offset for being-dragged
    // nodes.
    let last_mouse_pos = use_hook(|| Rc::new(RefCell::new(Point2D::zero())));
    // Track mouse movements, applying changes to the signals where necessary.
    let mouse_move = move |e: Event<MouseData>| {
        let cur_pos = e.page_coordinates().cast_unit();
        let is_middle = e.held_buttons() == EnumSet::only(MouseButton::Auxiliary);
        let is_primary = e.held_buttons() == EnumSet::only(MouseButton::Primary);

        let diff = cur_pos - *last_mouse_pos.borrow();
        last_mouse_pos.replace(cur_pos);

        if is_middle {
            // This shifts the whole canvas view.
            shift.with_mut(|shift| *shift += diff);
        } else if is_primary {
            // This moves the dragged node.
            let node_id = dragged_node_id();
            if let Some(node_id) = node_id.only_valid() {
                nodes.with_mut(|nodes| {
                    if let Some(node) = nodes.get_mut(&node_id) {
                        node.offset += diff;

                        // Check whether to update z-index to put the node on top.
                        if node.z_index != u32::from(Id::latest()) {
                            // Note that we don't really change the ID of this node.
                            // We just use ID pool
                            // to generate z-index which is guaranteed to be
                            // above all current nodes.
                            node.z_index = Id::next().into();
                        }
                    } else {
                        error!("Node with ID {node_id} not found for move in the nodes map.");
                    }
                });
            }
        }
    };

    let mouse_up = move |e: Event<MouseData>| {
        let is_primary = e.data().trigger_button() == Some(MouseButton::Primary);

        *cursor.write() = "default";
        if is_primary {
            // Reset the dragged node ID.
            dragged_node_id.replace(Id::uninit());
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
        let has_tracking = !dragged_node_id().is_uninit();
        if has_tracking {
            *cursor.write() = "grabbing";
        }
    });

    let dimensions = *dimensions.read();
    let shift = *shift.read();
    let node_drag_notify = NodeDragNotify {
        node_id: dragged_node_id,
    };
    rsx! {
        div {
            onmounted: move |cx| div.set(Some(cx.data())),
            onresize: move |e| update_dims(e.get_content_box_size()),
            onmousemove: mouse_move,
            onmouseup: mouse_up,
            onmousedown: mouse_down,

            cursor: cursor,

            width: "100%",
            height: "100%",
            min_height: "100vh",
            position: "fixed",

            Grid { shift, grid: GridLines::calc_grid(dimensions) }
            div {
                position: "relative",
                transform: "{shift}",

                // Line { cfg: conn1 }
                for (_, cfg) in nodes() {
                    Node { cfg, drag: node_drag_notify.clone() }
                }
            }
        }
    }
}

/// Send the selected node ID to the parent component.
#[derive(Clone, PartialEq)]
struct NodeDragNotify {
    node_id: Signal<Id>,
}

impl NodeDragNotify {
    pub fn notify_mouse_down(&mut self, id: Id) {
        debug!("Register drag on node {id}");
        *self.node_id.write() = id;
    }
}

/// For testing, for now.
fn dummy_nodes() -> HashMap<Id, node::Cfg> {
    let n1 = CfgInner {
        inputs: 0,
        outputs: 1,
        icon: Icon::Process,
        class: Class::Input,
        label: "Person Record".into(),
    };
    let n2 = CfgInner {
        inputs: 3,
        outputs: 3,
        icon: Icon::Process,
        class: Class::Process,
        label: "Dummy Node".into(),
    };
    let n3 = CfgInner {
        inputs: 2,
        outputs: 4,
        icon: Icon::Comment,
        class: Class::Comment,
        label: "Dummy Node".into(),
    };
    let n4 = CfgInner {
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

    let n1 = Cfg {
        cfg: n1,
        offset: off1,
        id: Id::next(),
        z_index: Id::next().into(),
    };
    let n2 = Cfg {
        cfg: n2,
        offset: off2,
        id: Id::next(),
        z_index: Id::next().into(),
    };
    let n3 = Cfg {
        cfg: n3,
        offset: off3,
        id: Id::next(),
        z_index: Id::next().into(),
    };
    let n4 = Cfg {
        cfg: n4,
        offset: off4,
        id: Id::next(),
        z_index: Id::next().into(),
    };

    let mut m = HashMap::with_capacity(4);
    m.insert(n1.id, n1);
    m.insert(n2.id, n2);
    m.insert(n3.id, n3);
    m.insert(n4.id, n4);
    m
}

#[component]
pub fn Nodes(drag: NodeDragNotify, shift: Shift) -> Element {
    // let conn1 = LineCfg {
    //     start: n1.id,
    //     end: n2.id,
    //     start_pin: 0,
    //     end_pin: 0,
    // };

    rsx! {}
}

/// Amount of shift applied to the canvas view. This shifts all the elements and the grid
/// on the specified amount.
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
    /// the excess shift. Effectively this makes it looks like the grid is being
    /// shifted continuously, when really we just have the grid wrap around
    /// by coordinates, so that we don't need to recreate or move actual lines during
    /// normal shifting.
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

impl ops::AddAssign<Vector2D<f64, Pixels>> for Shift {
    fn add_assign(&mut self, rhs: Vector2D<f64, Pixels>) {
        self.point += rhs
    }
}

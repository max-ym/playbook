use core::fmt;
use std::{
    borrow::Cow,
    sync::{OnceLock, RwLock},
};

use dioxus::html::{
    geometry::{
        Pixels,
        euclid::{Length, Point2D, Size2D, Vector2D},
    },
    input_data::MouseButton,
};
use hashbrown::HashMap;

use crate::{
    canvas::{CanvasDrag, next_z_index},
    *,
};

use super::{GridUnit, GridUnitConvert};

/// Registry of all line connections on the canvas. Key is an ID of the node by the initial z-index
/// that was assigned to the node on creation. Signal is used to update the line's end's
/// position when the node is dragged. Since several lines can be connected to the same node,
/// the value is a vector of signals.
static LINE_CONN: RwLock<OnceLock<HashMap<u32, Vec<GlobalSignal<Point2D<f64, Pixels>>>>>> =
    RwLock::new(OnceLock::new());

/// Manipulation over [LINE_CONN] singleton.
struct LineConn {
    pub orig_z_index0: u32,
    pub signal0: GlobalSignal<Point2D<f64, Pixels>>,
    pub orig_z_index1: u32,
    pub signal1: GlobalSignal<Point2D<f64, Pixels>>,
}

impl LineConn {
    /// Ensure that the [LINE_CONN] singleton is initialized.
    /// Execute the closure on initialized singleton and return the result.
    #[inline]
    fn ensure_init<T>(
        f: impl FnOnce(&mut HashMap<u32, Vec<GlobalSignal<Point2D<f64, Pixels>>>>) -> T,
    ) -> T {
        let mut lock = LINE_CONN.write().unwrap();
        lock.get_or_init(HashMap::new);
        lock.get_mut().map(f).expect("initialized just above")
    }

    fn to_arr(self) -> [(u32, GlobalSignal<Point2D<f64, Pixels>>); 2] {
        [
            (self.orig_z_index0, self.signal0),
            (self.orig_z_index1, self.signal1),
        ]
    }

    /// Register a new line connection with the provided parameters.
    pub fn register(self) {
        Self::ensure_init(|map| {
            for (orig_z_index, signal) in self.to_arr() {
                map.entry(orig_z_index)
                    .or_insert_with(Vec::new)
                    .push(signal);
            }
        });
    }

    /// Unregister all line connections with the provided z-index.
    /// Needs to be called when the node is being removed from the canvas.
    pub fn unregister_all(orig_z_index: u32) {
        Self::ensure_init(|map| {
            map.remove(&orig_z_index);
        });
    }

    /// Unregister a line connection with the provided parameters.
    /// Needs to be called when the line is being removed from the canvas.
    ///
    /// # Panics
    /// Panics if the line connection does not exist.
    pub fn unregister(self) {
        Self::ensure_init(|map| {
            for (orig_z_index, signal) in self.to_arr() {
                let signals = map
                    .get_mut(&orig_z_index)
                    .expect("orig_z_index should exist");
                let idx = signals
                    .iter()
                    .position(|s| s.signal() == signal.signal())
                    .expect("signal should exist");
                signals.swap_remove(idx);
            }
        });
    }

    /// Notify all line connections with the provided z-index about the change in position.
    pub fn notify_all(orig_z_index: u32, diff: Vector2D<f64, Pixels>) {
        Self::ensure_init(|map| {
            if let Some(signals) = map.get(&orig_z_index) {
                for signal in signals {
                    signal.with_mut(|pos| {
                        *pos += diff;
                    });
                }
            }
        });
    }
}

/// Notify listeners about the change in position of a node.
pub fn notify_moved(orig_z_index: u32, diff: Vector2D<f64, Pixels>) {
    LineConn::notify_all(orig_z_index, diff);
}

#[component]
pub fn Node(cfg: Cfg) -> Element {
    let orig_z_index = cfg.orig_z_index;
    let orig_offset = cfg.offset;
    let cfg = cfg.cfg;

    let mut z_index = use_signal(|| orig_z_index);

    let offset = use_signal(|| CanvasDrag::new(orig_offset, orig_z_index));
    let mouse_down = move |e: Event<MouseData>| {
        let is_primary = e.data().trigger_button() == Some(MouseButton::Primary);
        if !is_primary {
            return;
        }
        e.prevent_default();

        *z_index.write() = next_z_index();
        CanvasDrag::track_new(offset);
    };

    // Disconnect all lines when the node is being removed.
    use_drop(move || {
        LineConn::unregister_all(orig_z_index);
    });

    let offset = offset.read().element_offset;
    rsx! {
        div {
            onmousedown: mouse_down,

            z_index: z_index,
            position: "absolute",
            top: "{offset.y}px",
            left: "{offset.x}px",
            NodeUnpositioned { cfg }
        }
    }
}

#[component]
pub fn NodeUnpositioned(cfg: CfgInner) -> Element {
    let size = cfg.min_size();
    let (w, h) = (size.width, size.height);
    let (add_top_inputs, add_top_outputs) = cfg.offset_pins();

    rsx! {
        div {
            class: "{cfg.class}",
            width: "{w}px",
            height: "{h}px",

            div {
                display: "flex",
                justify_content: "center",
                align_items: "center",
                height: "100%",
                margin_bottom: "4px",
                {cfg.label.as_ref()}
            }

            for i in 0..cfg.inputs {
                Pin { i, is_in: true, add_top: add_top_inputs }
            }
            for i in 0..cfg.outputs {
                Pin { i, is_in: false, add_top: add_top_outputs }
            }
        }
    }
}

#[component]
fn Pin(i: u8, is_in: bool, add_top: f64) -> Element {
    let top =
        i as f64 * CfgInner::PIN_SIZE + i as f64 * CfgInner::PIN_MARGIN + CfgInner::OUTER_MARGIN;
    let top = top + add_top;
    let side = -CfgInner::PIN_SIZE / 2.0;
    let padding: f64 = CfgInner::PIN_SIZE;

    rsx! {
        div {
            position: "absolute",
            class: "pin pin-input",

            top: "{top}px",
            left: if is_in { "{side}px" } else { "auto" },
            right: if is_in { "auto" } else { "{side}px" },
            padding_left: "{padding}px",
            padding_top: "{padding}px",
        }
    }
}

#[component]
pub fn Line(cfg: LineCfg) -> Element {
    let start = *cfg.start.read();
    let end = *cfg.end.read();
    let c2 = Point2D::<f64, Pixels>::new(start.x, end.y);
    let c1 = Point2D::<f64, Pixels>::new(end.x, start.y);

    rsx! {
        div {
            z_index: u32::MAX,
            position: "absolute",
            pointer_events: "none",

            svg {
                overflow: "visible",
                path {
                    d: "M {start.x} {start.y} C {c1.x} {c1.y} {c2.x} {c2.y} {end.x} {end.y}",
                    fill: "none",
                    stroke: "black",
                    stroke_width: "2",
                    opacity: "50%",
                }
            }
        }
    }
}

/// Configuration of inner node properties.
#[derive(Debug, Clone, PartialEq, Props)]
pub struct CfgInner {
    pub inputs: u8,
    pub outputs: u8,

    pub icon: Icon,
    pub class: Class,
    pub label: Cow<'static, str>,
}

/// Configuration of a node with its offset and original z-index to add on the canvas.
#[derive(Debug, Clone, PartialEq)]
pub struct Cfg {
    pub cfg: CfgInner,
    pub offset: Point2D<f64, Pixels>,
    pub orig_z_index: u32,
}

impl CfgInner {
    pub const MIN_H: Length<u32, GridUnit> = Length::new(2);
    pub const MIN_W: Length<u32, GridUnit> = Length::new(12);

    pub const PIN_SIZE: f64 = 16.0;
    pub const PIN_MARGIN: f64 = 16.0;
    pub const OUTER_MARGIN: f64 = 12.0;

    pub fn min_size(&self) -> Size2D<f64, Pixels> {
        let min_w = Self::MIN_W.to_pixels();
        let min_h = Self::MIN_H.to_pixels();

        let pins_on_side = self.inputs.max(self.outputs);
        let pin_size = Self::min_height_for_cnt(pins_on_side) + Self::OUTER_MARGIN * 2.0;

        Size2D::new(min_w, min_h.max(pin_size))
    }

    fn min_height_for_cnt(pins: u8) -> f64 {
        if pins == 0 {
            0.0
        } else {
            pins as f64 * Self::PIN_SIZE + (pins - 1) as f64 * Self::PIN_MARGIN
        }
    }

    /// The amount to offset the one kind of pins from the top to center relative to the
    /// other kind of pins.
    pub fn offset_pins(&self) -> (f64, f64) {
        let inputs = Self::min_height_for_cnt(self.inputs);
        let outputs = Self::min_height_for_cnt(self.outputs);

        if self.inputs < self.outputs {
            let diff = outputs - inputs;
            (diff / 2.0, 0.0)
        } else {
            let diff = inputs - outputs;
            (0.0, diff / 2.0)
        }
    }

    pub fn connect_pins(a: &Cfg, b: &Cfg, p1: u8, p2: u8) -> LineCfg {
        let a_pos = a.cfg.pin_pos(p1);
        let b_pos = b.cfg.pin_pos(p2);

        let start = Point2D::new(a_pos.x + a.offset.x, a_pos.y + a.offset.y);
        let end = Point2D::new(b_pos.x + b.offset.x, b_pos.y + b.offset.y);

        let signal0 = Signal::global(Point2D::zero);
        let signal1 = Signal::global(Point2D::zero);

        *signal0.write() = start;
        *signal1.write() = end;

        let line_cfg = LineCfg {
            start: signal0.signal(),
            end: signal1.signal(),
        };

        LineConn {
            orig_z_index0: a.orig_z_index,
            signal0,
            orig_z_index1: b.orig_z_index,
            signal1,
        }
        .register();

        line_cfg
    }

    pub fn pin_pos(&self, i: u8) -> Point2D<f64, Pixels> {
        let sum = self.inputs + self.outputs;
        assert!(i < sum, "Pin index out of bounds");

        let is_in = i < self.inputs;
        let i = if is_in { i } else { i - self.inputs };

        let pin_center = Self::PIN_SIZE / 2.0;

        let offset = self.offset_pins();
        let top = i as f64 * Self::PIN_SIZE + i as f64 * Self::PIN_MARGIN + Self::OUTER_MARGIN;
        let top = top + pin_center + if is_in { offset.0 } else { offset.1 };

        let side = if is_in {
            -pin_center
        } else {
            Self::MIN_W.to_pixels() + pin_center
        };

        Point2D::new(side, top)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Icon {
    Input,
    Output,
    Process,
    Comment,
}

impl Icon {
    pub fn asset(self) -> Asset {
        use Icon::*;
        match self {
            Input => asset!("assets/node/input.svg"),
            Output => asset!("assets/node/output.svg"),
            Process => asset!("assets/node/process.svg"),
            Comment => asset!("assets/node/comment.svg"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Input,
    Output,
    Process,
    Comment,
    Warning,
}

impl AsRef<str> for Class {
    fn as_ref(&self) -> &str {
        use Class::*;
        match self {
            Input => "node node-input",
            Output => "node node-output",
            Process => "node node-process",
            Comment => "node node-comment",
            Warning => "node node-warning",
        }
    }
}

impl fmt::Display for Class {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_ref())
    }
}

/// A configuration for a line that connects two node pins.
#[derive(Debug, Clone, PartialEq, Props)]
pub struct LineCfg {
    start: Signal<Point2D<f64, Pixels>>,
    end: Signal<Point2D<f64, Pixels>>,
}

use core::fmt;
use std::borrow::Cow;

use dioxus::html::{
    geometry::{
        Pixels,
        euclid::{Length, Point2D, Size2D},
    },
    input_data::MouseButton,
};

use crate::*;

use super::{GridUnit, GridUnitConvert, Id, NodeDragNotify};

#[component]
pub fn Node(cfg: Cfg, drag: NodeDragNotify) -> Element {
    let id = cfg.id;
    let offset = cfg.offset;
    let z_index = cfg.z_index;

    let onmousedown = move |e: Event<MouseData>| {
        let is_primary = e.data().trigger_button() == Some(MouseButton::Primary);
        if !is_primary {
            return;
        }

        e.prevent_default();
        drag.notify_mouse_down(id);
    };

    rsx! {
        div {
            onmousedown,
            z_index,
            position: "absolute",
            top: "{offset.y}px",
            left: "{offset.x}px",
            NodeUnpositioned { cfg: cfg.cfg }
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
    let top = i as f64 * (CfgInner::PIN_SIZE + CfgInner::PIN_MARGIN);
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
    let start = cfg.pos0;
    let end = cfg.pos1;
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
                    opacity: "40%",
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
    pub id: Id,
    pub z_index: u32,
}

impl CfgInner {
    pub const MIN_H: Length<u32, GridUnit> = Length::new(2);
    pub const MIN_W: Length<u32, GridUnit> = Length::new(12);

    pub const PIN_SIZE: f64 = 16.0;
    pub const PIN_MARGIN: f64 = 16.0;
    pub const OUTER_MARGIN: f64 = 8.0;

    pub fn min_size(&self) -> Size2D<f64, Pixels> {
        let min_w = Self::MIN_W.to_pixels();
        let min_h = Self::MIN_H.to_pixels();

        let pins_on_bigger_side = self.inputs.max(self.outputs);
        let pin_h = Self::min_height_for_cnt(pins_on_bigger_side) + Self::OUTER_MARGIN * 2.0;

        Size2D::new(min_w, min_h.max(pin_h).round())
    }

    /// Minimum size selected ceiled to the grid unit.
    pub fn min_size_to_grid(&self) -> Size2D<f64, Pixels> {
        let size = self.min_size();

        let rem_h = (size.height as f64 % GridUnit::F).round();
        let rem_w = (size.width as f64 % GridUnit::F).round();
        let add_h = if rem_h > 0.0 {
            GridUnit::F - rem_h
        } else {
            0.0
        };
        let add_w = if rem_w > 0.0 {
            GridUnit::F - rem_w
        } else {
            0.0
        };

        let h = size.height + add_h;
        let w = size.width + add_w;
        Size2D::new(w.round(), h.round())
    }

    fn min_height_for_cnt(pins: u8) -> f64 {
        if pins == 0 {
            0.0
        } else {
            pins as f64 * Self::PIN_SIZE + (pins - 1) as f64 * Self::PIN_MARGIN
        }
    }

    fn actual_size(&self) -> Size2D<f64, Pixels> {
        Self::min_size_to_grid(&self)
    }

    /// The amount to offset the one kind of pins from the top to center relative to the
    /// other kind of pins.
    pub fn offset_pins(&self) -> (f64, f64) {
        let size = self.actual_size();

        let outputs_h = Self::min_height_for_cnt(self.outputs);
        let inputs_h = Self::min_height_for_cnt(self.inputs);

        (
            (size.height - inputs_h) / 2.0,
            (size.height - outputs_h) / 2.0,
        )
    }

    // pub fn connect_pins(a: &Cfg, b: &Cfg, p1: u8, p2: u8) -> LineCfg {
    //     let a_pos = a.cfg.pin_pos(p1);
    //     let b_pos = b.cfg.pin_pos(p2);

    //     let start = Point2D::new(a_pos.x + a.offset.x, a_pos.y + a.offset.y);
    //     let end = Point2D::new(b_pos.x + b.offset.x, b_pos.y + b.offset.y);

    //     let signal0 = Signal::global(Point2D::zero);
    //     let signal1 = Signal::global(Point2D::zero);

    //     *signal0.write() = start;
    //     *signal1.write() = end;

    //     let line_cfg = LineCfg {
    //         start: signal0.signal(),
    //         end: signal1.signal(),
    //     };

    //     LineConn {
    //         orig_z_index0: a.orig_z_index,
    //         signal0,
    //         orig_z_index1: b.orig_z_index,
    //         signal1,
    //     }
    //     .register();

    //     line_cfg
    // }

    /// Position of the line for the pin with the given index.
    pub fn pin_line_pos(&self, i: u8) -> Point2D<f64, Pixels> {
        let sum = self.inputs + self.outputs;
        assert!(i < sum, "Pin index out of bounds");

        let is_in = i < self.inputs;
        let i = if is_in { i } else { i - self.inputs };

        let pin_center = Self::PIN_SIZE / 2.0;

        let offset = self.offset_pins();
        let top = i as f64 * (Self::PIN_SIZE + Self::PIN_MARGIN);
        let top = pin_center + top + if is_in { offset.0 } else { offset.1 };

        // TODO can we compile with CSS value directly?
        const BORDER_MARGIN: f64 = 4.0;

        // Since pins should be centered relative to the node's edge, we shift
        // them by half of the pin size.
        let side = if is_in {
            -pin_center
        } else {
            Self::MIN_W.to_pixels() + pin_center + BORDER_MARGIN
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

/// Configuration for calculating line parameters that connects node pins.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeLineCalc {
    /// Start node.
    pub start: Cfg,

    /// Pin number in the start node.
    pub start_pin: u8,

    /// End node.
    pub end: Cfg,

    /// Pin number in the end node.
    pub end_pin: u8,
}

/// A configuration for a line that connects two node pins.
#[derive(Debug, Clone, PartialEq, Props)]
pub struct LineCfg {
    pub pos0: Point2D<f64, Pixels>,
    pub pos1: Point2D<f64, Pixels>,
}

impl NodeLineCalc {
    pub fn calc(self) -> LineCfg {
        let start = self.start.cfg.pin_line_pos(self.start_pin);
        let end = self.end.cfg.pin_line_pos(self.end_pin);

        LineCfg {
            pos0: Point2D::new(start.x + self.start.offset.x, start.y + self.start.offset.y),
            pos1: Point2D::new(end.x + self.end.offset.x, end.y + self.end.offset.y),
        }
    }
}

impl From<NodeLineCalc> for LineCfg {
    fn from(calc: NodeLineCalc) -> Self {
        calc.calc()
    }
}

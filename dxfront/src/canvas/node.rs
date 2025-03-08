use core::fmt;
use std::borrow::Cow;

use dioxus::html::geometry::{
    Pixels,
    euclid::{Length, Point2D, Size2D},
};

use crate::*;

use super::{GridUnit, GridUnitConvert};

#[component]
pub fn Node(cfg: CfgAndOffset) -> Element {
    let offset = cfg.offset;
    let cfg = cfg.cfg;

    rsx! {
        div {
            position: "absolute",
            top: "{offset.y}px",
            left: "{offset.x}px",
            NodeUnpositioned { cfg }
        }
    }
}

#[component]
pub fn NodeUnpositioned(cfg: Cfg) -> Element {
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
    let top = i as f64 * Cfg::PIN_SIZE + i as f64 * Cfg::PIN_MARGIN + Cfg::OUTER_MARGIN;
    let top = top + add_top;
    let side = -Cfg::PIN_SIZE / 2.0;
    let padding: f64 = Cfg::PIN_SIZE;

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
    let start = cfg.start;
    let end = cfg.end;
    let c1 = Point2D::<f64, Pixels>::new(start.x, end.y);
    let c2 = Point2D::<f64, Pixels>::new(end.x, start.y);

    rsx! {
        svg {
            overflow: "visible",
            path {
                d: "M {start.x} {start.y} C {c1.x} {c1.y} {c2.x} {c2.y} {end.x} {end.y}",
                fill: "none",
                stroke: "black",
                stroke_width: "2",
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Props)]
pub struct Cfg {
    pub inputs: u8,
    pub outputs: u8,

    pub icon: Icon,
    pub class: Class,
    pub label: Cow<'static, str>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CfgAndOffset {
    pub cfg: Cfg,
    pub offset: Point2D<f64, Pixels>,
}

impl Cfg {
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

    pub fn connect_pins(a: &CfgAndOffset, b: &CfgAndOffset, p1: u8, p2: u8) -> LineCfg {
        let a_pos = a.cfg.pin_pos(p1);
        let b_pos = b.cfg.pin_pos(p2);

        let start = Point2D::new(a_pos.x + a.offset.x, a_pos.y + a.offset.y);
        let end = Point2D::new(b_pos.x + b.offset.x, b_pos.y + b.offset.y);

        LineCfg { start, end }
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
    start: Point2D<f64, Pixels>,
    end: Point2D<f64, Pixels>,
}

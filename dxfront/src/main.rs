use std::rc::Rc;

use dioxus::prelude::*;
use tracing::{trace, debug, info};

const FAVICON: Asset = asset!("/assets/favicon.ico");
const MAIN_CSS: Asset = asset!("/assets/main.css");
const TAILWIND_CSS: Asset = asset!("/assets/tailwind.css");

pub mod canvas;

fn main() {
    tracing_log::LogTracer::init().unwrap();
    use dioxus::logger::tracing::Level;
    let level = if cfg!(debug_assertions) {
        Level::DEBUG
    } else {
        Level::INFO
    };
    dioxus::logger::init(level).expect("Failed to initialize logger");
    info!("Log level set to {level}");

    launch(App);
}

#[component]
fn App() -> Element {
    rsx! {
        document::Link { rel: "icon", href: FAVICON }
        document::Link { rel: "stylesheet", href: MAIN_CSS }
        document::Link { rel: "stylesheet", href: TAILWIND_CSS }

        canvas::Canvas {}
    }
}

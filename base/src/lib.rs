use compact_str::CompactString;

pub mod vcs;

pub mod canvas;

pub mod table_data;

// pub mod valid;

#[cfg(test)]
mod tests {
    use fern::colors::{Color, ColoredLevelConfig};

    pub fn init() {
        let colors = ColoredLevelConfig::new()
        .error(Color::Red)
        .warn(Color::Yellow)
        .trace(Color::BrightMagenta)
        .info(Color::BrightGreen)
        .debug(Color::BrightBlue);

    let _ = fern::Dispatch::new()
        .format(move |out, message, record| {
            out.finish(format_args!(
                "\x1B[{dc}m{date}\x1B[{tc}m[{target}]\x1B[{mc}m[{level}] {message}\x1B[0m",
                date = chrono::Local::now().format("[%Y-%m-%d %H:%M:%S]"),
                target = record.target(),
                level = record.level(),
                message = message,
                mc = colors.get_color(&record.level()).to_fg_str(),
                tc = Color::Magenta.to_fg_str(),
                dc = Color::BrightMagenta.to_fg_str(),
            ))
        })
        .level(log::LevelFilter::Trace)
        .chain(std::io::stdout())
        .apply();
    }
}

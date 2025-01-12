use super::*;

#[derive(Debug)]
pub struct HistoryItem {
    pub timestamp: chrono::NaiveDateTime,
    pub author: CompactString,
    pub message: CompactString,
    pub canvas: canvas::Canvas<()>,
}

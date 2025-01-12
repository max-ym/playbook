use crate::*;

/// A CSV parser implementation.
// This should work well on WASM, but for server we would want to rewrite this with SIMD,
// which would be available on native targets.
pub mod csv;

/// A table with columns and rows.
/// This data is used as a source of values to run into the data flow, with validations
/// and transformations.
pub struct Table {
    columns: Vec<CompactString>,
    rows: Rows,

    /// This not only contains the unique strings per column, but also how many times they reoccur.
    /// The outer vector is indexed by column, the inner vector is indexed by the unique string.
    /// The inner vector is sorted by the string.
    /// Not only we save memory by storing the strings only once, but count can
    /// be useful for logging, and reducing computations. We will only validate one value once,
    /// not per row, which in turn makes validations faster for tables with many repeated values.
    unique_per_col: Vec<Uniques>,
}

/// Rows implementation that allocates one array for all cells in all rows.
/// Since we know the number of columns, we use its length to divide the array into rows.
pub struct Rows {
    /// Index into [Table::unique_per_col].
    cells: Vec<u32>,
}

impl Rows {
    pub fn row(&self, i: usize, columns: usize) -> &[u32] {
        let start = i * columns;
        let end = start + columns;
        &self.cells[start..end]
    }
}

/// Unique strings in a column.
pub struct Uniques(Vec<Counted>);

impl Uniques {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn get(&self, i: u32) -> &Counted {
        &self.0[i as usize]
    }

    pub fn add(&mut self, s: &str) -> u32 {
        let i = match self.0.binary_search_by(|c| c.s.as_str().cmp(&s)) {
            Ok(i) => i,
            Err(i) => {
                self.0.insert(i, Counted { count: 0, s: s.into() });
                i
            }
        };
        self.0[i].count += 1;
        debug_assert!(i <= u32::MAX as usize);
        i as u32
    }
}

/// Counted string.
pub struct Counted {
    /// How many times the string reoccurs.
    count: u32,

    /// The string.
    s: CompactString,
}

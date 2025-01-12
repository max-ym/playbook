use std::iter::Peekable;

use super::*;

/// Load a CSV file into a Table with the first row as header.
/// This will also check whether the bytes are actually a CSV.
pub fn with_header(bytes: &[u8], delim: char) -> Result<Table, LoadError> {
    let string = std::str::from_utf8(bytes).map_err(|_| LoadError::InvalidUtf8)?;
    let mut chars = string.chars().peekable();

    let mut columns = Vec::new();
    let mut strbuf = String::with_capacity(256);
    loop {
        let (acc, eol) = scan_str(&mut chars, delim, &mut strbuf)?;
        columns.push(acc);
        if eol {
            break;
        }
    }

    let mut uniques: Vec<Uniques> = Vec::with_capacity(columns.len());
    let mut cells = Vec::new();
    while chars.peek().is_some() {
        let mut count = 0;
        loop {
            let (acc, eol) = scan_str(&mut chars, delim, &mut strbuf)?;
            if let Some(uniques) = uniques.get_mut(count) {
                cells.push(uniques.add(acc.as_str()));
            } else {
                // Too many columns.
                return Err(LoadError::UnmatchingColumnCount);
            }

            count += 1;

            if eol {
                if count != columns.len() {
                    return Err(LoadError::UnmatchingColumnCount);
                }
                break;
            }
        }
    }

    Ok(Table {
        columns,
        rows: Rows { cells },
        unique_per_col: uniques,
    })
}

fn scan_str(
    chars: &mut Peekable<impl Iterator<Item = char>>,
    delim: char,
    buf: &mut String,
) -> Result<(CompactString, bool), LoadError> {
    buf.clear();
    let mut eol = false;

    let mut quoted = false;
    if chars.peek() == Some(&'"') {
        chars.next();
        quoted = true;
    }

    loop {
        match chars.next() {
            Some(c) if c == delim && quoted => break,
            Some(c) if c == '\n' => {
                if quoted {
                    return Err(LoadError::UnclosedQuote);
                } else {
                    eol = true;
                    break;
                }
            }
            Some(c) if c == '\r' => {
                if quoted {
                    return Err(LoadError::UnclosedQuote);
                } else if let Some('\n') = chars.peek() {
                    chars.next();
                }
                eol = true;
                break;
            }
            Some(c) if c == '"' => {
                if quoted {
                    if chars.peek() == Some(&'"') {
                        chars.next();
                        if chars.peek() == Some(&'"') {
                            chars.next();
                        } else {
                            return Err(LoadError::BrokenEscapeQuote);
                        }
                    } else {
                        break;
                    }
                }
                buf.push('"');
            }
            Some(c) => buf.push(c),
            None => break,
        }
    }

    Ok((CompactString::new(buf), eol))
}

pub enum LoadError {
    InvalidUtf8,
    UnmatchingColumnCount,
    BrokenEscapeQuote,
    UnclosedQuote,
}

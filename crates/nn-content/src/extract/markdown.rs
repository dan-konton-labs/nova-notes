//! Small Markdown writers shared by the HTML, spreadsheet and Office extractors.

/// Render `rows` as a Markdown pipe table. The first row is the header; rows
/// are padded to the widest row. Returns an empty string when there are no
/// rows or no columns.
pub(crate) fn table(rows: &[Vec<String>]) -> String {
    let width = rows.iter().map(Vec::len).max().unwrap_or(0);
    if width == 0 {
        return String::new();
    }
    let mut out = String::new();
    for (index, row) in rows.iter().enumerate() {
        out.push('|');
        for column in 0..width {
            out.push(' ');
            out.push_str(&cell(row.get(column).map_or("", String::as_str)));
            out.push_str(" |");
        }
        out.push('\n');
        if index == 0 {
            out.push('|');
            for _ in 0..width {
                out.push_str(" --- |");
            }
            out.push('\n');
        }
    }
    out.pop();
    out
}

/// Make `text` safe inside a table cell: one line, pipes escaped.
pub(crate) fn cell(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut pending_space = false;
    for c in text.trim().chars() {
        if c == '\n' || c == '\r' || c == '\t' {
            pending_space = true;
            continue;
        }
        if pending_space {
            if !out.ends_with(' ') {
                out.push(' ');
            }
            pending_space = false;
        }
        if c == '|' {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Collapse runs of three or more newlines to a blank line, strip trailing
/// spaces on each line, and trim the whole text.
pub(crate) fn tidy(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut blank_run = 0;
    for line in text.lines() {
        let line = line.trim_end();
        if line.is_empty() {
            blank_run += 1;
            continue;
        }
        if !out.is_empty() {
            out.push('\n');
            if blank_run > 0 {
                out.push('\n');
            }
        }
        blank_run = 0;
        out.push_str(line);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(cells: &[&str]) -> Vec<String> {
        cells.iter().map(|c| c.to_string()).collect()
    }

    #[test]
    fn renders_table_with_padding_and_escaping() {
        let rows = vec![row(&["Name", "Note"]), row(&["a|b", "line one\nline two"]), row(&["only"])];
        assert_eq!(
            table(&rows),
            "| Name | Note |\n| --- | --- |\n| a\\|b | line one line two |\n| only |  |"
        );
        assert_eq!(table(&[]), "");
        assert_eq!(table(&[vec![]]), "");
    }

    #[test]
    fn tidy_collapses_blank_lines() {
        assert_eq!(tidy("\n\na  \n\n\n\nb\nc\n\n"), "a\n\nb\nc");
    }
}

//! Spreadsheets and delimited text as Markdown tables.
//!
//! Workbooks (`xlsx`, `xlsm`, `xlsb`, `xls`, `ods`) are read with calamine:
//! one `## sheet name` section per sheet that has data, cell values as the
//! workbook stores them (formula results, not formulas). CSV and TSV become a
//! single table; the delimiter of a CSV file is detected.

use std::io::Cursor;

use calamine::{Data, Ods, Range, Reader, Xls, Xlsb, Xlsx};

use super::{decode_text, DocOutput};
use crate::error::ContentError;
use crate::extract::detect::FileKind;
use crate::extract::markdown;

pub(super) fn extract_workbook(kind: &FileKind, bytes: &[u8]) -> Result<DocOutput, ContentError> {
    let cursor = Cursor::new(bytes);
    let unreadable = |e: String| ContentError::FileOperation(format!("cannot read the workbook: {e}"));
    let sheets = match kind {
        FileKind::Xlsx => read_sheets(Xlsx::new(cursor).map_err(|e| unreadable(e.to_string()))?),
        FileKind::Xlsb => read_sheets(Xlsb::new(cursor).map_err(|e| unreadable(e.to_string()))?),
        FileKind::Xls => read_sheets(Xls::new(cursor).map_err(|e| unreadable(e.to_string()))?),
        _ => read_sheets(Ods::new(cursor).map_err(|e| unreadable(e.to_string()))?),
    };

    let mut sections = Vec::new();
    let mut total_rows = 0usize;
    let sheet_count = sheets.len();
    for (name, range) in sheets {
        let rows = range_rows(&range);
        if rows.is_empty() {
            continue;
        }
        total_rows += rows.len();
        sections.push(format!("## {}\n\n{}", name.trim(), markdown::table(&rows)));
    }

    let mut out = DocOutput { content: sections.join("\n\n"), ..DocOutput::default() };
    out.set("sheets", sheet_count);
    out.set("rows", total_rows);
    Ok(out)
}

fn read_sheets<'a, R>(mut workbook: R) -> Vec<(String, Range<Data>)>
where
    R: Reader<Cursor<&'a [u8]>>,
{
    workbook.worksheets()
}

/// Non-empty rows of a sheet as display strings, trailing empty cells removed.
fn range_rows(range: &Range<Data>) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = range
        .rows()
        .map(|row| {
            let mut cells: Vec<String> = row.iter().map(cell_text).collect();
            while cells.last().is_some_and(|c| c.is_empty()) {
                cells.pop();
            }
            cells
        })
        .filter(|cells| !cells.is_empty())
        .collect();
    // Drop columns that are empty in every row.
    let width = rows.iter().map(Vec::len).max().unwrap_or(0);
    let used: Vec<bool> =
        (0..width).map(|column| rows.iter().any(|row| row.get(column).is_some_and(|c| !c.is_empty()))).collect();
    if used.iter().any(|u| !u) {
        for row in &mut rows {
            let mut column = 0;
            row.retain(|_| {
                let keep = used.get(column).copied().unwrap_or(true);
                column += 1;
                keep
            });
        }
    }
    rows
}

fn cell_text(cell: &Data) -> String {
    match cell {
        Data::Empty => String::new(),
        Data::String(text) => text.trim().to_string(),
        Data::Int(value) => value.to_string(),
        Data::Float(value) => {
            if value.fract() == 0.0 && value.abs() < 1e15 {
                format!("{}", *value as i64)
            } else {
                value.to_string()
            }
        }
        Data::Bool(value) => if *value { "TRUE" } else { "FALSE" }.to_string(),
        Data::DateTime(value) => match value.as_datetime() {
            Some(datetime) if !value.is_duration() => {
                let text = datetime.format("%Y-%m-%d %H:%M:%S").to_string();
                text.strip_suffix(" 00:00:00").map(str::to_string).unwrap_or(text)
            }
            _ => value.to_string(),
        },
        Data::DateTimeIso(text) | Data::DurationIso(text) => text.clone(),
        Data::Error(error) => error.to_string(),
    }
}

/// Render delimited text as one table. `delimiter` of `None` detects it from
/// the first line (comma, semicolon, tab or pipe).
pub(super) fn extract_delimited(bytes: &[u8], delimiter: Option<u8>) -> Result<DocOutput, ContentError> {
    let text = decode_text(bytes);
    let delimiter = delimiter.unwrap_or_else(|| sniff_delimiter(&text));
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .delimiter(delimiter)
        .from_reader(text.as_bytes());

    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|e| ContentError::FileOperation(format!("cannot parse the delimited file: {e}")))?;
        let cells: Vec<String> = record.iter().map(|field| field.trim().to_string()).collect();
        if cells.iter().any(|c| !c.is_empty()) {
            rows.push(cells);
        }
    }

    let mut out = DocOutput { content: markdown::table(&rows), ..DocOutput::default() };
    out.set("rows", rows.len());
    out.set("delimiter", (delimiter as char).to_string());
    Ok(out)
}

/// The candidate that occurs most often outside quotes in the first non-empty line.
fn sniff_delimiter(text: &str) -> u8 {
    let Some(line) = text.lines().find(|line| !line.trim().is_empty()) else {
        return b',';
    };
    let mut counts = [(b',', 0usize), (b';', 0), (b'\t', 0), (b'|', 0)];
    let mut quoted = false;
    for byte in line.bytes() {
        if byte == b'"' {
            quoted = !quoted;
        } else if !quoted {
            if let Some(entry) = counts.iter_mut().find(|(candidate, _)| *candidate == byte) {
                entry.1 += 1;
            }
        }
    }
    // max_by_key keeps the last maximum; reverse so comma wins ties.
    counts.iter().rev().max_by_key(|(_, count)| *count).map_or(b',', |(delimiter, _)| *delimiter)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::extract::documents::testutil::zip_bytes;

    /// A two-sheet workbook: shared strings, numbers, a boolean, an inline
    /// string, an empty row and an empty second sheet.
    pub(crate) fn xlsx_bytes() -> Vec<u8> {
        let content_types = r#"<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/></Types>"#;
        let root_rels = r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#;
        let workbook = r#"<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Inventory" sheetId="1" r:id="rId1"/><sheet name="Empty" sheetId="2" r:id="rId2"/></sheets></workbook>"#;
        let workbook_rels = r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/></Relationships>"#;
        let shared = r#"<?xml version="1.0" encoding="UTF-8"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="4" uniqueCount="4"><si><t>Item</t></si><si><t>Qty</t></si><si><t>In stock</t></si><si><t>Widget | large</t></si></sst>"#;
        let sheet1 = r#"<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c></row>
<row r="2"><c r="A2" t="s"><v>3</v></c><c r="B2"><v>12</v></c><c r="C2" t="b"><v>1</v></c></row>
<row r="4"><c r="A4" t="inlineStr"><is><t>Gadget</t></is></c><c r="B4"><v>2.5</v></c></row>
</sheetData></worksheet>"#;
        let sheet2 = r#"<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData/></worksheet>"#;
        zip_bytes(&[
            ("[Content_Types].xml", content_types.as_bytes()),
            ("_rels/.rels", root_rels.as_bytes()),
            ("xl/workbook.xml", workbook.as_bytes()),
            ("xl/_rels/workbook.xml.rels", workbook_rels.as_bytes()),
            ("xl/sharedStrings.xml", shared.as_bytes()),
            ("xl/worksheets/sheet1.xml", sheet1.as_bytes()),
            ("xl/worksheets/sheet2.xml", sheet2.as_bytes()),
        ])
    }

    #[test]
    fn xlsx_sheets_become_tables() {
        let out = extract_workbook(&FileKind::Xlsx, &xlsx_bytes()).unwrap();
        assert_eq!(
            out.content,
            "## Inventory\n\n| Item | Qty | In stock |\n| --- | --- | --- |\n| Widget \\| large | 12 | TRUE |\n| Gadget | 2.5 |  |"
        );
        assert_eq!(out.metadata["sheets"], 2);
        assert_eq!(out.metadata["rows"], 3);
        assert_eq!(out.title, None);
    }

    #[test]
    fn unreadable_workbook_is_a_file_error() {
        let error = extract_workbook(&FileKind::Xlsx, b"PK\x03\x04not a workbook").unwrap_err();
        assert!(matches!(error, ContentError::FileOperation(_)));
        let error = extract_workbook(&FileKind::Xls, b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1 nope").unwrap_err();
        assert!(matches!(error, ContentError::FileOperation(_)));
    }

    #[test]
    fn cell_formatting() {
        assert_eq!(cell_text(&Data::Float(3.0)), "3");
        assert_eq!(cell_text(&Data::Float(-0.125)), "-0.125");
        assert_eq!(cell_text(&Data::Int(42)), "42");
        assert_eq!(cell_text(&Data::Bool(false)), "FALSE");
        assert_eq!(cell_text(&Data::String("  padded ".into())), "padded");
        assert_eq!(cell_text(&Data::Empty), "");
        assert_eq!(cell_text(&Data::DateTimeIso("2024-05-01T10:00:00".into())), "2024-05-01T10:00:00");
        let date = Data::DateTime(calamine::ExcelDateTime::new(45413.0, calamine::ExcelDateTimeType::DateTime, false));
        assert_eq!(cell_text(&date), "2024-05-01");
        let stamp = Data::DateTime(calamine::ExcelDateTime::new(45413.5, calamine::ExcelDateTimeType::DateTime, false));
        assert_eq!(cell_text(&stamp), "2024-05-01 12:00:00");
    }

    #[test]
    fn csv_with_quotes_and_ragged_rows() {
        let csv = "name,role,notes\n\"Lovelace, Ada\",Engineer,\"Wrote the first\nprogram\"\n\nGrace,Admiral\n";
        let out = extract_delimited(csv.as_bytes(), None).unwrap();
        assert_eq!(
            out.content,
            "| name | role | notes |\n| --- | --- | --- |\n| Lovelace, Ada | Engineer | Wrote the first program |\n| Grace | Admiral |  |"
        );
        assert_eq!(out.metadata["rows"], 3);
    }

    #[test]
    fn delimiters_are_detected() {
        let semicolon = extract_delimited("a;b;c\n1,5;2;3\n".as_bytes(), None).unwrap();
        assert_eq!(semicolon.content, "| a | b | c |\n| --- | --- | --- |\n| 1,5 | 2 | 3 |");
        let tab = extract_delimited(b"a\tb\n1\t2\n", Some(b'\t')).unwrap();
        assert_eq!(tab.content, "| a | b |\n| --- | --- |\n| 1 | 2 |");
        assert_eq!(sniff_delimiter("x|y|z\n"), b'|');
        assert_eq!(sniff_delimiter("\"a;b;c\",d\n"), b',');
        assert_eq!(sniff_delimiter("single\n"), b',');
        assert_eq!(sniff_delimiter(""), b',');
        let latin1 = extract_delimited(b"nom,ville\nRen\xe9,Orl\xe9ans\n", None).unwrap();
        assert!(latin1.content.contains("| René | Orléans |"));
    }
}

//! Spreadsheets and delimited text as Markdown tables.
//!
//! Workbooks (`xlsx`, `xlsm`, `xlsb`, `xls`, `ods`) are read with calamine:
//! one `## sheet name` section per sheet that has data, cell values as the
//! workbook stores them (formula results, not formulas). CSV and TSV become a
//! single table; the delimiter of a CSV file is detected.

use std::io::Cursor;

use calamine::{Data, Ods, Reader, Xls, Xlsb, Xlsx};

use super::{check_zip_budget, decode_text, DocOutput};
use crate::error::ContentError;
use crate::extract::detect::FileKind;
use crate::extract::markdown;

/// Most non-empty cells read from one workbook, all sheets together. Further
/// cells are left out and the output says so.
const MAX_WORKBOOK_CELLS: usize = 2_000_000;
/// Largest table rendered for one sheet, in rows x columns that hold data.
const MAX_SHEET_GRID: usize = 5_000_000;

pub(super) fn extract_workbook(kind: &FileKind, bytes: &[u8]) -> Result<DocOutput, ContentError> {
    let unreadable = |e: String| ContentError::FileOperation(format!("cannot read the workbook: {e}"));
    if matches!(kind, FileKind::Xlsx | FileKind::Xlsb | FileKind::Ods) {
        check_zip_budget(bytes)?;
    }
    if matches!(kind, FileKind::Ods) {
        ods::check_size(bytes)?;
    }
    let cursor = Cursor::new(bytes);
    let mut budget = CellBudget { remaining: MAX_WORKBOOK_CELLS, truncated: false };
    // Sheets are read cell by cell where the format allows it: calamine's
    // `worksheets()` expands each sheet to a dense grid over the bounding box
    // of its cells, so two cells in opposite corners would cost 17 billion.
    let sheets = match kind {
        FileKind::Xlsx => {
            let mut book = Xlsx::new(cursor).map_err(|e| unreadable(e.to_string()))?;
            let mut sheets = Vec::new();
            for name in book.sheet_names() {
                let Ok(mut reader) = book.worksheet_cells_reader(&name) else { continue };
                let mut cells = Vec::new();
                while let Some(cell) = reader.next_cell().map_err(|e| unreadable(e.to_string()))? {
                    if !push_cell(&mut cells, cell.get_position(), &cell.get_value().clone().into(), &mut budget) {
                        break;
                    }
                }
                sheets.push((name, cells));
            }
            sheets
        }
        FileKind::Xlsb => {
            let mut book = Xlsb::new(cursor).map_err(|e| unreadable(e.to_string()))?;
            let mut sheets = Vec::new();
            for name in book.sheet_names() {
                let Ok(mut reader) = book.worksheet_cells_reader(&name) else { continue };
                let mut cells = Vec::new();
                while let Some(cell) = reader.next_cell().map_err(|e| unreadable(e.to_string()))? {
                    if !push_cell(&mut cells, cell.get_position(), &cell.get_value().clone().into(), &mut budget) {
                        break;
                    }
                }
                sheets.push((name, cells));
            }
            sheets
        }
        // The format caps these sheets (XLS at 65,536 x 256 cells, ODS by the
        // size check above); only their used cells are kept.
        FileKind::Xls => dense_sheets(Xls::new(cursor).map_err(|e| unreadable(e.to_string()))?, &mut budget),
        _ => dense_sheets(Ods::new(cursor).map_err(|e| unreadable(e.to_string()))?, &mut budget),
    };

    let mut sections = Vec::new();
    let mut total_rows = 0usize;
    let sheet_count = sheets.len();
    for (name, cells) in sheets {
        let (rows, cut) = sparse_rows(cells);
        budget.truncated |= cut;
        if rows.is_empty() {
            continue;
        }
        total_rows += rows.len();
        sections.push(format!("## {}\n\n{}", name.trim(), markdown::table(&rows)));
    }

    let mut out = DocOutput { content: sections.join("\n\n"), ..DocOutput::default() };
    out.set("sheets", sheet_count);
    out.set("rows", total_rows);
    if budget.truncated {
        tracing::warn!("workbook is larger than the import limits; part of it was left out");
        out.set("truncated", true);
    }
    Ok(out)
}

/// Cells left to read from the workbook.
struct CellBudget {
    remaining: usize,
    truncated: bool,
}

impl CellBudget {
    /// Whether another cell may be read; records that the workbook was cut short.
    fn take(&mut self) -> bool {
        if self.remaining == 0 {
            self.truncated = true;
            return false;
        }
        self.remaining -= 1;
        true
    }
}

/// A non-empty cell: `(row, column, text)`.
type SparseCell = (u32, u32, String);

/// Keep `value` if it shows anything. False when the budget is spent and
/// reading must stop.
fn push_cell(cells: &mut Vec<SparseCell>, (row, column): (u32, u32), value: &Data, budget: &mut CellBudget) -> bool {
    let text = cell_text(value);
    if text.is_empty() {
        return true;
    }
    if !budget.take() {
        return false;
    }
    cells.push((row, column, text));
    true
}

/// Used cells of every sheet of a workbook whose reader only offers whole ranges.
fn dense_sheets<'a, R>(mut workbook: R, budget: &mut CellBudget) -> Vec<(String, Vec<SparseCell>)>
where
    R: Reader<Cursor<&'a [u8]>>,
{
    let mut sheets = Vec::new();
    for name in workbook.sheet_names() {
        let Ok(range) = workbook.worksheet_range(&name) else { continue };
        let mut cells = Vec::new();
        let (start_row, start_column) = range.start().unwrap_or((0, 0));
        for (row, column, value) in range.used_cells() {
            let position = (start_row.saturating_add(row as u32), start_column.saturating_add(column as u32));
            if !push_cell(&mut cells, position, value, budget) {
                break;
            }
        }
        sheets.push((name, cells));
    }
    sheets
}

/// Rows of a sheet as display strings: rows without data are left out,
/// columns without data in any row are left out, trailing empty cells are
/// trimmed. Built from the used cells only, never from the bounding box.
/// The second value is true when the table had to be cut to [`MAX_SHEET_GRID`].
fn sparse_rows(mut cells: Vec<SparseCell>) -> (Vec<Vec<String>>, bool) {
    // Stable sort: of two cells at one position, the later one wins.
    cells.sort_by_key(|(row, column, _)| (*row, *column));
    let mut columns: Vec<u32> = cells.iter().map(|(_, column, _)| *column).collect();
    columns.sort_unstable();
    columns.dedup();
    let width = columns.len().max(1);

    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut current_row = None;
    let mut cut = false;
    for (row, column, text) in cells {
        if current_row != Some(row) {
            if (rows.len() + 1).saturating_mul(width) > MAX_SHEET_GRID {
                cut = true;
                break;
            }
            current_row = Some(row);
            rows.push(Vec::new());
        }
        let Some(line) = rows.last_mut() else { break };
        let index = columns.binary_search(&column).unwrap_or_default();
        if line.len() <= index {
            line.resize(index + 1, String::new());
        }
        line[index] = text;
    }
    (rows, cut)
}

/// Size check for OpenDocument spreadsheets, which calamine expands in full
/// (repeated rows and columns included) as soon as the file is opened.
mod ods {
    use std::io::{BufReader, Cursor, Read};

    use quick_xml::events::{BytesStart, Event};
    use quick_xml::Reader;

    use super::MAX_SHEET_GRID;
    use crate::error::ContentError;
    use crate::extract::documents::MAX_DOCUMENT_BYTES;

    const MAX_ROWS: usize = 1_048_576;
    const MAX_COLUMNS: usize = 16_384;

    fn repeats(element: &BytesStart, name: &str, cap: usize) -> usize {
        element
            .try_get_attribute(name)
            .ok()
            .flatten()
            .and_then(|attr| attr.value.trim().parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, cap.max(1))
    }

    fn has_value(element: &BytesStart) -> bool {
        ["office:value-type", "table:formula"].iter().any(|name| element.try_get_attribute(name).ok().flatten().is_some())
    }

    /// Reject the workbook when calamine would materialise more than
    /// [`MAX_SHEET_GRID`] cells: the cells it stores while reading rows, or
    /// the grid it then builds from the first to the last cell with data.
    /// Errs on the side of counting a cell as holding data.
    pub(super) fn check_size(bytes: &[u8]) -> Result<(), ContentError> {
        let Ok(mut archive) = zip::ZipArchive::new(Cursor::new(bytes)) else { return Ok(()) };
        let Ok(content) = archive.by_name("content.xml") else { return Ok(()) };
        let mut reader = Reader::from_reader(BufReader::new(content.take(MAX_DOCUMENT_BYTES)));
        let too_large = || {
            ContentError::FileOperation(format!("the spreadsheet has more than {MAX_SHEET_GRID} cells to import"))
        };

        let mut buf = Vec::new();
        let (mut stored, mut grid) = (0usize, 0usize);
        // Per sheet: next row index, first and last row with data, column span.
        let mut row_index = 0usize;
        let mut rows: Option<(usize, usize)> = None;
        let mut columns: Option<(usize, usize)> = None;
        // Per row: repeat count, next column, first column with data, end of the last one.
        let mut row: Option<(usize, usize, Option<usize>, usize)> = None;
        loop {
            let event = match reader.read_event_into(&mut buf) {
                Ok(event) => event,
                // Broken XML: calamine reports it.
                Err(_) => return Ok(()),
            };
            match event {
                Event::Eof => break,
                Event::Start(e) | Event::Empty(e) if e.name().as_ref() == "table:table" => {
                    row_index = 0;
                    rows = None;
                    columns = None;
                }
                Event::Start(e) if e.name().as_ref() == "table:table-row" => {
                    let repeat = repeats(&e, "table:number-rows-repeated", MAX_ROWS.saturating_sub(row_index));
                    row = Some((repeat, 0, None, 0));
                }
                Event::Empty(e) if e.name().as_ref() == "table:table-row" => {
                    row_index += repeats(&e, "table:number-rows-repeated", MAX_ROWS.saturating_sub(row_index));
                }
                ref event @ (Event::Start(ref e) | Event::Empty(ref e))
                    if matches!(e.name().as_ref(), "table:table-cell" | "table:covered-table-cell") =>
                {
                    let Some((_, column, first, end)) = row.as_mut() else { continue };
                    let repeat = repeats(e, "table:number-columns-repeated", MAX_COLUMNS.saturating_sub(*column));
                    if matches!(event, Event::Start(_)) || has_value(e) {
                        first.get_or_insert(*column);
                        *end = *column + repeat;
                    }
                    *column = (*column + repeat).min(MAX_COLUMNS);
                }
                Event::End(e) if e.name().as_ref() == "table:table-row" => {
                    let Some((repeat, _, first, end)) = row.take() else { continue };
                    if let Some(first) = first {
                        stored = stored.saturating_add(end);
                        let (top, _) = rows.unwrap_or((row_index, 0));
                        rows = Some((top, row_index + repeat - 1));
                        let (left, right) = columns.unwrap_or((first, end));
                        columns = Some((left.min(first), right.max(end)));
                    }
                    row_index = (row_index + repeat).min(MAX_ROWS);
                }
                Event::End(e) if e.name().as_ref() == "table:table" => {
                    if let (Some((top, bottom)), Some((left, right))) = (rows, columns) {
                        grid = grid.saturating_add((bottom - top + 1).saturating_mul(right - left));
                    }
                }
                _ => {}
            }
            if stored > MAX_SHEET_GRID || grid > MAX_SHEET_GRID {
                return Err(too_large());
            }
            buf.clear();
        }
        Ok(())
    }
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

    /// A one-sheet workbook whose cells are `(reference, text)` inline strings.
    fn xlsx_with_cells(cells: &[(String, String)]) -> Vec<u8> {
        let workbook = r#"<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Data" sheetId="1" r:id="rId1"/></sheets></workbook>"#;
        let rels = r#"<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>"#;
        let root_rels = r#"<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#;
        let mut sheet = String::from(r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:XFD1048576"/><sheetData>"#);
        for (reference, text) in cells {
            sheet.push_str(&format!(r#"<c r="{reference}" t="inlineStr"><is><t>{text}</t></is></c>"#));
        }
        sheet.push_str("</sheetData></worksheet>");
        zip_bytes(&[
            ("_rels/.rels", root_rels.as_bytes()),
            ("xl/workbook.xml", workbook.as_bytes()),
            ("xl/_rels/workbook.xml.rels", rels.as_bytes()),
            ("xl/worksheets/sheet1.xml", sheet.as_bytes()),
        ])
    }

    fn column_name(mut index: usize) -> String {
        let mut name = Vec::new();
        loop {
            name.push(b'A' + (index % 26) as u8);
            if index < 26 {
                break;
            }
            index = index / 26 - 1;
        }
        name.reverse();
        String::from_utf8(name).unwrap()
    }

    #[test]
    fn sparse_workbook_is_read_from_its_used_cells() {
        // Two cells in opposite corners: a dense grid would be 16,384 x
        // 1,048,576 cells (about 550 GB of `Data`).
        let bytes = xlsx_with_cells(&[("A1".into(), "corner".into()), ("XFD1048576".into(), "far".into())]);
        let started = std::time::Instant::now();
        let out = extract_workbook(&FileKind::Xlsx, &bytes).unwrap();
        assert!(started.elapsed() < std::time::Duration::from_secs(10));
        assert_eq!(out.content, "## Data\n\n| corner |  |\n| --- | --- |\n|  | far |");
        assert_eq!(out.metadata["rows"], 2);
        assert!(out.metadata.get("truncated").is_none());
    }

    #[test]
    fn diagonal_workbook_is_cut_to_the_table_limit() {
        // Every row and every column used once: a 3,000 x 3,000 table.
        let cells: Vec<(String, String)> = (0..3000).map(|i| (format!("{}{}", column_name(i), i + 1), format!("v{i}"))).collect();
        let out = extract_workbook(&FileKind::Xlsx, &xlsx_with_cells(&cells)).unwrap();
        assert_eq!(out.metadata["truncated"], true);
        let rows = out.metadata["rows"].as_u64().unwrap() as usize;
        assert!(rows > 0 && rows * 3000 <= MAX_SHEET_GRID, "{rows}");
        assert_eq!(column_name(0), "A");
        assert_eq!(column_name(26), "AA");
        assert_eq!(column_name(16383), "XFD");
    }

    fn ods_with_table(rows: &str) -> Vec<u8> {
        let content = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?><office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" office:version="1.2"><office:body><office:spreadsheet><table:table table:name="Sheet1">{rows}</table:table></office:spreadsheet></office:body></office:document-content>"#
        );
        let manifest = r#"<?xml version="1.0" encoding="UTF-8"?><manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.2"><manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.spreadsheet"/><manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/></manifest:manifest>"#;
        zip_bytes(&[
            ("mimetype", b"application/vnd.oasis.opendocument.spreadsheet"),
            ("META-INF/manifest.xml", manifest.as_bytes()),
            ("content.xml", content.as_bytes()),
        ])
    }

    #[test]
    fn ods_is_read_and_a_repeated_grid_is_refused() {
        let small = ods_with_table(
            r#"<table:table-row><table:table-cell office:value-type="string"><text:p>Name</text:p></table:table-cell><table:table-cell table:number-columns-repeated="3"/><table:table-cell office:value-type="float" office:value="7"><text:p>7</text:p></table:table-cell></table:table-row>
<table:table-row table:number-rows-repeated="1048575"><table:table-cell table:number-columns-repeated="1024"/></table:table-row>"#,
        );
        let out = extract_workbook(&FileKind::Ods, &small).unwrap();
        assert_eq!(out.content, "## Sheet1\n\n| Name | 7 |\n| --- | --- |");

        // A few hundred bytes that calamine would expand to 16,384 x 6,000
        // cells (about 5 GB with its parallel formula grid), just under its
        // own 100-million-cell cap.
        let huge = ods_with_table(
            r#"<table:table-row table:number-rows-repeated="6000"><table:table-cell office:value-type="float" office:value="1"><text:p>1</text:p></table:table-cell><table:table-cell table:number-columns-repeated="16382"/><table:table-cell office:value-type="float" office:value="1"><text:p>1</text:p></table:table-cell></table:table-row>"#,
        );
        let started = std::time::Instant::now();
        let error = extract_workbook(&FileKind::Ods, &huge).unwrap_err();
        assert!(matches!(error, ContentError::FileOperation(ref m) if m.contains("cells")), "{error:?}");
        assert!(started.elapsed() < std::time::Duration::from_secs(10));

        // Many stored rows, each wide: refused on what calamine stores while reading.
        let row = r#"<table:table-row><table:table-cell table:number-columns-repeated="16383"/><table:table-cell office:value-type="float" office:value="1"><text:p>1</text:p></table:table-cell></table:table-row>"#;
        let error = extract_workbook(&FileKind::Ods, &ods_with_table(&row.repeat(400))).unwrap_err();
        assert!(matches!(error, ContentError::FileOperation(_)), "{error:?}");
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

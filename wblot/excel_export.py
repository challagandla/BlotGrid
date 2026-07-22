"""Small, dependency-free XLSX writer used for Streamlit downloads.

The normal openpyxl writer serializes every worksheet through a temporary file.
That can make an otherwise in-memory download fail when the system temporary
drive is full.  The tables exported by this app only need scalar cells, so this
module writes the corresponding OOXML parts directly to an in-memory ZIP.
"""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Mapping
from datetime import date, datetime, time, timezone
from decimal import Decimal
from io import BytesIO
from pathlib import PurePosixPath
from xml.sax.saxutils import escape, quoteattr
from zipfile import ZIP_DEFLATED, ZipFile


_INVALID_XML = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]"
)
_INVALID_SHEET_NAME = re.compile(r"[\\/*?:\[\]]")
_MAX_CELL_TEXT = 32_767


def _clean_text(value: object) -> str:
    text = _INVALID_XML.sub("", str(value))
    return text[:_MAX_CELL_TEXT]


def _excel_column(number: int) -> str:
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _normalise_scalar(value: object) -> object | None:
    if value is None or type(value).__name__ in {"NAType", "NaTType"}:
        return None

    # Convert NumPy scalar values without importing NumPy here.
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = item()
        except (TypeError, ValueError):
            pass

    if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
        try:
            if not math.isfinite(float(value)):
                return None
        except (TypeError, ValueError, OverflowError):
            pass
    if isinstance(value, Decimal) and (value.is_nan() or value.is_infinite()):
        return None
    return value


def _cell_xml(reference: str, value: object, *, header: bool = False) -> str:
    value = _normalise_scalar(value)
    style = ' s="1"' if header else ""
    if value is None:
        return f'<c r="{reference}"{style}/>'
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"{style}><v>{int(value)}</v></c>'
    if isinstance(value, numbers.Integral):
        return f'<c r="{reference}"{style}><v>{int(value)}</v></c>'
    if isinstance(value, (numbers.Real, Decimal)):
        return f'<c r="{reference}"{style}><v>{value}</v></c>'
    if isinstance(value, (datetime, date, time)):
        value = value.isoformat()
    elif isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8", errors="replace")
    text = escape(_clean_text(value))
    return (
        f'<c r="{reference}" t="inlineStr"{style}>'
        f'<is><t xml:space="preserve">{text}</t></is></c>'
    )


def _safe_sheet_names(names: list[object]) -> list[str]:
    used: set[str] = set()
    result: list[str] = []
    for index, original in enumerate(names, start=1):
        base = _INVALID_SHEET_NAME.sub("_", _clean_text(original)).strip("'")
        base = (base or f"Sheet{index}")[:31]
        candidate = base
        suffix = 2
        while candidate.casefold() in used:
            marker = f"_{suffix}"
            candidate = f"{base[: 31 - len(marker)]}{marker}"
            suffix += 1
        used.add(candidate.casefold())
        result.append(candidate)
    return result


def _worksheet_xml(frame: object) -> str:
    columns = list(frame.columns)
    body: list[str] = []
    if columns:
        header_cells = "".join(
            _cell_xml(f"{_excel_column(column)}1", name, header=True)
            for column, name in enumerate(columns, start=1)
        )
        body.append(f'<row r="1">{header_cells}</row>')

    row_number = 2
    for values in frame.itertuples(index=False, name=None):
        cells = "".join(
            _cell_xml(f"{_excel_column(column)}{row_number}", value)
            for column, value in enumerate(values, start=1)
        )
        body.append(f'<row r="{row_number}">{cells}</row>')
        row_number += 1

    last_column = _excel_column(max(1, len(columns)))
    last_row = max(1, row_number - 1)
    auto_filter = (
        f'<autoFilter ref="A1:{last_column}{last_row}"/>' if columns else ""
    )
    frozen_header = (
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '</sheetView></sheetViews>'
        if columns
        else '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="A1:{last_column}{last_row}"/>{frozen_header}'
        '<sheetFormatPr defaultRowHeight="15"/>'
        f'<sheetData>{"".join(body)}</sheetData>{auto_filter}'
        '</worksheet>'
    )


_STYLES_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="0"/>
  <fonts count="2">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="0" borderId="0" applyFont="1" xfId="0"/>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''


def dataframes_to_xlsx(sheets: Mapping[str, object]) -> bytes:
    """Return an XLSX workbook containing the supplied pandas-like frames.

    Each frame must expose ``columns`` and ``itertuples(index=False, name=None)``.
    Missing and non-finite values become blank cells, and XML-illegal control
    characters are removed from text values.
    """

    if not sheets:
        raise ValueError("At least one worksheet is required")

    frames = list(sheets.values())
    names = _safe_sheet_names(list(sheets.keys()))
    sheet_overrides = "".join(
        '<Override PartName="/xl/worksheets/sheet{0}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'.format(index)
        for index in range(1, len(names) + 1)
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        f'{sheet_overrides}</Types>'
    )
    workbook_sheets = "".join(
        f'<sheet name={quoteattr(name)} sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(names, start=1)
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{workbook_sheets}</sheets><calcPr calcId="191029"/></workbook>'
    )
    workbook_rels = "".join(
        '<Relationship Id="rId{0}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet{0}.xml"/>'.format(index)
        for index in range(1, len(names) + 1)
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{workbook_rels}'
        f'<Relationship Id="rId{len(names) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        '</Relationships>'
    )
    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    core = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:creator>Western Blot Quantification</dc:creator>'
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>'
        '</cp:coreProperties>'
    )
    titles = "".join(f'<vt:lpstr>{escape(name)}</vt:lpstr>' for name in names)
    app = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        '<Application>Western Blot Quantification</Application>'
        f'<TitlesOfParts><vt:vector size="{len(names)}" baseType="lpstr">{titles}</vt:vector></TitlesOfParts>'
        '</Properties>'
    )

    output = BytesIO()
    with ZipFile(output, mode="w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr(PurePosixPath("[Content_Types].xml").as_posix(), content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("docProps/core.xml", core)
        archive.writestr("docProps/app.xml", app)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", _STYLES_XML)
        for index, frame in enumerate(frames, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _worksheet_xml(frame))
    return output.getvalue()

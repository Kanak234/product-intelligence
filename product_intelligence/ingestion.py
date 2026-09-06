"""
Multi-source catalogue ingestion.

The challenge names three sources explicitly — websites, catalogues, and
technical documents — so the pipeline has to accept all of them, not just a
text box. Each reader turns its format into a list of `RawProduct`, and from
that point the rest of the pipeline is source-agnostic.

Hard rule: readers extract, they never invent. A column the CSV does not have
becomes an absent field, not an empty string, so that downstream completeness
metrics stay honest.

Optional dependencies degrade rather than crash: `openpyxl` for .xlsx and
`pypdf` for PDF text. Without them the other formats still work, which keeps
the offline install light.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import RawProduct


#: Column headers that map onto the three first-class fields. Everything else
#: becomes a structured attribute, which is usually the more useful outcome.
NAME_COLUMNS = ("name", "product", "product_name", "title", "item", "item_name",
                "description_short", "model", "part", "part_name")
DESC_COLUMNS = ("description", "details", "long_description", "specs", "specification",
                "specifications", "features", "notes", "remarks", "product_description")
CATEGORY_COLUMNS = ("category", "product_category", "type", "product_type", "family",
                    "group", "segment", "class")
SKU_COLUMNS = ("sku", "part_number", "part_no", "partno", "model_number", "model_no",
               "item_code", "code", "mpn", "catalog_number")


@dataclass
class IngestionResult:
    products: List[RawProduct]
    source_type: str
    source_ref: str
    row_count: int
    skipped: List[Dict[str, Any]]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "row_count": self.row_count,
            "ingested": len(self.products),
            "skipped": self.skipped,
            "warnings": self.warnings,
        }


class IngestionError(ValueError):
    """Raised when a source cannot be parsed at all."""


# =========================================================================
# CSV / TSV
# =========================================================================


def from_csv(
    content: str,
    source_ref: str = "upload.csv",
    delimiter: Optional[str] = None,
) -> IngestionResult:
    """Parse a delimited catalogue export. Delimiter is sniffed if not given."""
    if not content.strip():
        raise IngestionError("The file is empty.")

    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(content[:4096], delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","

    reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
    if not reader.fieldnames:
        raise IngestionError("No header row could be identified.")

    headers = [(h or "").strip() for h in reader.fieldnames]
    mapping = _map_columns(headers)
    warnings: List[str] = []
    if mapping["name"] is None:
        raise IngestionError(
            "No product name column found. Expected one of: "
            + ", ".join(NAME_COLUMNS[:6])
        )
    if mapping["description"] is None:
        warnings.append(
            "No description column found; enrichment will rely on the product name alone."
        )

    products: List[RawProduct] = []
    skipped: List[Dict[str, Any]] = []
    row_count = 0

    for index, row in enumerate(reader, start=2):   # row 1 is the header
        row_count += 1
        name = _clean(row.get(mapping["name"]))
        if not name:
            skipped.append({"row": index, "reason": "empty product name"})
            continue

        attributes = {
            header: _clean(row.get(header))
            for header in headers
            if header
            and header not in mapping.values()
            and _clean(row.get(header))
        }

        products.append(RawProduct(
            name=name,
            description=_clean(row.get(mapping["description"])) if mapping["description"] else "",
            category=_clean(row.get(mapping["category"])) if mapping["category"] else "",
            source="csv",
            source_ref=f"{source_ref}#row{index}",
            attributes=attributes,
        ))

    return IngestionResult(products, "csv", source_ref, row_count, skipped, warnings)


# =========================================================================
# JSON / JSONL
# =========================================================================


def from_json(content: str, source_ref: str = "upload.json") -> IngestionResult:
    """Accept a JSON array, a JSONL stream, or an object wrapping a list."""
    content = content.strip()
    if not content:
        raise IngestionError("The file is empty.")

    records: List[Dict[str, Any]] = []
    warnings: List[str] = []

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            records = [r for r in parsed if isinstance(r, dict)]
        elif isinstance(parsed, dict):
            for key in ("products", "items", "data", "records", "results"):
                if isinstance(parsed.get(key), list):
                    records = [r for r in parsed[key] if isinstance(r, dict)]
                    break
            else:
                records = [parsed]
    except json.JSONDecodeError:
        # JSONL fallback
        for line_no, line in enumerate(content.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                warnings.append(f"Line {line_no} is not valid JSON; skipped.")

    if not records:
        raise IngestionError("No product objects found in the JSON payload.")

    products: List[RawProduct] = []
    skipped: List[Dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        flat = _flatten(record)
        mapping = _map_columns(list(flat))
        name = _clean(flat.get(mapping["name"])) if mapping["name"] else ""
        if not name:
            skipped.append({"row": index, "reason": "no identifiable name field"})
            continue

        attributes = {
            key: _clean(value)
            for key, value in flat.items()
            if key not in mapping.values() and _clean(value)
        }

        products.append(RawProduct(
            name=name,
            description=_clean(flat.get(mapping["description"])) if mapping["description"] else "",
            category=_clean(flat.get(mapping["category"])) if mapping["category"] else "",
            source="json",
            source_ref=f"{source_ref}#{index}",
            attributes=attributes,
        ))

    return IngestionResult(products, "json", source_ref, len(records), skipped, warnings)


# =========================================================================
# HTML (product pages and spec tables)
# =========================================================================


class _ProductPageParser(HTMLParser):
    """
    Pulls a product title and any two-column spec tables out of a web page.

    Deliberately conservative: it reads <h1>/<title> for the name and <table>
    or <dl> pairs for specs, and ignores everything else. Scraping heuristics
    that guess harder than this produce noise that then has to be validated
    back out.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str = ""
        self.h1: str = ""
        self.pairs: List[Tuple[str, str]] = []
        self.paragraphs: List[str] = []

        self._stack: List[str] = []
        self._cell_buffer: List[str] = []
        self._row_cells: List[str] = []
        self._text_buffer: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip_depth += 1
            return
        self._stack.append(tag)
        if tag == "tr":
            self._row_cells = []
        if tag in ("td", "th", "dt", "dd", "p", "h1", "title"):
            self._cell_buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        text = " ".join(" ".join(self._cell_buffer).split())

        if tag in ("td", "th", "dt", "dd"):
            self._row_cells.append(text)
        elif tag == "tr":
            if len(self._row_cells) == 2 and self._row_cells[0]:
                self.pairs.append((self._row_cells[0], self._row_cells[1]))
            self._row_cells = []
        elif tag == "h1" and text and not self.h1:
            self.h1 = text
        elif tag == "title" and text and not self.title:
            self.title = text
        elif tag == "p" and len(text) > 40:
            self.paragraphs.append(text)

        self._cell_buffer = []
        if self._stack and self._stack[-1] == tag:
            self._stack.pop()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if data.strip():
            self._cell_buffer.append(data.strip())

    def finish_dl(self) -> None:
        """Pair up dt/dd cells collected outside a table row."""
        cells = self._row_cells
        for i in range(0, len(cells) - 1, 2):
            if cells[i]:
                self.pairs.append((cells[i], cells[i + 1]))
        self._row_cells = []


def from_html(content: str, source_ref: str = "page.html") -> IngestionResult:
    """Extract a single product from a product detail page."""
    parser = _ProductPageParser()
    try:
        parser.feed(content)
        parser.finish_dl()
    except Exception as exc:  # malformed markup is common in the wild
        raise IngestionError(f"Could not parse HTML: {exc}") from exc

    name = parser.h1 or parser.title
    if not name:
        raise IngestionError("No <h1> or <title> found; cannot identify the product.")

    # Strip a trailing site-name segment ("... | Acme"). The separator must be
    # surrounded by whitespace, or the hyphen inside a part number like
    # "CP-200" gets treated as one and the model number is lost.
    name = re.sub(r"\s+[|\u2013\u2014]\s+.{0,40}$", "", name).strip()
    name = re.sub(r"\s+-\s+.{0,40}$", "", name).strip()

    attributes = {
        _clean(key).rstrip(":"): _clean(value)
        for key, value in parser.pairs
        if _clean(key) and _clean(value) and len(key) < 60
    }

    description = " ".join(parser.paragraphs[:3])

    product = RawProduct(
        name=name,
        description=description,
        source="html",
        source_ref=source_ref,
        attributes=attributes,
    )

    warnings: List[str] = []
    if not attributes:
        warnings.append("No specification table was found on the page.")

    return IngestionResult([product], "html", source_ref, 1, [], warnings)


# =========================================================================
# Plain text / technical documents
# =========================================================================


def from_text(
    content: str,
    source_ref: str = "document.txt",
    split_on_headings: bool = True,
) -> IngestionResult:
    """
    Parse a technical document or datasheet dump.

    A single datasheet becomes one product. A multi-product catalogue dump is
    split on heading-like lines (all-caps or numbered, short, no terminal
    period), which is how most PDF-to-text catalogue exports are shaped.
    """
    text = content.strip()
    if not text:
        raise IngestionError("The document is empty.")

    blocks: List[Tuple[str, str]] = []

    if split_on_headings:
        current_heading: Optional[str] = None
        current_body: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if _looks_like_heading(stripped):
                if current_heading:
                    blocks.append((current_heading, "\n".join(current_body).strip()))
                current_heading = stripped
                current_body = []
            elif current_heading:
                current_body.append(stripped)
        if current_heading:
            blocks.append((current_heading, "\n".join(current_body).strip()))

    warnings: List[str] = []
    if not blocks:
        first_line = next((l.strip() for l in text.splitlines() if l.strip()), "Untitled document")
        blocks = [(first_line[:120], text)]
        warnings.append(
            "No product headings detected; the document was ingested as a single record."
        )

    products = [
        RawProduct(
            name=heading,
            description=body,
            source="document",
            source_ref=f"{source_ref}#block{i}",
            attributes=_key_value_lines(body),
        )
        for i, (heading, body) in enumerate(blocks, start=1)
        if heading
    ]

    return IngestionResult(products, "document", source_ref, len(blocks), [], warnings)


def from_pdf_bytes(data: bytes, source_ref: str = "document.pdf") -> IngestionResult:
    """Extract text from a PDF, then hand off to the document reader."""
    try:
        from pypdf import PdfReader  # optional dependency
    except ImportError as exc:  # pragma: no cover
        raise IngestionError(
            "PDF ingestion needs the 'pypdf' package. Install it, or convert the "
            "document to text first."
        ) from exc

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise IngestionError(f"Could not read the PDF: {exc}") from exc

    text = "\n".join(pages).strip()
    if not text:
        raise IngestionError(
            "No extractable text found. The PDF is probably a scan — route it through "
            "the OCR service first."
        )

    result = from_text(text, source_ref=source_ref)
    result.source_type = "pdf"
    if len(pages) > 1:
        result.warnings.append(f"Extracted text from {len(pages)} page(s).")
    return result


def from_xlsx_bytes(data: bytes, source_ref: str = "catalog.xlsx") -> IngestionResult:
    """Read the first worksheet of an Excel catalogue export."""
    try:
        from openpyxl import load_workbook  # optional dependency
    except ImportError as exc:  # pragma: no cover
        raise IngestionError(
            "Excel ingestion needs the 'openpyxl' package. Export the sheet as CSV instead."
        ) from exc

    try:
        workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        sheet = workbook[workbook.sheetnames[0]]
        rows = list(sheet.iter_rows(values_only=True))
    except Exception as exc:
        raise IngestionError(f"Could not read the workbook: {exc}") from exc

    if not rows:
        raise IngestionError("The worksheet is empty.")

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    for row in rows:
        writer.writerow(["" if cell is None else str(cell) for cell in row])

    result = from_csv(buffer.getvalue(), source_ref=source_ref)
    result.source_type = "xlsx"
    return result


# =========================================================================
# Dispatch
# =========================================================================


def ingest(
    content: Any, filename: str = "upload", content_type: Optional[str] = None
) -> IngestionResult:
    """Route to the right reader based on the filename or MIME type."""
    lower = filename.lower()
    ctype = (content_type or "").lower()

    if isinstance(content, bytes):
        if lower.endswith(".pdf") or "pdf" in ctype:
            return from_pdf_bytes(content, filename)
        if lower.endswith((".xlsx", ".xlsm")) or "spreadsheet" in ctype:
            return from_xlsx_bytes(content, filename)
        try:
            content = content.decode("utf-8")
        except UnicodeDecodeError:
            content = content.decode("latin-1", errors="replace")

    if lower.endswith((".csv", ".tsv")) or "csv" in ctype:
        return from_csv(content, filename, delimiter="\t" if lower.endswith(".tsv") else None)
    if lower.endswith((".json", ".jsonl", ".ndjson")) or "json" in ctype:
        return from_json(content, filename)
    if lower.endswith((".html", ".htm")) or "html" in ctype:
        return from_html(content, filename)
    if lower.endswith((".txt", ".md", ".text")) or "text/plain" in ctype:
        return from_text(content, filename)

    # Unknown extension: sniff the content itself.
    head = content.lstrip()[:200].lower()
    if head.startswith("<") and ("html" in head or "<!doctype" in head):
        return from_html(content, filename)
    if head.startswith(("{", "[")):
        return from_json(content, filename)
    if "," in content.splitlines()[0] if content.splitlines() else False:
        return from_csv(content, filename)
    return from_text(content, filename)


SUPPORTED_FORMATS: Tuple[str, ...] = (
    ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".html", ".htm",
    ".txt", ".md", ".pdf", ".xlsx", ".xlsm",
)


# =========================================================================
# helpers
# =========================================================================


def _map_columns(headers: Sequence[str]) -> Dict[str, Optional[str]]:
    """Match headers to our three first-class fields, case/format insensitive."""
    normalised = {h: re.sub(r"[^a-z0-9]+", "_", h.lower()).strip("_") for h in headers if h}

    def find(candidates: Sequence[str]) -> Optional[str]:
        for candidate in candidates:
            for original, norm in normalised.items():
                if norm == candidate:
                    return original
        # Suffix fallback only. A plain substring test is too greedy: it maps
        # "specs_voltage" onto the "specs" description column and swallows a
        # real attribute. Suffix matching still catches "product_name" and
        # "product_description" without that failure mode.
        for candidate in candidates:
            for original, norm in normalised.items():
                if norm.endswith("_" + candidate):
                    return original
        return None

    return {
        "name": find(NAME_COLUMNS),
        "description": find(DESC_COLUMNS),
        "category": find(CATEGORY_COLUMNS),
        "sku": find(SKU_COLUMNS),
    }


def _flatten(record: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """One level of nesting is common in API exports; flatten it."""
    flat: Dict[str, Any] = {}
    for key, value in record.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, prefix=f"{full_key}_"))
        elif isinstance(value, list):
            flat[full_key] = ", ".join(str(v) for v in value if not isinstance(v, (dict, list)))
        else:
            flat[full_key] = value
    return flat


def _key_value_lines(text: str) -> Dict[str, str]:
    """Pick up 'Key: value' lines inside a datasheet body."""
    found: Dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*([A-Za-z][A-Za-z0-9 /_().-]{2,40})\s*[:\t]\s*(.+?)\s*$", line)
        if match:
            key, value = match.group(1).strip(), match.group(2).strip()
            if key.lower() not in found and len(value) < 120:
                found[key] = value
    return found


def _looks_like_heading(line: str) -> bool:
    if not (4 <= len(line) <= 90):
        return False
    if line.endswith((".", ",", ";", ":")):
        return False
    if re.match(r"^\d+(\.\d+)*[\.\)]\s+\S", line):     # "3.1 Centrifugal Pumps"
        return True
    letters = [c for c in line if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.75:
        return True
    return False


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()

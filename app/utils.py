import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Document:
    text: str
    source: str
    file_type: str
    metadata: dict = field(default_factory=dict)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_title(text: str, file_type: str, fallback_name: Optional[str] = None) -> str:
    if file_type == "md":
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("[Page "):
            return stripped[:120]

    return fallback_name or "Untitled"


def _read_plain_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_txt_file(file_path: str) -> Document:
    text = _read_plain_text(file_path)
    cleaned_text = clean_text(text)
    filename = os.path.basename(file_path)
    return Document(
        text=cleaned_text,
        source=file_path,
        file_type="txt",
        metadata={
            "filename": filename,
            "title": infer_title(cleaned_text, "txt", fallback_name=filename),
        },
    )


def read_md_file(file_path: str) -> Document:
    text = _read_plain_text(file_path)
    cleaned_text = clean_text(text)
    filename = os.path.basename(file_path)
    return Document(
        text=cleaned_text,
        source=file_path,
        file_type="md",
        metadata={
            "filename": filename,
            "title": infer_title(cleaned_text, "md", fallback_name=filename),
        },
    )


def read_pdf_file(file_path: str) -> Document:
    reader = PdfReader(file_path)
    page_texts = []

    for page_index, page in enumerate(reader.pages, start=1):
        page_text = clean_text(page.extract_text() or "")
        if page_text:
            page_texts.append(f"[Page {page_index}]\n{page_text}")

    merged_text = "\n\n".join(page_texts).strip()
    filename = os.path.basename(file_path)
    return Document(
        text=merged_text,
        source=file_path,
        file_type="pdf",
        metadata={
            "filename": filename,
            "page_count": len(reader.pages),
            "title": infer_title(merged_text, "pdf", fallback_name=filename),
        },
    )


def read_file(file_path: str) -> Document:
    ext = Path(file_path).suffix.lower()

    if ext == ".txt":
        return read_txt_file(file_path)
    if ext == ".md":
        return read_md_file(file_path)
    if ext == ".pdf":
        return read_pdf_file(file_path)

    raise ValueError(f"unsupported file type: {ext}")


def list_supported_files(data_dir: str) -> list[str]:
    files = []
    for root, _, filenames in os.walk(data_dir):
        for name in filenames:
            ext = Path(name).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(root, name))
    return sorted(files)


# 兼容旧代码命名
list_text_files = list_supported_files

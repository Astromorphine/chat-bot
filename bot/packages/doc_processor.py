
from bot.packages.i_classes.i_logger import ILogger

import fitz
from docx import Document
import os
from pathlib import Path

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

class DocumentProcessor():

    def __init__(self, logger : ILogger):
        self.logger = logger

    def read_pdf(self, file_path: str) -> str:
        doc = fitz.open(file_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text

    def read_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text

    def read_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
        
    def extract_text_from_file(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".txt":
            return self.read_txt(file_path)
        elif ext == ".pdf":
            return self.read_pdf(file_path)
        elif ext == ".docx":
            return self.read_docx(file_path)
        else:
            raise ValueError(f"Данный тип файла не поддерживается: {ext}")

    
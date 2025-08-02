from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

BANK_DATA_OUTPUT = BASE_DIR / "bank_data_output"

PDF_DIR = BANK_DATA_OUTPUT / "pdf"
TXT_DIR = BANK_DATA_OUTPUT / "text"

LOG_DIR = BASE_DIR / "logs"

PDF_FILES = PDF_DIR.glob("*.pdf")
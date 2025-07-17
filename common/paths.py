from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_DIR = BASE_DIR / "bank_data_output" / "pdf"
LOG_DIR = BASE_DIR / "logs"

PDF_FILES = PDF_DIR.glob("*.pdf")
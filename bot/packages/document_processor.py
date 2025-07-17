import os, sys
import pdfplumber
from pathlib import Path

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from packages.i_classes.i_document_proccesor import IDocumentProccesor
from packages.i_classes.i_logger import ILogger
from packages.embedding_generator import OpenAITokenizerWrapper

from common.paths import PDF_DIR, LOG_DIR, PDF_FILES

class DocumentProccesor(IDocumentProccesor):

    def __init__(self, logger : ILogger, tokenizer : OpenAITokenizerWrapper, max_tokens : int):
        self.logger = logger
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def convert_pdf_to_document(self, pdf_path: Path):
        """
        Конвертирует PDF в формат документа docling.
        
        Args:
            pdf_path: путь к PDF-файлу
            
        Returns:
            Объект docling с конвертированным документом
        """
        converter = DocumentConverter()
        
        if not str(pdf_path).startswith('http'):
            pdf_path = Path(pdf_path).resolve()
        
        return converter.convert(pdf_path)
    
    def pdf_validation(self, pdf_path):
    
        with pdfplumber.open(pdf_path) as pdf:

            if len(pdf.pages) >= 20:
                raise ValueError("Можно загрузить pdf размерностью не более 20 страниц")

        return pdf_path


    def process_pdf_documents(self, pdf_files: list[Path] = None):
        """
        Обрабатывает все PDF-документы из стандартной директории.
        Можно указать новый pdf_files или будет использоватся PDF_FILES из common/paths.py
        Returns:
            Список чанков из всех документов
        """
        pdf_files = [pdf_files] or list(PDF_FILES)

        if not pdf_files:
            self.logger.critical("❌ В директории не найдено PDF-файлов")
            return []
        
        all_chunks = []
        
        for pdf_file in pdf_files:
            
            # Инициализируем заранее. Если получим ошибку до объявления chunks то try except получит ошибку в блоке finally
            chunks = []

            try:
                self.logger.info(f"📄 Обработка файла: {pdf_file}")
                result = self.convert_pdf_to_document(pdf_file)

                chunker = HybridChunker(
                    tokenizer=self.tokenizer,
                    max_tokens=self.max_tokens,
                    merge_peers=True,
                )
                

                chunks = list(chunker.chunk(dl_doc=result.document))
                
                all_chunks.extend(chunks)
            except Exception as e:
                self.logger.error(f"❌ Ошибка при обработке {pdf_file.name}: {e}")
            finally:
                self.logger.info(f"🔹 Извлечено {len(chunks)} чанков из: {pdf_file.name}")

        self.logger.info(f"✅ Всего извлечено {len(all_chunks)} чанков из {len(pdf_files)} файлов")
        return all_chunks
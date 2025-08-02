from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
#from bot.packages.embedding_generator import OpenAITokenizerWrapper
from bot.packages.i_classes.i_logger import ILogger
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class TextProcessor():
    def __init__(self, logger : ILogger):
        self.logger = logger

    def chunk_text(self, filepath : str | Path, chunk_size : int, chunk_overlap : int)-> List[Document] | None:
        try:
            text : str
            with open(file=filepath, mode="r", encoding='utf-8') as f:
                text = f.read()
            text_splitter = text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,  # Размер чанка в символах
            chunk_overlap=chunk_overlap,  # Перекрытие 200 символов для сохранения контекста
            separators=[".", "\n", ", "],  # Разделители по точке, новой строке и запятой
            keep_separator="end",  # Сохраняем разделители в конце чанков
            is_separator_regex=False  # Без использования регулярных выражений
            )
            chunks = text_splitter.create_documents([text])
            self.logger.info(f"Текст из директории: {filepath} был успешно разделён на чанки, размером {len(chunks)} элементов")
            return chunks
        except Exception as e:
            self.logger.critical(f"Ошибка при разбиении текста на чанки, Trace: {e}")
            return None
        
            

            


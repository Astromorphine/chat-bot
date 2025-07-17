
from packages.my_logger import StandardLogger
from packages.rag_bot import RAGAgent, RAGBotHandler

from packages.document_processor import DocumentProccesor
from packages.embedding_generator import OpenAIEmbeddingGenerator, OpenAITokenizerWrapper
from packages.my_logger import StandardLogger
from packages.lance_vector_db import LanceVectorDB
from common.paths import PDF_DIR, LOG_DIR, PDF_FILES
from packages.html_processing import HTMLProcessing


class AppContext:
    def __init__(self):
        self.logger = StandardLogger(name="RAGBot")
        self.agent = RAGAgent(logger=self.logger)
        self.bot_handler = RAGBotHandler(
            agent=self.agent,
            logger=self.logger,
            db_path="./lancedb"
        )
        self.tokenizer = OpenAITokenizerWrapper()
        self.doc_proccesor = DocumentProccesor(self.logger, self.tokenizer, 8191)
        self.embedding_generator = OpenAIEmbeddingGenerator(self.logger)
        self.html_poccessor = HTMLProcessing()
        self.lance_db = LanceVectorDB(self.logger, self.embedding_generator, "pdf_chunks")
from bot.packages.my_logger import StandardLogger
from bot.packages.rag_bot import RAGAgent, RAGBotHandler
from bot.packages.html_processing import HTMLDownloader, HTMLCleaner
from bot.packages.embedding_generator import OpenAIEmbeddingGenerator
from bot.packages.my_logger import StandardLogger
from bot.packages.lance_vector_db import LanceVectorDB
from bot.packages.text_pocessor import TextProcessor
from bot.packages.travily_agent import TravilyAgent
from bot.packages.qa_simple_bot import QAgent
from bot.packages.doc_processor import DocumentProcessor

from common.file_utils import FileUtilities

class AppContext:
    def __init__(self):
        self.logger = StandardLogger(name="RAGBot")
        self.embedding_generator = OpenAIEmbeddingGenerator(self.logger)
        self.lance_db = LanceVectorDB(self.logger, self.embedding_generator)
        self.qa_agent = QAgent(logger=self.logger)
        self.rag_agent = RAGAgent(logger=self.logger)
        self.bot_handler = RAGBotHandler(
            agent=self.rag_agent,
            logger=self.logger,
            db_path="./data/lancedb"
        )
        self.travily_agent = TravilyAgent(self.logger)
        self.html_processor = HTMLDownloader(self.logger)
        self.html_cleaner = HTMLCleaner(self.logger)
        self.text_processor = TextProcessor(self.logger)
        self.document_processor = DocumentProcessor(self.logger)
        self.file_utilities = FileUtilities(self.logger)
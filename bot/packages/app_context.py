from bot.packages.my_logger import StandardLogger
from bot.packages.rag_bot import RAGAgent, RAGBotHandler

from bot.packages.html_processing import HTMLDownloader, HTMLCleaner
from bot.packages.embedding_generator import OpenAIEmbeddingGenerator
from bot.packages.my_logger import StandardLogger
from bot.packages.lance_vector_db import LanceVectorDB
from bot.packages.text_pocessor import TextProcessor
from bot.packages.travily_agent import TravilyAgent

class AppContext:
    def __init__(self):
        self.logger = StandardLogger(name="RAGBot")
        self.agent = RAGAgent(logger=self.logger)
        self.bot_handler = RAGBotHandler(
            agent=self.agent,
            logger=self.logger,
            db_path="./data/lancedb"
        )
        self.travily_agent = TravilyAgent(self.logger)
        self.embedding_generator = OpenAIEmbeddingGenerator(self.logger)
        self.html_processor = HTMLDownloader(self.logger)
        self.html_cleaner = HTMLCleaner(self.logger)
        self.text_processor = TextProcessor(self.logger)
        self.lance_db = LanceVectorDB(self.logger, self.embedding_generator)
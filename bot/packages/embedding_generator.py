from typing import List, Dict, Any, Optional, Tuple, Protocol
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from tiktoken import get_encoding
from openai import OpenAI
import openai
import os

from packages.i_classes.i_embedding_generator import IEmbeddingGenerator
from packages.i_classes.i_embedding_generator import ILogger

class OpenAITokenizerWrapper(PreTrainedTokenizerBase):
    """Минимальная обертка для токенизатора OpenAI."""

    def __init__(
        self, model_name: str = "cl100k_base", max_length: int = 8191, **kwargs
    ):
        """Инициализация токенизатора.

        Args:
            model_name: Название кодировки OpenAI для использования
            max_length: Максимальная длина последовательности
        """
        super().__init__(model_max_length=max_length, **kwargs)
        self.tokenizer = get_encoding(model_name)
        self._vocab_size = self.tokenizer.max_token_value

    def __len__(self):
        return self.vocab_size

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Основной метод, используемый HybridChunker."""
        return [str(t) for t in self.tokenizer.encode(text)]

    def _tokenize(self, text: str) -> List[str]:
        return self.tokenize(text)

    def _convert_token_to_id(self, token: str) -> int:
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        return str(index)

    def get_vocab(self) -> Dict[str, int]:
        return dict(enumerate(range(self.vocab_size)))

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def save_vocabulary(self, *args) -> Tuple[str]:
        return ()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Классовый метод для соответствия интерфейсу HuggingFace."""
        return cls()

class OpenAIEmbeddingGenerator(IEmbeddingGenerator):
    
    def __init__(self, logger : ILogger):

        self.logger = logger

        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        load_dotenv(env_path)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY не найден в .env")

        self.client = OpenAI(api_key=api_key)
        self.max_tokens = 8191  # пока не используется, но может пригодиться

    @retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIError, openai.APIConnectionError))
    )
    def create_embedding(self, text : str) -> list[float]:
        """
        Создает эмбеддинг для текста с использованием OpenAI API.
        
        Функция использует декоратор retry для автоматического повтора
        при ошибках API (rate limits, timeout и т.д.)
        
        Args:
            text: текст для создания эмбеддинга
            
        Returns:
            Вектор эмбеддинга
        """
        try:
            response = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            dimensions=1536
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.critical(f"Произошла ошибка при создании вектора эмбеддинга, Trace:, {e}")
            raise

    
    
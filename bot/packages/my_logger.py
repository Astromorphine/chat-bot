from typing import Protocol
import logging
from packages.i_classes.i_logger import ILogger

class StandardLogger(ILogger):
    def __init__(self, name: str = __name__, level: int = logging.INFO, log_to_file: bool = True):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)

        if not self._logger.handlers:  # чтобы не добавлять хендлеры повторно
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

            if log_to_file:
                file_handler = logging.FileHandler("bot.log", encoding="utf-8")
                file_handler.setFormatter(formatter)
                self._logger.addHandler(file_handler)

            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self._logger.addHandler(stream_handler)

    def info(self, msg: str):
        self._logger.info(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def debug(self, msg: str):
        self._logger.debug(msg)

    def critical(self, msg: str):
        self._logger.critical(msg)

    def get_logger(self) -> logging.Logger:
        return self._logger


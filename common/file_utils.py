import base64
from pathlib import Path
from common.paths import TXT_DIR
from bot.packages.i_classes.i_logger import ILogger


class FileUtilities():

    def __init__(self, logger : ILogger):
        self.logger = logger

    def encode_filename_base64(self, url: str) -> str:
        return base64.urlsafe_b64encode(url.encode('utf-8')).decode('utf-8')

    def decode_filename_base64(self, encoded_url: str) -> str:
        return base64.urlsafe_b64decode(encoded_url.encode('utf-8')).decode('utf-8')

    def create_txt(self, text : str, filename : str) -> Path | None:
        try:
            file_path = TXT_DIR / f"{self.encode_filename_base64(filename)}.txt"
            with open(file=file_path,mode="w", encoding="utf-8") as f:
                data = f.write(text)
                self.logger.info(f"Файл с названием: {filename} и размером в {data} символов был успешно сохранён, путь -> {file_path}")
            return file_path
        except Exception as e:
            self.logger.critical(f"Произошла ошибка при сохранении txt файла, Trace: {e}")
            return None

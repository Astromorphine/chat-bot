#from bot.packages.document_processor import DocumentProccesor
from bot.packages.embedding_generator import OpenAIEmbeddingGenerator#, OpenAITokenizerWrapper
from bot.packages.my_logger import StandardLogger
from bot.packages.lance_vector_db import LanceVectorDB
from common.paths import PDF_DIR, LOG_DIR, PDF_FILES
from bot.packages.html_processing import HTMLDownloader
from bot.packages.rag_bot import RAGBotHandler, RAGAgent

#lance_db = LanceVectorDB(StandardLogger("LanceVectorDB"), embedding_generator)

from bot.packages.html_processing import *
import asyncio

urls = [
    "https://habr.com/ru/articles/930746/",
    "https://lancedb.com/documentation/",
    "https://mail.google.com/mail/u/0/#inbox/FMfcgzQbgRpRPJPwdkCnQRrjpWKPMMQV",
    "https://reactjs.org",
    ""
]

from bot.packages.text_pocessor import DocumentProccesor

logger = StandardLogger()
downloader = HTMLDownloader(logger)
cleaner = HTMLCleaner(logger)
doc_proccesor = DocumentProccesor(logger)
embedding_generator = OpenAIEmbeddingGenerator(logger)
lance_db = LanceVectorDB(logger=logger,embedding_generator=embedding_generator)
agent = RAGAgent(logger=logger)
bot = RAGBotHandler(agent=agent, logger=logger, db_path="data/lancedb/")

print(bot.handle_question("Расскажи про протеин"))

'''
https://habr.com/ru/articles/932640/
https://habr.com/ru/articles/932430/
'''

'''
lance_db.connect_db()

if not "from_txt" in lance_db.connection.table_names():
    lance_db.create_table("from_txt")

lance_db.select_table("from_txt")
table = lance_db.get_table()
chunks = doc_proccesor.chunk_text(TXT_DIR / "aHR0cHM6Ly9oYWJyLmNvbS9ydS9hcnRpY2xlcy85MzA3NDYv.txt")
if chunks:
    lance_db.fill_table(filename="aHR0cHM6Ly9oYWJyLmNvbS9ydS9hcnRpY2xlcy85MzA3NDYv.txt", chunks=chunks, current_table=table)

    print("Успех!")

'''


'''
lance_db.connect_db()

if not "from_txt" in lance_db.connection.table_names():
    lance_db.create_table("from_txt")

lance_db.select_table("from_txt")



print(lance_db.search_in_table("Протеиновая лихорадка"))


'''

'''

    embeddings = embedding_generator.create_embeddings_for_chunks(chunks)

print(embeddings)

'''

'''
for url in urls:
    clean_text = ""
    text = asyncio.run(downloader.download(url))
    if text[0] != None :
        clean_text = cleaner.clean(text[0])
        downloader.create_txt(clean_text,url)

'''
'''
gen = TXT_DIR.iterdir()

for i in gen:
    print(downloader.decode_filename_base64(str(i).replace(f"{TXT_DIR}","").replace(".txt", "").replace("\\","")))
    

    #print(downloader.decode_filename_base64(str(i).replace(".txt", "")))
'''
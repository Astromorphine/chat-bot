from packages.document_processor import DocumentProccesor
from packages.embedding_generator import OpenAIEmbeddingGenerator, OpenAITokenizerWrapper
from packages.my_logger import StandardLogger
from packages.lance_vector_db import LanceVectorDB
from common.paths import PDF_DIR, LOG_DIR, PDF_FILES
from packages.html_processing import HTMLProcessing

'''
tokenizer = OpenAITokenizerWrapper()
doc_proccesor = DocumentProccesor(StandardLogger("DocumentProccesor"),tokenizer , 8191)
'''
embedding_generator = OpenAIEmbeddingGenerator(StandardLogger("OpenAIEmbeddingGenerator"))
'''
pipeline = HTMLProcessing()
pipeline.process_urls_from_sitemap("https://habr.com/ru/articles/887322/")

chunks = doc_proccesor.process_pdf_documents(pipeline.doc_path)
'''
lance_db = LanceVectorDB(StandardLogger("LanceVectorDB"), embedding_generator, "pdf_chunks")


'''
#lance_db.create_table()

lance_db.fill_table("pdf_chunks",chunks)
'''
lance_db.connect_db()
lance_db.select_table()
lance_db.display_search_results(lance_db.search_in_table("Средняя чистая прибыль"))

'''
tokenizer = OpenAITokenizerWrapper()
doc_proccesor = DocumentProccesor(StandardLogger("DocumentProccesor"),tokenizer , 8191)

embedding_generator = OpenAIEmbeddingGenerator(StandardLogger("OpenAIEmbeddingGenerator"))

pipeline = HTMLProcessing()
pipeline.process_urls_from_sitemap("https://habr.com/ru/articles/887322/")

chunks = doc_proccesor.process_pdf_documents(pipeline.doc_path)

lance_db = LanceVectorDB(StandardLogger("LanceVectorDB"), embedding_generator, "pdf_chunks")



#lance_db.create_table()

lance_db.fill_table("pdf_chunks",chunks)

lance_db.connect_db()
lance_db.select_table()
lance_db.display_search_results(lance_db.search_in_table("Средняя чистая прибыль"))
'''
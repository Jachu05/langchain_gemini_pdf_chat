from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

EMBEDD_DIM_SIZE = 768

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    index = faiss.IndexFlatL2(EMBEDD_DIM_SIZE)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore= InMemoryDocstore(),
        index_to_docstore_id={}
    )

    chunks = vector_store.from_texts(text_chunks, embeddings)
    print(text_chunks[0])
    print(chunks)
    return chunks

# test
if __name__ == '__main__':

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # result = embeddings.embed_query("What's our Q1 revenue?")
    # print(len(result))

    index = faiss.IndexFlatL2(EMBEDD_DIM_SIZE)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore= InMemoryDocstore(),
        index_to_docstore_id={}
    )

    document_1 = Document(page_content="foo", metadata={"baz": "bar"})
    document_2 = Document(page_content="thuder", metadata={"bar": "baz"})
    document_3 = Document(page_content="i will be deleted :(")

    documents = [document_1, document_2, document_3]
    ids = ["1", "2", "3"]
    vector_store.add_documents(documents=documents, ids=ids)

    results = vector_store.similarity_search_with_score(query="thud",k=1)
    for doc, score in results:
        print(f"* {doc.page_content} [{doc.metadata}] score:{score}")

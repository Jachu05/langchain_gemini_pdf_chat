from typing import Sequence

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def get_rag_chain(retriever):

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    class State(TypedDict):
        input: str
        chat_history: Annotated[Sequence[BaseMessage], add_messages]
        context: str
        answer: str

    def call_model(state: State):
        response = rag_chain.invoke(state)
        return {
            "chat_history": [
                HumanMessage(state["input"]),
                AIMessage(response["answer"]),
            ],
            "context": response["context"],
            "answer": response["answer"],
        }

    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app


# test
if __name__ == '__main__':
    from dotenv import load_dotenv
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    import faiss
    from langchain_community.vectorstores import FAISS
    from langchain_community.docstore.in_memory import InMemoryDocstore

    load_dotenv()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    index = faiss.IndexFlatL2(768)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore= InMemoryDocstore(),
        index_to_docstore_id={}
    )

    chain = get_rag_chain(vector_store.as_retriever())
    # config = {"configurable": {"thread_id": "abc123"}}

    # result = chain.invoke(
    #     {"input": "What is Task Decomposition?"},
    #     config=config,
    # )
    # print(result["answer"])
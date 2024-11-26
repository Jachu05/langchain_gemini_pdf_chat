import streamlit as st
import utils
import rag_chain
import vector_db
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import RemoveMessage


def handle_user_question(user_question):
    # st.session_state.conversation.append('you: ' + user_question)

    # prompt = f"""
    #     You are helping assistant. Make your answers short, one sentence the best.
    #     If asked something you dont, act like you it and be friednly.

    #     {user_question}
    # """
    # message = st.session_state.llm.invoke(prompt)
    # st.session_state.conversation.append('bot: ' + str(message.content))
    if st.session_state.chain:
        response = st.session_state.chain.invoke(
            {"input": user_question},
            config=st.session_state.config,
        )
        st.session_state.chat_history = response['chat_history']   
        for message in st.session_state.chat_history:
            if message.type == 'human':
                st.markdown(f"<div style='text-align: right; color: white; background-color: #1DA1F2; padding: 10px; border-radius: 10px; margin: 5px;'>{message.content}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align: left; color: white; background-color: #555; padding: 10px; border-radius: 10px; margin: 5px;'>{message.content}</div>", unsafe_allow_html=True)
    else:
        response = st.session_state.base_llm.invoke(user_question)
        if st.session_state.chat_history is None:
            st.session_state.chat_history = []
        st.session_state.chat_history += [response.content]   
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.markdown(f"<div style='text-align: right; color: white; background-color: #1DA1F2; padding: 10px; border-radius: 10px; margin: 5px;'>{message}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align: left; color: white; background-color: #555; padding: 10px; border-radius: 10px; margin: 5px;'>{message}</div>", unsafe_allow_html=True)


def clear():
    # not working :(:(:(
    chat_history = st.session_state.chain.get_state(st.session_state.config).values['chat_history']
    st.session_state.chain.update_state(st.session_state.config, {
                "chat_history": [RemoveMessage(id=m.id) for m in chat_history]
    })

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")

    if 'base_llm' not in st.session_state:
        st.session_state.base_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    if 'chain' not in st.session_state:
        st.session_state.chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None
    if 'config' not in st.session_state:
        st.session_state.config = {"configurable": {"thread_id": "abc123"}}
    
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_user_question(user_question)
        user_question = ''

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = utils.get_pdf_text(pdf_docs)
                text_chunks = utils.get_text_chunks(raw_text)
                vectorstore = vector_db.get_vectorstore(text_chunks)
                st.session_state.chain = rag_chain.get_rag_chain(vectorstore.as_retriever())
                # st.write(text_chunks)

        if st.button("Clear") and ('chain', 'config') in st.session_state: 
            # clear()
            ...

if __name__ == '__main__':
    # streamlit run c:/repos/langchain_gemini_pdf_chat/app.py
    main()

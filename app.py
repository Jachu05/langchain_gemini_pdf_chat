import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    
    if 'llm' not in st.session_state:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        st.session_state.llm = llm
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        st.session_state.conversation.append('you: ' + user_question)

        prompt = f"""
            You are helping assistant. Make your answers short, one sentence the best.
            If asked something you dont, act like you it and be friednly.

            {user_question}
        """
        message = st.session_state.llm.invoke(prompt)
        st.session_state.conversation.append('bot: ' + str(message.content))

    with st.sidebar:
        st.subheader("Your documents")
        st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        st.button("Process")
        if st.button("Clear"): st.session_state.conversation = []

    for message in st.session_state.conversation:
        if message.startswith('bot'):
            st.markdown(f"<div style='text-align: right; color: white; background-color: #1DA1F2; padding: 10px; border-radius: 10px; margin: 5px;'>{message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left; color: white; background-color: #555; padding: 10px; border-radius: 10px; margin: 5px;'>{message}</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()

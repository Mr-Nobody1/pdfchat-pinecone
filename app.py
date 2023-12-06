import os
import streamlit as st
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENVIRONMENT
from constants import PINECONE_INDEX_NAME
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import pinecone
from streamlit_chat import message as st_message

# Set API key for OpenAI and Pinecone
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENVIRONMENT
)
# Streamlit title
st.title("Document Answering Chat")

# Initialize session state for storing messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to handle the logic of generating a response
def generate_response(user_input):
    # Document query logic
    llm_chat = ChatOpenAI(temperature=0.9, max_tokens=150, model='gpt-3.5-turbo-0613', client='')
    embeddings = OpenAIEmbeddings(client='')
    docsearch = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    chain = load_qa_chain(llm_chat)

    try:
        # Execute similarity search
        search_results = docsearch.similarity_search(user_input)
        
        # Generate response using search results
        response = chain.run(input_documents=search_results, question=user_input)
        return f"Response: {response}", search_results
    except Exception as e:
        return f'Error: {str(e)} Please try again.', None

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    # Display document similarity search results (if available)
    if message["role"] == "assistant" and "search_results" in message:
        with st.expander('Document Similarity Search Results'):
            st.write(message["search_results"])

# User input section
with st.container():
    user_input = st.chat_input("Ask your document query:")

    if user_input:
        response_message, search_results = generate_response(user_input)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Add assistant response to chat history
        assistant_message = {"role": "assistant", "content": response_message}
        if search_results:
            assistant_message["search_results"] = search_results
        st.session_state.messages.append(assistant_message)

        st.rerun()
    
# # Setting Streamlit
# st.title('Chatbot Interface with Langchain and Pinecone')

# # Initialize OpenAI LLM and embeddings
# llm_chat = ChatOpenAI(temperature=0.9, max_tokens=500, model='gpt-4-1106-preview', client='')
# embeddings = OpenAIEmbeddings(client='')

# # Initialize Pinecone index
# docsearch = Pinecone.from_existing_index(
#     index_name=PINECONE_INDEX_NAME, embedding=embeddings)

# # Load QA chain
# chain = load_qa_chain(llm_chat)

# # Initialize session state for chat history
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []

# # Chat interface
# usr_input = st.text_input('Your question:', key='question')

# def handle_send():
#     question = st.session_state.question
#     if question:
#         # Perform document search and generate response
#         try:
#             search = docsearch.similarity_search(question)
#             response = chain.run(input_documents=search, question=question)
#             st.session_state.chat_history.append(f"You: {question}")
#             st.session_state.chat_history.append(f"Bot: {response}")
#         except Exception as e:
#             st.error("Error in processing your request.")
#             print(e)

#         # Clear input
#         st.session_state.question = ""


# if st.button("Send", on_click=handle_send):
#     pass


# # Display chat history
# for message in st.session_state.chat_history:
#     st.text(message)

# # Optional: Document Similarity Search Expander
# with st.expander('Document Similarity Search'):
#     if usr_input:
#         # Display results
#         search = docsearch.similarity_search(usr_input)
#         st.write(search)

from dotenv import load_dotenv
import streamlit as st
from user_utils import *
import os

def main():
    load_dotenv()

    st.header("Document Question & Answer")
    #Capture user input
    st.write("We are here to help you, please ask your question:")
    user_input = st.text_input("")

    if user_input:

        #creating embeddings instance...
        embeddings=create_embeddings()

        #Function to pull index data from Pinecone
        
        
        index=pull_from_pinecone(os.getenv("PINECONE_API_KEY"),"gcp-starter","document",embeddings)
        
        #This function will help us in fetching the top relevent documents from our vector store - Pinecone Index
        relavant_docs=get_similar_docs(index,user_input)

        #This will return the fine tuned response by LLM
        response=get_answer(relavant_docs,user_input)
        st.write(response)

if __name__ == '__main__':
    main()
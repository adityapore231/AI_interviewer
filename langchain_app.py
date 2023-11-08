import streamlit as st
import requests
import PyPDF2
import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


load_dotenv()
def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ''
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            text += page.extract_text().strip()
        return text
    except Exception as e:
        st.error(f'Error extracting text from PDF: {e}')
        return None


def app():
# Set page title and icon
  st.set_page_config(page_title='AI Interviewer Assistant', page_icon=':robot_face:')

  st.write('## AI Interviewer Assistant')
  st.write('##### Fill the below information to start generating questions for your candidate.')

# Initial Prompt
  uploaded_file = st.file_uploader('''Upload candidate's resume''')
  job_role = st.text_input('Job Role/Position')
  job_description = st.text_input('Job Description')
  difficulty_level = st.slider('Select Difficulty Level', min_value=1, max_value=5, step=1, value=3)
  years_of_experience = st.selectbox('Years of Experience', ['0-2', '2-5', '5+'])
  num_questions = st.number_input('How many questions?', min_value=1, max_value=10)
  text = extract_text_from_pdf(uploaded_file)
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  chunks  = text_splitter.split_text(text)
  embeddings = OpenAIEmbeddings()
  knowledgeBase = FAISS.from_texts(chunks, embeddings)
  query = f"You are interviewer today. You are taking interview for the position {job_role}. The job description is {job_description}. you are hiring for {years_of_experience} year experience people. difficulty level will be {difficulty_level} out of 5. Ask Question from knowledgeBase. Ask {num_questions} question related to project / work experience from resume first."
  print(query)
  if st.button('Generate'):
    docs = knowledgeBase.similarity_search(query)
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type='stuff')
            
    with get_openai_callback() as cost:
        response = chain.run(input_documents=docs, question=query)
        print(cost)
                
    st.write(response)

if __name__ == "__main__":
    app()
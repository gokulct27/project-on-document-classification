#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np
import streamlit as st
import docx2txt,textract
import pdfplumber
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot  as plt
stop=set(stopwords.words('english'))
from pickle import load
import pickle
model=load(open('Randomforest.sav','rb'))
vectors = pickle.load(open('tfidf.pkl','rb'))

nltk.download('wordnet')
nltk.download('stopwords')


# In[43]:


resume = []

def display(doc_file):
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))
    else :
        with pdfplumber.open(doc_file) as pdf:
            pages=pdf.pages[0]
            resume.append(pages.extract_text())
            
    return resume
    


# In[44]:


def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words)  


# In[45]:


def main():
    menu = ["Prediction page","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    html_temp = """
    <div style ="background-color:red;padding:13px">
    <h1 style ="color:black;text-align:center;"> DOCUMENT CLASSIFICATION </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    
    if choice == "Prediction page":
        st.subheader("prediction app")
    upload_file = st.file_uploader('Hey,Upload Your Resume ',
                                type= ['docx','pdf'],accept_multiple_files=True)
        
    if st.button("Process"):
        for doc_file in upload_file:
            if doc_file is not None:
               
                file_details = {'filename':[doc_file.name],
                               'filetype':doc_file.type.split('.')[-1].upper(),
                               'filesize':str(doc_file.size)+' KB'}
                file_type=pd.DataFrame(file_details)
                st.write(file_type.set_index('filename'))
                displayed=display(doc_file)
                cleaned=preprocess(display(doc_file))
                predicted= model.predict(vectors.transform([cleaned]))

                if int(predicted) == 0:
                    st.header("The Resume Is From Peoplessoft Resumes")
                elif int(predicted) == 1:
                    st.header("The Resume Is From SQL_Developer_Lightning _Insight ")
                elif int(predicted) == 2:
                    st.header("The Resume Is From  React Js Developers ")
                else:
                    st.header("The Resume Is From  Workday Resumes ")


    elif choice == "About":
        st.subheader("About") 
        st.info("Document classification project from group5")
        st.info("Built with Streamlit")

                
    
if __name__ == '__main__':
     main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





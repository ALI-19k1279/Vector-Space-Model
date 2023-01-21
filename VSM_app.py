import streamlit as st
import pandas as pd
import numpy as np
from VSM import processQeury
from VSM import filereader

data={
    'Doc Ids':[],
    'Cosine Sim':[]
}
st.markdown("<h1 style='text-align: center; color: red;'>Vector Space Model</h1>", unsafe_allow_html=True)
st.subheader('Query Input')
query=st.text_input("Enter your Query","")
if st.button('Search'):
    docs,docvals=processQeury(str(query))
    data['Doc Ids']=docs
    data['Cosine Sim']=docvals
    if len(docs)!=0:
        st.subheader('Query Output')
        st.success(docs)
        df = pd.DataFrame(data)
        df.sort_values(by=['Cosine Sim'],inplace=True, ascending=False)
        st.markdown("<h3 style='text-align: center; color: lightblue;'>Rankings</h3>", unsafe_allow_html=True)
        st.table(df)
    else:
        st.err('Couldnot find any results :(')
    
    
    
    
    
    
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Detect Language", page_icon=":tada:",layout="wide")

df = pd.read_csv("Language Detection.csv")

s = set(df['Language'])
s = list(s)
s = ", ".join(i for i in s)

x = np.array(df['Text'])
y = np.array(df['Language'])

cv = CountVectorizer(lowercase=True)
X = cv.fit_transform(x)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = MultinomialNB(alpha = 0.05)
model.fit(X_train,y_train)
model.score(X_test,y_test)

with st.container():
    st.write("---")
    st.write("Wanna know in which language the sentence is..??")
    input = st.text_input("Enter a sentence: ", "I'm a dumb machine, teach me senpai..!!")
    ans = np.array([input])
    ans = cv.transform(ans)
    output = model.predict(ans)
    st.write("---")
    st.write("#")
    st.write("The detected language is: ")
    st.text(output[0])
    st.write("---")
    st.write("##")
    st.write("Supported languages are: ")
    st.write(s)
    hide_streamlit_style = """
        <style>
        .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
        .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
        .viewerBadge_text__1JaDK {
            display: none;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
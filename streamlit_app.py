import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
from PIL import Image

data = pd.read_csv("news.csv" , encoding= 'unicode_escape')

x = np.array(data["title"])
y = np.array(data["label"])

cv = CountVectorizer()
x = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(xtrain, ytrain)



#opening the image
image = Image.open('fakeimage2.png')

image = image.resize((1000, 400))


#displaying the image on streamlit app

st.image(image, caption='Detect The Fake news')


st.title("Fake News Detection System")
def fakenewsdetection():
    user = st.text_area("Enter Any News Headline: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        data = cv.transform([sample]).toarray()
        a = model.predict(data)
        st.title(a)
fakenewsdetection()

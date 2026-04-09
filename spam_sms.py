import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

data = {
    "message":[
        "Win money now",
        "Hi how are you",
        "Claim your free prize",
        "Let's meet tomorrow",
        "Congratulations you won"
    ],
    "label":[1,0,1,0,1]
}

df = pd.DataFrame(data)

X = df["message"]
y = df["label"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = MultinomialNB()
model.fit(X_train,y_train)

msg = vectorizer.transform(["You won free money"])
print("Prediction:", model.predict(msg))
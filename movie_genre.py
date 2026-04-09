import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = {
    "plot":[
        "hero saves world",
        "love story between two people",
        "ghost scares family",
        "police catch criminal"
    ],
    "genre":[
        "Action",
        "Romance",
        "Horror",
        "Action"
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["plot"])
y = df["genre"]

model = LogisticRegression()
model.fit(X,y)

test = vectorizer.transform(["romantic love story"])
print("Predicted Genre:",model.predict(test))
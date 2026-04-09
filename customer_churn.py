import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = {
    "tenure":[1,5,2,8,10,3,6],
    "monthly_charges":[100,200,150,300,400,120,220],
    "churn":[1,0,1,0,0,1,0]
}

df = pd.DataFrame(data)

X = df[["tenure","monthly_charges"]]
y = df["churn"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))
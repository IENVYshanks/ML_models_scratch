import numpy as np
import pandas as pd

import kagglehub
path = kagglehub.dataset_download("dileep070/heart-disease-prediction-using-logistic-regression")

print("Path to dataset files:", path)

df = pd.read_csv(path+"/framingham.csv")
df = df.fillna(df.mean(numeric_only=True))

columns = df.columns

class LogisticRegression:
    def __init__(self, pred_col_name:str, dataset):
        if dataset is None or dataset.empty:
            raise ValueError("no dataset give")
        if pred_col_name not in dataset.columns:
            raise ValueError("no column name given")
        self.X = dataset.drop(columns = pred_col_name).astype(float).copy()
        self.y = dataset[pred_col_name].astype(float).values    
        self.weights = np.random.default_rng(1).random(self.X.shape[1])
        self.bias = float(np.random.default_rng(1).random())
        for name in self.X.columns:
            col_min = self.X[name].min()
            col_max = self.X[name].max()
            if col_max != col_min:
                self.X[name] = (self.X[name] - col_min) / (col_max - col_min)
            else:
                self.X[name] = 0.0


    def negLogLikelihood(self, predicted, real):
        predicted = np.clip(predicted, 1e-15, 1-1e-15)
        return -((real*np.log(predicted)) + (1 - real)*np.log(1 - (predicted))).mean()

    def sig(self, z):
        return 1/(1+np.exp(-z))
        
        
    def train(self, epochs=200, lr = 0.02):
        X = self.X.values
        y = self.y
        n = len(y)
        loss = 0
        for epoch in range(epochs):
            z =np.dot(X, self.weights) + self.bias
            output = self.sig(z)
            loss = self.negLogLikelihood(output, y)
            dw = np.dot(X.T, (output -y))/ n
            db = np.sum(output - y) / n
            self.weights -= dw
            self.bias -= db
            print(f"epoch {epoch+1} : loss = {loss}")

    def predict_prob(self, X):
        X = X.astype(float).copy()
        for name in X.columns:
            col_min = X[name].min()
            col_max = X[name].max()
            if col_max != col_min:
                X[name] = (X[name] - col_min) / (col_max - col_min)
            else:
                X[name] = 0.0

            z = np.dot(X.values, self.weights) + self.bias
            return self.sig(z)
        
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
        

model = LogisticRegression('TenYearCHD', df)

model.train()



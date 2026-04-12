import pandas as pd
import numpy as np
import kagglehub

path = kagglehub.dataset_download("msjaiclub/regression", output_dir='data')
file_path = path + "/advanced.csv"

class LinearRegression:
    def __init__(self, file_path:str, pred_col:str, epochs:int):
        self.epochs = epochs
        self.df = pd.read_csv(file_path)
        self.X = self.df.drop(columns = pred_col)
        self.y = self.df[pred_col]
        self.lr = 0.1
        self.slope = np.random.default_rng(42).random(len(self.X.columns))
        self.bias = np.random.default_rng(42).random()
        self.error = 0

        for names in self.X.columns:
            self.X[names] = (self.X[names] - self.X[names].min())/(self.X[names].max()-self.X[names].min())
        
    
    def train(self):
        for epoch in range(self.epochs):
            self.pred_val = (self.X * self.slope).sum(axis = 1) + self.bias
            self.error = self.y - self.pred_val
            n = len(self.y)
            dw = (-2 / n) * (self.X * self.error.values.reshape(-1,1)).sum(axis=0)
            db = (-2 / n) * self.error.sum()
            self.slope -= self.lr * dw
            self.bias -= self.lr * db
            loss = ((self.y - self.pred_val) ** 2).mean()
            print(f"Epoch {epoch}: Loss = {loss}")

        
        return (self.bias, self.slope)
        
model = LinearRegression(file_path=file_path, pred_col='quality', epochs=200)
bias, slope = model.train()
print(bias, slope)


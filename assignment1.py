import pandas as pd
import numpy as np
import pickle
from statsmodels.tsa.api import ExponentialSmoothing
import plotly.express as px


#Load Data
train = pd.read_csv("assignment_data_train.csv")
test  = pd.read_csv("assignment_data_test.csv")

y_train = train["trips"].astype(float)

model = ExponentialSmoothing(
    y_train,
    trend="add",
    damped_trend=True,
    seasonal="add",
    seasonal_periods=168
)

modelFit = model.fit()

forecast = modelFit.forecast(744)

combined = pd.concat([
    y_train.reset_index(drop=True),
    pd.Series(forecast).reset_index(drop=True)
], ignore_index=True)

flag = pd.Series([0]*len(y_train) + [1]*744)

trends = pd.DataFrame({"trips": combined, "forecast": flag})

px.line(trends, y='trips', color='forecast').show()

import os
import pandas as pd
import plotly.express as px
import numpy as np
# import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pio.renderers.default = "browser"

file_path = "ca_training_data_hs23/"

df = pd.DataFrame([])

# use this if local
dir_list = os.listdir(file_path)


for i in range(len(dir_list)):
    k =  pd.read_csv(file_path + dir_list[i])
    df = pd.concat([df, k])

df['yearweek_start'] = pd.to_datetime(df['yearweek_start'])



# fig = make_subplots(rows=1, cols=1, subplot_titles=("Residual"))
fig = make_subplots(rows=4, cols=1, subplot_titles=("Sales","Seasonality", "Trend", "Residual"))



for name, group in df.groupby('article_name'):
    # Decompose the time series
    sd = seasonal_decompose(group['sales'], period=52)
    t_mean = np.mean(sd.trend)

    # Add the seasonality and trend traces to the subplots
    fig.add_trace(go.Scatter(x=group['yearweek_start'], y=sd.observed, mode='lines', name=name), row=1, col=1)
    fig.add_trace(go.Scatter(x=group['yearweek_start'], y=sd.seasonal, mode='lines', name=name), row=2, col=1)
    fig.add_trace(go.Scatter(x=group['yearweek_start'], y=sd.trend - t_mean, mode='lines', name=name), row=3, col=1)
    fig.add_trace(go.Scatter(x=group['yearweek_start'], y=sd.resid, mode='lines', name=name), row=4, col=1)
    print(f"{name} : {t_mean}")

# Customize the layout
fig.update_layout(title='Seasonality and Trend by Product',
                  xaxis_title='Year-Week Start',
                  yaxis_title='Sales',
                  legend_title='Product')

# Show the figure
fig.show()






import pickle
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


with open('train_set.pickle', 'rb') as handle:
    train = pickle.load(handle)

with open('valid_set.pickle', 'rb') as handle:
    valid = pickle.load(handle)

with open('test_set.pickle', 'rb') as handle:
    test = pickle.load(handle)

with open('predictions.pickle', 'rb') as handle:
    pred = pickle.load(handle)

with open('predictions_formatted.pickle', 'rb') as handle:
    pred_form = pickle.load(handle)

with open('p50_formatted.pickle', 'rb') as handle:
    p50_form = pickle.load(handle)

with open('p90_formatted.pickle', 'rb') as handle:
    p90_form = pickle.load(handle)

with open('p10_formatted.pickle', 'rb') as handle:
    p10_form = pickle.load(handle)

targets = pred_form
targets.drop(['forecast_time', 'identifier'], axis=1)
target_ts = targets.iloc[0,2:]

p50_forecast = p50_form
p50_forecast.drop(['forecast_time', 'identifier'], axis=1)
p50_forecast_ts = p50_forecast.iloc[0,2:]

p90_forecast = p90_form
p90_forecast.drop(['forecast_time', 'identifier'], axis=1)
p90_forecast_ts = p90_forecast.iloc[0,2:]

p10_forecast = p10_form
p10_forecast.drop(['forecast_time', 'identifier'], axis=1)
p10_forecast_ts = p10_forecast.iloc[0,2:]

df = pd.DataFrame(dict(time=np.arange(target_ts.size),
                       target=target_ts))

df["p50"] = p50_forecast_ts
df["p90"] = p90_forecast_ts
df["p10"] = p10_forecast_ts
df["time"] = df["time"]
df = df.reset_index(drop=True).astype(float)
data = pd.melt(df, ['time'])
# g = sns.lineplot(data=df)
data = data.rename(columns={"variable": "label"})
g = sns.lineplot(x='time', y='value', hue='label',
             data=data)

plt.show()



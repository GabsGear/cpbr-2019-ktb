import requests, json
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import numpy as np

fig = plt.figure(figsize=(10,10))
ax = fig.gca()

if os.path.isfile('prices.csv'):
    pass
else:
    pFile = open("prices.csv","w")
    pFile.write("bidPrice,askPrice\n") 
    pFile.close()

def spread(bid,ask):
    return (ask - bid) / ( ask / 100)

def bbands(tickers, window, nDerivation):
    if len(tickers) > window:
        mean = tickers.rolling(window=window).mean()
        std = tickers.rolling(window=window).std()
        bband = {
            'mean':mean,
            'upperBand':mean + (std * nDerivation),
            'lowerBand':mean - (std * nDerivation)
        }
        return bband

def getTickers():
    prices = "https://www.binance.com/api/v3/ticker/bookTicker"
    data = requests.get(url=prices)
    data = data.json()
    mfile = open("prices.csv","a")
    mfile.write("{},{}\n".format(data[1]['bidPrice'], data[1]['askPrice']))
    mfile.close()

def plot():
    df = pd.read_csv("prices.csv")
    if len(df) > 1:
        ax.clear()
        bid = df['bidPrice']
        ask = df['askPrice']
        diferenca = ask[-1:] - bid[-1:]
        plt.title("Litecoin/BTC")
        ax.set_xlim(len(bid)/10, len(bid)+(len(bid)/4)+5)
        ax.plot(bid, label = "Bid - Venda LTC {}".format(np.around(float(bid[-1:]),8)), color = 'blue', alpha = 0.5)
        #ax.plot(ask, label = "Ask - Compra LTC "+ str(np.around(float(ask[-1:]),8)), color = 'red', alpha = 0.5)
        
        bband = bbands(bid, 30, 2)
        ax.plot(bband['mean'], "--", color = "gray", label='SMA')
        ax.plot(bband['upperBand'], "--", color = "green", label='upperBand')
        ax.plot(bband['lowerBand'], "--", color = "green", label='lowerBand')
        plt.legend()
        plt.pause(2)

while True:
    getTickers()
    plot()
 


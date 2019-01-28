#!/usr/bin/env python
# coding: utf-8
import requests, json
from binance.client import Client
from binance.enums import *

import pandas as pd
import numpy as np
import time
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from math import pi
from bokeh.plotting import figure, show, output_file
from bokeh.sampledata.stocks import MSFT

from pyti.exponential_moving_average import exponential_moving_average as ema
from pyti.aroon import aroon_down, aroon_up
from pyti.ichimoku_cloud import tenkansen, kijunsen, chiku_span, senkou_a, senkou_b
from pyti.momentum import momentum

class Aquisition(object):
    
    def __init__(self):
        self.client = Client('', '')
        self.df = pd.DataFrame(columns= ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time'])
           
    def cast_to_dataframe(self, opentime, lopen, lhigh, llow, lclose, lvol, closetime):
        self.df['Open_time'] = opentime
        self.df["date"] = opentime
        self.df['Open'] = np.array(lopen).astype(np.float)
        self.df['High'] = np.array(lhigh).astype(np.float)
        self.df['Low'] = np.array(llow).astype(np.float)
        self.df['Close'] = np.array(lclose).astype(np.float)
        self.df['Volume'] = np.array(lvol).astype(np.float)
        self.df['Close_time'] = closetime
        self.df["date"] = pd.to_datetime(self.df['date'],unit='ms')
        
    def parse(self, candles):   
        opentime, lopen, lhigh, llow, lclose, lvol, closetime = [], [], [], [], [], [], []
        for candle in candles:
            opentime.append(candle[0])
            lopen.append(candle[1])
            lhigh.append(candle[2])
            llow.append(candle[3])
            lclose.append(candle[4])
            lvol.append(candle[5])
            closetime.append(candle[6])    
        self.cast_to_dataframe(opentime, lopen, lhigh, llow, lclose, lvol, closetime)
        
    def get_candles(self):
        #candles = self.client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "1 Ago, 2018")
        candles = self.client.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1DAY)
        self.parse(candles)
        return self.df
    
    def get_price_now(self):
        r = requests.get("https://www.binance.com/api/v3/ticker/price?symbol=BTCUSDT")
        r = r.content
        jsonResponse = json.loads(r.decode('utf-8'))
        return float(jsonResponse['price'])
    
    
    def plot_candles(self):
        df = self.df[450:]
        df["date"] = df["Open_time"]
        df["date"] = pd.to_datetime(self.df['date'],unit='ms')
        inc = df.Close > df.Open
        dec = df.Open > df.Close
        w = 12*60*60*1000 # half day in ms
        TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
        p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=1000, title = "BITCOIN Candlestick")
        p.xaxis.major_label_orientation = pi/4
        p.grid.grid_line_alpha=0.3
        p.segment(df.date, df.High, df.date, df.Low, color="black")
        p.vbar(df.date[inc], w, df.Open[inc], df.Close[inc], fill_color="#006400", line_color="black")
        p.vbar(df.date[dec], w, df.Open[dec], df.Close[dec], fill_color="#F2583E", line_color="black")
        output_file("candlestick.html", title="candlestick.py Grafico de Candles")
        show(p)  


class Bbands(Aquisition):  
    
    def __init__(self, nDer = 2, period = 20):
        super(Bbands, self).__init__()
        self.candles = super().get_candles()
        self.nDer = nDer
        self.period = period
    
    def eval_boillinger_bands(self):
        self.candles['30 Day MA'] = self.candles['Close'].rolling(window=self.period).mean()
        self.candles['30 Day STD'] = self.candles['Close'].rolling(window=self.period).std()
        self.candles['Upper Band'] = self.candles['30 Day MA'] + (self.candles['30 Day STD'] * self.nDer)
        self.candles['Lower Band'] = self.candles['30 Day MA'] - (self.candles['30 Day STD'] * self.nDer)
        
    def configure_plot(self, df):
        TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
        p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=1000, title = "Bbands Chart")
        p.xaxis.major_label_orientation = pi/4
        p.grid.grid_line_alpha=0.3
        p.line(df.date, df.Close, line_color="black")
        p.line(df.date, df['30 Day MA'], line_color="red", legend="30 Day MA", muted_alpha=0.2)
        p.line(df.date, df['Upper Band'], line_color="blue", legend="Upper Band", muted_alpha=0.2)
        p.line(df.date, df['Lower Band'], line_color="green", legend="Lower Band", muted_alpha=0.2)
        
        p.legend.location = "top_left"
        p.legend.click_policy="mute"
        return p
        
    def plot_bands(self):
        df = self.candles
        p = self.configure_plot(df)
        output_file("candlestick.html", title="candlestick.py Bbands")
        show(p)  
        
    def plot_cross_points(self):
        self.eval_boillinger_bands()
        
        self.candles['up_cross'] = np.where((self.candles['Close'] >= self.candles['Upper Band'])
                                            , self.candles['Close'], None)       
        self.candles['down_cross'] = np.where((self.candles['Close'] <= self.candles['Lower Band'])
                                              , self.candles['Close'], None)
        
        p = self.configure_plot(self.candles)        
        p.circle(self.candles.date, self.candles['up_cross'], size=5, color="red", alpha=1, legend="Up cross")
        p.circle(self.candles.date, self.candles['down_cross'], size=5, color="green", alpha=1,legend="Down cross")   
        output_file("candlestick.html", title="candlestick.py Bbands")
        show(p)  
        
    def detect_cross(self):
        self.eval_boillinger_bands()   
        self.candles['up_cross'] = np.where((self.candles['Close'] >= self.candles['Upper Band']) , 1, 0)
        self.candles['down_cross'] = np.where((self.candles['Close'] <= self.candles['Lower Band']) , 1,  0)



class Indicators():

    def __init__(self, candles):
        self.candles = candles
    
    def add_indicators(self):
        self.candles = self.candles.drop(['Open_time', 'Close_time'], axis=1)
        self.candles['EMA - 15'] = ema(self.candles['Close'].tolist(), 15)
        self.candles['aaron down'] = aroon_down(self.candles['Close'].tolist(), 25)
        self.candles['aaron up'] = aroon_up(self.candles['Close'].tolist(), 25)
        self.candles['tenkansen'] = tenkansen(self.candles['Close'].tolist())
        self.candles['kijunsen'] = kijunsen(self.candles['Close'].tolist())
        self.candles['momentun'] = momentum(self.candles['Close'], 15)
        return self.candles


class Corr(object):
    def __init__(self, df):
        self.df = df
        try:
            self.df = self.df.drop(['High', 'Low', 'Open'], axis=1)
        except:
            pass
    
    def pearson(self):
        plt.figure(figsize=(12,8))
        kwargs = {'fontsize':12,'color':'black'}
        sns.heatmap(self.df.corr(),annot=True,robust=True)
        plt.title('Correlation Analysis',**kwargs)
        plt.tick_params(length=3,labelsize=12,color='black')
        plt.yticks(rotation=0)
        plt.show()


class Target(object):

    def __init__(self, candles):
        self.candles = candles
        self.buy = [] 
        self.sell = []
        self.hold = []
        self.close = self.candles['Close'].tolist() 
        
        
    def fill(self, vBuy, vSell):
        self.buy.append(vBuy) 
        self.sell.append(vSell)
        
    def test_target(self, shift):
        for i in range (0, len(self.close)-(shift+1)):
            if float(self.close[i]) < float(self.close[i+shift]):
                self.fill(self.close[i], 'NaN')
            elif float(self.close[i]) > float(self.close[i+shift]):
                self.fill('NaN', self.close[i])
        
        if (len(self.buy) != len(self.candles['Close'])):
            len_diff = (len(self.buy) - len(self.candles['Close']))
            for i in range (len(self.candles.Close)+len_diff, len(self.candles.Close)):
                self.fill('NaN', 'NaN')
                
        self.candles['buy'] = self.buy
        self.candles['sell'] = self.sell
        
    def plot_targets(self):   
        TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
        p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=1000, title = "Target Chart")
        p.xaxis.major_label_orientation = pi/4
        p.grid.grid_line_alpha=0.3

        p.line(self.candles.date, self.candles.Close, line_color="black")

        p.circle(self.candles.date, self.candles['sell'], size=5, color="red", alpha=1, legend='buy')
        p.circle(self.candles.date, self.candles['buy'], size=5, color="green", alpha=1, legend='sell')

        output_file("candlestick.html", title="candlestick.py Bbands")
        show(p)  


def createTarget(candles, shift):
    
    close = candles['Close'].tolist()
    target = []
    
    for i in range (0, len(close)-(shift+1)):
        if float(close[i]) < float(close[i+shift]):
            target.append(0)

        elif float(close[i]) > float(close[i+shift]):
            target.append(1)
            
    if (len(candles.Close) != len(target)):
        len_diff = (len(candles.Close) - len(target))
        for i in range (len(candles.Close), len(candles.Close)+ len_diff):
            target.append('NaN')
        
    candles['target'] = target
    return candles


class TestTarget():
    
    def __init__(self, candles):
        self.last_index = 0
        self.winner = 0 
        self.loser = 0
        self.last_buy = 0
        self.candles = candles
        self.win_percent = 0
        self.lose_percent = 0
        self.total_trades = 0
        
    def test(self):
        for candle in range (0, len(self.candles)):
            if(self.candles['target'][candle] == self.candles['target'][self.last_index]):
                pass
                  
            elif (self.candles['target'][candle] == 0 and self.candles['target'][self.last_index] == 1):
                self.last_index = candle
                self.last_buy = self.candles['Close'][candle]
            else:
                if (self.last_buy == 0):
                    pass
                elif (self.last_buy < self.candles['Close'][candle]):
                    self.winner += 1
                else:
                    self.loser += 1
                self.last_index = candle
    
    def eval_metrics(self):
        self.total_trades = self.winner + self.loser
        self.win_percent = self.winner / self.total_trades
        self.loss_percent = 1 - self.win_percent
        print('Numero de acertos: {} \nNumero de erros: {} \nPorcentagem de acerto: {} \nPorcentagem de erro: {}'
            .format(self.winner, self.loser, self.win_percent, self.loss_percent))    

class cleanData(object):
    def __init__(self, candles):
        self.candles = candles

    def clean(self):
        try:
            self.candles = self.candles.drop(['sell', 'buy'], axis=1)
        except:
            print('Erro apagando sell e buy')
            pass
        
        self.candles = self.candles.dropna()
        for column in self.candles:
            self.candles = self.candles[~self.candles[column].isin(['NaN'])]

        self.candles.to_csv('csv_ok', sep=',', encoding='utf-8')
        return self.candles

import requests, json
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import numpy as np
import seaborn as sns

fig = plt.figure(figsize=(10,10))
ax = fig.gca()
if os.path.isfile('prices.csv'):
    pass
else:
    pFile = open("prices.csv","w")
    pFile.write("bidPrice,askPrice\n") 
    pFile.close()

status = {
    'bid': [],
    'ask' : [], 
    'lb': [],
    'ub' : [],
    'buy': [],
    'sell': [],
    'signals': [],
    'buy_i': [],
    'sell_i' :[],
    'index': []
}

def getTickers():
    prices = "https://www.binance.com/api/v3/ticker/bookTicker"
    data = requests.get(url=prices)
    data = data.json()
    mfile = open("prices.csv","a")
    mfile.write("{},{}\n".format(data[1]['bidPrice'], data[1]['askPrice']))
    mfile.close()

class strategies():
    def __init__(self,bid, ask, window, std):
        self.bid = bid
        self.ask = ask
        self.window =  window
        self.std = std

    def spread(self):
        return (self.ask[-1:] - self.bid[-1:]) / ( self.ask[-1:] / 100)

    def rsi():
        pass

    def bbands(self):
        mean = self.bid.rolling( window=self.window).mean()
        std = self.bid.rolling(window=self.window).std()
        bband = {
            'mean':mean,
            'upperBand':mean + (std * self.std),
            'lowerBand':mean - (std * self.std)
        }
        spread = self.spread()
        delta = self.ask[-1:] - self.bid[-1:] 
        ax.text(len(self.ask) + (len(self.ask)/10), self.bid[-1:] + (delta/2), "Spread " + str(np.around(float(spread),3)) + "%")
        ax.plot(bband['upperBand'], '--', color = "green", alpha = .5)
        ax.plot(bband['lowerBand'], '--', color = "red", alpha = .5)
        ax.scatter(len(self.ask), bband['upperBand'][-1:], color = "green", alpha = .1)
        ax.scatter(len(self.ask), bband['upperBand'][-1:], color = "green", alpha = .1)
        return bband

    def bbandsCross(self, index):
        bband = self.bbands()
        if len(status['signals']) > 1:
            if self.bid[-1:] > bband['lowerBand'][-1:] and self.bid[-2:-1] <= bband['lowerBand'][-2:-1]:
                status['buy'].append(float(self.ask))
                status['buy_i'].append(index)
                signal = 1
            elif self.bid[-1:] <  bband['upperBand'][-1:] and self.bid[-2:-1] <= bband['upperBand'][-2:-1]:
                status['sell'].append(float(self.bid))
                status['sell_i'].append(index)
                signal = 2
            else:
                sinal = 0
                status['signals'].append(signal)
        else:
            signal = 0
            status['signals'].append(signal)
        return 0

    def clearStatus(self):
        status = {
            'bid': [],
            'ask' : [], 
            'lb': [],
            'ub' : [],
            'buy': [],
            'sell': [],
            'signals': [],
            'buy_i': [],
            'sell_i' :[],
        }

    def plotSignals(self):
        self.clearStatus()
        # for i in range(len(self.bid) - (30*2), len(self.bid)):
        #     _ = self.bbandsCross(i)  
        if len(status['buy']) > 0:
            ax.scatter(status['buy_i'], status['buy'], marker = 'v', color = 'red')
            for c in range(len(status['buy_i'])):
                ax.text(status['buy_i'][c], status['buy'][c], ' - buy', color = 'black', alpha = .5)
       
        if len(status['sell']) > 0:
            ax.scatter(status['sell_i'], status['sell'], marker = '^', color = 'green')
            for v in range(len(status['sell_i'])):
                ax.text(status['sell_i'][v], status['sell'][v], ' - sell', color = 'black', alpha = .5)

def main():
    df = pd.read_csv("prices.csv")
    conf = {
        'window': 30,
        'bid': df['bidPrice'],
        'ask': df['askPrice'],
        'delta': df['askPrice'][:-1] - df['askPrice'][:-1] 
    }
    if len(df) > conf['window']:
        ax.clear()
        plt.title("LTC / USD") 
        if len(conf['bid']) < conf['window'] *2 :
            ax.set_xlim(0,len(conf['bid']) + (len(conf['bid'])/10))
        else:
            ax.set_xlim(len(conf['bid'])-conf['window']*2, len(conf['bid'])+10)
            bid_min = np.array(conf['bid'][-conf['window']*2:]).min()
            ask_max = np.array(conf['bid'][-conf['window']*2:]).max()
            ax.set_ylim(bid_min-(bid_min * .01), ask_max+(ask_max * .01))

        ax.plot(conf['bid'], label = "BID - VENDA" + str(np.around(float(conf['bid'][-1:]),2)), color = 'black', alpha = .5)
        ax.plot(conf['ask'], label = "ASK - COMPRA" + str(np.around(float(conf['ask'][-1:]),2)), color = 'gray', alpha = .5)

        if len(conf['bid']) > conf['window'] *2:
            strat = strategies(conf['bid'], conf['ask'], 30, 2)
            bband = strat.bbands()
            strat.plotSignals()
        plt.pause(2)


while True:
    getTickers()
    main()
 



# coding: utf-8

# # Importando os módulos

# In[2]:


import botcpbr as bt
import pandas as pd
from binance.client import Client
from binance.enums import ORDER_TYPE_LIMIT, TIME_IN_FORCE_GTC, SIDE_BUY, SIDE_SELL, TIME_IN_FORCE_FOK

import io
from scipy import misc
from sklearn import tree # pack age tree 
from sklearn.metrics import accuracy_score # medir % acerto
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split # cortar dataset
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')

import logging


# In[6]:


LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s" 
logging.basicConfig(filename = 'logFile.log',
                   level = logging.INFO,
                   format = LOG_FORMAT,
                   filemode = 'w')
logger = logging.getLogger()
logger.info("LogFile created")


# # Encapsulando o modelo

# In[7]:


class Dct(object):
    def __init__(self, candles):
        self.candles = candles
        self.scaler = MinMaxScaler()
        self.train = None
        self.test = None
        self.model = None
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume', '30 Day MA',
                   '30 Day STD', 'Upper Band', 'Lower Band', 'up_cross', 'down_cross',
                   'EMA - 15', 'aaron down', 'aaron up', 'tenkansen', 'kijunsen',
                   'momentun']
        self.xtrain = None
        self.ytrain = None
        self.xtest = None
        self.ytest = None
        self.fitted = None
        self.state = None

    def split_and_scale(self):
        self.candles = self.candles.drop(['date'], axis=1)    
        self.candles[self.features] = self.scaler.fit_transform(self.candles[self.features])      
        self.train, self.test = train_test_split(self.candles, test_size=round(len(self.candles)*0.2))
    
    def cast_target(self):
        self.train['target'] = pd.to_numeric(self.train['target'], downcast='float')
        self.test['target'] = pd.to_numeric(self.test['target'], downcast='float')
        
    def create_model(self):
        self.model = GradientBoostingClassifier(criterion='friedman_mse', init=None,
                    learning_rate=0.1, loss='deviance', max_depth=3, max_features=None, 
                    max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                    min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
                    n_estimators=100, presort='auto', random_state=None, subsample=1.0, verbose=0, 
                    warm_start=False)
    
    def mount_model(self):
        self.x_train = self.train[self.features]
        self.y_train = self.train['target']
        self.x_test = self.test[self.features]
        self.y_test = self.test['target']
        
    def fit_model(self):
        self.fitted = self.model.fit(self.x_train, self.y_train) 
        
    def get_prediction(self):
        self.y_pred = self.fitted.predict(self.x_test)
        return accuracy_score(self.y_test, self.y_pred)*100
        
    def make_prediction(self, state):
        return self.fitted.predict(state)
        
    def get_model(self):
        self.split_and_scale()
        self.cast_target()
        self.create_model()
        self.mount_model()
        self.fit_model()
        
    def create_signal(self):
        self.get_model()
        acc = self.get_prediction()
        state = self.candles.tail(1)
        self.state = state.drop(['target'], axis=1)
        pred = self.make_prediction(self.state)
        
        if pred == 0:
            return 'buy', acc
        else: 
            return 'sell', acc


# # Definindo a estratégia

# In[8]:


def run_strategy():
    #pegando os candles e calculando bbands e os cross das bandas
    bb = bt.Bbands()
    bb.eval_boillinger_bands()
    bb.detect_cross()

    #adicionando os outros indicadores
    ind = bt.Indicators(bb.candles)
    candles = ind.add_indicators()
    
    #Definindo o target
    candles = bt.createTarget(candles, 5)

    #Limpando o dataframe
    dt = bt.cleanData(candles)
    candles = dt.clean()
    
    model = Dct(candles)
    sig, acc = model.create_signal()
    return sig, acc, model.state


# # Bot

# In[9]:


class Orders(object):
    
    def loginApi(self):
        return Client("RhjQHqKNMG865W12yN2DkQER6oBofnreOuOAjMqwtdopJgTpd6WN9x7OAmAECW3W",
                        "omHjQxSIe4BV9YjtlhQGrM2SbxInM6ouu5kFAzptUdIz8HiT9qlWPWkQrVYZuSGT")
    
    def create_buy_order(self):      
        client = self.loginApi()
        try:
            order = client.create_test_order(
            symbol='BTCUSDT',
            side=SIDE_BUY,
            type=ORDER_TYPE_LIMIT,
            timeInForce=TIME_IN_FORCE_FOK,
            quantity=100,
            price='3420.06')
            logger.info("Sucefull creating a buy order")
            return True
        except:
            logger.error("Error creating a buy order")
            return False
        
    def create_sell_order(self):
        client = self.loginApi()
        try:
            order = client.create_test_order(
            symbol='BTCUSDT',
            side=SIDE_SELL,
            type=ORDER_TYPE_LIMIT,
            timeInForce=TIME_IN_FORCE_FOK,
            quantity=100,
            price='3420.06')
            logger.info("Sucefull creating a sell order")
            return True
        except:
            logger.error("Error creating a sell order")
            return False

class Bot(Orders):
    def __init__(self):
        self.botStatus = {
            'openOrder':'none',
            'lastOperation':'sell',
            'classifierAccuracy': None
        }
        self.trades = 0
        self.signal = 'none'

    
    def check_signal(self):
        self.signal, acc, state = run_strategy()
        self.botStatus['classifierAccuracy'] = acc
        
        if (self.botStatus['lastOperation'] == 'buy'):
            logger.info("We have open orders, waiting for sell signal")
            if (self.signal == 'sell'):
                f= open("historic.txt","a")
                f.write('sell at: ' + str(state['Close'])+ ',\n')
                self.sell()
        else:
            logger.info("Waiting for a buy signal")
            if (self.signal == 'buy'):
                f= open("historic.txt","a")
                f.write('buy at: ' + str(state['Close']) + ',\n')
                self.buy()
    
    def buy(self): 
        if super().create_buy_order():
            self.updateStatus('yes', self.signal)

    def sell(self):
        if super().create_sell_order():
            self.updateStatus('no', self.signal)
            self.trades += 1
        
    def updateStatus(self, status, operation):
        self.botStatus['openOrder'] = status
        self.botStatus['lastOperation'] = operation


# # Rodando

# In[12]:


from IPython.display import clear_output
import time

data = bt.Aquisition()
candles = data.get_candles()

bot = Bot()
runtime = 0
while(True):
    clear_output() 
    print('Runtime = %s seconds' % (runtime))
    print('Numero de trades: {}'.format(bot.trades))
    print('Sinal atual: ' + str(bot.signal))
    
    print(bot.botStatus)
    
    bot.check_signal()
    time.sleep(30)
    runtime += 30


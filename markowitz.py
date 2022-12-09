import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimze
from datetime import date

stocks = ['AAPL','WMT','TSLA','GE','AMZN','DB']
startdate = date(2010,1,1)
enddate = date(2022,12,7)
#average no. of trading days in a year
trading_days = 252

#it will generate random weight for different portfolios (10000 times)
num_portfolio = 10000

def download_data():
    stocks_data = {}
    for i in stocks:
        ticker = yf.Ticker(i)
        stocks_data[i] = ticker.history(start=startdate,end=enddate)['Close']


    df = pd.DataFrame(stocks_data)
    print(df)
    return df

def show_data(df):
    df.plot()
    plt.show()

def calculate_data(df):
    #LOG is used to normalize the data to generate comparable matrics
    log_return = np.log(df/df.shift(1))
    print(log_return[1:])
    return log_return[1:]

def show_calculate_date(log_return):
    log_return.plot()
    plt.show()

def statistics(rt):
    #instead of daily returns it will focus on annual returns
    #mean of annual returns
    print("Mean annual return is \n",rt.mean() * trading_days)
    print("Covariance of annual return is \n",rt.cov() * trading_days)

def mean_variance(rt,wt):
    portfolio_return = np.sum(rt.mean()*wt) * trading_days
    portfolio_volatility = np.sqrt(np.dot(wt.T,np.dot(rt.cov() * trading_days,wt)))
    print("Expected portfolio return is :",portfolio_return)
    print("Expected portfolio volatility is :",portfolio_volatility)

def generate_portfolio(rt):
    portfolio_means = []
    portfolio_risk = []
    portfolio_weights = []
    for j in range(num_portfolio):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(rt.mean() * w ) * trading_days)
        portfolio_risk.append(np.sqrt(np.dot(w.T,np.dot(rt.cov() * trading_days,w))))
    return np.array(portfolio_weights) , np.array(portfolio_means), np.array(portfolio_risk)

def show_portfolio(rt,vt):
    plt.scatter(vt,rt,c=rt/vt,marker='*')
    plt.grid(True)
    plt.xlabel("Expected volatility")
    plt.ylabel("Expected returns")
    plt.show()

def stat_with_weight(wt,rt):
    portfolio_return = np.sum(rt.mean() * wt) * trading_days
    portfolio_volatility = np.sqrt(np.dot(wt.T,np.dot(rt.cov() * trading_days,wt)))
    return np.array([portfolio_return,portfolio_volatility,portfolio_return / portfolio_volatility])

def sharpe(wt,rt):
    return -stat_with_weight(wt,rt)[2]

def optimize_portfolio(wt,rt):
    constraints = {'type':'eq','fun':lambda x:np.sum(x)-1}
    bounds = tuple((0,1) for g in range(len(stocks)))
    optimze.minimize(fun=sharpe,x0=weight[0],args=rt,method='SLSQP',bounds=bounds,constraints=constraints)

def print_optimum_portfolio(opt,rt):
    print(opt['x'],round(3))
    statistics(stat_with_weight(opt['x'],round(3),rt))

def show_optimum_portfolio(opt,rets,rt,vt):
    plt.scatter(vt,rt,c=rt/vt,marker='*')
    plt.grid(True)
    plt.xlabel("Expected volatility")
    plt.ylabel("Expected returns")
    #plt.plot(stat_with_weight(opt['x'], rets)[1], stat_with_weight(opt['x'],rets)[0],'g*', markersize=20.0)
    plt.plot(stat_with_weight(opt['x'], rets)[1], stat_with_weight(opt['x'], rets)[0], 'g*', markersize=20.0)
    plt.show()





#download_data()
print("\n\n\n\n\n")
dataset = download_data()
show_data(dataset)
print("\n\n\n\n\n")
calculate_data(dataset)
print("\n\n\n\n\n")
log_data = calculate_data(dataset)
print("\n\n\n\n\n")
show_calculate_date(log_data)
print("\n\n\n\n\n")
statistics(log_data)
print("\n\n\n\n")
weight,mean,risk = generate_portfolio(log_data)
show_portfolio(mean,risk)
optimum = optimize_portfolio(weight,log_data)
show_optimum_portfolio(optimum,log_data,mean,risk)
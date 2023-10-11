from WindPy import w
import pandas as pd
import numpy
import os
from datetime import *
import uqer
from uqer import DataAPI
def make_dates_stocks0(date_start,dates_end,client):


    data=DataAPI.TradeCalGet(exchangeCD=u"XSHG,XSHE",beginDate=date_start.replace('-',''),endDate=dates_end.replace('-',''),isOpen=u"1",field=u"",pandas="1")
    dates=data['calendarDate'].values
    dates=numpy.sort(list(set(list(dates))))
    #print (dates)
    #pd.DataFrame(numpy.array([dates]).T,columns=['tradeDate']).to_csv('D:/data_predict_project/股票与交易日列表/tradeDatelist.csv',encoding='gbk')
    data=DataAPI.SecIDGet(partyID=u"",ticker=u"",cnSpell=u"",assetClass=u"E",exchangeCD="",listStatusCD="",field=u"",pandas="1")
    stocks0 = numpy.sort(list(set(list(data['ticker'].values))))
    stocks0 = [st_num for st_num in stocks0 if
               ((st_num[0] == '0') | (st_num[0] == '3') | (st_num[0] == '6')) & (len(st_num) == 6)]

    #print (stocks0)
    #pd.DataFrame(numpy.array([stocks0]).T, columns=['stock_num']).to_csv('D:/data_predict_project/股票与交易日列表/stocknumlist.csv', encoding='gbk')

    #print (data)



    return dates,stocks0
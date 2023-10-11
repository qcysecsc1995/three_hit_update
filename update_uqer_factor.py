from WindPy import w
import pandas as pd
import numpy
import os
from datetime import *
import uqer
from uqer import DataAPI


def get_MktStockFactorsDateRangeProGet(date_start,dates_end,stocks0,client):
    while (date_start<=dates_end):
        date_str = date_start.replace('-', '')
        data = DataAPI.MktStockFactorsOneDayProGet(tradeDate=date_str,secID=u"",ticker=stocks0,field=u"",pandas="1")
        if len(data) > 0:
            # need_col=['ticker','endDate','reportType','actPubtime','updateTime']
            # data=data.reindex(columns=need_col)
            data.to_csv('D:/因子择时/择时数据/通联股票因子/' + date_start + '.csv', encoding='gbk')

        date_start = (datetime.strptime(date_start, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    return

def get_RMExposureDayGet(date_start,dates_end,client):
    while (date_start<=dates_end):
        date_str = date_start.replace('-', '')
        data = DataAPI.RMExposureDayGet(secID=u"",ticker=u"",tradeDate=u"",beginDate=date_str,endDate=date_str,field=u"",pandas="1")
        if len(data) > 0:
            # need_col=['ticker','endDate','reportType','actPubtime','updateTime']
            # data=data.reindex(columns=need_col)
            data.to_csv('D:/因子择时/择时数据/股票风险因子/' + date_start + '.csv', encoding='gbk')

        date_start = (datetime.strptime(date_start, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    return


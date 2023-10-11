# import pp
# from stock_member import *
import numpy
import math
import random
import matplotlib.pyplot as plt
import xlrd
import xlwt
import csv
import xlsxwriter

import pandas as pd
from pandas import Series
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import svm
from sklearn import gaussian_process
from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.decomposition import PCA

import time
from sklearn.ensemble import GradientBoostingClassifier
import os
import datetime
def read_raw_pk(file):
    data=pd.read_pickle(file)
    try:

        stocks=data.columns

        dates=data.index
        data = data.values

        #if numpy.shape(data)==(1,1):
        #    data=[]
    except KeyError:
        data=numpy.array([])
        stocks=[]
        dates=[]


    return data,stocks,dates
def read_raw_h5(file):
    store = pd.HDFStore(file, mode='r')

    data = store.select('data')

    store.close()
    try:

        stocks = data.columns
        stocks=[st_num for st_num in stocks if (st_num[0]=='0')|(st_num[0]=='3')|(st_num[0]=='6')]
        data=data.reindex(columns=stocks)
        stocks = data.columns
        dates = data.index
        data = data.values

        # if numpy.shape(data)==(1,1):
        #    data=[]
    except KeyError:
        data = numpy.array([])
        stocks = []
        dates = []

    return data, stocks, dates


def make_uqer_factor_mat(stocks0):
    dir_name='D:/因子择时/择时数据/通联股票因子/'
    file_list=os.listdir(dir_name)

    factor_name_list=['DividendCover','NetAssetGrowRate','OperatingCycle','TotalAssetGrowRate',
                      'FinancialExpenseRate','OperatingProfitGrowRate','NPParentCompanyGrowRate',
                      'TotalAssetsTRate','NPParentCompanyCutYOY','ETOP','PBIndu','PSIndu','PEIndu',
                      'DividendPS','CapitalSurplusFundPS','NOCFToOperatingNILatest','NetProfitRatio','OperatingProfitRatio',
                      'Skewness','AccountsPayablesTRate','OperatingNIToTP']
    factor_dict={}
    for factor_name in factor_name_list:
        factor_dict[factor_name]=[]

    trade_date_list=[]
    for file in file_list:
        print (file)
        date_data=pd.read_csv(dir_name+file,encoding = 'gbk')
        date_data=date_data.reindex(columns=['ticker','tradeDate']+factor_name_list)
        date_data['ticker']=[str(int(ticker)).zfill(6) for ticker in date_data['ticker'].values]
        date_data.index=date_data['ticker'].values
        date_data=date_data.reindex(index=stocks0)
        #print(date_data)
        trade_date_list.append(file[:-4])
        for factor_name in factor_name_list:

            factor_dict[factor_name].append(date_data[factor_name].values)

    for factor_name in factor_name_list:
        out_df=pd.DataFrame(factor_dict[factor_name],index=trade_date_list,columns=stocks0).fillna(method='ffill')
        out_df.to_csv('D:/基本面三击策略每日更新/uqer基本面因子/'+factor_name+'.csv',encoding = 'gbk')

    return

def make_barra_mat(stocks0):
    dir_name = 'D:/因子择时/择时数据/股票风险因子/'
    file_list = os.listdir(dir_name)
    factor_name_list = ['BETA', 'MOMENTUM', 'SIZE', 'EARNYILD', 'RESVOL', 'GROWTH', 'BTOP', 'LEVERAGE', 'LIQUIDTY',
                        'SIZENL']
    factor_dict = {'BETA': [], 'MOMENTUM': [], 'SIZE': [], 'EARNYILD': [], 'RESVOL': [], 'GROWTH': [], 'BTOP': [],
                   'LEVERAGE': [], 'LIQUIDTY': [], 'SIZENL': []}
    trade_date_list = []
    for file in file_list:
        print(file)
        date_data = pd.read_csv(dir_name + file, encoding='gbk')
        trade_date_list.append(file[:-4])
        for factor_name in factor_name_list:
            factor_data = date_data.pivot(index='tradeDate', columns='ticker', values=factor_name)
            factor_data.columns = [ticker.zfill(6) for ticker in factor_data.columns.astype(str)]
            factor_data = factor_data.reindex(columns=stocks0)
            factor_dict[factor_name].append(factor_data.values[0])

    for factor_name in factor_name_list:
        out_df = pd.DataFrame(factor_dict[factor_name], index=trade_date_list, columns=stocks0)
        out_df.to_csv('D:/因子择时/择时因子计算/多因子归因择时/风险因子矩阵/' + factor_name + '.csv', encoding='gbk')
    return
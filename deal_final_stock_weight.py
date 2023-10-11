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
from scipy import optimize
from decimal import Decimal
import pandas as pd
from pandas import Series
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import svm
from sklearn import gaussian_process
from sklearn.naive_bayes import GaussianNB
import random
from sklearn import tree

from sklearn.decomposition import PCA

import time
from sklearn.ensemble import GradientBoostingClassifier
import os
import datetime
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

def read_raw(file):
    store=pd.HDFStore(file,mode='r')

    data=store.select('data')


    store.close()
    return data

def read_raw_pk(file):
    data = pd.read_pickle(file)
    try:

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

def deal_final_stock_weight(stocks0, dates):

    factor_file_path='D:/基本面三击策略每日更新/三者兼优选股仓位/'

    file_weight_path = 'D:/基本面三击策略每日更新/各因子消除波动内在权重/'
    factor_weight_data=pd.read_csv('D:/基本面三击策略每日更新/重仓各子策略权重.csv',index_col=0,encoding='gbk')\
        .reindex(index=dates,method='ffill').fillna(value=0)

    print (factor_weight_data)

    factor_file_list=factor_weight_data.columns


    all_factor_mat=[]
    index_minus_mat=[]
    index_minus_mat_all=[]
    for file_name in factor_file_list:
        print (file_name)
        factor_data=pd.read_csv(factor_file_path+file_name,index_col=0,encoding='gbk')\
            .reindex(index=dates,method='ffill').reindex(columns=stocks0).fillna(value=0)
        weight_list=factor_weight_data[file_name].values

        single_weight_data = pd.read_csv(file_weight_path + file_name, index_col=0, encoding='gbk') \
            .reindex(index=dates, method='ffill').fillna(value=0)
        single_weight_data=pd.DataFrame(single_weight_data.values[:-1],index=dates[1:],columns=single_weight_data.columns)
        single_weight_data=single_weight_data.reindex(index=dates, method='ffill').fillna(value=0)
        #print (factor_data)
        adj_fator_data=[factor_data.values[ii]/numpy.sum(factor_data.values[ii])*weight_list[ii]
                        if numpy.sum(factor_data.values[ii])>0 else factor_data.values[ii] for ii in range (len(weight_list))]

        all_factor_mat.append(adj_fator_data)
        index_minus_weight = single_weight_data['index500'].values
        fin_index_min = (index_minus_weight) * weight_list
        index_minus_mat.append(fin_index_min)

        index_minus_mat_all.append(weight_list)


    all_factor=numpy.sum(all_factor_mat,axis=0)
    all_factor = numpy.where(numpy.array(all_factor) > 0.0001, all_factor, 0)
    #all_factor = [numpy.nan_to_num(line/numpy.sum(line)) for line in all_factor]
    all_factor_df=pd.DataFrame(all_factor,index=dates,columns=stocks0)
    all_factor_df.to_csv('D:/基本面三击策略每日更新/good_stock_finalweight.csv',encoding='gbk')
    all_factor_df.to_csv('D:/deal_factor_index_adjust/uqer基本面衍生因子/good_stock_finalweight.csv',
                               encoding='gbk')
    all_factor_df.to_csv('D:/因子择时/择时因子计算/uqer公告文本研究/uqer基本面衍生因子/good_stock_finalweight.csv',
                         encoding='gbk')

    for i in range (1,len(all_factor_df)):
        if sum(abs(all_factor_df.values[i-1]-all_factor_df.values[i]))>10**(-10):
            out_stocks = all_factor_df.columns
            out_factor = all_factor_df.values[i]
            out_stocks = out_stocks[out_factor > 0]
            out_factor = out_factor[out_factor > 0]
            out_factor = out_factor / numpy.sum(out_factor)
            act_out_date = all_factor_df.index[i]
            date_index = numpy.full(len(out_stocks), act_out_date)
            out_df = pd.DataFrame(numpy.array([out_stocks, out_factor]).T, index=date_index,
                                  columns=['ticker', 'weight'])
            out_df.index.name = 'Date'
            out_df.to_csv('D:/基本面三击策略每日更新/三击策略因子仓位/' + act_out_date + '.csv', encoding='gbk')
            out_df.to_csv('D:/wind金股因子每日更新/alpha项目文件夹/三击策略因子仓位/' + act_out_date + '.csv', encoding='gbk')
            #out_df.to_csv('//192.168.26.40/alpha项目文件夹/三击策略因子仓位/' + act_out_date + '.csv', encoding='gbk')

    index_minus = numpy.sum(index_minus_mat, axis=0)
    index_minus_df = pd.DataFrame()
    index_minus_df['weight'] = index_minus
    index_minus_df.index = dates
    index_minus_df.to_csv('D:/基本面三击策略每日更新/对冲权重.csv', encoding='gbk')
    return




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

def deal_single_factor_index():
    close, stocks, dates = read_raw_h5('D:\data_predict_project\data deal project\stock_price_adj_pyd/closePrice.h5')



    index_frame=read_raw(r'D:\data_predict_project\data deal project\index_data_pyd\index_price_pyd\000852\closeIndex.h5')
    index_frame=pd.DataFrame(index_frame.values,index=index_frame.index.astype(str),columns=index_frame.columns)
    index_frame=index_frame.reindex(index=dates)
    index_frame.columns=['ret']
    ret_index_frame1000=numpy.log(index_frame/index_frame.shift(1)).fillna(value=0)
    index_frame=read_raw(r'D:\data_predict_project\data deal project\index_data_pyd\index_price_pyd\000905\closeIndex.h5')
    index_frame=pd.DataFrame(index_frame.values,index=index_frame.index.astype(str),columns=index_frame.columns)
    index_frame=index_frame.reindex(index=dates)
    index_frame.columns=['ret']
    ret_index_frame500=numpy.log(index_frame/index_frame.shift(1)).fillna(value=0)
    index_frame=read_raw(r'D:\data_predict_project\data deal project\index_data_pyd\index_price_pyd\000300\closeIndex.h5')
    index_frame=pd.DataFrame(index_frame.values,index=index_frame.index.astype(str),columns=index_frame.columns)
    index_frame=index_frame.reindex(index=dates)
    index_frame.columns=['ret']
    ret_index_frame300=numpy.log(index_frame/index_frame.shift(1)).fillna(value=0)





    ret_index_frame1000=numpy.exp(ret_index_frame1000)-1
    ret_index_frame500=numpy.exp(ret_index_frame500)-1
    ret_index_frame300=numpy.exp(ret_index_frame300)-1
    #print (ret_index_frame500)


    def cov_f(weights,ret_mat):
        #weights=numpy.array(weights)*numpy.array(weights)*numpy.array([1,-1,-1,-1])
        #weights[0]=weights[0]*weights[0]
        weights=numpy.array(weights)
        ret_mat = numpy.array(ret_mat)
        ret_all=numpy.dot(weights,ret_mat)/numpy.sum(abs(weights))
        std_ret=numpy.std(ret_all)
        return std_ret


    def cov_fun(ret_mat):
        #cov_ret=numpy.cov(ret_mat.T)
        fun_re=lambda weights: cov_f(weights,ret_mat)
        return fun_re



    def con_fun(weight):
        #weight[0]=weight[0]*weight[0]
        con_s=numpy.sum(abs(weight))-1
        return con_s


    fund_ret_file_path='D:/基本面三击策略每日更新/选股仓位多头收益/'
    file_list=os.listdir(fund_ret_file_path)
    fund_ret_df_list=[]
    for file_name in file_list:

        factor_df2=pd.read_csv(fund_ret_file_path+file_name,index_col=0,encoding = 'gbk')
        factor_df2=factor_df2.reindex(index=dates,fill_value=numpy.nan)
        factor_df2 = numpy.exp(factor_df2) - 1

        ret_daily=[]
        ret_date=[]
        weight_daily=[]
        weight_daily_sign=[]
        for i in range (len(dates)+1):
            len_range=250
            if i>len_range:

                ret_f2 = factor_df2['ret'].values[i - len_range:i-1]


                ret_300=ret_index_frame300['ret'].values[i-len_range:i-1]
                ret_500 = ret_index_frame500['ret'].values[i - len_range:i-1 ]
                ret_1000 = ret_index_frame1000['ret'].values[i - len_range:i-1 ]
                ret_mat=[ret_f2,ret_500]#,ret_1000,ret_index_chose1,ret_index_chose300
                ret_name=['factor','index500']#,'index1000','index_chose','index300_chose'
                #weights0 = [0.1,-0.5,-0.1,-0.1]
                sign_ii=[1,0]#,0,1,1
                bonds=[(-1.0,-0.0) if sign_ii[ii]!=1 else (0,1.0) for ii in range (len(ret_mat))]
                if numpy.isfinite(numpy.std(ret_f2)):
                    def con_cov_shgo(weights):
                        re = numpy.std(numpy.dot(numpy.array(weights), numpy.array(ret_mat)))# / numpy.sum(abs(weights)))

                        return re
                    # print (make_con(weights0))
                    cons = ({'type': 'eq', 'fun': con_fun})
                    #res = optimize.minimize(cov_fun(ret_mat), weights0,method='SLSQP')#,constraints=cons
                    res=optimize.shgo(con_cov_shgo,bonds,constraints=cons,n=64, iters=3, sampling_method='sobol')#
                    success_sign=res.success
                    res_x = res.x  # numpy.array([num if abs(num)>10**(-16) else 0 for num in res.x])
                    print (res_x)
                    model_output=1#numpy.average(numpy.dot(numpy.array(res_x), numpy.array(ret_mat))[-120:])
                    if success_sign*1 >0:
                        model_output_sign=numpy.average(numpy.dot(numpy.array(res_x), numpy.array(ret_mat))[-120:])
                    else:
                        model_output_sign = 0
                else:
                    model_output=-1
                    model_output_sign=0
                if model_output<0:
                    res_x=numpy.full(len(ret_mat),0)
                #res_x=res_x*res_x*numpy.array([1,-1,-1,-1])
                #res_x[0]=res_x[0]*res_x[0]
                if i<len(dates):
                    ret_mat_n=[factor_df2['ret'].values[i],ret_index_frame500['ret'].values[i]
                        ]
                    #,factor_index_chose1['ret'].values[i]
                    #print (ret_mat_n)
                    ret_n=numpy.sum(numpy.nan_to_num(numpy.array(res_x)*numpy.array(ret_mat_n)/numpy.sum(abs(res_x[:1]))))
                    print (dates[i],numpy.nan_to_num(ret_n) ,numpy.sum(abs(res_x[:1])),model_output)
                    ret_daily.append(numpy.nan_to_num(ret_n))
                    ret_date.append(factor_df2.index[i])
                weight_daily.append(numpy.nan_to_num(res_x/abs(res_x[0])))
                weight_daily_sign.append(model_output_sign)
        out_df=pd.DataFrame()
        out_df['ret']=ret_daily
        out_df.index=ret_date
        out_df.to_csv('D:/基本面三击策略每日更新/各因子消除波动后的多头收益/'+file_name,encoding='gbk')
        weight_out_df=pd.DataFrame(weight_daily[1:],index=ret_date,columns=ret_name)
        weight_out_df.to_csv('D:/基本面三击策略每日更新/各因子消除波动内在权重/' + file_name, encoding='gbk')
        weight_daily_sign_df = pd.DataFrame(weight_daily_sign[1:], index=ret_date, columns=['sign'])
        weight_daily_sign_df.to_csv('D:/基本面三击策略每日更新/各因子消除波动内在权重标志/' + file_name, encoding='gbk')
    return

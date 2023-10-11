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



def make_retline(factor_read,index_price,close_read,trade_steady,dates):
    print (len(close_read)-1)
    ret_rank = []
    ret_pos = []
    retindex = []
    rk=factor_read[0]
    ret_dates=[]
    for i in range(1, len(close_read)-1):#
        rk_1=rk.copy()
        rk=factor_read[i].copy()

        nan_move_cap=numpy.sum(rk_1[numpy.array(trade_steady[i])==0])
        nan_move_cap_now = numpy.sum(rk[numpy.array(trade_steady[i])==0])

        if nan_move_cap_now!=1 and nan_move_cap!=1:
            rk=numpy.array(rk)/(1-nan_move_cap_now)*(1-nan_move_cap)
            rk=numpy.where(numpy.array(trade_steady[i])==0,rk_1,rk)

        else:
            rk=rk_1.copy()
        last_ret=close_read[i]/close_read[i-1]
        last_ret=numpy.where(numpy.isfinite(last_ret),last_ret,1)
        now_ret = close_read[i+1] / close_read[i ]
        now_ret = numpy.where(numpy.isfinite(now_ret), now_ret,1)
        all_cap=numpy.sum(rk_1)#numpy.sum(last_ret*rk_1)

        cost=numpy.sum(abs(rk*all_cap-last_ret*rk_1))*0.0015
        if numpy.sum(factor_read[i-1])==0:
            all_cap=1
            cost=0.00

        count_len=500

        if len(ret_pos)>count_len:
            ret_p=numpy.array(ret_pos[-count_len:])
            ret_i=numpy.array(retindex[-count_len:])
            ret_p=ret_p-numpy.mean(ret_p)
            ret_i = ret_i - numpy.mean(ret_i)
            cov_R=numpy.cov(ret_p,ret_i)
            posi_pos=2*cov_R[0][1]/(cov_R[0][0]-cov_R[1][1]+cov_R[0][1]*2)
            posi_index=2-abs(posi_pos)
        else:
            posi_pos=1
            posi_index=1
        if (not numpy.isfinite(posi_pos)) or (not numpy.isfinite(posi_index)):
            posi_pos = 1
            posi_index = 1
        all_cap_ret=(numpy.sum(now_ret*rk*all_cap)-cost)/all_cap
        all_ret=numpy.log(1+((all_cap_ret-1)*posi_pos-(index_price[i+1][0]/index_price[i][0]-1)*posi_index)/1)
        all_ret_pos = numpy.log(1 + (all_cap_ret - index_price[i ][0] / index_price[i][0]) / 1)
        all_retindex = numpy.log((index_price[i+1][0] / index_price[i][0]) / 1)
        print (numpy.sum((rk)),numpy.sum(rk_1),all_cap,all_cap_ret,all_ret,posi_pos)

        if numpy.sum(factor_read[i])==0 or numpy.isnan(all_ret) or numpy.sum(rk)<0.999 or numpy.sum(rk_1)<0.999:
            all_ret=-0.00
            all_ret_pos=0
            all_retindex=0
            if numpy.sum(factor_read[i])==0 and numpy.sum(factor_read[i-1])!=0:
                all_ret = -0.00#15
                all_ret_pos = -0.00#15
                all_retindex = 0
        ret_rank.append(all_ret)
        ret_pos.append(all_ret_pos)
        retindex.append(all_retindex)
        ret_dates.append(dates[i+1])

    #print (ret_rank[ret_rank!=0])
    ret_rank1=numpy.array(ret_rank)[numpy.array(ret_rank)!=0]
    ret_sharpe=numpy.average(ret_rank1)/numpy.std(ret_rank1)*numpy.sqrt(250)

    #ret_rank = numpy.array([ret_rank]).T
    return ret_sharpe,ret_rank,ret_pos,retindex,ret_dates

def industry_one(factor,industry_mat,industry_list):
    out_list=[]
    for industry_name in industry_list:
        factor_ind=numpy.where(industry_mat==industry_name,factor,0)
        factor_ind_sum=numpy.sum(factor_ind,axis=1)
        factor_ind_one=[factor_ind[i]/factor_ind_sum[i] for i in range (len(factor_ind_sum))]
        out_list.append(factor_ind_one)
    out_mat=numpy.sum(out_list,axis=0)
    return out_mat

def arank1(r,reverse=1):
    r=numpy.array(r)
    rr = numpy.argsort(reverse*r)
    for rank in range (0,len(rr)):
        r[rr[rank]]=rank+1
    return r

def rank1(x,reverse=1):
    x_col=x.columns
    x_index=x.index
    x_value=x.values
    x_out=[]
    for x_line in x_value:
        x_col_part=x_col[numpy.isfinite(x_line)]

        x_out_line=arank1(x_line[numpy.isfinite(x_line)],reverse=reverse)
        x_out_line_all=pd.DataFrame([x_out_line],columns=x_col_part).reindex(columns=x_col,fill_value=numpy.nan).values[0]
        x_out.append(x_out_line_all)
    x_out_df=pd.DataFrame(x_out,index=x_index,columns=x_col)
    return x_out_df

def arank(r,reverse=1):
    r=numpy.array(r)
    rr = numpy.argsort(reverse*r)
    for rank in range (0,len(rr)):
        r[rr[rank]]=rank+1
    return r

def rank(x,reverse=1):
    x_col=x.columns
    x_index=x.index
    x_value=x.values
    x_out=[]
    for x_line in x_value:
        x_col_part=x_col[numpy.isfinite(x_line)]

        x_out_line=arank(x_line[numpy.isfinite(x_line)],reverse=reverse)
        x_out_line_all=pd.DataFrame([x_out_line],columns=x_col_part).reindex(columns=x_col,fill_value=numpy.nan).values[0]
        x_out.append(x_out_line_all)
    x_out_df=pd.DataFrame(x_out,index=x_index,columns=x_col)
    return x_out_df

def deal_all_factor_asset_return():
    #factor_df=pd.DataFrame(factor_df.values[:-1],index=factor_df.index[:-1],columns=factor_df.columns)
    close, stocks, dates = read_raw_h5('D:\data_predict_project\data deal project\stock_price_adj_pyd/closePrice.h5')



    close_frame=read_raw(r'D:\data_predict_project\data deal project\stock_price_adj_pyd\closePrice.h5')

    close_frame=close_frame.reindex(columns=stocks,index=dates)

    index_frame=read_raw(r'D:\data_predict_project\data deal project\index_data_pyd\index_price_pyd\000905\closeIndex.h5')
    index_frame=pd.DataFrame(index_frame.values,index=index_frame.index.astype(str),columns=index_frame.columns)
    index_frame=index_frame.reindex(index=dates)

    trade_frame=read_raw(r'D:\data_predict_project\data deal project\stock_price_adj_pyd\isTrade.h5')
    trade_frame=trade_frame.reindex(columns=stocks,index=dates)

    close_read=close_frame.values
    trade_read=trade_frame.values
    index_price=index_frame.values

    trade_steady=[]
    for i in range (0,len(close_read)):
        rk=[]
        for j in range (len(trade_read[i])):

                if (numpy.isnan(trade_read[i][j]) or trade_read[i][j] == 0):# or (round(close_read[i - 1][j] * 1.097, 2) <= close_read[i][j] or round(close_read[i - 1][j] * 0.903,2) >= close_read[i][j]):

                    rk.append(0)
                else:
                    rk.append(1)

        trade_steady.append(rk)

    ##############################################################################################################
    print ('cal factor')

    fac_path='D:/基本面三击策略每日更新/三者兼优选股仓位/'
    fac_list=os.listdir(fac_path)
    for file in fac_list:
        factor_df1 = pd.read_csv(fac_path+file, index_col=0, encoding='gbk').reindex(index=dates,method='ffill').reindex(columns=stocks,fill_value=0)
        factor_df = factor_df1.fillna(value=0)  #
        factor_df = pd.DataFrame(factor_df.values, index=factor_df.index, columns=factor_df.columns)
        print (factor_df)
        factor_or=factor_df.values
        factor_read=[]
        change_read=[]
        for i in range (len(factor_or)):
            rk=[]
            if numpy.sum(factor_or[i])>0 and numpy.isfinite(numpy.sum(factor_or[i])):
                fac_rk=numpy.array(factor_or[i])/numpy.sum(factor_or[i])
            else:
                fac_rk = numpy.full(len(trade_steady[0]),0)#numpy.where(numpy.isfinite(factor_or[i]),factor_or[i],0)#trade_steady[0] / numpy.sum(trade_steady[0])
            if i==0 :#or (chose_time_mat[i]==0 ):

                fac_rk = numpy.full(len(trade_steady[0]),0)#trade_steady[0] / numpy.sum(trade_steady[0])

            print (numpy.sum(fac_rk))

            factor_read.append(fac_rk)
            if i>0:
                change_read.append(numpy.sum(abs(factor_read[-1]-factor_read[-2])))


        ret_sharpe,ret_rank,ret_pos,retindex,ret_dates=make_retline(factor_read,index_price,close_read,trade_steady,dates)

        pos_ret_df=pd.DataFrame()

        pos_ret_df['ret']=ret_pos
        pos_ret_df.index=ret_dates
        pos_ret_df.to_csv('D:/基本面三击策略每日更新/选股仓位多头收益/'+file,encoding='gbk')
    return


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


def deal_combined_golden_factor(stocks0, dates):


    fund_ret_file_path='D:/基本面三击策略每日更新/各因子消除波动后的多头收益/'
    fund_weight_file_path='D:/基本面三击策略每日更新/各因子消除波动内在权重/'
    fund_sign_file_path='D:/基本面三击策略每日更新/各因子消除波动内在权重标志/'
    file_list=os.listdir(fund_ret_file_path)
    file_list=['PBinliqlow_rank_mat_nobad_stock_big.csv','PEreturninliqlow_rank_mat_nobad_stock_small.csv',
        'PBincaphigh_rank_mat_nobad_stock_big.csv','PEreturnincaplow_rank_mat_nobad_stock_small.csv',
               'PBinnlsizehigh_rank_mat_nobad_stock_big.csv', 'PEreturninnlsizehigh_rank_mat_nobad_stock_small.csv'
               ]
    #file_list=['PBinliqlow_rank_mat_nobad_stock_small.csv','PBincaphigh_rank_mat_nobad_stock_small.csv','PBinnlsizehigh_rank_mat_nobad_stock_small.csv',
    #           'PBinliqlow_rank_mat_nobad_stock_big.csv', 'PBincaphigh_rank_mat_nobad_stock_big.csv',
    #           'PBinnlsizehigh_rank_mat_nobad_stock_big.csv']
    #
    #
    #,
    using_dates = []
    for year in range(2013, 2050):
        for mon in range(1, 13):
            using_dates.append(str(int(year)) + '-' + str(int(mon)).zfill(2) + '-01')
    trade_date_list = [dates[dates >= date_str][0] for date_str in using_dates if len(dates[dates >= date_str]) > 0]



    fund_ret_df_list=[]
    fund_weight_df_list=[]
    fund_sign_df_list=[]
    for file_name in file_list:

        factor_df2=pd.read_csv(fund_ret_file_path+file_name,index_col=0,encoding = 'gbk')
        factor_df2=factor_df2.reindex(index=dates,fill_value=0)
        #factor_df2 = numpy.exp(factor_df2) - 1

        fund_ret_df_list.append(factor_df2)
        factor_df3 = pd.read_csv(fund_weight_file_path + file_name, index_col=0, encoding='gbk')
        factor_df3 = factor_df3.reindex(index=dates, fill_value=0)
        fund_weight_df_list.append(factor_df3)

        factor_df4 = pd.read_csv(fund_sign_file_path + file_name, index_col=0, encoding='gbk')
        factor_df4 = factor_df4.reindex(index=dates, fill_value=0)
        fund_sign_df_list.append(factor_df4)




    def cov_f(weights,ret_mat):
        #weights=numpy.array(weights)*numpy.array(weights)*numpy.array([1,-1,-1,-1])
        #weights[0]=weights[0]*weights[0]
        weights=abs(numpy.array(weights))
        ret_mat = numpy.array(ret_mat)
        ret_all=numpy.dot(weights,ret_mat)/numpy.sum(abs(weights))
        std_ret=-numpy.mean(ret_all)/numpy.std(ret_all)
        return std_ret


    def cov_fun(ret_mat):
        #cov_ret=numpy.cov(ret_mat.T)
        fun_re=lambda weights: cov_f(weights,ret_mat)
        return fun_re



    def con_fun(weight):
        #weight[0]=weight[0]*weight[0]
        con_s=numpy.sum(abs(weight))-1
        return con_s

    ret_daily=[]
    weight_mat=[]
    date_list=[]
    res_x=[]
    for i in range (len(dates)):
        len_range=250
        if i>len_range:
            date_list.append(dates[i])
            mon_str = dates[i - 1][5:7]
            year = int(dates[i - 1][:4])
            close_date = dates[i - 251]
            if (mon_str == '01') | (mon_str == '02') | (mon_str == '03'):
                season_first = '-12-01'
                season_last = '-09-30'
            if (mon_str == '04') | (mon_str == '05') | (mon_str == '06'):
                season_first = '-12-01'
                season_last = '-09-30'
            if (mon_str == '07') | (mon_str == '08') | (mon_str == '09'):
                season_first = '-12-01'
                season_last = '-09-30'
            if (mon_str == '10') | (mon_str == '11') | (mon_str == '12'):
                season_first = '-10-01'
                season_last = '-12-31'
            dates_need = dates[((dates < dates[i - 1]) & (dates >= close_date)) |
                               ((dates >= str(year - 1) + season_first) & (dates <= str(year - 1) + season_last)) |
                               ((dates >= str(year - 2) + season_first) & (dates <= str(year - 2) + season_last)) |
                               ((dates >= str(year - 3) + season_first) & (dates <= str(year - 3) + season_last))]

            ret_mat=[]#,ret_f,ret_index_chose1,ret_300,ret_500,ret_1000
            #weights0 = [0.1,-0.5,-0.1,-0.1]
            sign_ii=[]#0,0,0
            ret_mat_n = []
            # #ret_index_frame300['ret'].values[i], ret_index_frame500['ret'].values[i],ret_index_frame1000['ret'].values[i],
            weights0 = []
            name_list=[]
            for ii in range (len(fund_ret_df_list)):
                ret_df=fund_ret_df_list[ii]
                ret_f2 = ret_df['ret'].values[i - len_range:i]
                #ret_f2 = ret_df[ret_df.index.isin(dates_need)]['ret'].values
                ret_w_df=fund_weight_df_list[ii]
                w_sign=ret_w_df['factor'].values[i-1]
                ret_sign_df = fund_sign_df_list[ii]
                w_sign2=ret_sign_df['sign'].values[i-1]
                if numpy.std(ret_f2)!=0 and w_sign!=0 :# and numpy.sum(ret_f2[:-int(len_range/2)])>0 and numpy.sum(ret_f2[-int(len_range/2):])>0:
                    ret_mat.append(ret_f2)
                    sign_ii.append(1)
                    weights0.append(1)
                    name_list.append(file_list[ii])
                ret_mat_n.append(ret_df['ret'].values[i])
            #print(ret_mat_n)

            if len(sign_ii)>0:
                bonds=[(-1.0,0) if sign_ii[ii]!=1 else (0,1) for ii in range (len(ret_mat))]
                mr_mat=numpy.array(numpy.average(ret_mat,axis=1))
                weights0[0]=0.1
                weights0=numpy.array(weights0)/numpy.sum(abs(numpy.array(weights0)))

                #print (cov_mat)

                # print (make_con(weights0))
                cons = ({'type': 'eq', 'fun': con_fun})
                if dates[i] in trade_date_list:
                    res_fun = []
                    res_x_list = []
                    for times in range(300):
                        weights0 = [random.random() for unit in weights0]
                        weights0 = numpy.array(weights0) / numpy.sum(abs(numpy.array(weights0)))
                        res = optimize.minimize(cov_fun(ret_mat), weights0, method='SLSQP', constraints=cons,
                                                options={'maxiter': 2000, 'disp': False},bounds=bonds)  # ,constraints=cons
                        # res=optimize.shgo(con_cov_shgo,bonds,constraints=cons)#
                        # res_x=res_x*res_x*numpy.array([1,-1,-1,-1])
                        # res_x[0]=res_x[0]*res_x[0]
                        res_fun.append(res.fun)
                        res_x_list.append(res.x)
                        print(res.success)
                    res_x = res_x_list[
                        numpy.argmin(res_fun)]  # numpy.array([num if abs(num)>10**(-16) else 0 for num in res.x])
                    res_x_namelist = name_list

                if len(res_x)>0:
                    res_x_change=pd.DataFrame([abs(res_x)/numpy.sum(abs(res_x))],columns=res_x_namelist).reindex(columns=file_list,fill_value=0).values[0]
                    weight_mat.append(res_x_change)
                    ret_mat_n_change = \
                    pd.DataFrame([ret_mat_n], columns=file_list).reindex(columns=file_list, fill_value=0).values[0]
                    ret_n = numpy.sum(numpy.array(res_x_change) * numpy.array(ret_mat_n_change))
                    print(dates[i], numpy.nan_to_num(ret_n), numpy.sum(abs(res_x_change)),res.success)
                    ret_daily.append(numpy.nan_to_num(ret_n))

                else:

                    ret_daily.append(numpy.nan_to_num(0))
                    weight_mat.append(numpy.full(len(file_list), 0))
            else:
                if dates[i] in trade_date_list:
                    res_x=[]

                ret_daily.append(numpy.nan_to_num(0))
                weight_mat.append(numpy.full(len(file_list),0))


    out_weight_df=pd.DataFrame(weight_mat,index=date_list,columns=file_list)
    print (out_weight_df)
    out_weight_df.to_csv('D:/基本面三击策略每日更新/重仓各子策略权重.csv',encoding='gbk')


    return


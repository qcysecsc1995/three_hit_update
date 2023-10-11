import pandas as pd
import numpy
from sklearn import tree
import uqer
from uqer import DataAPI
import pymysql
import warnings
import os
import datetime
import csv

goldstock_factor_path='D:/基本面三击策略每日更新/三击策略因子仓位/'
file_list=os.listdir(goldstock_factor_path)
can_file_name=str(datetime.datetime.today())[:10]+'.csv'
if 1>0:
    data_rep = pd.read_csv(goldstock_factor_path + '2023-10-09.csv',  encoding='gbk')
    t_day='20230927'
    out_df=pd.DataFrame()
    out_df['证券代码']=[str(num).zfill(6)+'.SH' if str(num).zfill(6)[0]=='6' else str(num).zfill(6)+'.SZ' for num in data_rep['ticker'].values]
    out_df['持仓权重']=data_rep['weight'].values
    out_df['调整日期'] = t_day
    out_df=out_df.reindex(columns=['调整日期','证券代码','持仓权重','成本价格','是否融资融券'])
    out_df.to_csv('D:/基本面三击策略每日更新/single_factor_import_wind.csv',encoding='gbk')
    #with open('D:/基本面三击策略每日更新/single_factor_import_wind.csv', 'w', newline='') as csvfile:
    #    spamwriter = csv.writer(csvfile)
    #    spamwriter.writerows(out_df.values)

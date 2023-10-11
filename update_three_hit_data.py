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
#can_file_name='2023-08-01.csv'
if can_file_name in file_list:
    data_rep = pd.read_csv(goldstock_factor_path + can_file_name,  encoding='gbk')
    out_df=pd.DataFrame()
    out_df['ticker']=[str(num).zfill(6)+'.SH' if str(num).zfill(6)[0]=='6' else str(num).zfill(6)+'.SZ' for num in data_rep['ticker'].values]
    out_df['weight']=data_rep['weight'].values
    with open('D:/wind金股因子每日更新/alpha项目文件夹/三击策略_'+can_file_name[:10].replace('-','')+'.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerows(out_df.values)
    #with open('//192.168.26.40/alpha项目文件夹/三击策略_'+can_file_name[:10].replace('-','')+'.csv', 'w', newline='') as csvfile:
    #    spamwriter = csv.writer(csvfile)
    #    spamwriter.writerows(out_df.values)
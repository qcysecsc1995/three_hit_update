import pandas as pd
import numpy
import os
from sklearn import tree
import uqer
from uqer import DataAPI
import pymysql
import warnings
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

def read_raw(file):
    store=pd.HDFStore(file,mode='r')

    data=store.select('data')


    store.close()
    return data
#########################################################################3
def arank(r,reverse=1):
    r=numpy.array(r)
    rr = numpy.argsort(reverse*r)
    for rank in range (0,len(rr)):
        r[rr[rank]]=rank+1
    return r

def standard_line(line):
    line=(line-numpy.nanmean(line))/numpy.nanstd(line)
    return line

def make_three_hit_factor_pool_total(stocks0, dates):

    PB_df=read_raw('D:/data_predict_project/data deal project/stock_price_pyd/PB.h5')\
        .reindex(index=dates,method='ffill').reindex(columns=stocks0).shift(1)

    PB_df = pd.DataFrame(numpy.where(PB_df.values > 0, PB_df.values, numpy.nan), index=PB_df.index, columns=PB_df.columns)
    fund_file_path='D:/因子择时/择时数据/通联股票因子/'
    file_dict=os.listdir(fund_file_path)
    DividendCover_df = pd.read_csv('D:/基本面三击策略每日更新/uqer基本面因子/DividendCover.csv', index_col=0, encoding='gbk') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    NetAssetGrowRate_df = pd.read_csv('D:/基本面三击策略每日更新/uqer基本面因子/NetAssetGrowRate.csv', index_col=0, encoding='gbk') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    OperatingCycle_df = pd.read_csv('D:/基本面三击策略每日更新/uqer基本面因子/OperatingCycle.csv', index_col=0, encoding='gbk') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    TotalAssetGrowRate_df = pd.read_csv('D:/基本面三击策略每日更新/uqer基本面因子/TotalAssetGrowRate.csv', index_col=0, encoding='gbk') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    FinancialExpenseRate_df = pd.read_csv('D:/基本面三击策略每日更新/uqer基本面因子/FinancialExpenseRate.csv', index_col=0, encoding='gbk') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    OperatingProfitGrowRate_df = pd.read_csv('D:/基本面三击策略每日更新/uqer基本面因子/OperatingProfitGrowRate.csv', index_col=0, encoding='gbk') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    NPParentCompanyGrowRate_df = pd.read_csv('D:/基本面三击策略每日更新/uqer基本面因子/NPParentCompanyGrowRate.csv', index_col=0, encoding='gbk') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    TotalAssetsTRate_df = pd.read_csv('D:/基本面三击策略每日更新/uqer基本面因子/TotalAssetsTRate.csv', index_col=0, encoding='gbk') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    ETOP_df = pd.read_csv('D:/基本面三击策略每日更新/uqer基本面因子/ETOP.csv', index_col=0, encoding='gbk') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)

    #################################################

    ##############################################

    trade_date_list = []
    for year in range(2013, 2050):
        for mon in range(1, 13):
            trade_date_list.append(str(int(year)) + '-' + str(int(mon)).zfill(2) + '-01')
    trade_date_list = [dates[dates >= date_str][0] for date_str in trade_date_list if len(dates[dates >= date_str]) > 0]

    factor_mat = []
    for trade_date_num_i in range(len(trade_date_list)):
        trade_date_num = trade_date_list[trade_date_num_i]
        i = list(dates).index(trade_date_num)
        print(dates[i])
        ###############################################
        # uqer
        f_part_line = -DividendCover_df.values[i]
        DividendCover_line = arank(f_part_line, reverse=-1) / len(
            f_part_line[numpy.isfinite(f_part_line)])


        f_part_line = ETOP_df.values[i]
        ETOP_line = arank(f_part_line, reverse=-1) / len(
            f_part_line[numpy.isfinite(f_part_line)])

        op_line = standard_line(-OperatingCycle_df.values[i]) + standard_line(-FinancialExpenseRate_df.values[i]) + \
                  standard_line(TotalAssetsTRate_df.values[i])
        op_line = arank(op_line, reverse=-1) / len(
            op_line[numpy.isfinite(op_line)])
        growth_line = standard_line(OperatingProfitGrowRate_df.values[i]) + standard_line(
            NPParentCompanyGrowRate_df.values[i])
        growth_line = arank(growth_line, reverse=-1) / len(
            growth_line[numpy.isfinite(growth_line)])
        asset_over_growth = standard_line(-NetAssetGrowRate_df.values[i]) + standard_line(
            -TotalAssetGrowRate_df.values[i])
        asset_over_growth = arank(asset_over_growth, reverse=-1) / len(
            asset_over_growth[numpy.isfinite(asset_over_growth)])



        judge_line_list2 = [DividendCover_line, asset_over_growth, op_line, growth_line, ETOP_line]
        # NetAssetGrowRate_line,TotalAssetGrowRate_line
        # ,EGRO_line,InvestCashGrowRate_line
        judge_line_list2 = [line if len(line[numpy.isfinite(line)]) > 0 else numpy.full(len(line), 0.1) for line in
                            judge_line_list2]
        judge_line_list2 = [numpy.where(line < 0.8, 1, numpy.nan) for line in judge_line_list2]

        judge_line = numpy.sum(judge_line_list2, axis=0)

        #######################################
        judge_line_absolute_list = [NetAssetGrowRate_df.values[i], TotalAssetsTRate_df.values[i], PB_df.values[i]]
        judge_line_absolute_list = [line if len(line[numpy.isfinite(line)]) > 0 else numpy.full(len(line), 0.1) for line
                                    in
                                    judge_line_absolute_list]
        judge_line_absolute_list = [numpy.where(line > 0, 1, numpy.nan) for line in judge_line_absolute_list]
        judge_absolute_line = numpy.sum(judge_line_absolute_list, axis=0)
        ##########################################
        judge_line = numpy.where(numpy.isfinite(judge_line) & numpy.isfinite(judge_absolute_line), 1, 0)
        ##########################################

        ############################################

        factor_mat.append(judge_line)

    factor_df = pd.DataFrame(factor_mat, index=trade_date_list, columns=stocks0)

    factor_df.to_csv('D:/基本面三击策略每日更新/nobad_stock_total.csv', encoding='gbk')

    return
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

#########################################################################3
pool_name='nobad_stock_big'
pool_name_small='nobad_stock_small'
def get_PBincaphigh(stocks0,dates):
    PB_df = read_raw('D:/data_predict_project/data deal project/stock_price_pyd/PB.h5')\
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    marketvalue_df = read_raw('D:\data_predict_project\data deal project\stock_price_adj_pyd/marketValue.h5') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    PB_df = pd.DataFrame(numpy.where(PB_df.values > 0, PB_df.values, numpy.nan), index=PB_df.index,
                         columns=PB_df.columns)
    store = pd.HDFStore('D:\data_predict_project\data deal project\industry_pyd/stock_industry_zhongxin_matrix.h5',
                        mode='r')

    ind_mat_df = store.select('Lv1')
    store.close()
    ind_mat_df = ind_mat_df.reindex(columns=stocks0, fill_value='').reindex(index=dates, method='ffill').shift(1)
    ind_mat_df = ind_mat_df.fillna(value='')
    file_path = 'D:/data_predict_project/data deal project/blacklist_pyd/warnlist.h5'
    store = pd.HDFStore(file_path, mode='r')
    warnlist = store['data']
    store.close()
    warnlist_df = warnlist.reindex(columns=stocks0).reindex(index=dates, method='ffill').shift(1)
    ind_line_last = ind_mat_df.values[-1]
    ind_name_list = list(set(list(ind_line_last)))
    ind_name_list = [ind_name for ind_name in ind_name_list if ind_name != '']
    ind_stock_chose = {}
    for ind_name in ind_name_list:
        ind_stock_chose[ind_name] = []

    ind_stock_chose_data = {}
    for ind_name in ind_name_list:
        ind_stock_chose_data[ind_name] = []

    white_stock = pd.read_csv('D:/wind金股因子每日更新/股票池20230605.csv', encoding='gbk')
    white_stock = list(white_stock['证券代码'].values)
    white_stock = [num[:-3] for num in white_stock]

    pool_file = [pool_name,pool_name_small]
    choose_num = [50,50]
    factor_all_list = []
    for pool_num in range(len(pool_file)):
        golden_stock_mat = pd.read_csv('D:/基本面三击策略每日更新/' + pool_file[pool_num] + '.csv', index_col=0,
                                       encoding='gbk').reindex(columns=stocks0)
        trade_date_list = [dates[dates <= pool_date][-1] for pool_date in golden_stock_mat.index]
        factor_mat = []
        for trade_date_num_i in range(len(trade_date_list)):
            trade_date_num = trade_date_list[trade_date_num_i]
            i = list(dates).index(trade_date_num)
            print(dates[i])
            PB_line = PB_df.values[i]
            ind_line = ind_mat_df.values[i]
            warn_line = numpy.nan_to_num(warnlist_df.values[i])
            mktvalues_line = marketvalue_df.values[i]

            mktvalues_line_part_len = len(mktvalues_line[(numpy.isfinite(mktvalues_line))])
            mktvalues_rank = (arank(mktvalues_line) / mktvalues_line_part_len)
            PB_line_part_len = len(PB_line[(numpy.isfinite(PB_line))])
            PB_rank = (arank(PB_line, reverse=-1) / PB_line_part_len)
            cb_line_PB = (PB_line - numpy.nanmean(PB_line) / numpy.nanstd(PB_line))
            cb_line_mktvalue = (mktvalues_line - numpy.nanmean(mktvalues_line) / numpy.nanstd(mktvalues_line))
            cbline = -cb_line_mktvalue + cb_line_PB
            chose_stock = []
            chose_data = []
            for ind_name in ind_name_list:
                chose_stock_part = stocks0[
                    (ind_line == ind_name) & (numpy.isfinite(PB_line)) & (numpy.isfinite(mktvalues_line))]
                pbline_part = PB_line[
                    (ind_line == ind_name) & (numpy.isfinite(PB_line)) & (numpy.isfinite(mktvalues_line))]
                mktvaluesline_part = mktvalues_line[
                    (ind_line == ind_name) & (numpy.isfinite(mktvalues_line)) & (numpy.isfinite(PB_line))]

                chose_data_part = cbline[(ind_line == ind_name) & (numpy.isfinite(cbline))]
                if len(chose_data_part) > 0:
                    cb_line_PB_part = (pbline_part - numpy.nanmean(pbline_part) / numpy.nanstd(pbline_part))
                    cb_line_mktvalue_part = (
                        mktvaluesline_part - numpy.nanmean(mktvaluesline_part) / numpy.nanstd(mktvaluesline_part))
                    cbline_part = -cb_line_mktvalue_part + cb_line_PB_part
                    # chose_data_part = list(arank(cbline_part) / len(cbline_part))
                    chose_data_part = list(arank(chose_data_part) / len(chose_data_part))
                    chose_stock = chose_stock + list(chose_stock_part)
                    chose_data = chose_data + list(chose_data_part)

            pb_rank_line = [chose_data[chose_stock.index(st_num)] if (st_num in chose_stock) & (st_num in white_stock)
                            else numpy.nan for st_num in stocks0]

            golden_stock_line = golden_stock_mat.loc[golden_stock_mat.index[trade_date_num_i]].values
            pb_rank_line = numpy.where((golden_stock_line > 0) & (warn_line == 0), pb_rank_line, numpy.nan)
            pb_rank_line = arank(pb_rank_line)
            pb_rank_line = numpy.where(pb_rank_line <= choose_num[pool_num], 1, 0)
            factor_mat.append(pb_rank_line)

        # factor_df=rank(factor_df,reverse=-1)
        factor_df = pd.DataFrame(factor_mat, index=golden_stock_mat.index, columns=stocks0)
        factor_df.to_csv('D:/基本面三击策略每日更新/三者兼优选股仓位/PBincaphigh_rank_mat_' + pool_file[pool_num] + '.csv', encoding='gbk')
        factor_all_list.append(factor_df)
    return

def get_PBincaplow(stocks0,dates):
    PB_df = read_raw('D:/data_predict_project/data deal project/stock_price_pyd/PB.h5')\
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    marketvalue_df = read_raw('D:\data_predict_project\data deal project\stock_price_adj_pyd/marketValue.h5') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    PB_df = pd.DataFrame(numpy.where(PB_df.values > 0, PB_df.values, numpy.nan), index=PB_df.index,
                         columns=PB_df.columns)
    store = pd.HDFStore('D:\data_predict_project\data deal project\industry_pyd/stock_industry_zhongxin_matrix.h5',
                        mode='r')

    ind_mat_df = store.select('Lv1')
    store.close()
    ind_mat_df = ind_mat_df.reindex(columns=stocks0, fill_value='').reindex(index=dates, method='ffill').shift(1)
    ind_mat_df = ind_mat_df.fillna(value='')
    file_path = 'D:/data_predict_project/data deal project/blacklist_pyd/warnlist.h5'
    store = pd.HDFStore(file_path, mode='r')
    warnlist = store['data']
    store.close()
    warnlist_df = warnlist.reindex(columns=stocks0).reindex(index=dates, method='ffill').shift(1)
    ind_line_last = ind_mat_df.values[-1]
    ind_name_list = list(set(list(ind_line_last)))
    ind_name_list = [ind_name for ind_name in ind_name_list if ind_name != '']
    ind_stock_chose = {}
    for ind_name in ind_name_list:
        ind_stock_chose[ind_name] = []

    ind_stock_chose_data = {}
    for ind_name in ind_name_list:
        ind_stock_chose_data[ind_name] = []

    white_stock = pd.read_csv('D:/wind金股因子每日更新/股票池20230605.csv', encoding='gbk')
    white_stock = list(white_stock['证券代码'].values)
    white_stock = [num[:-3] for num in white_stock]

    pool_file = [pool_name,pool_name_small]
    choose_num = [50, 50]
    factor_all_list = []
    for pool_num in range(len(pool_file)):
        golden_stock_mat = pd.read_csv('D:/基本面三击策略每日更新/' + pool_file[pool_num] + '.csv', index_col=0,
                                       encoding='gbk').reindex(columns=stocks0)
        trade_date_list = [dates[dates <= pool_date][-1] for pool_date in golden_stock_mat.index]
        factor_mat = []
        for trade_date_num_i in range(len(trade_date_list)):
            trade_date_num = trade_date_list[trade_date_num_i]
            i = list(dates).index(trade_date_num)
            print(dates[i])
            PB_line = PB_df.values[i]
            ind_line = ind_mat_df.values[i]
            warn_line = numpy.nan_to_num(warnlist_df.values[i])
            mktvalues_line = marketvalue_df.values[i]

            mktvalues_line_part_len = len(mktvalues_line[(numpy.isfinite(mktvalues_line))])
            mktvalues_rank = (arank(mktvalues_line) / mktvalues_line_part_len)
            PB_line_part_len = len(PB_line[(numpy.isfinite(PB_line))])
            PB_rank = (arank(PB_line, reverse=-1) / PB_line_part_len)
            cb_line_PB = (PB_line - numpy.nanmean(PB_line) / numpy.nanstd(PB_line))
            cb_line_mktvalue = (mktvalues_line - numpy.nanmean(mktvalues_line) / numpy.nanstd(mktvalues_line))
            cbline = cb_line_mktvalue + cb_line_PB
            chose_stock = []
            chose_data = []
            for ind_name in ind_name_list:
                chose_stock_part = stocks0[
                    (ind_line == ind_name) & (numpy.isfinite(PB_line)) & (numpy.isfinite(mktvalues_line))]
                pbline_part = PB_line[
                    (ind_line == ind_name) & (numpy.isfinite(PB_line)) & (numpy.isfinite(mktvalues_line))]
                mktvaluesline_part = mktvalues_line[
                    (ind_line == ind_name) & (numpy.isfinite(mktvalues_line)) & (numpy.isfinite(PB_line))]

                chose_data_part = cbline[(ind_line == ind_name) & (numpy.isfinite(cbline))]
                if len(chose_data_part) > 0:
                    cb_line_PB_part = (pbline_part - numpy.nanmean(pbline_part) / numpy.nanstd(pbline_part))
                    cb_line_mktvalue_part = (
                        mktvaluesline_part - numpy.nanmean(mktvaluesline_part) / numpy.nanstd(mktvaluesline_part))
                    cbline_part = -cb_line_mktvalue_part + cb_line_PB_part
                    # chose_data_part = list(arank(cbline_part) / len(cbline_part))
                    chose_data_part = list(arank(chose_data_part) / len(chose_data_part))
                    chose_stock = chose_stock + list(chose_stock_part)
                    chose_data = chose_data + list(chose_data_part)

            pb_rank_line = [chose_data[chose_stock.index(st_num)] if (st_num in chose_stock) & (st_num in white_stock)
                            else numpy.nan for st_num in stocks0]

            golden_stock_line = golden_stock_mat.loc[golden_stock_mat.index[trade_date_num_i]].values
            pb_rank_line = numpy.where((golden_stock_line > 0) & (warn_line == 0), pb_rank_line, numpy.nan)
            pb_rank_line = arank(pb_rank_line)
            pb_rank_line = numpy.where(pb_rank_line <= choose_num[pool_num], 1, 0)
            factor_mat.append(pb_rank_line)

        # factor_df=rank(factor_df,reverse=-1)
        factor_df = pd.DataFrame(factor_mat, index=golden_stock_mat.index, columns=stocks0)
        factor_df.to_csv('D:/基本面三击策略每日更新/三者兼优选股仓位/PBincaplow_rank_mat_' + pool_file[pool_num] + '.csv', encoding='gbk')
        factor_all_list.append(factor_df)
    return

def get_PBinnlsizehigh(stocks0,dates):
    PB_df = read_raw('D:/data_predict_project/data deal project/stock_price_pyd/PB.h5')\
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    marketvalue_df = read_raw('D:\data_predict_project\data deal project\stock_price_adj_pyd/marketValue.h5') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    SIZENL_df = pd.read_csv('D:/因子择时/择时因子计算/多因子归因择时/风险因子矩阵/SIZENL.csv', index_col=0, encoding='gbk') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    PB_df = pd.DataFrame(numpy.where(PB_df.values > 0, PB_df.values, numpy.nan), index=PB_df.index,
                         columns=PB_df.columns)
    store = pd.HDFStore('D:\data_predict_project\data deal project\industry_pyd/stock_industry_zhongxin_matrix.h5',
                        mode='r')

    ind_mat_df = store.select('Lv1')
    store.close()
    ind_mat_df = ind_mat_df.reindex(columns=stocks0, fill_value='').reindex(index=dates, method='ffill').shift(1)
    ind_mat_df = ind_mat_df.fillna(value='')
    file_path = 'D:/data_predict_project/data deal project/blacklist_pyd/warnlist.h5'
    store = pd.HDFStore(file_path, mode='r')
    warnlist = store['data']
    store.close()
    warnlist_df = warnlist.reindex(columns=stocks0).reindex(index=dates, method='ffill').shift(1)
    ind_line_last = ind_mat_df.values[-1]
    ind_name_list = list(set(list(ind_line_last)))
    ind_name_list = [ind_name for ind_name in ind_name_list if ind_name != '']
    ind_stock_chose = {}
    for ind_name in ind_name_list:
        ind_stock_chose[ind_name] = []

    ind_stock_chose_data = {}
    for ind_name in ind_name_list:
        ind_stock_chose_data[ind_name] = []

    white_stock = pd.read_csv('D:/wind金股因子每日更新/股票池20230605.csv', encoding='gbk')
    white_stock = list(white_stock['证券代码'].values)
    white_stock = [num[:-3] for num in white_stock]

    pool_file = [pool_name,pool_name_small]
    choose_num = [50, 50]
    factor_all_list = []
    for pool_num in range(len(pool_file)):
        golden_stock_mat = pd.read_csv('D:/基本面三击策略每日更新/' + pool_file[pool_num] + '.csv', index_col=0,
                                       encoding='gbk').reindex(columns=stocks0)
        trade_date_list = [dates[dates <= pool_date][-1] for pool_date in golden_stock_mat.index]
        factor_mat = []
        for trade_date_num_i in range(len(trade_date_list)):
            trade_date_num = trade_date_list[trade_date_num_i]
            i = list(dates).index(trade_date_num)
            print(dates[i])
            PB_line = PB_df.values[i]
            ind_line = ind_mat_df.values[i]
            warn_line = numpy.nan_to_num(warnlist_df.values[i])
            mktvalues_line = SIZENL_df.values[i]

            mktvalues_line_part_len = len(mktvalues_line[(numpy.isfinite(mktvalues_line))])
            mktvalues_rank = (arank(mktvalues_line) / mktvalues_line_part_len)
            PB_line_part_len = len(PB_line[(numpy.isfinite(PB_line))])
            PB_rank = (arank(PB_line, reverse=-1) / PB_line_part_len)
            cb_line_PB = (PB_line - numpy.nanmean(PB_line) / numpy.nanstd(PB_line))
            cb_line_mktvalue = (mktvalues_line - numpy.nanmean(mktvalues_line) / numpy.nanstd(mktvalues_line))
            cbline = -cb_line_mktvalue + cb_line_PB
            chose_stock = []
            chose_data = []
            for ind_name in ind_name_list:
                chose_stock_part = stocks0[
                    (ind_line == ind_name) & (numpy.isfinite(PB_line)) & (numpy.isfinite(mktvalues_line))]
                pbline_part = PB_line[
                    (ind_line == ind_name) & (numpy.isfinite(PB_line)) & (numpy.isfinite(mktvalues_line))]
                mktvaluesline_part = mktvalues_line[
                    (ind_line == ind_name) & (numpy.isfinite(mktvalues_line)) & (numpy.isfinite(PB_line))]

                chose_data_part = cbline[(ind_line == ind_name) & (numpy.isfinite(cbline))]
                if len(chose_data_part) > 0:
                    cb_line_PB_part = (pbline_part - numpy.nanmean(pbline_part) / numpy.nanstd(pbline_part))
                    cb_line_mktvalue_part = (
                        mktvaluesline_part - numpy.nanmean(mktvaluesline_part) / numpy.nanstd(mktvaluesline_part))
                    cbline_part = -cb_line_mktvalue_part + cb_line_PB_part
                    # chose_data_part = list(arank(cbline_part) / len(cbline_part))
                    chose_data_part = list(arank(chose_data_part) / len(chose_data_part))
                    chose_stock = chose_stock + list(chose_stock_part)
                    chose_data = chose_data + list(chose_data_part)

            pb_rank_line = [chose_data[chose_stock.index(st_num)] if (st_num in chose_stock) & (st_num in white_stock)
                            else numpy.nan for st_num in stocks0]

            golden_stock_line = golden_stock_mat.loc[golden_stock_mat.index[trade_date_num_i]].values
            pb_rank_line = numpy.where((golden_stock_line > 0) & (warn_line == 0), pb_rank_line, numpy.nan)
            pb_rank_line = arank(pb_rank_line)
            pb_rank_line = numpy.where(pb_rank_line <= choose_num[pool_num], 1, 0)
            factor_mat.append(pb_rank_line)

        # factor_df=rank(factor_df,reverse=-1)
        factor_df = pd.DataFrame(factor_mat, index=golden_stock_mat.index, columns=stocks0)
        factor_df.to_csv('D:/基本面三击策略每日更新/三者兼优选股仓位/PBinnlsizehigh_rank_mat_' + pool_file[pool_num] + '.csv', encoding='gbk')
        factor_all_list.append(factor_df)
    return

def get_PBinliqlow(stocks0,dates):
    PB_df = read_raw('D:/data_predict_project/data deal project/stock_price_pyd/PB.h5')\
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    marketvalue_df = read_raw('D:\data_predict_project\data deal project\stock_price_adj_pyd/marketValue.h5') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    LIQUIDTY_df = pd.read_csv('D:/因子择时/择时因子计算/多因子归因择时/风险因子矩阵/LIQUIDTY.csv', index_col=0, encoding='gbk') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    PB_df = pd.DataFrame(numpy.where(PB_df.values > 0, PB_df.values, numpy.nan), index=PB_df.index,
                         columns=PB_df.columns)
    store = pd.HDFStore('D:\data_predict_project\data deal project\industry_pyd/stock_industry_zhongxin_matrix.h5',
                        mode='r')

    ind_mat_df = store.select('Lv1')
    store.close()
    ind_mat_df = ind_mat_df.reindex(columns=stocks0, fill_value='').reindex(index=dates, method='ffill').shift(1)
    ind_mat_df = ind_mat_df.fillna(value='')
    file_path = 'D:/data_predict_project/data deal project/blacklist_pyd/warnlist.h5'
    store = pd.HDFStore(file_path, mode='r')
    warnlist = store['data']
    store.close()
    warnlist_df = warnlist.reindex(columns=stocks0).reindex(index=dates, method='ffill').shift(1)
    ind_line_last = ind_mat_df.values[-1]
    ind_name_list = list(set(list(ind_line_last)))
    ind_name_list = [ind_name for ind_name in ind_name_list if ind_name != '']
    ind_stock_chose = {}
    for ind_name in ind_name_list:
        ind_stock_chose[ind_name] = []

    ind_stock_chose_data = {}
    for ind_name in ind_name_list:
        ind_stock_chose_data[ind_name] = []

    white_stock = pd.read_csv('D:/wind金股因子每日更新/股票池20230605.csv', encoding='gbk')
    white_stock = list(white_stock['证券代码'].values)
    white_stock = [num[:-3] for num in white_stock]

    pool_file = [pool_name,pool_name_small]
    choose_num = [50, 50]
    factor_all_list = []
    for pool_num in range(len(pool_file)):
        golden_stock_mat = pd.read_csv('D:/基本面三击策略每日更新/' + pool_file[pool_num] + '.csv', index_col=0,
                                       encoding='gbk').reindex(columns=stocks0)
        trade_date_list = [dates[dates <= pool_date][-1] for pool_date in golden_stock_mat.index]
        factor_mat = []
        for trade_date_num_i in range(len(trade_date_list)):
            trade_date_num = trade_date_list[trade_date_num_i]
            i = list(dates).index(trade_date_num)
            print(dates[i])
            PB_line = PB_df.values[i]
            ind_line = ind_mat_df.values[i]
            warn_line = numpy.nan_to_num(warnlist_df.values[i])
            mktvalues_line = LIQUIDTY_df.values[i]

            mktvalues_line_part_len = len(mktvalues_line[(numpy.isfinite(mktvalues_line))])
            mktvalues_rank = (arank(mktvalues_line) / mktvalues_line_part_len)
            PB_line_part_len = len(PB_line[(numpy.isfinite(PB_line))])
            PB_rank = (arank(PB_line, reverse=-1) / PB_line_part_len)
            cb_line_PB = (PB_line - numpy.nanmean(PB_line) / numpy.nanstd(PB_line))
            cb_line_mktvalue = (mktvalues_line - numpy.nanmean(mktvalues_line) / numpy.nanstd(mktvalues_line))
            cbline = cb_line_mktvalue + cb_line_PB
            chose_stock = []
            chose_data = []
            for ind_name in ind_name_list:
                chose_stock_part = stocks0[
                    (ind_line == ind_name) & (numpy.isfinite(PB_line)) & (numpy.isfinite(mktvalues_line))]
                pbline_part = PB_line[
                    (ind_line == ind_name) & (numpy.isfinite(PB_line)) & (numpy.isfinite(mktvalues_line))]
                mktvaluesline_part = mktvalues_line[
                    (ind_line == ind_name) & (numpy.isfinite(mktvalues_line)) & (numpy.isfinite(PB_line))]

                chose_data_part = cbline[(ind_line == ind_name) & (numpy.isfinite(cbline))]
                if len(chose_data_part) > 0:
                    cb_line_PB_part = (pbline_part - numpy.nanmean(pbline_part) / numpy.nanstd(pbline_part))
                    cb_line_mktvalue_part = (
                        mktvaluesline_part - numpy.nanmean(mktvaluesline_part) / numpy.nanstd(mktvaluesline_part))
                    cbline_part = -cb_line_mktvalue_part + cb_line_PB_part
                    # chose_data_part = list(arank(cbline_part) / len(cbline_part))
                    chose_data_part = list(arank(chose_data_part) / len(chose_data_part))
                    chose_stock = chose_stock + list(chose_stock_part)
                    chose_data = chose_data + list(chose_data_part)

            pb_rank_line = [chose_data[chose_stock.index(st_num)] if (st_num in chose_stock) & (st_num in white_stock)
                            else numpy.nan for st_num in stocks0]

            golden_stock_line = golden_stock_mat.loc[golden_stock_mat.index[trade_date_num_i]].values
            pb_rank_line = numpy.where((golden_stock_line > 0) & (warn_line == 0), pb_rank_line, numpy.nan)
            pb_rank_line = arank(pb_rank_line)
            pb_rank_line = numpy.where(pb_rank_line <= choose_num[pool_num], 1, 0)
            factor_mat.append(pb_rank_line)

        # factor_df=rank(factor_df,reverse=-1)
        factor_df = pd.DataFrame(factor_mat, index=golden_stock_mat.index, columns=stocks0)
        factor_df.to_csv('D:/基本面三击策略每日更新/三者兼优选股仓位/PBinliqlow_rank_mat_' + pool_file[pool_num] + '.csv', encoding='gbk')
        factor_all_list.append(factor_df)
    return

def get_PEreturnincaphigh(stocks0,dates):
    PB_df = read_raw('D:/data_predict_project/data deal project/stock_price_pyd/PB.h5')\
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    marketvalue_df = read_raw('D:\data_predict_project\data deal project\stock_price_adj_pyd/marketValue.h5') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    PB_df = pd.DataFrame(numpy.where(PB_df.values > 0, PB_df.values, numpy.nan), index=PB_df.index,
                         columns=PB_df.columns)
    ############
    PE_df = read_raw('D:/data_predict_project/data deal project/stock_price_pyd/PE.h5') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    PE_df = pd.DataFrame(numpy.where(PE_df.values > 0, PE_df.values, numpy.nan), index=PE_df.index,
                         columns=PE_df.columns)
    PE_df_div = (1 / PE_df)
    mktvalue_df_div = ((1 / marketvalue_df) * 10 ** (10))
    PEreturn_factor_df = (PE_df_div - PE_df_div.rolling(60).mean()) / (PE_df_div.rolling(60).std()) \
                         - (mktvalue_df_div - mktvalue_df_div.rolling(60).mean()) / (mktvalue_df_div.rolling(60).std())

    PEreturn_factor_df_values = numpy.where(numpy.isfinite(PEreturn_factor_df.values), PEreturn_factor_df.values,
                                            numpy.nan)
    PEreturn_factor_df = pd.DataFrame(PEreturn_factor_df_values, index=PEreturn_factor_df.index,
                                      columns=PEreturn_factor_df.columns)
    store = pd.HDFStore('D:\data_predict_project\data deal project\industry_pyd/stock_industry_zhongxin_matrix.h5',
                        mode='r')

    ind_mat_df = store.select('Lv1')
    store.close()
    ind_mat_df = ind_mat_df.reindex(columns=stocks0, fill_value='').reindex(index=dates, method='ffill').shift(1)
    ind_mat_df = ind_mat_df.fillna(value='')
    file_path = 'D:/data_predict_project/data deal project/blacklist_pyd/warnlist.h5'
    store = pd.HDFStore(file_path, mode='r')
    warnlist = store['data']
    store.close()
    warnlist_df = warnlist.reindex(columns=stocks0).reindex(index=dates, method='ffill').shift(1)
    ind_line_last = ind_mat_df.values[-1]
    ind_name_list = list(set(list(ind_line_last)))
    ind_name_list = [ind_name for ind_name in ind_name_list if ind_name != '']
    ind_stock_chose = {}
    for ind_name in ind_name_list:
        ind_stock_chose[ind_name] = []

    ind_stock_chose_data = {}
    for ind_name in ind_name_list:
        ind_stock_chose_data[ind_name] = []

    white_stock = pd.read_csv('D:/wind金股因子每日更新/股票池20230605.csv', encoding='gbk')
    white_stock = list(white_stock['证券代码'].values)
    white_stock = [num[:-3] for num in white_stock]

    pool_file = [pool_name,pool_name_small]
    choose_num = [50, 50]
    factor_all_list = []
    for pool_num in range(len(pool_file)):
        golden_stock_mat = pd.read_csv('D:/基本面三击策略每日更新/' + pool_file[pool_num] + '.csv', index_col=0,
                                       encoding='gbk').reindex(columns=stocks0)
        trade_date_list = [dates[dates <= pool_date][-1] for pool_date in golden_stock_mat.index]
        factor_mat = []
        for trade_date_num_i in range(len(trade_date_list)):
            trade_date_num = trade_date_list[trade_date_num_i]
            i = list(dates).index(trade_date_num)
            print(dates[i])
            PB_line = -PEreturn_factor_df.values[i]
            ind_line = ind_mat_df.values[i]
            warn_line = numpy.nan_to_num(warnlist_df.values[i])
            mktvalues_line = marketvalue_df.values[i]

            mktvalues_line_part_len = len(mktvalues_line[(numpy.isfinite(mktvalues_line))])
            mktvalues_rank = (arank(mktvalues_line) / mktvalues_line_part_len)
            PB_line_part_len = len(PB_line[(numpy.isfinite(PB_line))])
            PB_rank = (arank(PB_line, reverse=-1) / PB_line_part_len)
            cb_line_PB = (PB_line - numpy.nanmean(PB_line) / numpy.nanstd(PB_line))
            cb_line_mktvalue = (mktvalues_line - numpy.nanmean(mktvalues_line) / numpy.nanstd(mktvalues_line))
            cbline = -cb_line_mktvalue + cb_line_PB
            chose_stock = []
            chose_data = []
            for ind_name in ind_name_list:
                chose_stock_part = stocks0[
                    (ind_line == ind_name) & (numpy.isfinite(PB_line)) & (numpy.isfinite(mktvalues_line))]
                pbline_part = PB_line[
                    (ind_line == ind_name) & (numpy.isfinite(PB_line)) & (numpy.isfinite(mktvalues_line))]
                mktvaluesline_part = mktvalues_line[
                    (ind_line == ind_name) & (numpy.isfinite(mktvalues_line)) & (numpy.isfinite(PB_line))]

                chose_data_part = cbline[(ind_line == ind_name) & (numpy.isfinite(cbline))]
                if len(chose_data_part) > 0:
                    cb_line_PB_part = (pbline_part - numpy.nanmean(pbline_part) / numpy.nanstd(pbline_part))
                    cb_line_mktvalue_part = (
                        mktvaluesline_part - numpy.nanmean(mktvaluesline_part) / numpy.nanstd(mktvaluesline_part))
                    cbline_part = -cb_line_mktvalue_part + cb_line_PB_part
                    # chose_data_part = list(arank(cbline_part) / len(cbline_part))
                    chose_data_part = list(arank(chose_data_part) / len(chose_data_part))
                    chose_stock = chose_stock + list(chose_stock_part)
                    chose_data = chose_data + list(chose_data_part)

            pb_rank_line = [chose_data[chose_stock.index(st_num)] if (st_num in chose_stock) & (st_num in white_stock)
                            else numpy.nan for st_num in stocks0]

            golden_stock_line = golden_stock_mat.loc[golden_stock_mat.index[trade_date_num_i]].values
            pb_rank_line = numpy.where((golden_stock_line > 0) & (warn_line == 0), pb_rank_line, numpy.nan)
            pb_rank_line = arank(pb_rank_line)
            pb_rank_line = numpy.where(pb_rank_line <= choose_num[pool_num], 1, 0)
            factor_mat.append(pb_rank_line)

        # factor_df=rank(factor_df,reverse=-1)
        factor_df = pd.DataFrame(factor_mat, index=golden_stock_mat.index, columns=stocks0)
        factor_df.to_csv('D:/基本面三击策略每日更新/三者兼优选股仓位/PEreturnincaphigh_rank_mat_' + pool_file[pool_num] + '.csv', encoding='gbk')
        factor_all_list.append(factor_df)
    return

def get_PEreturnincaplow(stocks0,dates):
    PB_df = read_raw('D:/data_predict_project/data deal project/stock_price_pyd/PB.h5')\
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    marketvalue_df = read_raw('D:\data_predict_project\data deal project\stock_price_adj_pyd/marketValue.h5') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    PB_df = pd.DataFrame(numpy.where(PB_df.values > 0, PB_df.values, numpy.nan), index=PB_df.index,
                         columns=PB_df.columns)
    PE_df = read_raw('D:/data_predict_project/data deal project/stock_price_pyd/PE.h5') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    PE_df = pd.DataFrame(numpy.where(PE_df.values > 0, PE_df.values, numpy.nan), index=PE_df.index,
                         columns=PE_df.columns)
    PE_df_div = (1 / PE_df)
    mktvalue_df_div = ((1 / marketvalue_df) * 10 ** (10))
    PEreturn_factor_df = (PE_df_div - PE_df_div.rolling(60).mean()) / (PE_df_div.rolling(60).std()) \
                         - (mktvalue_df_div - mktvalue_df_div.rolling(60).mean()) / (mktvalue_df_div.rolling(60).std())

    PEreturn_factor_df_values = numpy.where(numpy.isfinite(PEreturn_factor_df.values), PEreturn_factor_df.values,
                                            numpy.nan)
    PEreturn_factor_df = pd.DataFrame(PEreturn_factor_df_values, index=PEreturn_factor_df.index,
                                      columns=PEreturn_factor_df.columns)
    store = pd.HDFStore('D:\data_predict_project\data deal project\industry_pyd/stock_industry_zhongxin_matrix.h5',
                        mode='r')

    ind_mat_df = store.select('Lv1')
    store.close()
    ind_mat_df = ind_mat_df.reindex(columns=stocks0, fill_value='').reindex(index=dates, method='ffill').shift(1)
    ind_mat_df = ind_mat_df.fillna(value='')
    file_path = 'D:/data_predict_project/data deal project/blacklist_pyd/warnlist.h5'
    store = pd.HDFStore(file_path, mode='r')
    warnlist = store['data']
    store.close()
    warnlist_df = warnlist.reindex(columns=stocks0).reindex(index=dates, method='ffill').shift(1)
    ind_line_last = ind_mat_df.values[-1]
    ind_name_list = list(set(list(ind_line_last)))
    ind_name_list = [ind_name for ind_name in ind_name_list if ind_name != '']
    ind_stock_chose = {}
    for ind_name in ind_name_list:
        ind_stock_chose[ind_name] = []

    ind_stock_chose_data = {}
    for ind_name in ind_name_list:
        ind_stock_chose_data[ind_name] = []

    white_stock = pd.read_csv('D:/wind金股因子每日更新/股票池20230605.csv', encoding='gbk')
    white_stock = list(white_stock['证券代码'].values)
    white_stock = [num[:-3] for num in white_stock]

    pool_file = [pool_name,pool_name_small]
    choose_num = [50, 50]
    factor_all_list = []
    for pool_num in range(len(pool_file)):
        golden_stock_mat = pd.read_csv('D:/基本面三击策略每日更新/' + pool_file[pool_num] + '.csv', index_col=0,
                                       encoding='gbk').reindex(columns=stocks0)
        trade_date_list = [dates[dates <= pool_date][-1] for pool_date in golden_stock_mat.index]
        factor_mat = []
        for trade_date_num_i in range(len(trade_date_list)):
            trade_date_num = trade_date_list[trade_date_num_i]
            i = list(dates).index(trade_date_num)
            print(dates[i])
            PB_line = -PEreturn_factor_df.values[i]
            ind_line = ind_mat_df.values[i]
            warn_line = numpy.nan_to_num(warnlist_df.values[i])
            mktvalues_line = marketvalue_df.values[i]

            mktvalues_line_part_len = len(mktvalues_line[(numpy.isfinite(mktvalues_line))])
            mktvalues_rank = (arank(mktvalues_line) / mktvalues_line_part_len)
            PB_line_part_len = len(PB_line[(numpy.isfinite(PB_line))])
            PB_rank = (arank(PB_line, reverse=-1) / PB_line_part_len)
            cb_line_PB = (PB_line - numpy.nanmean(PB_line) / numpy.nanstd(PB_line))
            cb_line_mktvalue = (mktvalues_line - numpy.nanmean(mktvalues_line) / numpy.nanstd(mktvalues_line))
            cbline = cb_line_mktvalue + cb_line_PB
            chose_stock = []
            chose_data = []
            for ind_name in ind_name_list:
                chose_stock_part = stocks0[
                    (ind_line == ind_name) & (numpy.isfinite(PB_line)) & (numpy.isfinite(mktvalues_line))]
                pbline_part = PB_line[
                    (ind_line == ind_name) & (numpy.isfinite(PB_line)) & (numpy.isfinite(mktvalues_line))]
                mktvaluesline_part = mktvalues_line[
                    (ind_line == ind_name) & (numpy.isfinite(mktvalues_line)) & (numpy.isfinite(PB_line))]

                chose_data_part = cbline[(ind_line == ind_name) & (numpy.isfinite(cbline))]
                if len(chose_data_part) > 0:
                    cb_line_PB_part = (pbline_part - numpy.nanmean(pbline_part) / numpy.nanstd(pbline_part))
                    cb_line_mktvalue_part = (
                        mktvaluesline_part - numpy.nanmean(mktvaluesline_part) / numpy.nanstd(mktvaluesline_part))
                    cbline_part = -cb_line_mktvalue_part + cb_line_PB_part
                    # chose_data_part = list(arank(cbline_part) / len(cbline_part))
                    chose_data_part = list(arank(chose_data_part) / len(chose_data_part))
                    chose_stock = chose_stock + list(chose_stock_part)
                    chose_data = chose_data + list(chose_data_part)

            pb_rank_line = [chose_data[chose_stock.index(st_num)] if (st_num in chose_stock) & (st_num in white_stock)
                            else numpy.nan for st_num in stocks0]

            golden_stock_line = golden_stock_mat.loc[golden_stock_mat.index[trade_date_num_i]].values
            pb_rank_line = numpy.where((golden_stock_line > 0) & (warn_line == 0), pb_rank_line, numpy.nan)
            pb_rank_line = arank(pb_rank_line)
            pb_rank_line = numpy.where(pb_rank_line <= choose_num[pool_num], 1, 0)
            factor_mat.append(pb_rank_line)

        # factor_df=rank(factor_df,reverse=-1)
        factor_df = pd.DataFrame(factor_mat, index=golden_stock_mat.index, columns=stocks0)
        factor_df.to_csv('D:/基本面三击策略每日更新/三者兼优选股仓位/PEreturnincaplow_rank_mat_' + pool_file[pool_num] + '.csv', encoding='gbk')
        factor_all_list.append(factor_df)
    return

def get_PEreturninnlsizehigh(stocks0,dates):
    PB_df = read_raw('D:/data_predict_project/data deal project/stock_price_pyd/PB.h5')\
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    marketvalue_df = read_raw('D:\data_predict_project\data deal project\stock_price_adj_pyd/marketValue.h5') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    SIZENL_df = pd.read_csv('D:/因子择时/择时因子计算/多因子归因择时/风险因子矩阵/SIZENL.csv', index_col=0, encoding='gbk') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    PB_df = pd.DataFrame(numpy.where(PB_df.values > 0, PB_df.values, numpy.nan), index=PB_df.index,
                         columns=PB_df.columns)
    PE_df = read_raw('D:/data_predict_project/data deal project/stock_price_pyd/PE.h5') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    PE_df = pd.DataFrame(numpy.where(PE_df.values > 0, PE_df.values, numpy.nan), index=PE_df.index,
                         columns=PE_df.columns)
    PE_df_div = (1 / PE_df)
    mktvalue_df_div = ((1 / marketvalue_df) * 10 ** (10))
    PEreturn_factor_df = (PE_df_div - PE_df_div.rolling(60).mean()) / (PE_df_div.rolling(60).std()) \
                         - (mktvalue_df_div - mktvalue_df_div.rolling(60).mean()) / (mktvalue_df_div.rolling(60).std())

    PEreturn_factor_df_values = numpy.where(numpy.isfinite(PEreturn_factor_df.values), PEreturn_factor_df.values,
                                            numpy.nan)
    PEreturn_factor_df = pd.DataFrame(PEreturn_factor_df_values, index=PEreturn_factor_df.index,
                                      columns=PEreturn_factor_df.columns)
    store = pd.HDFStore('D:\data_predict_project\data deal project\industry_pyd/stock_industry_zhongxin_matrix.h5',
                        mode='r')

    ind_mat_df = store.select('Lv1')
    store.close()
    ind_mat_df = ind_mat_df.reindex(columns=stocks0, fill_value='').reindex(index=dates, method='ffill').shift(1)
    ind_mat_df = ind_mat_df.fillna(value='')
    file_path = 'D:/data_predict_project/data deal project/blacklist_pyd/warnlist.h5'
    store = pd.HDFStore(file_path, mode='r')
    warnlist = store['data']
    store.close()
    warnlist_df = warnlist.reindex(columns=stocks0).reindex(index=dates, method='ffill').shift(1)
    ind_line_last = ind_mat_df.values[-1]
    ind_name_list = list(set(list(ind_line_last)))
    ind_name_list = [ind_name for ind_name in ind_name_list if ind_name != '']
    ind_stock_chose = {}
    for ind_name in ind_name_list:
        ind_stock_chose[ind_name] = []

    ind_stock_chose_data = {}
    for ind_name in ind_name_list:
        ind_stock_chose_data[ind_name] = []

    white_stock = pd.read_csv('D:/wind金股因子每日更新/股票池20230605.csv', encoding='gbk')
    white_stock = list(white_stock['证券代码'].values)
    white_stock = [num[:-3] for num in white_stock]

    pool_file = [pool_name,pool_name_small]
    choose_num = [50, 50]
    factor_all_list = []
    for pool_num in range(len(pool_file)):
        golden_stock_mat = pd.read_csv('D:/基本面三击策略每日更新/' + pool_file[pool_num] + '.csv', index_col=0,
                                       encoding='gbk').reindex(columns=stocks0)
        trade_date_list = [dates[dates <= pool_date][-1] for pool_date in golden_stock_mat.index]
        factor_mat = []
        for trade_date_num_i in range(len(trade_date_list)):
            trade_date_num = trade_date_list[trade_date_num_i]
            i = list(dates).index(trade_date_num)
            print(dates[i])
            PB_line = -PEreturn_factor_df.values[i]
            ind_line = ind_mat_df.values[i]
            warn_line = numpy.nan_to_num(warnlist_df.values[i])
            mktvalues_line = SIZENL_df.values[i]

            mktvalues_line_part_len = len(mktvalues_line[(numpy.isfinite(mktvalues_line))])
            mktvalues_rank = (arank(mktvalues_line) / mktvalues_line_part_len)
            PB_line_part_len = len(PB_line[(numpy.isfinite(PB_line))])
            PB_rank = (arank(PB_line, reverse=-1) / PB_line_part_len)
            cb_line_PB = (PB_line - numpy.nanmean(PB_line) / numpy.nanstd(PB_line))
            cb_line_mktvalue = (mktvalues_line - numpy.nanmean(mktvalues_line) / numpy.nanstd(mktvalues_line))
            cbline = -cb_line_mktvalue + cb_line_PB
            chose_stock = []
            chose_data = []
            for ind_name in ind_name_list:
                chose_stock_part = stocks0[
                    (ind_line == ind_name) & (numpy.isfinite(PB_line)) & (numpy.isfinite(mktvalues_line))]
                pbline_part = PB_line[
                    (ind_line == ind_name) & (numpy.isfinite(PB_line)) & (numpy.isfinite(mktvalues_line))]
                mktvaluesline_part = mktvalues_line[
                    (ind_line == ind_name) & (numpy.isfinite(mktvalues_line)) & (numpy.isfinite(PB_line))]

                chose_data_part = cbline[(ind_line == ind_name) & (numpy.isfinite(cbline))]
                if len(chose_data_part) > 0:
                    cb_line_PB_part = (pbline_part - numpy.nanmean(pbline_part) / numpy.nanstd(pbline_part))
                    cb_line_mktvalue_part = (
                        mktvaluesline_part - numpy.nanmean(mktvaluesline_part) / numpy.nanstd(mktvaluesline_part))
                    cbline_part = -cb_line_mktvalue_part + cb_line_PB_part
                    # chose_data_part = list(arank(cbline_part) / len(cbline_part))
                    chose_data_part = list(arank(chose_data_part) / len(chose_data_part))
                    chose_stock = chose_stock + list(chose_stock_part)
                    chose_data = chose_data + list(chose_data_part)

            pb_rank_line = [chose_data[chose_stock.index(st_num)] if (st_num in chose_stock) & (st_num in white_stock)
                            else numpy.nan for st_num in stocks0]

            golden_stock_line = golden_stock_mat.loc[golden_stock_mat.index[trade_date_num_i]].values
            pb_rank_line = numpy.where((golden_stock_line > 0) & (warn_line == 0), pb_rank_line, numpy.nan)
            pb_rank_line = arank(pb_rank_line)
            pb_rank_line = numpy.where(pb_rank_line <= choose_num[pool_num], 1, 0)
            factor_mat.append(pb_rank_line)

        # factor_df=rank(factor_df,reverse=-1)
        factor_df = pd.DataFrame(factor_mat, index=golden_stock_mat.index, columns=stocks0)
        factor_df.to_csv('D:/基本面三击策略每日更新/三者兼优选股仓位/PEreturninnlsizehigh_rank_mat_' + pool_file[pool_num] + '.csv', encoding='gbk')
        factor_all_list.append(factor_df)
    return

def get_PEreturninliqlow(stocks0,dates):
    PB_df = read_raw('D:/data_predict_project/data deal project/stock_price_pyd/PB.h5')\
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    marketvalue_df = read_raw('D:\data_predict_project\data deal project\stock_price_adj_pyd/marketValue.h5') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    LIQUIDTY_df = pd.read_csv('D:/因子择时/择时因子计算/多因子归因择时/风险因子矩阵/LIQUIDTY.csv', index_col=0, encoding='gbk') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    PB_df = pd.DataFrame(numpy.where(PB_df.values > 0, PB_df.values, numpy.nan), index=PB_df.index,
                         columns=PB_df.columns)
    PE_df = read_raw('D:/data_predict_project/data deal project/stock_price_pyd/PE.h5') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    PE_df = pd.DataFrame(numpy.where(PE_df.values > 0, PE_df.values, numpy.nan), index=PE_df.index,
                         columns=PE_df.columns)
    PE_df_div = (1 / PE_df)
    mktvalue_df_div = ((1 / marketvalue_df) * 10 ** (10))
    PEreturn_factor_df = (PE_df_div - PE_df_div.rolling(60).mean()) / (PE_df_div.rolling(60).std()) \
                         - (mktvalue_df_div - mktvalue_df_div.rolling(60).mean()) / (mktvalue_df_div.rolling(60).std())

    PEreturn_factor_df_values = numpy.where(numpy.isfinite(PEreturn_factor_df.values), PEreturn_factor_df.values,
                                            numpy.nan)
    PEreturn_factor_df = pd.DataFrame(PEreturn_factor_df_values, index=PEreturn_factor_df.index,
                                      columns=PEreturn_factor_df.columns)
    store = pd.HDFStore('D:\data_predict_project\data deal project\industry_pyd/stock_industry_zhongxin_matrix.h5',
                        mode='r')

    ind_mat_df = store.select('Lv1')
    store.close()
    ind_mat_df = ind_mat_df.reindex(columns=stocks0, fill_value='').reindex(index=dates, method='ffill').shift(1)
    ind_mat_df = ind_mat_df.fillna(value='')
    file_path = 'D:/data_predict_project/data deal project/blacklist_pyd/warnlist.h5'
    store = pd.HDFStore(file_path, mode='r')
    warnlist = store['data']
    store.close()
    warnlist_df = warnlist.reindex(columns=stocks0).reindex(index=dates, method='ffill').shift(1)
    ind_line_last = ind_mat_df.values[-1]
    ind_name_list = list(set(list(ind_line_last)))
    ind_name_list = [ind_name for ind_name in ind_name_list if ind_name != '']
    ind_stock_chose = {}
    for ind_name in ind_name_list:
        ind_stock_chose[ind_name] = []

    ind_stock_chose_data = {}
    for ind_name in ind_name_list:
        ind_stock_chose_data[ind_name] = []

    white_stock = pd.read_csv('D:/wind金股因子每日更新/股票池20230605.csv', encoding='gbk')
    white_stock = list(white_stock['证券代码'].values)
    white_stock = [num[:-3] for num in white_stock]

    pool_file = [pool_name,pool_name_small]
    choose_num = [50, 50]
    factor_all_list = []
    for pool_num in range(len(pool_file)):
        golden_stock_mat = pd.read_csv('D:/基本面三击策略每日更新/' + pool_file[pool_num] + '.csv', index_col=0,
                                       encoding='gbk').reindex(columns=stocks0)
        trade_date_list = [dates[dates <= pool_date][-1] for pool_date in golden_stock_mat.index]
        factor_mat = []
        for trade_date_num_i in range(len(trade_date_list)):
            trade_date_num = trade_date_list[trade_date_num_i]
            i = list(dates).index(trade_date_num)
            print(dates[i])
            PB_line = -PEreturn_factor_df.values[i]
            ind_line = ind_mat_df.values[i]
            warn_line = numpy.nan_to_num(warnlist_df.values[i])
            mktvalues_line = LIQUIDTY_df.values[i]

            mktvalues_line_part_len = len(mktvalues_line[(numpy.isfinite(mktvalues_line))])
            mktvalues_rank = (arank(mktvalues_line) / mktvalues_line_part_len)
            PB_line_part_len = len(PB_line[(numpy.isfinite(PB_line))])
            PB_rank = (arank(PB_line, reverse=-1) / PB_line_part_len)
            cb_line_PB = (PB_line - numpy.nanmean(PB_line) / numpy.nanstd(PB_line))
            cb_line_mktvalue = (mktvalues_line - numpy.nanmean(mktvalues_line) / numpy.nanstd(mktvalues_line))
            cbline = cb_line_mktvalue + cb_line_PB
            chose_stock = []
            chose_data = []
            for ind_name in ind_name_list:
                chose_stock_part = stocks0[
                    (ind_line == ind_name) & (numpy.isfinite(PB_line)) & (numpy.isfinite(mktvalues_line))]
                pbline_part = PB_line[
                    (ind_line == ind_name) & (numpy.isfinite(PB_line)) & (numpy.isfinite(mktvalues_line))]
                mktvaluesline_part = mktvalues_line[
                    (ind_line == ind_name) & (numpy.isfinite(mktvalues_line)) & (numpy.isfinite(PB_line))]

                chose_data_part = cbline[(ind_line == ind_name) & (numpy.isfinite(cbline))]
                if len(chose_data_part) > 0:
                    cb_line_PB_part = (pbline_part - numpy.nanmean(pbline_part) / numpy.nanstd(pbline_part))
                    cb_line_mktvalue_part = (
                        mktvaluesline_part - numpy.nanmean(mktvaluesline_part) / numpy.nanstd(mktvaluesline_part))
                    cbline_part = -cb_line_mktvalue_part + cb_line_PB_part
                    # chose_data_part = list(arank(cbline_part) / len(cbline_part))
                    chose_data_part = list(arank(chose_data_part) / len(chose_data_part))
                    chose_stock = chose_stock + list(chose_stock_part)
                    chose_data = chose_data + list(chose_data_part)

            pb_rank_line = [chose_data[chose_stock.index(st_num)] if (st_num in chose_stock) & (st_num in white_stock)
                            else numpy.nan for st_num in stocks0]

            golden_stock_line = golden_stock_mat.loc[golden_stock_mat.index[trade_date_num_i]].values
            pb_rank_line = numpy.where((golden_stock_line > 0) & (warn_line == 0), pb_rank_line, numpy.nan)
            pb_rank_line = arank(pb_rank_line)
            pb_rank_line = numpy.where(pb_rank_line <= choose_num[pool_num], 1, 0)
            factor_mat.append(pb_rank_line)

        # factor_df=rank(factor_df,reverse=-1)
        factor_df = pd.DataFrame(factor_mat, index=golden_stock_mat.index, columns=stocks0)
        factor_df.to_csv('D:/基本面三击策略每日更新/三者兼优选股仓位/PEreturninliqlow_rank_mat_' + pool_file[pool_num] + '.csv', encoding='gbk')
        factor_all_list.append(factor_df)
    return

def get_lowstd(stocks0,dates):
    close_df = read_raw('D:/data_predict_project/data deal project/stock_price_pyd/closePrice.h5')\
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    marketvalue_df = read_raw('D:\data_predict_project\data deal project\stock_price_adj_pyd/marketValue.h5') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    std_df = (close_df.diff(1)/close_df.shift(1)).rolling(window=20).std()
    store = pd.HDFStore('D:\data_predict_project\data deal project\industry_pyd/stock_industry_zhongxin_matrix.h5',
                        mode='r')

    ind_mat_df = store.select('Lv1')
    store.close()
    ind_mat_df = ind_mat_df.reindex(columns=stocks0, fill_value='').reindex(index=dates, method='ffill').shift(1)
    ind_mat_df = ind_mat_df.fillna(value='')
    file_path = 'D:/data_predict_project/data deal project/blacklist_pyd/warnlist.h5'
    store = pd.HDFStore(file_path, mode='r')
    warnlist = store['data']
    store.close()
    warnlist_df = warnlist.reindex(columns=stocks0).reindex(index=dates, method='ffill').shift(1)
    ind_line_last = ind_mat_df.values[-1]
    ind_name_list = list(set(list(ind_line_last)))
    ind_name_list = [ind_name for ind_name in ind_name_list if ind_name != '']
    ind_stock_chose = {}
    for ind_name in ind_name_list:
        ind_stock_chose[ind_name] = []

    ind_stock_chose_data = {}
    for ind_name in ind_name_list:
        ind_stock_chose_data[ind_name] = []

    white_stock = pd.read_csv('D:/wind金股因子每日更新/股票池20230605.csv', encoding='gbk')
    white_stock = list(white_stock['证券代码'].values)
    white_stock = [num[:-3] for num in white_stock]

    pool_file = [pool_name,pool_name_small]
    choose_num = [50, 50]
    factor_all_list = []
    for pool_num in range(len(pool_file)):
        golden_stock_mat = pd.read_csv('D:/基本面三击策略每日更新/' + pool_file[pool_num] + '.csv', index_col=0,
                                       encoding='gbk').reindex(columns=stocks0)
        trade_date_list = [dates[dates <= pool_date][-1] for pool_date in golden_stock_mat.index]
        factor_mat = []
        for trade_date_num_i in range(len(trade_date_list)):
            trade_date_num = trade_date_list[trade_date_num_i]
            i = list(dates).index(trade_date_num)
            print(dates[i])
            PB_line = std_df.values[i]
            ind_line = ind_mat_df.values[i]
            warn_line = numpy.nan_to_num(warnlist_df.values[i])
            mktvalues_line = marketvalue_df.values[i]

            mktvalues_line_part_len = len(mktvalues_line[(numpy.isfinite(mktvalues_line))])
            mktvalues_rank = (arank(mktvalues_line) / mktvalues_line_part_len)
            PB_line_part_len = len(PB_line[(numpy.isfinite(PB_line))])
            PB_rank = (arank(PB_line, reverse=-1) / PB_line_part_len)
            cb_line_PB = (PB_line - numpy.nanmean(PB_line) / numpy.nanstd(PB_line))
            cb_line_mktvalue = (mktvalues_line - numpy.nanmean(mktvalues_line) / numpy.nanstd(mktvalues_line))
            cbline = cb_line_PB
            chose_stock = []
            chose_data = []
            for ind_name in ind_name_list:
                chose_stock_part = stocks0[
                    (ind_line == ind_name) & (numpy.isfinite(PB_line)) & (numpy.isfinite(mktvalues_line))]
                pbline_part = PB_line[
                    (ind_line == ind_name) & (numpy.isfinite(PB_line)) & (numpy.isfinite(mktvalues_line))]
                mktvaluesline_part = mktvalues_line[
                    (ind_line == ind_name) & (numpy.isfinite(mktvalues_line)) & (numpy.isfinite(PB_line))]

                chose_data_part = cbline[(ind_line == ind_name) & (numpy.isfinite(cbline))]
                if len(chose_data_part) > 0:
                    cb_line_PB_part = (pbline_part - numpy.nanmean(pbline_part) / numpy.nanstd(pbline_part))
                    cb_line_mktvalue_part = (
                        mktvaluesline_part - numpy.nanmean(mktvaluesline_part) / numpy.nanstd(mktvaluesline_part))
                    cbline_part = -cb_line_mktvalue_part + cb_line_PB_part
                    # chose_data_part = list(arank(cbline_part) / len(cbline_part))
                    chose_data_part = list(arank(chose_data_part) / len(chose_data_part))
                    chose_stock = chose_stock + list(chose_stock_part)
                    chose_data = chose_data + list(chose_data_part)

            pb_rank_line = [chose_data[chose_stock.index(st_num)] if (st_num in chose_stock) & (st_num in white_stock)
                            else numpy.nan for st_num in stocks0]

            golden_stock_line = golden_stock_mat.loc[golden_stock_mat.index[trade_date_num_i]].values
            pb_rank_line = numpy.where((golden_stock_line > 0) & (warn_line == 0), pb_rank_line, numpy.nan)
            pb_rank_line = arank(pb_rank_line)
            pb_rank_line = numpy.where(pb_rank_line <= choose_num[pool_num], 1, 0)
            factor_mat.append(pb_rank_line)

        # factor_df=rank(factor_df,reverse=-1)
        factor_df = pd.DataFrame(factor_mat, index=golden_stock_mat.index, columns=stocks0)
        factor_df.to_csv('D:/基本面三击策略每日更新/三者兼优选股仓位/lowstd_rank_mat_' + pool_file[pool_num] + '.csv', encoding='gbk')
        factor_all_list.append(factor_df)
    return

def get_retreverse(stocks0,dates):
    close_df = read_raw('D:/data_predict_project/data deal project/stock_price_pyd/closePrice.h5')\
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    marketvalue_df = read_raw('D:\data_predict_project\data deal project\stock_price_adj_pyd/marketValue.h5') \
        .reindex(index=dates, method='ffill').reindex(columns=stocks0).shift(1)
    ret20_df = (close_df.diff(20)/close_df.shift(20))
    store = pd.HDFStore('D:\data_predict_project\data deal project\industry_pyd/stock_industry_zhongxin_matrix.h5',
                        mode='r')

    ind_mat_df = store.select('Lv1')
    store.close()
    ind_mat_df = ind_mat_df.reindex(columns=stocks0, fill_value='').reindex(index=dates, method='ffill').shift(1)
    ind_mat_df = ind_mat_df.fillna(value='')
    file_path = 'D:/data_predict_project/data deal project/blacklist_pyd/warnlist.h5'
    store = pd.HDFStore(file_path, mode='r')
    warnlist = store['data']
    store.close()
    warnlist_df = warnlist.reindex(columns=stocks0).reindex(index=dates, method='ffill').shift(1)
    ind_line_last = ind_mat_df.values[-1]
    ind_name_list = list(set(list(ind_line_last)))
    ind_name_list = [ind_name for ind_name in ind_name_list if ind_name != '']
    ind_stock_chose = {}
    for ind_name in ind_name_list:
        ind_stock_chose[ind_name] = []

    ind_stock_chose_data = {}
    for ind_name in ind_name_list:
        ind_stock_chose_data[ind_name] = []

    white_stock = pd.read_csv('D:/wind金股因子每日更新/股票池20230605.csv', encoding='gbk')
    white_stock = list(white_stock['证券代码'].values)
    white_stock = [num[:-3] for num in white_stock]

    pool_file = [pool_name,pool_name_small]
    choose_num = [50, 50]
    factor_all_list = []
    for pool_num in range(len(pool_file)):
        golden_stock_mat = pd.read_csv('D:/基本面三击策略每日更新/' + pool_file[pool_num] + '.csv', index_col=0,
                                       encoding='gbk').reindex(columns=stocks0)
        trade_date_list = [dates[dates <= pool_date][-1] for pool_date in golden_stock_mat.index]
        factor_mat = []
        for trade_date_num_i in range(len(trade_date_list)):
            trade_date_num = trade_date_list[trade_date_num_i]
            i = list(dates).index(trade_date_num)
            print(dates[i])
            PB_line = ret20_df.values[i]
            ind_line = ind_mat_df.values[i]
            warn_line = numpy.nan_to_num(warnlist_df.values[i])
            mktvalues_line = marketvalue_df.values[i]

            mktvalues_line_part_len = len(mktvalues_line[(numpy.isfinite(mktvalues_line))])
            mktvalues_rank = (arank(mktvalues_line) / mktvalues_line_part_len)
            PB_line_part_len = len(PB_line[(numpy.isfinite(PB_line))])
            PB_rank = (arank(PB_line, reverse=-1) / PB_line_part_len)
            cb_line_PB = (PB_line - numpy.nanmean(PB_line) / numpy.nanstd(PB_line))
            cb_line_mktvalue = (mktvalues_line - numpy.nanmean(mktvalues_line) / numpy.nanstd(mktvalues_line))
            cbline = cb_line_PB
            chose_stock = []
            chose_data = []
            for ind_name in ind_name_list:
                chose_stock_part = stocks0[
                    (ind_line == ind_name) & (numpy.isfinite(PB_line)) & (numpy.isfinite(mktvalues_line))]
                pbline_part = PB_line[
                    (ind_line == ind_name) & (numpy.isfinite(PB_line)) & (numpy.isfinite(mktvalues_line))]
                mktvaluesline_part = mktvalues_line[
                    (ind_line == ind_name) & (numpy.isfinite(mktvalues_line)) & (numpy.isfinite(PB_line))]

                chose_data_part = cbline[(ind_line == ind_name) & (numpy.isfinite(cbline))]
                if len(chose_data_part) > 0:
                    cb_line_PB_part = (pbline_part - numpy.nanmean(pbline_part) / numpy.nanstd(pbline_part))
                    cb_line_mktvalue_part = (
                        mktvaluesline_part - numpy.nanmean(mktvaluesline_part) / numpy.nanstd(mktvaluesline_part))
                    cbline_part = -cb_line_mktvalue_part + cb_line_PB_part
                    # chose_data_part = list(arank(cbline_part) / len(cbline_part))
                    chose_data_part = list(arank(chose_data_part) / len(chose_data_part))
                    chose_stock = chose_stock + list(chose_stock_part)
                    chose_data = chose_data + list(chose_data_part)

            pb_rank_line = [chose_data[chose_stock.index(st_num)] if (st_num in chose_stock) & (st_num in white_stock)
                            else numpy.nan for st_num in stocks0]

            golden_stock_line = golden_stock_mat.loc[golden_stock_mat.index[trade_date_num_i]].values
            pb_rank_line = numpy.where((golden_stock_line > 0) & (warn_line == 0), pb_rank_line, numpy.nan)
            pb_rank_line = arank(pb_rank_line)
            pb_rank_line = numpy.where(pb_rank_line <= choose_num[pool_num], 1, 0)
            factor_mat.append(pb_rank_line)

        # factor_df=rank(factor_df,reverse=-1)
        factor_df = pd.DataFrame(factor_mat, index=golden_stock_mat.index, columns=stocks0)
        factor_df.to_csv('D:/基本面三击策略每日更新/三者兼优选股仓位/retreverse_rank_mat_' + pool_file[pool_num] + '.csv', encoding='gbk')
        factor_all_list.append(factor_df)
    return
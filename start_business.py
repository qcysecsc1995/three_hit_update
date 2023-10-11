import uqer
from WindPy import w
import pandas as pd
import numpy
import datetime
from make_dates_stocks0 import make_dates_stocks0
from update_uqer_factor import get_MktStockFactorsDateRangeProGet
from update_uqer_factor import get_RMExposureDayGet
from make_uqer_factor_mat import make_uqer_factor_mat
from make_uqer_factor_mat import make_barra_mat
from make_three_hit_factor_pool import make_three_hit_factor_pool
from make_three_hit_pool_big import make_three_hit_factor_pool_big
from make_three_hit_pool_rdexp import make_three_hit_factor_pool_rdexp
from make_three_hit_factor_pool_total import make_three_hit_factor_pool_total
from get_each_factor_in_pool import get_PBincaphigh
from get_each_factor_in_pool import get_PBincaplow
from get_each_factor_in_pool import get_PBinliqlow
from get_each_factor_in_pool import get_PBinnlsizehigh
from get_each_factor_in_pool import get_PEreturnincaphigh
from get_each_factor_in_pool import get_PEreturnincaplow
from get_each_factor_in_pool import get_PEreturninliqlow
from get_each_factor_in_pool import get_PEreturninnlsizehigh
from get_each_factor_in_pool import get_lowstd
from get_each_factor_in_pool import get_retreverse
from deal_combined_all_factor import deal_combined_golden_factor
from deal_final_stock_weight import deal_final_stock_weight

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


if __name__ == '__main__':
    w.start()
    client = uqer.Client(token='97088f3cbad0327f76126d33b97ec0e0a31043a78dd6756b5cbf346345e56472')
    today= (str(datetime.date.today()))
    dates_all, stocks0_all = make_dates_stocks0('2007-01-01', today, client)

    close, stocks0, dates = read_raw_h5('D:\data_predict_project\data deal project\stock_price_adj_pyd/closePrice.h5')

    dates_add=dates_all[dates_all>dates[-1]]
    dates_all=numpy.array(list(dates)+list(dates_add))
    white_stock = pd.read_csv('D:/wind金股因子每日更新/股票池20230605.csv', encoding='gbk')
    white_stock = list(white_stock['证券代码'].values)
    white_stock = [num[:-3] for num in white_stock]
    close=pd.DataFrame(close,columns=stocks0).reindex(columns=white_stock)
    stocks0=close.columns
    close=close.values
    dates_end0=dates[-1]
    dates_end_all=dates_all[-1]
    date_start_all=dates_all[-1]
    date_start_add_uqer=dates[-20]
    date_start_add_uqer_1=dates[-20]
    print (dates_end_all,date_start_all)

    get_MktStockFactorsDateRangeProGet(date_start_add_uqer_1, dates_end0, stocks0_all, client)
    get_RMExposureDayGet(date_start_add_uqer_1, dates_end0, client)
    make_uqer_factor_mat(stocks0_all)
    make_barra_mat(stocks0_all)

    make_three_hit_factor_pool(stocks0_all, dates_all)
    make_three_hit_factor_pool_rdexp(stocks0_all, dates_all)
    make_three_hit_factor_pool_big(stocks0_all, dates_all)
    make_three_hit_factor_pool_total(stocks0_all, dates_all)

    get_PBincaphigh(stocks0, dates_all)
    get_PBincaplow(stocks0, dates_all)
    get_PBinliqlow(stocks0, dates_all)
    get_PBinnlsizehigh(stocks0, dates_all)

    get_PEreturninliqlow(stocks0, dates_all)
    get_PEreturninnlsizehigh(stocks0, dates_all)
    get_PEreturnincaplow(stocks0, dates_all)
    get_PEreturnincaphigh(stocks0, dates_all)
    get_lowstd(stocks0, dates_all)
    get_retreverse(stocks0, dates_all)


    deal_combined_golden_factor(stocks0, dates_all)
    deal_final_stock_weight(stocks0, dates_all)




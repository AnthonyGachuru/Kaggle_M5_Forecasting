# ===========================================================================
# KAGGLE: M5 FORECAST CHALLENGE
# Daily forecast modelling with linear regression
# ===========================================================================
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# ===========================================================================
# FUNCTIONS: IMPORT DATA
# ===========================================================================

def import_data():
    
    # training set
    filename = 'data/sales_train_validation.csv'
    df_raw = pd.read_csv(filename)

    # calendar
    filename = 'data/calendar.csv'
    calendar = pd.read_csv(filename, parse_dates=['date'], infer_datetime_format=True)

    # price
    filename = 'data/sell_prices.csv'
    price = pd.read_csv(filename)

    return df_raw, calendar, price


def map_calendar(calendar):
    return calendar[['date', 'd', 'wm_yr_wk', 'weekday', 'month', 'year']].copy()


def map_holidays(calendar):
    holidays = calendar[['d', 'event_name_1']].copy()
    holidays.dropna(inplace=True)
    filter_holidays = ['SuperBowl', 'ValentinesDay', 'Easter', 'NBAFinalsEnd', 'Thanksgiving', 'Christmas', 'NewYear', "Mother's day", 'NBAFinalsStart','NBAFinalsEnd', "Father's day", 'IndependenceDay', 'Ramadan starts', 'Eid al-Fitr', 'LaborDay', 'ColumbusDay', 'Halloween', 'EidAlAdha']
    flag = holidays['event_name_1'].apply(lambda x: x in filter_holidays)
    holidays = holidays[flag]
    holidays.rename(columns={'event_name_1':'holiday'}, inplace=True)
    return holidays


def rank_products(df_raw):
    df = df_raw.drop(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], axis=1).copy()
    df.set_index('id', inplace=True)
    df['total'] = df.sum(axis=1)
    df_sorted = df[['total']].sort_values(by='total', ascending=False)
    return df_sorted


# ===========================================================================
# FUNCTIONS: PREPROCESSING
# ===========================================================================

def filter_product(df_raw, id):
    flag = df_raw['id'] == id
    return df_raw[flag]



# ===========================================================================
# FUNCTIONS: MODELLING
# ===========================================================================


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == '__main__':
    print()
from datetime import datetime
import pandas as pd
import numpy as np
import quandl 
import requests
import matplotlib.pyplot as plt
import warnings
import yfinance as yf
warnings.filterwarnings('ignore')

############# Data Fetching #############
# API 
myAPIkey = 'VPTMgg7k44QB9_2PCKWD'
quandl.ApiConfig.api_key = myAPIkey # for Nasdaq 
api_key = 'cf4f0867024084f82351267f25f9d1c6' # for FRED 


#get stock/ ETF data 
def get_stock_data(name, start, end): 
    raw_data = quandl.get_table('QUOTEMEDIA/PRICES', ticker = name, date = {'gte':start, 'lte':end}) 
    df = raw_data.sort_values(by = 'date')
    df.set_index('date', inplace = True)
    df  =  df.loc[:,['adj_close']]
    return df

# get bond data 
def fetch_data_from_FRED(api_key, series_id, file_type, start_date):
    '''
    This function will fetch the bond data given the api key, series id, file type, and start date
    '''

    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type={file_type}&observation_start={start_date}"

    # Make the API request
    response = requests.get(url)

    if response.status_code == 200:
        # Extract data if the request is successful
        data_raw = response.json()
        observations = data_raw['observations']

        # Convert the data into a list of dictionaries (date and value pairs)
        data = [{'date': entry['date'], 'value': entry['value']} for entry in observations]

        # Create a DataFrame and clean the data
        data_df = pd.DataFrame(data)
        data_df['date'] = pd.to_datetime(data_df['date'])
        data_df['value'] = pd.to_numeric(data_df['value'], errors='coerce')  # Convert values to numeric, making non-numeric into NaN
        data_df = data_df.dropna()  # Remove rows with NaN values

        return data_df
    else:
        print('Failed to retrieve data:', response.status_code)
        return pd.DataFrame()
    
def get_bond_data(name, start): 
    df = fetch_data_from_FRED(api_key, series_id = name, file_type = 'json'
                              , start_date = start).set_index('date')
    return df 

########## Data Processing ############
def yield_to_price(df, region, coupon_freq = 2):
    '''
    This function will calculate the price of the bond given the yield

    Parameters:
    df (pd.DataFrame): DataFrame containing the bond yields in a column named 'value'.
    region (str): Region of the bond ('US' or 'EU').
    coupon_freq (int): Frequency of coupon payments per year (default is 2).

    Returns:
    pd.DataFrame: DataFrame containing the bond yield, price, modified duration, convexity, coupon income, and total return.
    '''
    # compute the duration and convexity for price approximation
    # assume each bond is priced at par, so the coupon is equal to the yield
    def row_operation(row):
        face_value = 100
        total_periods = coupon_freq * 10
        coupon = row['value'] / coupon_freq
        y = row['value']/100
        #mod_d = sum([coupon*(-i/coupon_freq)*np.exp(y*(-i/coupon_freq)) for i in range(1, total_periods + 1)])
        mod_d = total_periods/(1+y/2)
        conv = sum([coupon*(i/coupon_freq)**2*np.exp(y*(-i/coupon_freq)) for i in range(1, total_periods + 1)])
        return mod_d, conv

    df[['mod_d', 'conv']] = df.apply(lambda row:row_operation(row), axis=1, result_type='expand')
    df['delta_y'] = df['value'].diff() / 100

    bond_price = [100.0]
    for i in range(1, len(df)):
        new_price = bond_price[-1] + (df['delta_y'].iloc[i] * df['mod_d'].iloc[i-1] + 1/2 * df['delta_y'].iloc[i]**2 * df['conv'].iloc[i-1]) * bond_price[-1]
        bond_price.append(new_price)

    df['price'] = bond_price

    if coupon_freq == 2:
        sample_interval = '6M'
    else:
        sample_interval = '3M'
    mean_coupon = df[['value']].resample(sample_interval).mean()
    mean_coupon.columns = ['coupon_income']

    merged_df = df.join(mean_coupon, how='left')
    merged_df['coupon_income'] = merged_df['coupon_income'].fillna(0).cumsum()
    merged_df['total'] = merged_df['coupon_income'] + merged_df['price']
    return merged_df[['value', 'price', 'mod_d', 'conv','coupon_income', 'total']]

def calculate_weekly_return(df): 
    data = df.copy()
    data['Weekly Return'] = ((data['price'] - data['price'].shift(5)) / data['price'].shift(5)) 
    weekly_data = data.loc[:,['Weekly Return']]
    weekly_data.dropna(inplace=True)
    return weekly_data

############### Plotting ################
def plot_rolling_corr(rolling_corr,constant_corr):
    plt.figure(figsize=(12,6))
    rolling_corr.plot()
    plt.title('Rolling Correlation between SPY and Bond')
    plt.axhline(constant_corr, color='black', lw=1, ls='--')
    plt.xlim(pd.to_datetime(rolling_corr.index.min()), pd.to_datetime(rolling_corr.index.max()))
    max_corr = rolling_corr.idxmax()
    min_corr = rolling_corr.idxmin()
    plt.scatter(max_corr, rolling_corr[max_corr], color='blue', s=50, label='Max Correlation', zorder=5)
    plt.scatter(min_corr, rolling_corr[min_corr], color='orange', s=50, label='Min Correlation', zorder=5)

    plt.show()
    

############# Risk Parity Strategy ################
def risk_parity_weights(volatility1, volatility2, correlation):
    # Calculate the inverse of the Marginal Risk Contributions 
    mrc1 = volatility1 * (volatility1 + correlation * volatility2)
    mrc2 = volatility2 * (volatility2 + correlation * volatility1)
    
    inv_mrc1 = 1 / mrc1
    inv_mrc2 = 1 / mrc2
    
    # Normalize the inverse MRCs to sum to 1 to get the portfolio weights
    weight1 = inv_mrc1 / (inv_mrc1 + inv_mrc2)
    weight2 = inv_mrc2 / (inv_mrc1 + inv_mrc2)
    
    return weight1, weight2

def equal_risk_parity(vol, cov): 
    if not vol.index.equals(cov.index): 
        raise ValueError("Index of vol and cov must be the same")
    n = len(vol)
    weight = pd.DataFrame(index = vol.index, columns = ['weight stock', 'weight bond'])

    for i in range(n):
        weight.iloc[i] = risk_parity_weights(vol.iloc[i,0], vol.iloc[i,1], cov.iloc[i,0])
    return weight


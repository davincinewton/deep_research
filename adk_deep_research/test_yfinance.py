import yfinance as yf
import json
import pandas as pd
# read response from REST API with `requests` library and format it as python dict
from sqlalchemy import create_engine,text    
from sqlalchemy.exc import SQLAlchemyError

username = 'root'
password = ''
host = 'localhost:3307'
database = 'stock_deep_research'
mysql_conn = f'mysql+mysqlconnector://{username}:{password}@{host}/{database}'

def display_findata(sym:str):
    try:
        engine = create_engine(mysql_conn)
    except ImportError:
        print("DBAPI driver (mysql-connector-python) not found. Please 'pip install mysql-connector-python'")
        exit()
    table_name = 'deep'
    sql_query = f"SELECT price_mo FROM {table_name} WHERE symbol = '{sym}'"

    # Initialize an empty dataframe
    result_df = pd.DataFrame()
    connection = engine.connect()
    result_df = pd.read_sql(sql_query, connection)
    print(result_df['price_mo'][0])  

def update_findata(num_row=9999):
    try:
        engine = create_engine(mysql_conn)
    except ImportError:
        print("DBAPI driver (mysql-connector-python) not found. Please 'pip install mysql-connector-python'")
        exit()
    table_name = 'deep'
    sql_query = f"SELECT symbol FROM {table_name} LIMIT {num_row}"

    # Initialize an empty dataframe
    result_df = pd.DataFrame()
    connection = engine.connect()
    result_df = pd.read_sql(sql_query, connection)
    print(result_df.head(10))
    # loop over cik
    row_cnt = 0
    for cik in result_df['symbol'].to_list():
        print(cik)
        # get historical market data
        dat = yf.Ticker(cik)
        kline = dat.history(period='3y', interval='1mo')
        # print(kline)
        # Ensure 'start' and 'end' are strings in YYYY-MM-DD format
        # kline['Date'] = pd.to_datetime(kline['Date']).dt.strftime('%Y-%m-%d')
        # kline.index = pd.to_datetime(kline.index).strftime('%Y-%m-%d')
        kline = kline.reset_index()

        # Format the 'Date' column to string in '%Y-%m-%d' format
        kline['Date'] = pd.to_datetime(kline['Date']).dt.strftime('%Y-%m-%d')

        # Convert DataFrame to JSON string (records orientation)
        json_string = kline.to_json(orient='records', lines=False)

        # Print or save the JSON string (e.g., to a file or send via API)
        print(len(json_string))
        # column_name = 'finance'
        # update_query = f"UPDATE {table_name} SET {column_name} = '{json_string}' WHERE id = {cik}"
        # SQL query using text() to safely handle the JSON string
        query = text("UPDATE deep SET price_mo = :json_data WHERE symbol = :id")
        # result = connection.execute(text(update_query))
        result = connection.execute(query, {"json_data": json_string, "id": cik})
        connection.commit()
        if result.rowcount < 1:
            print("row: ", cik, "fail to update finance!")
        else:
            row_cnt += 1
    engine.dispose()
    return row_cnt

if __name__ == '__main__':
    # print(update_findata())
    display_findata('MSFT')


    # dat = yf.Ticker('BRK')
    # kline = dat.history(period='3y', interval='1mo')
    # print(kline)
# options
# print(dat.option_chain(dat.options[0]).calls)

# get financials
# print(dat.income_stmt)
# print(dat.quarterly_income_stmt)

# # dates
# print(dat.calendar)

# # general info
# print(dat.info)

# # analysis
# print(dat.analyst_price_targets)

# # websocket
# # dat.live()
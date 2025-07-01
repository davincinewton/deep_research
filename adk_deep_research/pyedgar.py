import json
import pandas as pd
# read response from REST API with `requests` library and format it as python dict
from sqlalchemy import create_engine,text    
from sqlalchemy.exc import SQLAlchemyError
import requests

username = 'root'
password = ''
host = 'localhost:3307'
database = 'stock_deep_research'

# Create SQLAlchemy engine
mysql_conn = f'mysql+mysqlconnector://{username}:{password}@{host}/{database}'

Header = {
  "User-Agent": "your.email@email.com"#, # remaining fields are optional
#    "Accept-Encoding": "gzip, deflate",
#    "Host": "data.sec.gov"
}

# CIK_df = None

def get_cik_df():
    # https://www.sec.gov/files/company_tickers_exchange.json
    with open(r"G:\work\smolagent\adk_deep_research\company_tickers_exchange.json", "r") as f:
        CIK_dict = json.load(f)
        # convert CIK_dict to pandas
        CIK_df = pd.DataFrame(CIK_dict["data"], columns=CIK_dict["fields"])
        return CIK_df
    
def save_2_mysql(CIK_df):
    # MySQL connection parameters
    table_name = 'cik'
 
    # Create SQLAlchemy engine
    engine = create_engine(mysql_conn)

    # Save DataFrame to MySQL
    CIK_df.to_sql(table_name, con=engine, if_exists='replace', index=False)

    # Dispose of the engine
    engine.dispose()    

def get_sp500():
    # --- Step 1: Database Connection Details ---
    # Replace with your actual database credentials
    # The engine manages connections to the database.
    try:
        engine = create_engine(mysql_conn)
    except ImportError:
        print("DBAPI driver (mysql-connector-python) not found. Please 'pip install mysql-connector-python'")
        exit()


    # --- Step 3: Define the SQL Query with a JOIN ---
    # This query is identical to the previous one. It joins the two tables on the stock symbol.
    # Note: In the cik table sample, BRK.B is represented as BRK-B. 
    # We use REPLACE() to handle this discrepancy. Adjust if your data is different.
    # Using text() is a good practice to prevent SQL injection warnings with SQLAlchemy.
    sql_query = text("""
    SELECT 
        c.ticker AS symbol,
        c.cik,
        s.Company AS 'company_name'
    FROM 
        sp500 s
    INNER JOIN 
        cik c ON s.Symbol = REPLACE(c.ticker, '-', '.')
    """)

    # Initialize an empty dataframe
    result_df = pd.DataFrame()

    try:
        # --- Step 4: Connect, Fetch Data, and Close Connection ---
        # The 'with' statement ensures the connection is automatically closed
        with engine.connect() as connection:
            print("Connection successful via SQLAlchemy engine.")
            
            # Use pandas to read the SQL query results into a DataFrame
            result_df = pd.read_sql(sql_query, connection)

    except SQLAlchemyError as e:
        print(f"An error occurred while connecting or fetching data: {e}")
        return None

    # --- Step 5: Display the Resulting DataFrame ---
    if not result_df.empty:
        print("\n--- Resulting DataFrame ---")
        # Rename column to match the exact requirement "company name"
        # result_df = result_df.rename(columns={'company_name': 'company name'})
        # print(result_df.to_string()) # .to_string() prints the full dataframe
        print(result_df.head(20))
        result_df.to_sql(name='deep', con=engine, if_exists='append', index=False)

        # Dispose of the engine
        engine.dispose()
        return result_df
    else:
        print("\nNo data was fetched. The resulting DataFrame is empty.")
        return None
    
def update_findata(num_row=9999):
    try:
        engine = create_engine(mysql_conn)
    except ImportError:
        print("DBAPI driver (mysql-connector-python) not found. Please 'pip install mysql-connector-python'")
        exit()
    table_name = 'deep'
    sql_query = f"SELECT * FROM {table_name} LIMIT {num_row}"

    # Initialize an empty dataframe
    result_df = pd.DataFrame()
    connection = engine.connect()
    result_df = pd.read_sql(sql_query, connection)
    print(result_df.head(10))
    # loop over cik
    row_cnt = 0
    for cik in result_df['cik'].to_list():
        print(cik)
        total = get_total_xbs(cik=cik)
        # Ensure 'start' and 'end' are strings in YYYY-MM-DD format
        total['start'] = pd.to_datetime(total['start']).dt.strftime('%Y-%m-%d')
        total['end'] = pd.to_datetime(total['end']).dt.strftime('%Y-%m-%d')
        # Convert DataFrame to JSON string (records orientation)
        json_string = total.to_json(orient='records', lines=False)

        # Print or save the JSON string (e.g., to a file or send via API)
        print(len(json_string))
        # column_name = 'finance'
        # update_query = f"UPDATE {table_name} SET {column_name} = '{json_string}' WHERE id = {cik}"
        # SQL query using text() to safely handle the JSON string
        query = text("UPDATE deep SET finance = :json_data WHERE cik = :id")
        # result = connection.execute(text(update_query))
        result = connection.execute(query, {"json_data": json_string, "id": cik})
        connection.commit()
        if result.rowcount < 1:
            print("row: ", cik, "fail to update finance!")
        else:
            row_cnt += 1
    engine.dispose()
    return row_cnt

def display_findata(sym:str):
    try:
        engine = create_engine(mysql_conn)
    except ImportError:
        print("DBAPI driver (mysql-connector-python) not found. Please 'pip install mysql-connector-python'")
        exit()
    table_name = 'deep'
    sql_query = f"SELECT finance FROM {table_name} WHERE symbol = '{sym}'"

    # Initialize an empty dataframe
    result_df = pd.DataFrame()
    connection = engine.connect()
    result_df = pd.read_sql(sql_query, connection)
    print(result_df['finance'][0])    
        
def get_cik(ticker:str, CIK_df):
    row = CIK_df[CIK_df["ticker"].str.lower() == ticker.lower()]
    if len(row)>=1:
        print(row)
        # print("More than 1 cik retrieved, return first.")
        return row.cik.values[0]
    else:
        print("0 cik retrieved, return AAPL cik.")
        return 320193

def get_company_fillings(CIK:str):
    # preparation of input data, using ticker and CIK set earlier
    url = f"https://data.sec.gov/submissions/CIK{str(CIK).zfill(10)}.json"
    print(url)
    company_filings = requests.get(url, headers=Header).json()
    print(company_filings['filings']['recent'])
    company_filings_df = pd.DataFrame(company_filings["filings"]["recent"])
    print(company_filings_df[['filingDate','form','primaryDocument']])
    return company_filings_df

def get_company_fillings_filter(company_filings_df, filter):
    return company_filings_df[company_filings_df.form == filter]

def get_annual_report(CIK, company_filings_10k_df, year:str):
    row = company_filings_10k_df[company_filings_10k_df.reportDate.str.contains(year, case=False)]
    
    access_number = row.accessionNumber.values[0].replace("-", "")

    file_name = row.primaryDocument.values[0]

    url = f"https://www.sec.gov/Archives/edgar/data/{CIK}/{access_number}/{file_name}"
    print(url)
    return url

def get_company_facts(CIK:str): 
    '''
    finance:
        "asset": current assets
        "revenue": revenue
        "gprofit": gross profit
        "netincome": net income
    
    '''
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{str(CIK).zfill(10)}.json"
    print(url)
    company_facts = requests.get(url, headers=Header).json()
    return company_facts

def get_assets(company_facts):
    if "AssetsCurrent" in company_facts:
        tkey = list(company_facts["AssetsCurrent"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["AssetsCurrent"]["units"][tkey]) 
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["10-K","10-K/A","20-F","20-F/A"])]
    elif 'Assets' in company_facts:
        tkey = list(company_facts["Assets"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["Assets"]["units"][tkey])     
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["20-F","20-F/A"])]               
    else:
        print('cannot read AssetsCurrent')
        return None, None
    curr_assets_df = curr_assets_df.drop_duplicates(["end"], keep='last', inplace=False)
    # print(curr_assets_df)
    return curr_assets_df, "Assets "+tkey

def get_longtermdebt(company_facts):
    if "LongTermDebt" in company_facts:
        tkey = list(company_facts["LongTermDebt"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["LongTermDebt"]["units"][tkey])
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["10-K","10-K/A","20-F","20-F/A"])]
    elif 'LongtermBorrowings' in company_facts:
        tkey = list(company_facts["LongtermBorrowings"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["LongtermBorrowings"]["units"][tkey])     
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["20-F","20-F/A"])]              
    else:
        print('cannot read LongTermDebt')
        return None, None    
     
    # curr_assets_df = curr_assets_df[curr_assets_df.form == "10-K"]
    curr_assets_df = curr_assets_df.drop_duplicates(["end"], keep='last', inplace=False)
    # print(curr_assets_df)
    return curr_assets_df, "Long Term Debt " + tkey

def get_Liabilities(company_facts):
    if "LiabilitiesCurrent" in company_facts:
        tkey = list(company_facts["LiabilitiesCurrent"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["LiabilitiesCurrent"]["units"][tkey]) 
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["10-K","10-K/A","20-F","20-F/A"])]
    elif 'Liabilities' in company_facts:
        tkey = list(company_facts["Liabilities"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["Liabilities"]["units"][tkey])     
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["20-F","20-F/A"])]        
    else:
        print('cannot read LiabilitiesCurrent')
        return None, None    
    # curr_assets_df = curr_assets_df[curr_assets_df.form == "10-K"]
    curr_assets_df = curr_assets_df.drop_duplicates(["end"], keep='last', inplace=False)
    # print(curr_assets_df)
    return curr_assets_df, "Liabilities "+ tkey

def get_cash(company_facts):
    if "NetCashProvidedByUsedInOperatingActivities" in company_facts:
        tkey = list(company_facts["NetCashProvidedByUsedInOperatingActivities"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["NetCashProvidedByUsedInOperatingActivities"]["units"][tkey]) 
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["10-K","10-K/A","20-F","20-F/A"])]
        curr_assets_df = curr_assets_df.drop_duplicates(["start","end"], keep='last', inplace=False)
        curr_assets_df['duration'] = (pd.to_datetime(curr_assets_df['end'])-pd.to_datetime(curr_assets_df['start'])).dt.days
        curr_assets_df = curr_assets_df[curr_assets_df['duration']>300]            
    elif 'CashAndCashEquivalents' in company_facts:
        tkey = list(company_facts["CashAndCashEquivalents"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["CashAndCashEquivalents"]["units"][tkey])     
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["20-F","20-F/A"])]         
        curr_assets_df = curr_assets_df.drop_duplicates(["end"], keep='last', inplace=False)  
    else:
        print('cannot read NetCashProvidedByUsedInOperatingActivities')
        return None, None        
    
    #CashCashEquivalentsAndShortTermInvestments NetCashProvidedByUsedInOperatingActivities PaymentsToAcquirePropertyPlantAndEquipment
    # curr_assets_df = curr_assets_df[curr_assets_df.form == "10-K"]

    # print(curr_assets_df)
    return curr_assets_df, "Cash "+ tkey

def get_shequity(company_facts):
    if "StockholdersEquity" in company_facts:
        tkey = list(company_facts["StockholdersEquity"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["StockholdersEquity"]["units"][tkey]) 
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["10-K","10-K/A","20-F","20-F/A"])]
    elif 'Equity' in company_facts:
        tkey = list(company_facts["Equity"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["Equity"]["units"][tkey])     
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["20-F","20-F/A"])]           
    else:
        print('cannot read StockholdersEquity')
        return None, None      
    # curr_assets_df = curr_assets_df[curr_assets_df.form == "10-K"]
    curr_assets_df = curr_assets_df.drop_duplicates(["end"], keep='last', inplace=False)
    # print(curr_assets_df)
    return curr_assets_df, "Equity "+tkey

def get_revenue_old(company_facts):
    if "Revenues" in company_facts:
        tkey = list(company_facts["Revenues"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["Revenues"]["units"][tkey ]) 
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["10-K","10-K/A","20-F","20-F/A"])]
    elif 'Revenue' in company_facts:
        tkey = list(company_facts["Revenue"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["Revenue"]["units"][tkey])     
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["20-F","20-F/A"])]             
    else:
        print('cannot read revenue')
        return None, None
    
    #  RevenueFromContractWithCustomerExcludingAssessedTax
    # curr_assets_df = curr_assets_df[curr_assets_df.form == "10-K"]
    curr_assets_df = curr_assets_df.drop_duplicates(["start","end"], keep='last', inplace=False)
    curr_assets_df['duration'] = (pd.to_datetime(curr_assets_df['end'])-pd.to_datetime(curr_assets_df['start'])).dt.days
    curr_assets_df = curr_assets_df[curr_assets_df['duration']>300]
    # print(curr_assets_df)
    return curr_assets_df, "Revenues "+tkey

def get_revenue(company_facts):
    if 'RevenueFromContractWithCustomerExcludingAssessedTax' in company_facts:
        tkey = list(company_facts["RevenueFromContractWithCustomerExcludingAssessedTax"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["RevenueFromContractWithCustomerExcludingAssessedTax"]["units"][tkey]) 
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["10-K","10-K/A","20-F","20-F/A"])]
    elif 'RevenueFromContractsWithCustomers' in company_facts:
        tkey = list(company_facts["RevenueFromContractsWithCustomers"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["RevenueFromContractsWithCustomers"]["units"][tkey])     
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["20-F","20-F/A"])]     
    else:
        print('cannot read revenue')
        return None, None
    
    #  RevenueFromContractWithCustomerExcludingAssessedTax
    
    curr_assets_df = curr_assets_df.drop_duplicates(["start","end"], keep='last', inplace=False)
    curr_assets_df['duration'] = (pd.to_datetime(curr_assets_df['end'])-pd.to_datetime(curr_assets_df['start'])).dt.days
    curr_assets_df = curr_assets_df[curr_assets_df['duration']>300]
    # print(curr_assets_df)
    return curr_assets_df, "Revenue "+tkey

def get_netincome(company_facts):
    if 'NetIncomeLoss' in company_facts:
        tkey = list(company_facts["NetIncomeLoss"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["NetIncomeLoss"]["units"][tkey])
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["10-K","10-K/A","20-F","20-F/A"])]
    elif 'ProfitLoss' in company_facts:
        tkey = list(company_facts["ProfitLoss"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["ProfitLoss"]["units"][tkey])        
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["20-F","20-F/A"])] 
    else:
        print('cannot read NetIncomeLoss')
        return None, None    
    curr_assets_df = curr_assets_df.drop_duplicates(["start","end"], keep='last', inplace=False)
    curr_assets_df['duration'] = (pd.to_datetime(curr_assets_df['end'])-pd.to_datetime(curr_assets_df['start'])).dt.days
    curr_assets_df = curr_assets_df[curr_assets_df['duration']>300]
    # print(curr_assets_df)
    return curr_assets_df, "Profit "+tkey

#OperatingIncomeLoss
def get_operatingincome(company_facts):
    if 'OperatingIncomeLoss' in company_facts:
        tkey = list(company_facts["OperatingIncomeLoss"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["OperatingIncomeLoss"]["units"][tkey]) 
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["10-K","10-K/A","20-F","20-F/A"])]
    elif 'ProfitLossFromOperatingActivities' in company_facts:
        tkey = list(company_facts["ProfitLossFromOperatingActivities"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["ProfitLossFromOperatingActivities"]["units"][tkey]) 
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["20-F","20-F/A"])]
    else:
        print('cannot OperatingIncomeLoss')
        return None, None    
    curr_assets_df = curr_assets_df.drop_duplicates(["start","end"], keep='last', inplace=False)
    curr_assets_df['duration'] = (pd.to_datetime(curr_assets_df['end'])-pd.to_datetime(curr_assets_df['start'])).dt.days
    curr_assets_df = curr_assets_df[curr_assets_df['duration']>300]
    # print(curr_assets_df)
    return curr_assets_df, "Operation Income "+tkey

def get_grossprofit(company_facts):
    if 'GrossProfit' in company_facts:
        tkey = list(company_facts["GrossProfit"]["units"].keys())[0]
        if tkey == "USD":
            curr_assets_df = pd.DataFrame(company_facts["GrossProfit"]["units"]["USD"]) 
            curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["10-K","10-K/A","20-F","20-F/A"])]
        else:
            curr_assets_df = pd.DataFrame(company_facts["GrossProfit"]["units"][tkey]) 
            curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["20-F","20-F/A"])]
    else:
        print('cannot read gross profit')
        return None, None
    
    curr_assets_df = curr_assets_df.drop_duplicates(["start","end"], keep='last', inplace=False)
    curr_assets_df['duration'] = (pd.to_datetime(curr_assets_df['end'])-pd.to_datetime(curr_assets_df['start'])).dt.days
    curr_assets_df = curr_assets_df[curr_assets_df['duration']>300]
    # print(curr_assets_df)
    return curr_assets_df, "Gross Profit "+tkey

def get_costofrevenue(company_facts):
    if 'CostOfRevenue' in company_facts:
        tkey = list(company_facts["CostOfRevenue"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["CostOfRevenue"]["units"][tkey]) 
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["10-K","10-K/A","20-F","20-F/A"])]
    elif 'CostOfSales' in company_facts:
        tkey = list(company_facts["CostOfSales"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["CostOfSales"]["units"][tkey]) 
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["20-F","20-F/A"])]
    else:
        print('cannot read CostOfRevenue')
        return None, None
    
    curr_assets_df = curr_assets_df.drop_duplicates(["start","end"], keep='last', inplace=False)
    curr_assets_df['duration'] = (pd.to_datetime(curr_assets_df['end'])-pd.to_datetime(curr_assets_df['start'])).dt.days
    curr_assets_df = curr_assets_df[curr_assets_df['duration']>300]
    # print(curr_assets_df)
    return curr_assets_df, "Cost "+tkey

def get_eps_basic(company_facts):
    if 'EarningsPerShareBasic' in company_facts:
        tkey = list(company_facts["EarningsPerShareBasic"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["EarningsPerShareBasic"]["units"][tkey]) 
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["10-K","10-K/A","20-F","20-F/A"])]
    elif 'BasicEarningsLossPerShare' in company_facts:
        tkey = list(company_facts["BasicEarningsLossPerShare"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["BasicEarningsLossPerShare"]["units"][tkey]) 
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["20-F","20-F/A"])]
    else:
        print('cannot read EarningsPerShareBasic')
        return None, None    
    
    
    curr_assets_df = curr_assets_df.drop_duplicates(["start","end"], keep='last', inplace=False)
    curr_assets_df['duration'] = (pd.to_datetime(curr_assets_df['end'])-pd.to_datetime(curr_assets_df['start'])).dt.days
    curr_assets_df = curr_assets_df[curr_assets_df['duration']>300]
    # print(curr_assets_df)
    return curr_assets_df, "EPS "+tkey

def get_shareoutstanding(company_facts): #CommonStockSharesOutstanding
    if 'WeightedAverageNumberOfSharesOutstandingBasic' in company_facts:
        tkey = list(company_facts["WeightedAverageNumberOfSharesOutstandingBasic"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["WeightedAverageNumberOfSharesOutstandingBasic"]["units"][tkey]) 
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["10-K","10-K/A","20-F","20-F/A"])]
    elif 'WeightedAverageShares' in company_facts:
        tkey = list(company_facts["WeightedAverageShares"]["units"].keys())[0]
        curr_assets_df = pd.DataFrame(company_facts["WeightedAverageShares"]["units"][tkey]) 
        curr_assets_df = curr_assets_df[curr_assets_df.form.isin(["20-F","20-F/A"])]
    else:
        print('cannot read WeightedAverageNumberOfSharesOutstandingBasic')
        return None, None        
    # print(company_facts["WeightedAverageNumberOfSharesOutstandingBasic"]["units"]['shares'])
    
    
    curr_assets_df = curr_assets_df.drop_duplicates(["start","end"], keep='last', inplace=False)
    curr_assets_df['duration'] = (pd.to_datetime(curr_assets_df['end'])-pd.to_datetime(curr_assets_df['start'])).dt.days
    curr_assets_df = curr_assets_df[curr_assets_df['duration']>300]
    # print(curr_assets_df)
    return curr_assets_df, "Number of Shares "+tkey

    # #RevenueFromContractWithCustomerExcludingAssessedTax: remove querters, remove replicates, remove sigulars
    # # elif finance == "gprofit":

    # curr_netincome_df = pd.DataFrame(company_facts["NetIncomeLoss"]["units"]["USD"]) 
    
    # #Grossprofit sometime not exist, need to check costofrevenue

    # curr_epsbasic_df = pd.DataFrame(company_facts["EarningsPerShareBasic"]["units"]["USD/shares"])  #

    # # curr_assets_df = curr_assets_df[curr_assets_df.frame.notna()]
    # curr_assets_df = curr_assets_df[curr_assets_df.form == "10-K"]
    # curr_assets_df = curr_assets_df.drop_duplicates(["start","end"], keep='last', inplace=False)
    # curr_assets_df['duration'] = (pd.to_datetime(curr_assets_df['end'])-pd.to_datetime(curr_assets_df['start'])).dt.days
    # curr_assets_df = curr_assets_df[curr_assets_df['duration']>190]
    # # print(curr_assets_df.dtypes)
import pandas as pd
import numpy as np

def combine_financial_statements(revenue_df, revenue_tag, gross_profit_df, gross_profit_tag, cost_of_revenue_df, cost_of_revenue_tag, revenue_old_df, revenue_old_tag):
    """
    Combines financial dataframes (revenue, gross_profit, cost_of_revenue, and additional revenue_old)
    based on their 'start' and 'end' dates. Calculates missing values using the
    formula: revenue = gross_profit + cost_of_revenue, if the other two values are present.

    Prioritizes revenue from revenue_df over revenue_old_df for overlapping periods.

    Args:
        revenue_df (pd.DataFrame): DataFrame with primary revenue data.
        gross_profit_df (pd.DataFrame): DataFrame with gross profit data.
        cost_of_revenue_df (pd.DataFrame): DataFrame with cost of revenue data.
        revenue_old_df (pd.DataFrame): DataFrame with older revenue data (secondary source).
        Any of these dataframes can be None or empty.

    Returns:
        pd.DataFrame: A combined DataFrame with 'start', 'end', 'revenue_val',
                      'gross_profit_val', and 'cost_of_revenue_val' columns.
                      Missing values are calculated where possible.
    """

    def _process_single_df(df, val_col_name):
        """Helper to process a single financial dataframe."""
        if df is None or df.empty:
            # Return an empty DataFrame with the expected columns
            return pd.DataFrame(columns=['start', 'end', val_col_name])
        
        # Select relevant columns and create a copy to avoid SettingWithCopyWarning
        processed_df = df[['start', 'end', 'val']].copy()
        
        # Ensure 'start' and 'end' are datetime objects
        processed_df['start'] = pd.to_datetime(processed_df['start'])
        processed_df['end'] = pd.to_datetime(processed_df['end'])
        
        # Convert 'val' to numeric, coercing errors to NaN
        processed_df['val'] = pd.to_numeric(processed_df['val'], errors='coerce')
        
        # Rename 'val' column to the specified name
        processed_df.rename(columns={'val': val_col_name}, inplace=True)
        return processed_df

    # 1. Process each input DataFrame
    # Process primary revenue source
    rev_main_proc = _process_single_df(revenue_df, 'revenue_val_main')
    
    # Process secondary revenue source
    rev_old_proc = _process_single_df(revenue_old_df, 'revenue_val_old')

    # Consolidate revenue data, prioritizing 'revenue_df' over 'revenue_old_df'
    # Get all unique date ranges from both revenue dataframes
    # Filter out empty dataframes before concatenating to avoid FutureWarning
    dfs_for_rev_dates = [df[['start', 'end']] for df in [rev_main_proc, rev_old_proc] if not df.empty]
    
    if not dfs_for_rev_dates:
        all_rev_dates_index = pd.DataFrame(columns=['start', 'end'])
    else:
        all_rev_dates_index = pd.concat(dfs_for_rev_dates).drop_duplicates().sort_values(by=['start', 'end']).reset_index(drop=True)

    # Merge main revenue values
    consolidated_revenue = pd.merge(all_rev_dates_index, rev_main_proc, on=['start', 'end'], how='left')
    
    # Merge old revenue values and use them to fill NaNs from main revenue
    consolidated_revenue = pd.merge(consolidated_revenue, rev_old_proc, on=['start', 'end'], how='left')
    
    # Apply prioritization: if revenue_val_main is NaN, use revenue_val_old
    # This also handles cases where only revenue_old_df had data for a period
    # print(consolidated_revenue[['revenue_val_main', 'revenue_val_old']].dtypes)
    consolidated_revenue['revenue_val'] = consolidated_revenue['revenue_val_main'].fillna(consolidated_revenue['revenue_val_old']).astype('float64')
    
    # Keep only the final consolidated revenue column and date range columns
    consolidated_revenue = consolidated_revenue[['start', 'end', 'revenue_val']]
    # Ensure consolidated_revenue is treated as potentially empty if it resulted from empty inputs
    if consolidated_revenue.empty and not all_rev_dates_index.empty:
        # This case should ideally not happen if all_rev_dates_index has rows,
        # but defensively ensure it has the right structure if inputs were weird
        consolidated_revenue = pd.DataFrame(columns=['start', 'end', 'revenue_val'])
    elif consolidated_revenue.empty: # If all_rev_dates_index was empty too
         consolidated_revenue = pd.DataFrame(columns=['start', 'end', 'revenue_val'])


    # Process other financial dataframes
    gp_proc = _process_single_df(gross_profit_df, 'gross_profit_val')
    cor_proc = _process_single_df(cost_of_revenue_df, 'cost_of_revenue_val')

    # 2. Collect all unique date ranges from all consolidated dataframes
    # Filter out empty dataframes before concatenating to avoid FutureWarning
    dfs_for_all_dates = [df[['start', 'end']] for df in [consolidated_revenue, gp_proc, cor_proc] if not df.empty]

    if not dfs_for_all_dates:
        return pd.DataFrame(columns=['start', 'end', 'revenue_val', 'gross_profit_val', 'cost_of_revenue_val'])
    else:
        all_dates = pd.concat(dfs_for_all_dates).drop_duplicates().sort_values(by=['start', 'end']).reset_index(drop=True)

    # If all_dates is empty (e.g., all input DFs were empty/None), return an empty DataFrame
    if all_dates.empty: # This check is now mostly redundant due to the 'if not dfs_for_all_dates' above, but harmless.
        return pd.DataFrame(columns=['start', 'end', 'revenue_val', 'gross_profit_val', 'cost_of_revenue_val'])

    # 3. Merge processed dataframes onto the unique date ranges
    combined_df = all_dates.copy()
    combined_df = pd.merge(combined_df, consolidated_revenue, on=['start', 'end'], how='left')
    combined_df = pd.merge(combined_df, gp_proc, on=['start', 'end'], how='left')
    combined_df = pd.merge(combined_df, cor_proc, on=['start', 'end'], how='left')

    # Define column names for clarity
    rev_col = 'revenue_val'
    gp_col = 'gross_profit_val'
    cor_col = 'cost_of_revenue_val' 
    if revenue_tag is None:
        revenue_tag = 'Revenue'
    if gross_profit_tag is None:
        gross_profit_tag = 'Gross Profi'
    if revenue_old_tag is None:
        revenue_old_tag = 'Revenue'
    if cost_of_revenue_tag is None:
        cost_of_revenue_tag = 'Cost'

    # 4. Calculate missing values using the formula: revenue = gross_profit + cost_of_revenue
    # Calculations are performed only if two out of three values are available.

    # Case A: Revenue is missing (Revenue = Gross Profit + Cost of Revenue)
    mask_rev_missing = (combined_df[rev_col].isna()) & \
                       (combined_df[gp_col].notna()) & \
                       (combined_df[cor_col].notna())
    combined_df.loc[mask_rev_missing, rev_col] = combined_df[gp_col] + combined_df[cor_col]

    # Case B: Gross Profit is missing (Gross Profit = Revenue - Cost of Revenue)
    mask_gp_missing = (combined_df[gp_col].isna()) & \
                      (combined_df[rev_col].notna()) & \
                      (combined_df[cor_col].notna())
    combined_df.loc[mask_gp_missing, gp_col] = combined_df[rev_col] - combined_df[cor_col]

    # Case C: Cost of Revenue is missing (Cost of Revenue = Revenue - Gross Profit)
    mask_cor_missing = (combined_df[cor_col].isna()) & \
                       (combined_df[rev_col].notna()) & \
                       (combined_df[gp_col].notna())
    combined_df.loc[mask_cor_missing, cor_col] = combined_df[rev_col] - combined_df[gp_col]

    # Reorder columns for a clean output
    final_cols = ['start', 'end', rev_col, gp_col, cor_col]
    combined_df = combined_df[final_cols]
    return combined_df.rename(columns={rev_col:revenue_tag, gp_col:gross_profit_tag, cor_col:cost_of_revenue_tag})

def merge_full_financial_statements(combined_revenue_gp_cor_df, dfs_info):
    """
    Merges a combined revenue/GP/COR DataFrame with additional financial statements
    provided dynamically via dfs_info.

    Args:
        combined_revenue_gp_cor_df (pd.DataFrame): Output from combine_financial_statements_revenue_group.
                                                    Expected to have 'start', 'end', 'revenue_val',
                                                    'gross_profit_val', 'cost_of_revenue_val'.
        dfs_info (list of tuples): Each tuple is (dataframe, 'desired_column_name_for_val', is_period_data_boolean).
            - dataframe (pd.DataFrame or None): The input DataFrame.
            - 'desired_column_name_for_val' (str): The name for the 'val' column in the final output.
            - is_period_data_boolean (bool): True if DataFrame has 'start' and 'end' columns for periods,
                                             False if it has only an 'end' column for point-in-time.

    Returns:
        pd.DataFrame: A comprehensive combined DataFrame.
    """

    def _process_dynamic_df(df, val_col_name, is_period_data):
        """Helper to process individual dataframes dynamically."""
        if df is None or df.empty:
            if is_period_data:
                return pd.DataFrame(columns=['start', 'end', val_col_name])
            else:
                return pd.DataFrame(columns=['end', val_col_name])

        processed_df = df.copy()

        # Convert date columns and select relevant columns
        if is_period_data:
            processed_df['start'] = pd.to_datetime(processed_df['start'])
            processed_df['end'] = pd.to_datetime(processed_df['end'])
            processed_df = processed_df[['start', 'end', 'val']]
        else:
            processed_df['end'] = pd.to_datetime(processed_df['end'])
            processed_df = processed_df[['end', 'val']]

        # Convert 'val' to numeric, coercing errors
        processed_df['val'] = pd.to_numeric(processed_df['val'], errors='coerce')

        # Rename 'val' column
        return processed_df.rename(columns={'val': val_col_name})

    # 1. Prepare the base DataFrame (from previous function's output)
    base_df = combined_revenue_gp_cor_df.copy()
    if base_df is None or base_df.empty:
        base_df = pd.DataFrame(columns=['start', 'end', 'revenue_val', 'gross_profit_val', 'cost_of_revenue_val'])
    else:
        base_df['start'] = pd.to_datetime(base_df['start'])
        base_df['end'] = pd.to_datetime(base_df['end'])
    
    # Initialize lists to store processed DFs and their column names
    processed_period_dfs = []
    processed_point_in_time_dfs = []
    dynamic_val_cols_order = [] # To keep track of column order for the final output

    # 2. Process all dynamic input dataframes from dfs_info
    for df, col_name in dfs_info:
        if df is None:
            continue
        # processed_df = _process_dynamic_ df(df, col_name, is_period_data)
        if 'start' in df.columns:
            processed_df = _process_dynamic_df(df, col_name, True)
            processed_period_dfs.append(processed_df)
        else:
            processed_df = _process_dynamic_df(df, col_name, False)
            processed_point_in_time_dfs.append(processed_df)
        dynamic_val_cols_order.append(col_name) # Add to desired output order

    # 3. Consolidate all unique period ranges for the master index
    all_period_date_ranges_for_concat = []
    if not base_df.empty:
        all_period_date_ranges_for_concat.append(base_df[['start', 'end']])
    for df_proc in processed_period_dfs:
        if not df_proc.empty:
            all_period_date_ranges_for_concat.append(df_proc[['start', 'end']])
            
    if not all_period_date_ranges_for_concat:
        master_periods = pd.DataFrame(columns=['start', 'end'])
    else:
        master_periods = pd.concat(all_period_date_ranges_for_concat).drop_duplicates().sort_values(by=['start', 'end']).reset_index(drop=True)

    # If master_periods is empty, return an empty DataFrame with all expected columns
    if master_periods.empty:
        print("no enough record.")
        return None
        # Define all possible columns for an empty output DataFrame
        # base_cols = ['start', 'end', 'revenue_val', 'gross_profit_val', 'cost_of_revenue_val']
        # return pd.DataFrame(columns=list(set(base_cols + dynamic_val_cols_order))) # Use set to handle potential duplicates

    # 4. Merge all period-based data onto master_periods
    final_combined_df = master_periods.copy()
    
    # Start by merging the base combined revenue/GP/COR data
    final_combined_df = pd.merge(final_combined_df, base_df, on=['start', 'end'], how='left')

    # Merge other processed period-based dataframes
    for df_proc in processed_period_dfs:
        if not df_proc.empty:
            final_combined_df = pd.merge(final_combined_df, df_proc, on=['start', 'end'], how='left')

    # 5. Merge point-in-time dataframes based on 'end' date
    for df_proc in processed_point_in_time_dfs:
        if not df_proc.empty:
            # If there are duplicate 'end' dates in a source DF, keep the last one.
            df_proc_unique = df_proc.drop_duplicates(subset=['end'], keep='last')
            final_combined_df = pd.merge(final_combined_df, df_proc_unique, on='end', how='left')

    # 6. Final column ordering
    # # Start with the core revenue group columns
    # final_cols = ['start', 'end', 'revenue_val', 'gross_profit_val', 'cost_of_revenue_val']
    
    # # Append the dynamically provided column names in the order they were specified
    # for col_name in dynamic_val_cols_order:
    #     if col_name in final_combined_df.columns and col_name not in final_cols: # Avoid duplicates
    #         final_cols.append(col_name)
            
    # # Remove any columns from final_cols that might not exist in the actual dataframe
    # final_cols = [col for col in final_cols if col in final_combined_df.columns]

    # return final_combined_df[final_cols]
    return final_combined_df

import os

def get_total_xbs(ticker='', cik=None):
    if cik is None:
        CIK_df = get_cik_df()
        cik = get_cik(ticker, CIK_df)
    # filings = get_company_fillings(cik)
    # c10ks = get_company_fillings_filter(filings,"10-K")
    # print(c10ks.head(20))
    # get_annual_report(cik, c10ks,"2022")
    facts = get_company_facts(cik)
    # print(facts["facts"].keys())
    # print(facts["facts"]['dei'].keys())
    
    # print(facts["facts"]['invest'].keys())
    if "us-gaap" not in facts["facts"]:
        if "ifrs-full" in facts["facts"]:
            facts = facts["facts"]["ifrs-full"]
            # print(facts.keys())
            # os._exit(0)
        else:
            print("no us-gaap record")
            os._exit(0)
    else:
        facts = facts["facts"]["us-gaap"]
    df_list = []
    # print("\nassets")
    assets = get_assets(facts)
    df_list.append(assets)
    # print("\nlong term debt")
    longtermdebt = get_longtermdebt(facts)
    df_list.append(longtermdebt)
    # print("\nliabilities")
    liabilities = get_Liabilities(facts)    
    df_list.append(liabilities)
    # print("\nStockholders Equity")
    shequity = get_shequity(facts)
    df_list.append(shequity)
    # print("\nnet income")
    netincome = get_netincome(facts)   
    df_list.append(netincome) 
    # print("\noperating income")
    operatingincome = get_operatingincome(facts)    
    df_list.append(operatingincome)
    # print("\nrevenue")
    revenue = get_revenue(facts)
    # print("\nrevenue old")
    revenueold = get_revenue_old(facts)    
    # print("\ngross profit")
    grossprofit = get_grossprofit(facts)
    # print("\ncost of revenue")
    costofrevenue = get_costofrevenue(facts)
    # print("\nconbined revennue")
    conbinedevenue = combine_financial_statements(revenue[0],revenue[1],grossprofit[0],grossprofit[1],costofrevenue[0],costofrevenue[1],revenueold[0],revenueold[1])
    # print(conbinedevenue)
    # print("\ncash for operating ")
    cash = get_cash(facts)
    df_list.append(cash)
    # df_list.append((None,None))
    # print("\neps basic")
    epsbasic = get_eps_basic(facts)
    df_list.append(epsbasic)
    # print("\nnumber of shares")
    wshareoutstanding = get_shareoutstanding(facts)
    df_list.append(wshareoutstanding)
    total = merge_full_financial_statements(conbinedevenue,df_list)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)  # Adjusts width to fit all columns
    # print("\ntotal table")
    # Define how many columns to print per line
    # columns_per_line = 8

    # Iterate over column groups
    # for i in range(0, len(total.columns), columns_per_line):
    #     subset_cols = total.columns[i:i + columns_per_line]
    #     print(f"\nColumns {i+1} to {i+len(subset_cols)}:")
    #     print(total[subset_cols])

    return total

if __name__ == '__main__':
    # # finding companies containing substring in company name
    # substring = "apple inc"
    # print(cik_df[cik_df["name"].str.contains(substring, case=False)])

    # CIK_df = get_cik_df()
    # # print(CIK_df)
    # # save_2_mysql(CIK_df)
    # # finding company row with given ticker
    # # ticker = "tsm"
    # # print(cik_df[cik_df["ticker"] == ticker])
    # # print(get_cik(ticker, CIK_df))
    # cik = get_cik(ticker, CIK_df)
    # filings = get_company_fillings(cik)
    # print(filings.head(50))
    # c4s = get_company_fillings_filter(filings,"4")
    # print(c4s.head(50))
    # # get_annual_report(cik, c10ks,"2022")    

    # total = get_total_xbs("msft")

    # # from plot_charts import plot_financial_charts_with_bars_and_text_growth_styled
    # # import matplotlib.pyplot as plt
    # # # Generate the charts
    # # fig1, fig2, fig3 = plot_financial_charts_with_bars_and_text_growth_styled(total)

    # # # Display the charts
    # # plt.show()
    # # Ensure 'start' and 'end' are strings in YYYY-MM-DD format
    # total['start'] = pd.to_datetime(total['start']).dt.strftime('%Y-%m-%d')
    # total['end'] = pd.to_datetime(total['end']).dt.strftime('%Y-%m-%d')
    # # Convert DataFrame to JSON string (records orientation)
    # json_string = total.to_json(orient='records', lines=False)

    # # Print or save the JSON string (e.g., to a file or send via API)
    # print(json_string)

    # # Optionally, save to a file for PHP to read
    # with open('financial_data.json', 'w') as f:
    #     f.write(json_string)

    # get_sp500()
    # print(update_findata(1))
    display_findata('AAPL')
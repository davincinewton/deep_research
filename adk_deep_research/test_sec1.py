# 1. Import the library
from edgar import *
from edgar.xbrl import *

# 2. Tell the SEC who you are (required by SEC regulations)
set_identity("your.name@example.com")  # Replace with your email

# 3. Find a company
company = Company("TSLA")  # Microsoft

# 4. Get company filings
filings = company.get_filings(form="10-K",amendments=True).latest(5)
xbs = XBRLS.from_filings(filings)
income_statement = xbs.statements.income_statement()
income_df = income_statement.to_dataframe()
print(income_df)

# # 5. Filter by form 
# insider_filings = filings.filter(form="4")  # Insider transactions

# # 6. Get the latest filing
# insider_filing = insider_filings[0]

# # 7. Convert to a data object
# ownership = insider_filing.obj()
# print(ownership)


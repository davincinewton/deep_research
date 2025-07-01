import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.container import BarContainer # Import BarContainer for type checking
import re # For regular expression to find currency unit

def plot_financial_charts_with_bars_and_text_growth_styled(df: pd.DataFrame):
    """
    Generates three matplotlib figures showing financial data as bars and their growth rates as text,
    applying specific styling and dynamically detecting currency units from column names.
    Gracefully handles columns with all NaN values or missing columns.

    Chart 1: Revenue, Gross Profit, Profit (bars) & their growth rates (text on bars).
    Chart 2: Equity, Liabilities, Cash (bars) & their growth rates (text on bars).
    Chart 3: EPS (left y-axis bars), Number of Shares (right y-axis bars)
             & their growth rates (text on respective bars).

    Args:
        df (pd.DataFrame): A pandas DataFrame containing financial data with
                           columns like 'end', 'Revenue XXX', 'Gross Profit XXX',
                           'Profit XXX', 'Equity XXX', 'Liabilities XXX',
                           'Cash XXX', 'EPS XXX/shares', 'Number of Shares shares',
                           where XXX is the currency unit (e.g., 'USD', 'TWD').

    Returns:
        tuple: A tuple containing the three matplotlib Figure objects (fig1, fig2, fig3).
    """

    # --- Pre-processing the DataFrame ---
    df['end_date'] = pd.to_datetime(df['end'])
    df = df.sort_values(by='end_date').reset_index(drop=True)
    df['Year'] = df['end_date'].dt.year

    # --- Define colors based on reference code's palette and common finance elements ---
    # Map base metric names to specific colors for consistency across different currency columns
    BASE_METRIC_COLORS = {
        'Revenue': '#3498db',        # Blue
        'Gross Profit': '#A9A9A9',   # Light Grey (Adjusted to match provided image's Gross Profit bar color)
        'Profit': '#9b59b6',         # Purple (consistent with reference 'Net Income')
        'Equity': '#1abc9c',         # Turquoise (distinct for balance sheet)
        'Liabilities': '#e74c3c',    # Red (often associated with liabilities/debt, also negative growth color)
        'Cash': '#f1c40f',           # Yellow (distinct for cash)
        'EPS': '#3498db',            # Blue (for EPS on primary axis)
        'Number of Shares': '#9b59b6', # Purple (for Number of Shares on secondary axis)
    }

    # Growth rate text colors (from reference)
    GROWTH_POSITIVE_COLOR = '#27ae60' # Green
    GROWTH_NEGATIVE_COLOR = '#e74c3c' # Red

    # Map common currency codes to symbols for display (symbols for labels, not tick format)
    CURRENCY_SYMBOLS = {
        'USD': '$',
        'TWD': 'NT$',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥',
        'CNY': '¥', # Chinese Yuan
        # Add more as needed
    }

    # --- Define the sets of columns for each chart using BASE names ---
    # These are conceptual names, actual column names will be resolved dynamically
    chart_configs = [
        {
            'title': 'Revenue, Gross Profit, and Profit & Growth Rates',
            'value_cols_base': ['Revenue', 'Gross Profit', 'Profit'],
            'y_axis_label_template': '{}', # Changed: Just the currency unit (e.g., 'TWD')
            'y_formatter_template': 'yes' # Changed: Raw number, comma, 2 decimal places
            # 'y_formatter_template': lambda x, p: f'{x/1e9:.1f}B' # Format in Billions with dollar sign
        },
        {
            'title': 'Equity, Liabilities, and Cash & Growth Rates',
            'value_cols_base': ['Equity', 'Liabilities', 'Cash'],
            'y_axis_label_template': '{}', # Changed: Just the currency unit (e.g., 'TWD')
            'y_formatter_template': 'yes' # Changed: Raw number, comma, 2 decimal places
            # 'y_formatter_template': lambda x, p: f'{x/1e9:.1f}B' # Format in Billions with dollar sign
        },
        {
            'title': 'EPS and Number of Shares & Growth Rates',
            'value_cols_base': ['EPS', 'Number of Shares'], # Note: 'Number of Shares' often doesn't have currency suffix
            'y_axis_label_template': 'EPS ({}/share)', # Placeholder for currency/unit
            # 'y_formatter_template': lambda symbol, x, p: f'{x:,.2f}' # EPS is formatted differently, symbol mostly for label
        }
    ]

    figures = [] # List to store the generated figure objects

    # --- Pre-calculate actual column names and currency for each config ---
    processed_chart_configs = []
    for config in chart_configs:
        current_config = config.copy()
        
        # Try to detect a common currency unit from existing column names
        detected_currency_unit = None
        
        # Prioritize finding currency from money columns first (e.g., "Revenue TWD")
        for base_name in ['Revenue', 'Gross Profit', 'Profit', 'Equity', 'Liabilities', 'Cash', 'Assets', 'Long Term Debt']:
            for col_name in df.columns:
                if col_name.startswith(base_name + ' ') and len(col_name.split()) > 1:
                    potential_unit = col_name.split()[-1]
                    if potential_unit in CURRENCY_SYMBOLS or potential_unit.isalpha() and len(potential_unit) == 3:
                        detected_currency_unit = potential_unit
                        break
            if detected_currency_unit:
                break
        
        # If not found from money columns, try from EPS (e.g., EPS TWD/shares)
        if detected_currency_unit is None:
            eps_col_candidates = [col for col in df.columns if col.startswith('EPS ') and '/shares' in col]
            if eps_col_candidates:
                match = re.search(r'EPS (\w+)/shares', eps_col_candidates[0])
                if match:
                    detected_currency_unit = match.group(1)

        # Fallback: If no currency unit was found, default to 'USD'
        if detected_currency_unit is None:
            detected_currency_unit = 'USD'
            print(f"No specific currency unit detected in column names. Defaulting to '{detected_currency_unit}'.")
        # print(detected_currency_unit)
        currency_symbol = CURRENCY_SYMBOLS.get(detected_currency_unit, detected_currency_unit) # Use code if no common symbol
        # print(currency_symbol)

        actual_value_cols = []
        for base_name in config['value_cols_base']:
            actual_col_name = None
            
            # Formulate potential column names based on detected currency and common patterns
            potential_names = []
            if base_name == 'EPS':
                potential_names = [f'EPS {detected_currency_unit}/shares'] + [f'EPS {c}/shares' for c in CURRENCY_SYMBOLS.keys() if c != detected_currency_unit] + ['EPS USD/shares', 'EPS']
            elif base_name == 'Number of Shares':
                potential_names = ['Number of Shares shares', 'Number of Shares']
            else: # For other financial metrics (Revenue, Profit, Assets, Liabilities, Cash, etc.)
                potential_names = [f"{base_name} {detected_currency_unit}"]
                # Also check for base name followed by any known currency, or just the base name itself
                potential_names.extend([f"{base_name} {c}" for c in CURRENCY_SYMBOLS.keys() if f"{base_name} {c}" != potential_names[0]])
                potential_names.append(base_name) # Fallback to just the base name if no unit suffix

            for pn in potential_names:
                if pn in df.columns:
                    actual_col_name = pn
                    break
            
            if actual_col_name:
                actual_value_cols.append(actual_col_name)
            else:
                print(f"Warning: Could not find a suitable column for base metric '{base_name}' in the DataFrame. Skipping for this chart.")

        current_config['actual_value_cols'] = actual_value_cols
        current_config['currency_unit'] = detected_currency_unit
        current_config['currency_symbol'] = currency_symbol
        
        # Finalize y_axis_label and y_formatter using detected info
        if 'y_axis_label_template' in config:
            # For EPS, the label will specifically say "EPS (TWD/share)"
            if config['y_axis_label_template'] == 'EPS ({}/share)':
                current_config['y_axis_label'] = config['y_axis_label_template'].format(detected_currency_unit)
            else: # For other charts, just the currency unit
                current_config['y_axis_label'] = config['y_axis_label_template'].format(detected_currency_unit)
        else:
             current_config['y_axis_label'] = 'Value' # Fallback
        
        if 'y_formatter_template' in config:
            # This is a lambda that expects symbol, x, p.
            # We'll 'bind' the currency_symbol here, so the stored lambda only needs x, p.
            # current_config['y_formatter'] = lambda x, p: config['y_formatter_template'](x, p)
            current_config['y_formatter'] = lambda x, p: f'{currency_symbol}{x/1e9:.1f}B'
            # print(1)
        else:
            current_config['y_formatter'] = lambda x, p: f'{x:,.1f}' # Fallback
            # print(2)
        
        processed_chart_configs.append(current_config)

    # --- Helper function to create a single chart ---
    def _create_chart(data, config):
        # print(str(config['y_formatter']))
        fig, ax1 = plt.subplots(figsize=(12, 8))

        is_chart_3 = (config['title'] == 'EPS and Number of Shares & Growth Rates')
        ax2_for_shares = None 
        if is_chart_3:
            ax2_for_shares = ax1.twinx()

        legend_handles = []
        legend_labels = []

        # Filter out actual columns that were not found in the dataframe or are all NaN
        valid_cols_for_plotting = [col for col in config['actual_value_cols'] 
                                    if col in data.columns and not data[col].dropna().empty]
        num_valid_cols = len(valid_cols_for_plotting) 
        
        bar_group_total_width = 0.8
        
        if num_valid_cols > 0:
            bar_width = bar_group_total_width / num_valid_cols 
        else:
            bar_width = 0 # No bars if no columns

        x_years = data['Year'].unique()
        x_base_positions = np.arange(len(x_years)) 
        group_center_offset = (num_valid_cols - 1) / 2 * bar_width


        # Use an internal counter for plotting position, since some columns might be skipped
        plot_idx = 0 
        for col in valid_cols_for_plotting: # Iterate over *actual* column names that were confirmed to be valid
            # Extract base name from actual column name for color lookup and cleaner legend
            base_name = col.replace(f' {config["currency_unit"]}', '').replace('/shares', '').strip()
            # Special handling for 'Number of Shares shares'
            if base_name == 'Number of Shares shares':
                base_name = 'Number of Shares'

            x_bar_positions = np.array([x_base_positions[np.where(x_years == year)[0][0]] for year in data['Year']])
            x_pos = x_bar_positions - group_center_offset + plot_idx * bar_width # Use plot_idx for consistent spacing

            target_axis = ax1
            current_label = base_name # Use base name for cleaner legend
            bar_color = BASE_METRIC_COLORS.get(base_name, '#888888') # Get color by base name

            if is_chart_3 and base_name == 'Number of Shares': # Check base name for special logic
                target_axis = ax2_for_shares
                bar_container = target_axis.bar(x_pos, data[col], width=bar_width,
                                                label=current_label,
                                                color=bar_color, alpha=0.8)
                
                if bar_container.patches:
                    legend_handles.append(bar_container.patches[0])
                    legend_labels.append(current_label)

                # Set ax2_for_shares specific properties for 'Number of Shares'
                target_axis.set_ylabel('Number of Shares (Billions)', fontsize=12, fontweight='bold', color='purple')
                target_axis.tick_params(axis='y', labelcolor='purple')
                target_axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1e9:.1f}B'))
            else:
                bar_container = target_axis.bar(x_pos, data[col], width=bar_width,
                                                label=current_label,
                                                color=bar_color, alpha=0.8)
                
                if bar_container.patches:
                    legend_handles.append(bar_container.patches[0])
                    legend_labels.append(current_label)

            # Calculate and add growth rate as text on top of bars
            growth_rate_series = data[col].pct_change() * 100
            for j, rect in enumerate(bar_container.patches):
                # Ensure the current year's data point has a corresponding growth rate
                # and that the bar actually exists (relevant for years with NaN data points)
                if j < len(growth_rate_series) and not pd.isna(growth_rate_series.iloc[j]) and not pd.isna(data[col].iloc[j]):
                    current_growth_rate = growth_rate_series.iloc[j]
                    
                    y_text_pos = rect.get_height()
                    va = 'bottom' 
                    if y_text_pos < 0: 
                        y_text_pos = rect.get_height()
                        va = 'top'

                    text_color = GROWTH_NEGATIVE_COLOR if current_growth_rate < 0 else GROWTH_POSITIVE_COLOR

                    target_axis.text(rect.get_x() + rect.get_width() / 2, # X position: center of the bar
                                     y_text_pos,                          # Y position: top/bottom of the bar
                                     f'{current_growth_rate:+.1f}%',       # Text to display (e.g., +10.5%, -2.3%)
                                     ha='center', va=va,                  # Horizontal/Vertical alignment
                                     color=text_color, fontsize=8, fontweight='bold', 
                                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))
            
            plot_idx += 1 # Increment plot index only for successfully plotted columns

        # --- Chart Customization ---
        ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax1.set_ylabel(config['y_axis_label'], fontsize=12, fontweight='bold', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax1.set_title(config['title'], fontsize=16, fontweight='bold', pad=20)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        ax1.set_xticks(x_base_positions)
        ax1.set_xticklabels(x_years, rotation=45, ha='right')
        # if is_chart_3:
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(config['y_formatter']))
        # else:
        #     ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1e9:.1f}B'))

        if legend_handles: # Only show legend if there are items to show
            ax1.legend(legend_handles, legend_labels,
                       loc='upper left',       
                       fontsize=10,            
                       frameon=True,           
                       facecolor='white',      
                       edgecolor='black'       
                      )
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        return fig

    # --- Generate each chart ---
    # Loop over the *processed* configurations which now contain actual column names
    for config in processed_chart_configs: 
        # Only create a chart if there are actual columns identified and have non-NaN values for it
        # This check filters out charts where all relevant columns are missing or entirely NaN
        has_valid_data_for_chart = False
        for col_name in config['actual_value_cols']:
            if col_name in df.columns and not df[col_name].dropna().empty:
                has_valid_data_for_chart = True
                break
        
        if has_valid_data_for_chart:
            fig = _create_chart(df, config)
            figures.append(fig)
        else:
            print(f"Skipping chart '{config['title']}' as no relevant data columns with valid data were found.")

    return tuple(figures)

# --- Example Usage with your DataFrame Structure ---
# Example with TWD currency
data_twd = {
    'start': ['2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01', '2024-01-01'],
    'end': ['2015-12-31', '2016-12-31', '2017-12-31', '2018-12-31', '2019-12-31', '2020-12-31', '2021-12-31', '2022-12-31', '2023-12-31', '2024-12-31'],
    'Revenue TWD': [8.434974e+11, 9.479383e+11, 9.774472e+11, 1.031474e+12, 1.069985e+12, 1.339255e+12, 1.587415e+12, 2.263891e+12, 2.161736e+12, 2.894308e+12],
    'Gross Profit TWD': [4.103949e+11, 4.748321e+11, 4.948264e+11, 4.978743e+11, 4.927019e+11, 7.111301e+11, 8.195373e+11, 1.3483548e+12, 1.1751106e+12, 1.6243536e+12],
    'Cost TWD': [4.331176e+11, 4.730771e+11, 4.826162e+11, 5.334875e+11, 5.772835e+11, 6.281247e+11, 7.678777e+11, 9.155365e+11, 9.866252e+11, 1.2699541e+12],
    'Profit TWD': [3.028329e+11, 3.317973e+11, 3.450390e+11, 3.631062e+11, 3.540270e+11, 5.110080e+11, 5.928806e+11, 9.932947e+11, 8.510277e+11, 1.1575239e+12],
    'Operation Income TWD': [3.200478e+11, 3.779578e+11, 3.855592e+11, 3.836235e+11, 3.727011e+11, 5.667837e+11, 6.499809e+11, 1.1212789e+12, 9.214656e+11, 1.3220530e+12],
    'EPS TWD/shares': [11.68, 12.79, 13.30, 14.00, 13.65, 19.70, 22.84, 38.29, 32.85, 44.68],
    'Number of Shares shares': [25930300000, 25930300000, 25930300000, 25930300000, 25930300000, 25930300000, 25930300000, 25929200000, 25929200000, 25927600000],
    'Assets TWD': [np.nan, 1.886297e+12, 1.991732e+12, 2.090031e+12, 2.264725e+12, 2.760600e+12, 3.725302e+12, 4.964459e+12, 5.532197e+12, 6.691765e+12],
    'Long Term Debt TWD': [np.nan, np.nan, np.nan, np.nan, np.nan, 1.967600e+09, 3.309100e+09, 4.760000e+09, 4.383000e+09, 3.182440e+10],
    'Liabilities TWD': [np.nan, 5.264509e+11, 4.972855e+11, 4.289261e+11, 6.503377e+11, 9.248367e+11, 1.573620e+12, 2.046627e+12, 2.078330e+12, 2.412493e+12],
    'Equity TWD': [1.1949701e+12, 1.3598458e+12, 1.4956923e+12, 1.6611051e+12, 1.6143873e+12, 1.8357638e+12, 2.1516825e+12, 2.9178324e+12, 3.4538665e+12, 4.2792716e+12],
    'Cash TWD': [5.626889e+11, 5.412538e+11, 5.533917e+11, 5.778146e+11, 4.553993e+11, 6.601706e+11, 1.0649902e+11, 1.3428141e+12, 1.4654278e+12, 2.1276270e+12]
}
df_twd = pd.DataFrame(data_twd)

# Example with USD currency and a column that will be all NaN for testing
data_usd = {
    'start': ['2006-10-01', '2007-09-30', '2008-09-28'],
    'end': ['2007-09-29', '2008-09-27', '2009-09-26'],
    'Revenue USD': [np.nan, np.nan, np.nan], # This will be skipped in Chart 1
    'Gross Profit USD': [8.152e+09, 1.3197e+10, 1.7222e+10],
    'Profit USD': [3.495e+09, 6.119e+09, 8.235e+09],
    'Equity USD': [1.4531e+10, 2.2297e+10, 3.164e+10],
    'Liabilities USD': [np.nan, 1.1361e+10, 1.1506e+10],
    'Cash USD': [5.47e+09, 9.596e+09, 1.0159e+10],
    'EPS USD/shares': [4.04, 6.94, 9.22],
    'Number of Shares shares': [8.64595e+08, 8.81592e+08, 8.93016e+08],
    'Bogus Metric EUR': [np.nan, np.nan, np.nan] # A completely unused and empty column
}
df_usd = pd.DataFrame(data_usd)


print("Plotting charts for TWD data:")
fig_twd_1, fig_twd_2, fig_twd_3 = plot_financial_charts_with_bars_and_text_growth_styled(df_twd)
plt.show()

print("\nPlotting charts for USD data (with some NaNs):")
fig_usd_1, fig_usd_2, fig_usd_3 = plot_financial_charts_with_bars_and_text_growth_styled(df_usd)
plt.show()
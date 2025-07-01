import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.container import BarContainer # Import BarContainer for type checking
import re # For regular expression to find currency unit
import traceback # Import traceback to print detailed error info

def plot_financial_charts_with_bars_and_text_growth_styled(df: pd.DataFrame):
    """
    Generates three matplotlib figures showing financial data as bars and their growth rates as text,
    applying specific styling and dynamically detecting currency units from column names.
    Gracefully handles columns with all NaN values or missing columns.

    Chart 1: Revenue, Gross Profit, Profit (bars) & their growth rates (text on bars),
             plus Gross Margin and Net Margin (lines on right y-axis, displaying only value).
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
               Returns None if a critical error prevents chart generation.
    """

    try:
        # --- Pre-processing the DataFrame ---
        df_processed = df.copy() # Work on a copy to avoid modifying original df
        df_processed['end_date'] = pd.to_datetime(df_processed['end'])
        df_processed = df_processed.sort_values(by='end_date').reset_index(drop=True)
        df_processed['Year'] = df_processed['end_date'].dt.year

        # --- Define colors based on reference code's palette and common finance elements ---
        BASE_METRIC_COLORS = {
            'Revenue': '#3498db',        # Blue
            'Gross Profit': '#A9A9A9',   # Light Grey
            'Profit': '#9b59b6',         # Purple
            'Gross Margin': '#2ecc71',   # Green for Gross Margin (line)
            'Net Margin': '#f39c12',     # Orange for Net Margin (line)
            'Equity': '#1abc9c',         # Turquoise
            'Liabilities': '#e74c3c',    # Red
            'Cash': '#f1c40f',           # Yellow
            'EPS': '#3498db',            # Blue
            'Number of Shares': '#9b59b6', # Purple
        }

        # Growth rate text colors (from reference)
        GROWTH_POSITIVE_COLOR = '#27ae60' # Green
        GROWTH_NEGATIVE_COLOR = '#e74c3c' # Red
        VALUE_ONLY_COLOR = 'black' # Color for value-only text on the first bar/point

        # Map common currency codes to symbols for display
        CURRENCY_SYMBOLS = {
            'USD': '$',
            'TWD': 'NT$',
            'EUR': '€',
            'GBP': '£',
            'JPY': '¥',
            'CNY': '¥',
            # Add more as needed
        }

        # --- Detect global currency unit early for calculations and dynamic column finding ---
        detected_currency_unit_global = None
        for base_name in ['Revenue', 'Gross Profit', 'Profit', 'Equity', 'Liabilities', 'Cash', 'Assets', 'Long Term Debt']:
            for col_name in df_processed.columns:
                if col_name.startswith(base_name + ' ') and len(col_name.split()) > 1:
                    potential_unit = col_name.split()[-1]
                    if potential_unit in CURRENCY_SYMBOLS or (potential_unit.isalpha() and len(potential_unit) == 3):
                        detected_currency_unit_global = potential_unit
                        break
            if detected_currency_unit_global:
                break
        if detected_currency_unit_global is None:
            eps_col_candidates = [col for col in df_processed.columns if col.startswith('EPS ') and '/shares' in col]
            if eps_col_candidates:
                match = re.search(r'EPS (\w+)/shares', eps_col_candidates[0])
                if match:
                    detected_currency_unit_global = match.group(1)
        if detected_currency_unit_global is None:
            detected_currency_unit_global = 'USD'
            print(f"No specific currency unit detected in column names. Defaulting to '{detected_currency_unit_global}'.")

        # --- Calculate new margin columns (Gross Margin, Net Margin) ---
        # Find the actual revenue, gross profit, and profit columns using detected currency
        revenue_col_name = next((col for col in df_processed.columns if col == f'Revenue {detected_currency_unit_global}'),
                                next((col for col in df_processed.columns if col.startswith('Revenue ')), 'Revenue'))
        gross_profit_col_name = next((col for col in df_processed.columns if col == f'Gross Profit {detected_currency_unit_global}'),
                                     next((col for col in df_processed.columns if col.startswith('Gross Profit ')), 'Gross Profit'))
        profit_col_name = next((col for col in df_processed.columns if col == f'Profit {detected_currency_unit_global}'),
                               next((col for col in df_processed.columns if col.startswith('Profit ')), 'Profit'))


        if revenue_col_name in df_processed.columns and not df_processed[revenue_col_name].dropna().empty:
            if gross_profit_col_name in df_processed.columns and not df_processed[gross_profit_col_name].dropna().empty:
                df_processed['Gross Margin %'] = df_processed[gross_profit_col_name].div(df_processed[revenue_col_name]).replace([np.inf, -np.inf], np.nan) * 100
            else:
                df_processed['Gross Margin %'] = np.nan
                print(f"Warning: Gross Profit column '{gross_profit_col_name}' missing or empty, skipping Gross Margin calculation.")

            if profit_col_name in df_processed.columns and not df_processed[profit_col_name].dropna().empty:
                df_processed['Net Margin %'] = df_processed[profit_col_name].div(df_processed[revenue_col_name]).replace([np.inf, -np.inf], np.nan) * 100
            else:
                df_processed['Net Margin %'] = np.nan
                print(f"Warning: Profit column '{profit_col_name}' missing or empty, skipping Net Margin calculation.")
        else:
            df_processed['Gross Margin %'] = np.nan
            df_processed['Net Margin %'] = np.nan
            print(f"Warning: Revenue column '{revenue_col_name}' missing or empty, skipping all margin calculations for this chart.")

        # --- Define the sets of columns for each chart using BASE names ---
        chart_configs = [
            {
                'title': 'Revenue, Gross Profit, Profit & Margins, and Growth Rates',
                'value_cols_base': ['Revenue', 'Gross Profit', 'Profit', 'Gross Margin', 'Net Margin'],
                'y_axis_label_template_ax1': '{}', # For currency values
                'y_formatter_template_ax1': 'money_billions' # For primary axis (currency values)
            },
            {
                'title': 'Equity, Liabilities, and Cash & Growth Rates',
                'value_cols_base': ['Equity', 'Liabilities', 'Cash'],
                'y_axis_label_template_ax1': '{}',
                'y_formatter_template_ax1': 'money_billions'
            },
            {
                'title': 'EPS and Number of Shares & Growth Rates',
                'value_cols_base': ['EPS', 'Number of Shares'],
                'y_axis_label_template_ax1': 'EPS ({}/share)',
                'y_formatter_template_ax1': 'eps_shares'
            }
        ]

        figures = [] # List to store the generated figure objects

        # --- Process chart configurations ---
        processed_chart_configs = []
        for config in chart_configs:
            current_config = config.copy()
            current_config['currency_unit'] = detected_currency_unit_global
            current_config['currency_symbol'] = CURRENCY_SYMBOLS.get(detected_currency_unit_global, detected_currency_unit_global)
            
            actual_value_cols = []
            for base_name in config['value_cols_base']:
                actual_col_name = None
                
                # Special handling for calculated margin columns
                if base_name == 'Gross Margin':
                    actual_col_name = 'Gross Margin %'
                elif base_name == 'Net Margin':
                    actual_col_name = 'Net Margin %'
                elif base_name == 'EPS':
                    potential_names = [f'EPS {detected_currency_unit_global}/shares'] + [f'EPS {c}/shares' for c in CURRENCY_SYMBOLS.keys() if c != detected_currency_unit_global] + ['EPS USD/shares', 'EPS']
                    for pn in potential_names:
                        if pn in df_processed.columns:
                            actual_col_name = pn
                            break
                elif base_name == 'Number of Shares':
                    potential_names = ['Number of Shares shares', 'Number of Shares']
                    for pn in potential_names:
                        if pn in df_processed.columns:
                            actual_col_name = pn
                            break
                else: # For other financial metrics (Revenue, Profit, etc.)
                    potential_names = [f"{base_name} {detected_currency_unit_global}"]
                    potential_names.extend([f"{base_name} {c}" for c in CURRENCY_SYMBOLS.keys() if f"{base_name} {c}" != potential_names[0]])
                    potential_names.append(base_name)
                    for pn in potential_names:
                        if pn in df_processed.columns:
                            actual_col_name = pn
                            break
                
                if actual_col_name and actual_col_name in df_processed.columns:
                    actual_value_cols.append(actual_col_name)
                else:
                    pass

            current_config['actual_value_cols'] = actual_value_cols
            
            # Set up y-axis formatter for ticker for ax1
            if config['y_formatter_template_ax1'] == 'money_billions':
                current_config['y_formatter_ax1'] = lambda x, p: f'{current_config["currency_symbol"]}{x/1e9:.1f}B'
                current_config['y_axis_label_ax1'] = config['y_axis_label_template_ax1'].format(detected_currency_unit_global) + ' (Billions)'
            elif config['y_formatter_template_ax1'] == 'eps_shares':
                current_config['y_formatter_ax1'] = lambda x, p: f'{x:,.2f}'
                current_config['y_axis_label_ax1'] = config['y_axis_label_template_ax1'].format(detected_currency_unit_global)
            else: # Fallback
                current_config['y_formatter_ax1'] = lambda x, p: f'{x:,.1f}'
                current_config['y_axis_label_ax1'] = config['y_axis_label_template_ax1']

            processed_chart_configs.append(current_config)

        # --- Helper function to create a single chart ---
        def _create_chart(data, config):
            fig, ax1 = plt.subplots(figsize=(12, 8))

            is_chart_3 = (config['title'] == 'EPS and Number of Shares & Growth Rates')
            is_chart_1 = (config['title'] == 'Revenue, Gross Profit, Profit & Margins, and Growth Rates')

            ax2_for_shares = None
            ax2_for_margins = None

            if is_chart_3:
                ax2_for_shares = ax1.twinx()
            elif is_chart_1:
                ax2_for_margins = ax1.twinx() # New twinx for margins

            # Helper function for bar/line value text formatting
            def format_value_for_annotation(value, col_name, currency_symbol):
                """Formats the value for text annotation based on the column type and magnitude."""
                if pd.isna(value):
                    return ""
                
                if 'Margin %' in col_name:
                    return f'{value:,.1f}%'
                elif 'EPS' in col_name:
                    return f'{value:,.2f}'
                elif 'Number of Shares' in col_name:
                    if abs(value) >= 1e9: return f'{value/1e9:,.1f}B'
                    elif abs(value) >= 1e6: return f'{value/1e6:,.1f}M'
                    else: return f'{value:,.0f}'
                else: # Currency values
                    if abs(value) >= 1e12: return f'{currency_symbol}{value/1e12:,.1f}T'
                    elif abs(value) >= 1e9: return f'{currency_symbol}{value/1e9:,.1f}B'
                    elif abs(value) >= 1e6: return f'{currency_symbol}{value/1e6:,.1f}M'
                    elif abs(value) >= 1e3: return f'{currency_symbol}{value/1e3:,.1f}K'
                    else: return f'{currency_symbol}{value:,.0f}'
            
            legend_handles = []
            legend_labels = []

            valid_cols_for_plotting = [col for col in config['actual_value_cols'] 
                                        if col in data.columns and not data[col].dropna().empty]
            
            bar_group_total_width = 0.8
            
            # Determine which columns are bars and which are lines
            bar_cols = [col for col in valid_cols_for_plotting if 'Margin %' not in col]
            line_cols = [col for col in valid_cols_for_plotting if 'Margin %' in col]

            # Calculate bar width based on number of bar columns
            num_bar_cols = len(bar_cols)
            bar_width = bar_group_total_width / num_bar_cols if num_bar_cols > 0 else 0.0 # Ensure float zero

            x_years = data['Year'].unique()
            x_base_positions = np.arange(len(x_years)) 
            
            # Plot bars first (Revenue, Gross Profit, Profit)
            if num_bar_cols > 0:
                bar_group_center_offset = (num_bar_cols - 1) / 2 * bar_width
                bar_plot_idx = 0 
                for col in bar_cols:
                    base_name = col.replace(f' {config["currency_unit"]}', '').replace('/shares', '').strip()
                    if base_name == 'Number of Shares shares':
                        base_name = 'Number of Shares'

                    x_bar_positions_for_year = np.array([x_base_positions[np.where(x_years == year)[0][0]] for year in data['Year']])
                    x_pos = x_bar_positions_for_year - bar_group_center_offset + bar_plot_idx * bar_width

                    target_axis = ax1
                    if is_chart_3 and base_name == 'Number of Shares': # Chart 3 specific logic
                        target_axis = ax2_for_shares
                        target_axis.set_ylabel('Number of Shares (Billions)', fontsize=12, fontweight='bold', color='purple')
                        target_axis.tick_params(axis='y', labelcolor='purple')
                        target_axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1e9:.1f}B'))

                    bar_container = target_axis.bar(x_pos, data[col], width=bar_width,
                                                    label=base_name,
                                                    color=BASE_METRIC_COLORS.get(base_name, '#888888'), alpha=0.8)
                    
                    if bar_container.patches:
                        legend_handles.append(bar_container.patches[0])
                        legend_labels.append(base_name)

                    # Add text annotations for bars
                    growth_rate_series = data[col].pct_change() * 100
                    for j, rect in enumerate(bar_container.patches):
                        current_value = data[col].iloc[j]
                        if pd.isna(current_value): continue

                        text_to_display = format_value_for_annotation(current_value, col, config['currency_symbol'])
                        text_color_for_annotation = VALUE_ONLY_COLOR

                        # Display growth rate from the second bar onwards (j > 0)
                        if j > 0 and not pd.isna(growth_rate_series.iloc[j]):
                            current_growth_rate = growth_rate_series.iloc[j]
                            text_color_for_annotation = GROWTH_NEGATIVE_COLOR if current_growth_rate < 0 else GROWTH_POSITIVE_COLOR
                            text_to_display = f'{text_to_display}\n{current_growth_rate:+.1f}%'
                        
                        y_text_pos = rect.get_y() + rect.get_height()
                        va = 'bottom'
                        if current_value < 0:
                            y_text_pos = rect.get_y()
                            va = 'top'
                        
                        target_axis.text(rect.get_x() + rect.get_width() / 2, y_text_pos,
                                         text_to_display, ha='center', va=va,
                                         color=text_color_for_annotation, fontsize=8, fontweight='bold', 
                                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))
                    
                    bar_plot_idx += 1


            # Plot lines for margins if it's Chart 1
            if is_chart_1 and ax2_for_margins and line_cols:
                # Set up ax2 for margins
                ax2_for_margins.set_ylabel('Margin (%)', fontsize=12, fontweight='bold', color='#f39c12') # Orange
                ax2_for_margins.tick_params(axis='y', labelcolor='#f39c12')
                ax2_for_margins.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))

                for col in line_cols:
                    base_name = col.replace(' %', '') # 'Gross Margin %' -> 'Gross Margin'
                    line_color = BASE_METRIC_COLORS.get(base_name, '#888888')

                    line, = ax2_for_margins.plot(x_base_positions, data[col], 
                                                 label=base_name, 
                                                 color=line_color, 
                                                 marker='o', linestyle='-', linewidth=2, alpha=0.8)
                    legend_handles.append(line)
                    legend_labels.append(base_name)

                    # Add text annotations for lines
                    # For margins, we ONLY display the value, not the growth rate.
                    for j, current_value in enumerate(data[col]):
                        if pd.isna(current_value): continue

                        text_to_display = format_value_for_annotation(current_value, col, config['currency_symbol'])
                        text_color_for_annotation = VALUE_ONLY_COLOR # Always black for margins

                        # --- Adjusted offset for line annotations: Reduced value for closer placement ---
                        y_offset_abs = 0.5 # Fixed offset in percentage points (e.g., 0.5% above/below the point)
                        va_align = 'bottom' 
                        if current_value < 0: # If value is negative, place text below
                            y_offset_abs = -y_offset_abs
                            va_align = 'top'
                            
                        ax2_for_margins.text(x_base_positions[j], current_value + y_offset_abs,
                                             text_to_display, ha='center', va=va_align,
                                             color=text_color_for_annotation, fontsize=8, fontweight='bold', 
                                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))

            # --- Chart Customization ---
            ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax1.set_ylabel(config['y_axis_label_ax1'], fontsize=12, fontweight='bold', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            ax1.set_title(config['title'], fontsize=16, fontweight='bold', pad=20)
            ax1.grid(axis='y', linestyle='--', alpha=0.7)

            ax1.set_xticks(x_base_positions)
            ax1.set_xticklabels(x_years, rotation=45, ha='right')
            
            ax1.yaxis.set_major_formatter(ticker.FuncFormatter(config['y_formatter_ax1']))

            if ax2_for_shares:
                ax2_for_shares.autoscale_view()
                ax2_for_shares.grid(False)
            if ax2_for_margins:
                ax2_for_margins.autoscale_view()
                ax2_for_margins.grid(False)

            if legend_handles:
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
        for config in processed_chart_configs: 
            has_valid_data_for_chart = False
            for col_name in config['actual_value_cols']:
                if col_name in df_processed.columns and not df_processed[col_name].dropna().empty:
                    has_valid_data_for_chart = True
                    break
            
            if has_valid_data_for_chart:
                fig = _create_chart(df_processed, config)
                figures.append(fig)
            else:
                print(f"Skipping chart '{config['title']}' as no relevant data columns with valid data were found.")

        return tuple(figures)

    except Exception as e:
        print(f"FATAL ERROR: An unexpected error occurred in plot_financial_charts_with_bars_and_text_growth_styled:")
        traceback.print_exc()
        return None

# # --- Example Usage with your DataFrame Structure ---
# # Example with TWD currency
# data_twd = {
#     'start': ['2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01', '2024-01-01'],
#     'end': ['2015-12-31', '2016-12-31', '2017-12-31', '2018-12-31', '2019-12-31', '2020-12-31', '2021-12-31', '2022-12-31', '2023-12-31', '2024-12-31'],
#     'Revenue TWD': [8.434974e+11, 9.479383e+11, 9.774472e+11, 1.031474e+12, 1.069985e+12, 1.339255e+12, 1.587415e+12, 2.263891e+12, 2.161736e+12, 2.894308e+12],
#     'Gross Profit TWD': [4.103949e+11, 4.748321e+11, 4.948264e+11, 4.978743e+11, 4.927019e+11, 7.111301e+11, 8.195373e+11, 1.3483548e+12, 1.1751106e+12, 1.6243536e+12],
#     'Cost TWD': [4.331176e+11, 4.730771e+11, 4.826162e+11, 5.334875e+11, 5.772835e+11, 6.281247e+11, 7.678777e+11, 9.155365e+11, 9.866252e+11, 1.2699541e+12],
#     'Profit TWD': [3.028329e+11, 3.317973e+11, 3.450390e+11, 3.631062e+11, 3.540270e+11, 5.110080e+11, 5.928806e+11, 9.932947e+11, 8.510277e+11, 1.1575239e+12],
#     'Operation Income TWD': [3.200478e+11, 3.779578e+11, 3.855592e+11, 3.836235e+11, 3.727011e+11, 5.667837e+11, 6.499809e+11, 1.1212789e+12, 9.214656e+11, 1.3220530e+12],
#     'EPS TWD/shares': [11.68, 12.79, 13.30, 14.00, 13.65, 19.70, 22.84, 38.29, 32.85, 44.68],
#     'Number of Shares shares': [25930300000, 25930300000, 25930300000, 25930300000, 25930300000, 25930300000, 25930300000, 25929200000, 25929200000, 25927600000],
#     'Assets TWD': [np.nan, 1.886297e+12, 1.991732e+12, 2.090031e+12, 2.264725e+12, 2.760600e+12, 3.725302e+12, 4.964459e+12, 5.532197e+12, 6.691765e+12],
#     'Long Term Debt TWD': [np.nan, np.nan, np.nan, np.nan, np.nan, 1.967600e+09, 3.309100e+09, 4.760000e+09, 4.383000e+09, 3.182440e+10],
#     'Liabilities TWD': [np.nan, 5.264509e+11, 4.972855e+11, 4.289261e+11, 6.503377e+11, 9.248367e+11, 1.573620e+12, 2.046627e+12, 2.078330e+12, 2.412493e+12],
#     'Equity TWD': [1.1949701e+12, 1.3598458e+12, 1.4956923e+12, 1.6611051e+12, 1.6143873e+12, 1.8357638e+12, 2.1516825e+12, 2.9178324e+12, 3.4538665e+12, 4.2792716e+12],
#     'Cash TWD': [5.626889e+11, 5.412538e+11, 5.533917e+11, 5.778146e+11, 4.553993e+11, 6.601706e+11, 1.0649902e+11, 1.3428141e+12, 1.4654278e+12, 2.1276270e+12]
# }
# df_twd = pd.DataFrame(data_twd)

# # Example with USD currency and a column that will be all NaN for testing
# data_usd = {
#     'start': ['2006-10-01', '2007-09-30', '2008-09-28'],
#     'end': ['2007-09-29', '2008-09-27', '2009-09-26'],
#     'Revenue USD': [8.0e9, 1.0e10, 1.2e10], # Make revenue not NaN for margin calc
#     'Gross Profit USD': [2.0e9, 3.0e9, 4.0e9],
#     'Profit USD': [1.0e9, 1.5e9, 2.0e9],
#     'Equity USD': [1.4531e+10, 2.2297e+10, 3.164e+10],
#     'Liabilities USD': [np.nan, 1.1361e+10, 1.1506e+10],
#     'Cash USD': [5.47e+09, 9.596e+09, 1.0159e+10],
#     'EPS USD/shares': [4.04, 6.94, 9.22],
#     'Number of Shares shares': [8.64595e+08, 8.81592e+08, 8.93016e+08],
#     'Bogus Metric EUR': [np.nan, np.nan, np.nan] # A completely unused and empty column
# }
# df_usd = pd.DataFrame(data_usd)

# # Example with all NaN revenue for margin test
# data_nan_revenue = {
#     'start': ['2020-01-01', '2021-01-01'],
#     'end': ['2020-12-31', '2021-12-31'],
#     'Revenue USD': [np.nan, np.nan],
#     'Gross Profit USD': [100, 120],
#     'Profit USD': [50, 60],
#     'EPS USD/shares': [1.0, 1.2],
#     'Number of Shares shares': [100000000, 100000000]
# }
# df_nan_revenue = pd.DataFrame(data_nan_revenue)

# # Example with zero revenue for margin test
# data_zero_revenue = {
#     'start': ['2020-01-01', '2021-01-01', '2022-01-01'],
#     'end': ['2020-12-31', '2021-12-31', '2022-12-31'],
#     'Revenue USD': [1000, 0, 1200],
#     'Gross Profit USD': [200, 0, 300],
#     'Profit USD': [100, 0, 150],
#     'EPS USD/shares': [1.0, 0, 1.2],
#     'Number of Shares shares': [100000000, 100000000, 100000000]
# }
# df_zero_revenue = pd.DataFrame(data_zero_revenue)


# print("Plotting charts for TWD data:")
# result_twd = plot_financial_charts_with_bars_and_text_growth_styled(df_twd)
# if result_twd is not None:
#     fig_twd_1, fig_twd_2, fig_twd_3 = result_twd
#     plt.show()
# else:
#     print("Chart generation failed for TWD data.")


# print("\nPlotting charts for USD data (with some NaNs):")
# result_usd = plot_financial_charts_with_bars_and_text_growth_styled(df_usd)
# if result_usd is not None:
#     fig_usd_1, fig_usd_2, fig_usd_3 = result_usd
#     plt.show()
# else:
#     print("Chart generation failed for USD data.")


# print("\nPlotting charts for data with all NaN Revenue (margins should be NaN):")
# result_nan_revenue = plot_financial_charts_with_bars_and_text_growth_styled(df_nan_revenue)
# if result_nan_revenue is not None:
#     if len(result_nan_revenue) == 3:
#         fig_nan_revenue_1, fig_nan_revenue_2, fig_nan_revenue_3 = result_nan_revenue
#         plt.show()
#     else:
#         print(f"Expected 3 charts, but got {len(result_nan_revenue)}. Displaying generated charts.")
#         for fig in result_nan_revenue:
#             plt.show()
# else:
#     print("Chart generation failed for NaN Revenue data.")


# print("\nPlotting charts for data with zero Revenue (margins should be NaN for that year):")
# result_zero_revenue = plot_financial_charts_with_bars_and_text_growth_styled(df_zero_revenue)
# if result_zero_revenue is not None:
#     if len(result_zero_revenue) == 3:
#         fig_zero_revenue_1, fig_zero_revenue_2, fig_zero_revenue_3 = result_zero_revenue
#         plt.show()
#     else:
#         print(f"Expected 3 charts, but got {len(result_zero_revenue)}. Displaying generated charts.")
#         for fig in result_zero_revenue:
#             plt.show()
# else:
#     print("Chart generation failed for zero Revenue data.")
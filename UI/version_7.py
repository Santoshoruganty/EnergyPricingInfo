import streamlit as st
import pandas as pd
import os
import plotly.express as px
import openai
import google.generativeai as genai
from datetime import datetime, timezone, date, timedelta
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
import requests
import urllib3
from io import StringIO
from groq import Groq
import json
import pathlib
from streamlit_modal import Modal
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go

from eroct_data import get_ercot_data
from isone_data import get_isone_data
from miso_data import get_miso_data
from pjm_date import get_pjm_data
from nyiso_data import get_nyiso_data
from dotenv import load_dotenv
load_dotenv()


st.set_page_config(page_title="Forecast Comparison", layout="wide")

#########################################

st.image("Images/Company_New_1.png", width=300)
st.title("TrueLight Non-Energy News Letter")

# Create base data directory
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def get_cache_path(iso_name, cache_key):
    """Get the cache directory path for ISO and cache_key"""
    return os.path.join(DATA_DIR, iso_name, cache_key)

def save_cache_data(iso_name, cache_key, current_df, previous_df, summaries):
    """Save cache data using cache_key as folder name"""
    cache_dir = get_cache_path(iso_name, cache_key)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Save DataFrames as CSV files
    current_df.to_csv(os.path.join(cache_dir, "current_data.csv"), index=False)
    previous_df.to_csv(os.path.join(cache_dir, "previous_data.csv"), index=False)
    
    # Save summaries as JSON
    with open(os.path.join(cache_dir, "summaries.json"), "w") as f:
        json.dump(summaries, f, indent=2, default=str)

def load_cache_data(iso_name, cache_key):
    """Load cache data from cache_key folder"""
    cache_dir = get_cache_path(iso_name, cache_key)
    
    if not os.path.exists(cache_dir):
        return None
    
    try:
        # Load summaries
        with open(os.path.join(cache_dir, "summaries.json"), "r") as f:
            summaries = json.load(f)
        
        # Load DataFrames
        current_df = pd.read_csv(os.path.join(cache_dir, "current_data.csv"))
        previous_df = pd.read_csv(os.path.join(cache_dir, "previous_data.csv"))
        
        return {
            "summaries": summaries,
            "current_df": current_df,
            "previous_df": previous_df
        }
    except Exception as e:
        st.warning(f"Error loading cache data: {e}")
        return None

def cache_exists(iso_name, cache_key):
    """Check if cache data exists"""
    cache_dir = get_cache_path(iso_name, cache_key)
    return os.path.exists(os.path.join(cache_dir, "summaries.json"))

def get_latest_cache(iso_name):
    """Get the most recent cache for an ISO"""
    iso_dir = os.path.join(DATA_DIR, iso_name)
    if not os.path.exists(iso_dir):
        return None
    
    # Get all cache directories and sort by name (assuming cache_key includes date)
    cache_dirs = [d for d in os.listdir(iso_dir) if os.path.isdir(os.path.join(iso_dir, d))]
    if not cache_dirs:
        return None
    
    # Sort by cache_key and get the latest
    latest_cache_key = sorted(cache_dirs, reverse=True)[0]
    return load_cache_data(iso_name, latest_cache_key)

#########################################

# groq_api_key = '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'
groq_api_key = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key="gsk_GB33plKMiQ2C7YYCEDwZWGdyb3FYFc4ISoZMKthE1gUcofRXzPuby")

def get_groq_summary(prompt):
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def summarize(diff_df, pct_df, label):
    # Limit the data sent to API - take only first 20 rows
    limited_diff = diff_df.head(20) if len(diff_df) > 20 else diff_df
    
    merged = limited_diff.copy()
    for col in limited_diff.columns[1:]:
        if col in pct_df.columns:
            try:
                merged[col + " (%)"] = pct_df.set_index("Curve Start Month").loc[merged["Curve Start Month"], col].values
            except:
                # If indexing fails, skip the percentage column
                pass
    
    # Convert to string and limit length
    data_string = merged.round(2).to_string(index=False)
    
    # If still too long, truncate
    if len(data_string) > 2000:
        data_string = data_string[:2000] + "..."
    
    prompt = f"""
    You are an expert energy analyst. Write **3 crisp bullet points** summarizing trends from the following data. **Do not add any introductions** like "Here is the summary" — only the 3 points.

    Focus on:
    - Market direction (up/down/mixed)
    - Big increases/decreases in categories
    - Keep it professional, direct, and data-driven

    Data for {label}:
    {data_string}
    """
    return get_groq_summary(prompt)

########################################

def display_summaries(summaries):
    cols = st.columns(3)
    for i, (k, v) in enumerate(summaries.items()):
        bullets = [point.strip() for point in v.split("•") if point.strip()]
        bullet_html = "".join(f"<li>{b}</li>" for b in bullets)

        with cols[i % 3]:
            st.markdown(f"""
            <div style='background-color: var(--secondary-background-color); padding: 15px; border-radius: 10px;'>
                <h4>{k}</h4>
                <ul style='padding-left: 1.2em;'>{bullet_html}</ul>
            </div>
            """, unsafe_allow_html=True)

def align_dataframes(current_df, previous_df):
    """Align current and previous dataframes for plotting"""
    curr = current_df.copy()
    prev = previous_df.copy()
    
    # Convert to datetime
    curr['Curve Start Month'] = pd.to_datetime(curr['Curve Start Month'], utc=True)
    prev['Curve Start Month'] = pd.to_datetime(prev['Curve Start Month'], utc=True)
    
    # Shift previous data by one month forward to align
    prev['Curve Start Month'] = prev['Curve Start Month'] + pd.DateOffset(months=1)
    
    # Get cost columns (excluding date columns)
    cost_columns = [col for col in curr.columns if col not in ['Curve Start Month', 'Curve Update Date']]
    
    return curr, prev, cost_columns

def plot_current_vs_previous(current_df, previous_df, iso_name):
    """Plot current vs previous data overlaid"""
    curr, prev, cost_columns = align_dataframes(current_df, previous_df)
    
    if not cost_columns:
        st.warning(f"No cost components found for {iso_name.upper()}")
        return
    
    # Component selector
    selected_component = st.selectbox(
        f"🔍 Select Component for Comparison",
        cost_columns,
        key=f"{iso_name}_component_selector"
    )
    
    # Create the comparison plot
    fig = go.Figure()
    
    # Add current data line
    fig.add_trace(go.Scatter(
        x=curr['Curve Start Month'],
        y=curr[selected_component],
        mode='lines+markers',
        name='Current Forecast',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    # Add previous data line
    fig.add_trace(go.Scatter(
        x=prev['Curve Start Month'],
        y=prev[selected_component],
        mode='lines+markers',
        name='Previous Forecast',
        line=dict(color='red', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=f"📈 {selected_component} - Current vs Previous Forecast ({iso_name.upper()})",
        xaxis_title="Forecast Month",
        yaxis_title="Cost ($)",
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show some basic stats
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Current Forecast Stats")
        current_stats = {
            'Average': f"${curr[selected_component].mean():.2f}",
            'Max': f"${curr[selected_component].max():.2f}",
            'Min': f"${curr[selected_component].min():.2f}"
        }
        for key, value in current_stats.items():
            st.metric(f"Current {key}", value)
    
    with col2:
        st.markdown("### 📊 Previous Forecast Stats")
        previous_stats = {
            'Average': f"${prev[selected_component].mean():.2f}",
            'Max': f"${prev[selected_component].max():.2f}",
            'Min': f"${prev[selected_component].min():.2f}"
        }
        for key, value in previous_stats.items():
            st.metric(f"Previous {key}", value)

def create_volatility_analysis_table(current_df, previous_df, iso_name):
    """Create volatility analysis table using Streamlit table"""
    curr, prev, cost_columns = align_dataframes(current_df, previous_df)
    
    if not cost_columns:
        st.warning(f"No cost components found for {iso_name.upper()}")
        return
    
    # Get curve titles from the data
    current_op_date = current_df.get('Curve Update Date', ['Unknown']).iloc[0] if 'Curve Update Date' in current_df.columns else 'Unknown'
    previous_op_date = previous_df.get('Curve Update Date', ['Unknown']).iloc[0] if 'Curve Update Date' in previous_df.columns else 'Unknown'
    
    # Create the table structure
    st.markdown("---")
    
    # Header with three columns
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("### Summary Analysis")
    with col2:
        st.markdown("### Volatility ($/MWh)")
    with col3:
        st.markdown("### Volatility (%)")
    
    # Curve comparison header
    st.markdown(f"""
    <div style='text-align: center; margin: 20px 0; padding: 15px; background-color: #f0f2f6; border-radius: 5px;'>
        <strong style='color: #1f4e79; font-size: 14px;'>
            {iso_name.upper()} ZONE: {current_op_date}<br>
            vs<br>
            {iso_name.upper()} ZONE: {previous_op_date}
        </strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate volatility for each component using the same logic as summaries
    table_data = []
    
    # Get the start date from the data
    start_date = curr['Curve Start Month'].min()
    
    # Define the same time periods as in calculate_summaries_and_pct
    f_date = curr['Curve Start Month'].min()  # Use actual start date if f_date not available
    l_date = curr['Curve Start Month'].min() + pd.DateOffset(months=1)  # Use 1 month if l_date not available
    
    time_periods = {
        'Prompt Month Price': (f_date, l_date),
        '12 Month Price': (start_date, start_date + pd.DateOffset(months=12)),
        '24 Month Price': (start_date, start_date + pd.DateOffset(months=24)),
        'Cal Strip Price': (pd.Timestamp('2025-01-01', tz='UTC'), pd.Timestamp('2025-12-31', tz='UTC')),
        'Winter Strip Price': ('winter_months', None),  # Special case
        'Summer Strip Price': ('summer_months', None)   # Special case
    }
    
    for component_name, time_filter in time_periods.items():
        # Filter data for this component's time period
        if time_filter[0] == 'winter_months':
            curr_filtered = curr[curr['Curve Start Month'].dt.month.isin([12, 1, 2])]
            prev_filtered = prev[prev['Curve Start Month'].dt.month.isin([12, 1, 2])]
        elif time_filter[0] == 'summer_months':
            curr_filtered = curr[curr['Curve Start Month'].dt.month.isin([6, 7, 8, 9])]
            prev_filtered = prev[prev['Curve Start Month'].dt.month.isin([6, 7, 8, 9])]
        else:
            # Regular time filtering
            start_time, end_time = time_filter
            curr_mask = (curr['Curve Start Month'] >= start_time) & (curr['Curve Start Month'] <= end_time)
            prev_mask = (prev['Curve Start Month'] >= start_time) & (prev['Curve Start Month'] <= end_time)
            curr_filtered = curr.loc[curr_mask]
            prev_filtered = prev.loc[prev_mask]
        
        if len(curr_filtered) > 0 and len(prev_filtered) > 0:
            # Calculate average volatility across all columns for this time period
            total_curr_vol = 0
            total_prev_vol = 0
            
            for col in cost_columns:
                if col in curr_filtered.columns and col in prev_filtered.columns:
                    curr_vol = curr_filtered[col].std()
                    prev_vol = prev_filtered[col].std()
                    total_curr_vol += curr_vol if not pd.isna(curr_vol) else 0
                    total_prev_vol += prev_vol if not pd.isna(prev_vol) else 0
            
            avg_curr_vol = total_curr_vol / len(cost_columns)
            avg_prev_vol = total_prev_vol / len(cost_columns)
            
            # Calculate percentage change
            if avg_prev_vol != 0:
                pct_change = ((avg_curr_vol - avg_prev_vol) / avg_prev_vol) * 100
            else:
                pct_change = 0
                
        else:
            # No data for this time period
            avg_curr_vol = 0
            pct_change = 0
        
        table_data.append({
            'Summary Analysis': component_name,
            'Volatility ($/MWh)': f"${avg_curr_vol:.2f}",
            'Volatility (%)': f"{pct_change:.2f}%"
        })
    
    # Create DataFrame and display as Streamlit table
    df_table = pd.DataFrame(table_data)
    
    st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Summary Analysis": st.column_config.TextColumn("Summary Analysis", width="large"),
            "Volatility ($/MWh)": st.column_config.TextColumn("Volatility ($/MWh)", width="medium"),
            "Volatility (%)": st.column_config.TextColumn("Volatility (%)", width="medium"),
        }
    )

########################################

def filter_timeframe(df, start_date, end_date):
    mask = (df['Curve Start Month'] >= start_date) & (df['Curve Start Month'] <= end_date)
    return df.loc[mask]

def calculate_summaries_and_pct(df1, df2, f_date, l_date):
    """Calculate summaries and return percentage DataFrame for plotting"""
    df1 = df1.copy()
    df2 = df2.copy()
    df1['Curve Start Month'] = pd.to_datetime(df1['Curve Start Month'], utc=True)
    df2['Curve Start Month'] = pd.to_datetime(df2['Curve Start Month'], utc=True)

    cost_columns = df1.columns[2:]
    for col in cost_columns:
        df1[col] = pd.to_numeric(df1[col], errors='coerce')
        df2[col] = pd.to_numeric(df2[col], errors='coerce')

    df1.set_index('Curve Start Month', inplace=True)
    df2.set_index('Curve Start Month', inplace=True)
    df1.index = df1.index + pd.DateOffset(months=1)

    df_diff = df2[cost_columns] - df1[cost_columns]
    
    df_pct = df_diff.copy()
    for col in cost_columns:
        mask_zero = (df1[col] == 0) | (df1[col].isna())
        mask_nonzero = ~mask_zero
        
        if mask_nonzero.any():
            df_pct.loc[mask_nonzero, col] = ((df_diff.loc[mask_nonzero, col] / df1.loc[mask_nonzero, col]) * 100)
        
        if mask_zero.any():
            has_change = df_diff.loc[mask_zero, col].abs() > 0
            df_pct.loc[mask_zero & has_change, col] = 999
            df_pct.loc[mask_zero & ~has_change, col] = 0
    
    df_pct = df_pct.round(2)

    df_diff.reset_index(inplace=True)
    df_pct.reset_index(inplace=True)

    # Get the start date from the data (first available month)
    start_date = df_diff['Curve Start Month'].min()
    
    # Define time periods based on the new logic
    prompt_month = filter_timeframe(df_diff, f_date, l_date)  # Prompt Month Price
    
    # 12 Month Price: First 12 months from start date
    twelve_month_end = start_date + pd.DateOffset(months=12)
    twelve_month = filter_timeframe(df_diff, start_date, twelve_month_end)
    
    # 24 Month Price: First 24 months from start date  
    twenty_four_month_end = start_date + pd.DateOffset(months=24)
    twenty_four_month = filter_timeframe(df_diff, start_date, twenty_four_month_end)
    
    # Cal Strip Price: Calendar year (Jan-Dec 2025)
    cal_strip = filter_timeframe(df_diff, datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2025, 12, 31, tzinfo=timezone.utc))
    
    # Winter Strip Price: Winter months only (Dec, Jan, Feb)
    winter_strip = df_diff[df_diff['Curve Start Month'].dt.month.isin([12, 1, 2])]
    
    # Summer Strip Price: Summer months only (Jun, Jul, Aug, Sep)
    summer_strip = df_diff[df_diff['Curve Start Month'].dt.month.isin([6, 7, 8, 9])]
    
    summaries = {
        "Prompt Month Price": summarize(prompt_month, df_pct, "Prompt Month Price"),
        "12 Month Price": summarize(twelve_month, df_pct, "12 Month Price"),
        "24 Month Price": summarize(twenty_four_month, df_pct, "24 Month Price"),
        "Cal Strip Price": summarize(cal_strip, df_pct, "Cal Strip Price"),
        "Winter Strip Price": summarize(winter_strip, df_pct, "Winter Strip Price"),
        "Summer Strip Price": summarize(summer_strip, df_pct, "Summer Strip Price")
    }
    
    return summaries

def process_iso_data(iso_name, get_data_func):
    """Process data for selected ISO with simplified cache structure"""
    with st.spinner(f"Fetching {iso_name.upper()} data..."):
        try:
            result = get_data_func()
            date_key = result.get("current_op_date", "unknown")
            cache_key = f"{iso_name}_{date_key}"

            if not result.get("success"):
                st.warning(f"⚠️ {iso_name.upper()} data not available. Showing latest cached data.")
                fallback = get_latest_cache(iso_name)
                if fallback:
                    display_summaries(fallback["summaries"])
                    plot_current_vs_previous(fallback["current_df"], fallback["previous_df"], iso_name)
                    create_volatility_analysis_table(fallback["current_df"], fallback["previous_df"], iso_name)
                else:
                    st.error("❌ No cached data found.")
                return

            if cache_exists(iso_name, cache_key):
                st.success("✅ Using cached data.")
                cached_data = load_cache_data(iso_name, cache_key)
                summaries = cached_data["summaries"]
                current_df = cached_data["current_df"]
                previous_df = cached_data["previous_df"]
                
            else:
                st.warning("📊 New data found. Generating analysis...")
                summaries = calculate_summaries_and_pct(
                    result['previous_df'], 
                    result['current_df'], 
                    result['f_date'], 
                    result['l_date']
                )
                
                # Save cache data
                save_cache_data(
                    iso_name, 
                    cache_key, 
                    result['current_df'], 
                    result['previous_df'], 
                    summaries
                )
                
                current_df = result['current_df']
                previous_df = result['previous_df']

            # Display results
            display_summaries(summaries)
            plot_current_vs_previous(current_df, previous_df, iso_name)
            create_volatility_analysis_table(current_df, previous_df, iso_name)

        except Exception as e:
            st.error(f"Failed to fetch {iso_name.upper()} data: {e}")
            import traceback
            st.error(f"Error details: {traceback.format_exc()}")

########################################
# MAIN UI WITH DROPDOWN
########################################

iso_options = {
    "Select an ISO": None,
    "ERCOT": ("ercot", get_ercot_data),
    "NYISO": ("nyiso", get_nyiso_data), 
    "ISONE": ("isone", get_isone_data),
    "MISO": ("miso", get_miso_data),
    "PJM": ("pjm", get_pjm_data)
}

selected_iso = st.selectbox(
    "🌐 Select ISO for Analysis:",
    list(iso_options.keys()),
    index=0
)

if selected_iso != "Select an ISO":
    iso_name, get_data_func = iso_options[selected_iso]
    
    st.markdown("---")
    st.subheader(f"🔍 {selected_iso} Non-Energy Forecast Analysis")
    
    process_iso_data(iso_name, get_data_func)
    
else:
    st.markdown("""
    ### Welcome to TrueLight Non-Energy Analysis
    
    Please select an ISO from the dropdown above to begin analysis.
    
    **Available ISOs:**
    - **ERCOT** - Electric Reliability Council of Texas
    - **NYISO** - New York Independent System Operator  
    - **ISONE** - ISO New England
    - **MISO** - Midcontinent Independent System Operator
    - **PJM** - PJM Interconnection """
    )

# Footer
st.markdown("""
<hr style='margin-top: 3rem; margin-bottom: 2rem;'>
<div style='display: flex; justify-content: space-between; align-items: flex-start; background-color: black; padding: 2rem;'>
    <div>
        <p style='margin: 0 0 1rem 0; text-decoration: underline;'>
            <a href='https://www.truelightenergy.com/product' target='_blank' style='color: white; text-decoration: underline;'>Product</a>
        </p>
        <p style='margin: 0 0 1rem 0; text-decoration: underline;'>
            <a href='https://www.truelightenergy.com/about-1' target='_blank' style='color: white; text-decoration: underline;'>About</a>
        </p>
        <p style='margin: 0; text-decoration: underline;'>
            <a href='https://www.truelightenergy.com/contact' target='_blank' style='color: white; text-decoration: underline;'>Contact</a>
        </p>
    </div>
    <div style='text-align: right;'>
        <p style='color: white; margin: 0;'>sales@truelightenergy.com</p>
        <p style='color: white; margin: 0;'>(617) 209-2415</p>
        <p style='color: white; margin: 0;'>18 Shipyard Drive, Suite 2A</p>
        <p style='color: white; margin: 0;'>Hingham, MA 02043</p>
        <p style='color: white; margin: 0;'>&copy; 2025 TRUELight Energy</p>
    </div>
</div>
""", unsafe_allow_html=True)

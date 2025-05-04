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


urllib3.disable_warnings()
load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)

def get_groq_summary(prompt):
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",  # Or "mixtral-8x7b-32768"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


# ========== Data Fetching Utilities ==========
def get_latest_friday(today=None):
    today = today or date.today()
    offset = (today.weekday() - 4) % 7  # 4 = Friday
    return today - timedelta(days=offset)


def is_friday_today():
    return date.today().weekday() == 4  # 4 = Friday


def generate_ercot_dates(today=None, use_minus_3_weeks=False):
    today = today or date.today()
    latest_friday = get_latest_friday(today)
    if use_minus_3_weeks:
        latest_friday -= timedelta(weeks=4)
    op_start = op_end = latest_friday.strftime("%Y-%m-%d")
    op_plus_one_month = latest_friday + relativedelta(months=1)
    start_date = op_plus_one_month.replace(day=1).strftime("%Y-%m-%d")
    end_date = op_plus_one_month.replace(year=2030).replace(day=1).strftime("%Y-%m-%d")
    return start_date, end_date, op_start, op_end


def login(email="anwar@truelightenergy.com", password="anwar@truelightenergy.com"):
    url = "https://truepriceenergy.com/login"
    response = requests.post(url, params={"email": email, "password": password}, verify=False)
    return eval(response.text)["access_token"]


def get_data_df(access_token, start_date, end_date, operating_day, offset, curve, iso, strip, history, type):
    url = "https://truepriceenergy.com/get_data"
    query = {
        "start": start_date, "end": end_date,
        "operating_day": operating_day,
        "curve_type": curve, "iso": iso,
        "strip": strip, "history": history,
        "type": type
    }
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, params=query, headers=headers, verify=False)
    df = pd.read_csv(StringIO(response.content.decode('utf-8')), header=None)

    return df


def clean_forecast_df(df):
    meta_cols = df.iloc[7, 2:].tolist()
    fixed_cols = df.iloc[9, :2].tolist()
    full_cols = fixed_cols + meta_cols
    df_clean = df.iloc[10:].copy()
    df_clean.columns = full_cols
    df_clean.reset_index(drop=True, inplace=True)
    df_clean[fixed_cols[0]] = pd.to_datetime(df_clean[fixed_cols[0]], errors='coerce')
    df_clean[full_cols[1]] = pd.to_datetime(df_clean[full_cols[1]], errors='coerce', utc=True)

    # 🔢 Convert all remaining columns to float
    for col in df_clean.columns[2:]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    return df_clean


# ========== Analysis Logic ==========
def analyze_forecast(df1, df2):
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
    df_pct = ((df_diff / df1[cost_columns]) * 100).round(2)

    df_diff.reset_index(inplace=True)
    df_pct.reset_index(inplace=True)

    def filter_timeframe(df, start, end):
        return df[(df['Curve Start Month'] >= start) & (df['Curve Start Month'] <= end)]

    prompt_month = filter_timeframe(df_diff, datetime(2025, 5, 1, tzinfo=timezone.utc), datetime(2025, 5, 1, tzinfo=timezone.utc))
    q2 = filter_timeframe(df_diff, datetime(2025, 4, 1, tzinfo=timezone.utc), datetime(2025, 6, 30, tzinfo=timezone.utc))
    h1 = filter_timeframe(df_diff, datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2025, 6, 30, tzinfo=timezone.utc))
    y2025 = filter_timeframe(df_diff, datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2025, 12, 31, tzinfo=timezone.utc))
    summer = filter_timeframe(df_diff, datetime(2025, 6, 1, tzinfo=timezone.utc), datetime(2025, 9, 30, tzinfo=timezone.utc))

    def generate_summary(diff_df, pct_df, name):
        merged = diff_df.copy()
        for col in cost_columns:
            merged[col + " (%)"] = pct_df.set_index("Curve Start Month").loc[merged["Curve Start Month"], col].values

        prompt = f"""
        Act as an energy market editor. Summarize this {name} forecast in one high-level sentence combining trends:
        - Market direction
        - Notable cost category changes
        - Concise tone

        Data:
        {merged.round(2).to_string(index=False)}
        """
        return get_groq_summary(prompt)

    summaries = {
        "Prompt Month": generate_summary(prompt_month, df_pct, "Prompt Month"),
        "Q2": generate_summary(q2, df_pct, "Q2"),
        "H1": generate_summary(h1, df_pct, "H1"),
        "Year 2025": generate_summary(y2025, df_pct, "Year 2025"),
        "Summer": generate_summary(summer, df_pct, "Summer")
    }

    return summaries, df1, df2


# ========== Data Storage and Retrieval Functions ==========
def get_data_folder():
    """Create and return the data folder path"""
    data_folder = pathlib.Path("data")
    data_folder.mkdir(exist_ok=True)
    return data_folder


def get_iso_folder(iso_name):
    """Create and return the ISO folder path"""
    iso_folder = get_data_folder() / iso_name.lower()
    iso_folder.mkdir(exist_ok=True)
    return iso_folder


def get_latest_data_date(iso_name):
    """Get the date of the latest data folder for the specified ISO"""
    iso_folder = get_iso_folder(iso_name)
    date_folders = [f for f in iso_folder.iterdir() if f.is_dir()]
    
    if not date_folders:
        return None
    
    # Sort by date (folder name should be YYYY-MM-DD)
    date_folders.sort(reverse=True)
    return date_folders[0].name


def save_data(iso_name, df1, df2, summaries):
    """Save data to the appropriate folder using the latest Friday date"""
    latest_friday = get_latest_friday()
    friday_str = latest_friday.strftime("%Y-%m-%d")
    iso_folder = get_iso_folder(iso_name)
    friday_folder = iso_folder / friday_str
    friday_folder.mkdir(exist_ok=True)
    
    # Save dataframes
    df1.reset_index().to_csv(friday_folder / "df1.csv", index=False)
    df2.reset_index().to_csv(friday_folder / "df2.csv", index=False)

    # Save summaries
    with open(friday_folder / "summaries.json", "w") as f:
        json.dump(summaries, f)
    
    return friday_folder


def load_data(iso_name, date_str=None):
    """Load data from the specified date folder or the latest folder"""
    iso_folder = get_iso_folder(iso_name)
    
    if date_str is None:
        date_str = get_latest_data_date(iso_name)
        if date_str is None:
            return None, None, None
    
    date_folder = iso_folder / date_str
    
    if not date_folder.exists():
        return None, None, None
    
    try:
        df1 = pd.read_csv(date_folder / "df1.csv")
        df1['Curve Start Month'] = pd.to_datetime(df1['Curve Start Month'])
        df1.set_index('Curve Start Month', inplace=True)
        
        df2 = pd.read_csv(date_folder / "df2.csv")
        df2['Curve Start Month'] = pd.to_datetime(df2['Curve Start Month'])
        df2.set_index('Curve Start Month', inplace=True)
        
        with open(date_folder / "summaries.json", "r") as f:
            summaries = json.load(f)
        
        return df1, df2, summaries
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


def run_pipeline(iso_code):
    """Run the data fetching and analysis pipeline"""
    access_token = login()
    
    start2, end2, op2, _ = generate_ercot_dates()
    df2_raw = get_data_df(access_token, start2, end2, op2, op2, "nonenergy", iso_code, "standardized", False, "csv")
    
    start1, end1, op1, _ = generate_ercot_dates(use_minus_3_weeks=True)
    df1_raw = get_data_df(access_token, start1, end1, op1, op1, "nonenergy", iso_code, "standardized", False, "csv")

    df1 = clean_forecast_df(df1_raw)
    df2 = clean_forecast_df(df2_raw)

    summaries, df1_indexed, df2_indexed = analyze_forecast(df1, df2)

    # Save the data
    save_data(iso_code, df1_indexed, df2_indexed, summaries)
    
    return df1, df2, summaries


def display_summaries(summaries):
    """Display the summaries in a nice format"""
    cols = st.columns(3)
    for i, (key, val) in enumerate(summaries.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div style='background-color: var(--secondary-background-color); color: var(--text-color); padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
            <h4 style='margin-top: 0;'>{key}</h4>
            <p>{val}</p>
            </div>
            """, unsafe_allow_html=True)


# ========== Streamlit App ==========
st.set_page_config(page_title="Forecast Comparison", layout="wide")
st.title("📊 Non-Energy Forecast Comparison Dashboard")

if not hasattr(st, 'tabs'):
    st.warning("Please upgrade Streamlit to version 1.10 or above to use st.tabs.")
else:
    tabs = st.tabs(["ERCOT", "NYISO", "ISONE"])

    for tab_name, tab in zip(["ERCOT", "NYISO", "ISONE"], tabs):
        with tab:
            st.subheader(f"🔍 {tab_name} Forecast Analysis")
            
            iso_code = tab_name.lower()
            latest_date = get_latest_data_date(iso_code)
            
            # Check if we need to run the pipeline
            run_pipeline_automatically = False
            
            # Get the latest Friday date for comparison
            latest_friday = get_latest_friday()
            latest_friday_str = latest_friday.strftime("%Y-%m-%d")
            
            if is_friday_today() and (latest_date != latest_friday_str):
                # st.info("Today is Friday! Running pipeline for fresh data...")
                run_pipeline_automatically = True
            # elif is_friday_today() and (latest_date == latest_friday_str):
            #     st.success(f"Today's Friday data is already available! Using existing data from {latest_date}")
            elif latest_date is None:
                # st.warning(f"No existing data found for {tab_name}. Running pipeline...")
                run_pipeline_automatically = True
            elif latest_date != latest_friday_str:
                # st.warning(f"Data for latest Friday ({latest_friday_str}) not found. Running pipeline...")
                run_pipeline_automatically = True
            else:
                st.info(f"Here is the Summary for {tab_name} curve for {latest_date}")
                

            if run_pipeline_automatically :
                with st.spinner(f"Fetching and analyzing {tab_name} data..."):
                    try:
                        df1, df2, summaries = run_pipeline(iso_code)
                        # st.success(f"✅ {tab_name} analysis complete!")
                        
                        # Display the summaries
                        st.header("📰 Newsletter Summary")
                        display_summaries(summaries)
                        
                    except Exception as e:
                        st.error(f"Error running pipeline: {e}")
            else:
                # Load existing data
                df1, df2, summaries = load_data(iso_code)
                
                if summaries is not None:
                    # st.success(f"✅ Loaded existing {tab_name} data from {latest_date}")
                    
                    # Display the summaries
                    st.header("📰 Newsletter Summary")
                    display_summaries(summaries)
                else:
                    st.warning(f"No data found for {tab_name}. Click 'Force Refresh Data' to run the pipeline.")
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

st.set_page_config(page_title="Forecast Comparison", layout="wide")




st.image("/home/ubuntu/EnergyPricingInfo/UI/logo.png", width=300)

urllib3.disable_warnings()
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)

ISO_CONFIG = {
    "ercot": {"update_freq": "weekly", "date_func": "generate_ercot_dates"},
    "isone": {"update_freq": "monthly", "date_func": "generate_isone_dates"},
    "nyiso": {"update_freq": "monthly", "date_func": "generate_isone_dates"},
    "miso":  {"update_freq": "monthly", "date_func": "generate_isone_dates"},
    "pjm":   {"update_freq": "monthly", "date_func": "generate_isone_dates"},
}

def get_groq_summary(prompt):
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def get_latest_friday(today=None):
    today = today or date.today()
    offset = (today.weekday() - 4) % 7
    return today - timedelta(days=offset)

def is_friday_today():
    return date.today().weekday() == 4

def get_latest_month_end(today=None):
    today = today or date.today()
    return (today.replace(day=1) + relativedelta(months=1)) - timedelta(days=1)

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

def generate_isone_dates(today=None, use_prev_month=False):
    today = today or date.today()
    latest_month_end = get_latest_month_end(today)
    if use_prev_month:
        latest_month_end -= relativedelta(months=1)
    op_start = op_end = latest_month_end.strftime("%Y-%m-%d")
    forecast_start = (latest_month_end + relativedelta(months=1)).replace(day=1).strftime("%Y-%m-%d")
    forecast_end = (latest_month_end + relativedelta(months=1)).replace(year=2030, day=1).strftime("%Y-%m-%d")
    return forecast_start, forecast_end, op_start, op_end

def generate_dates(iso_code, version="current"):
    today = date.today()
    config = ISO_CONFIG[iso_code]
    func_name = config["date_func"]
    use_previous = version == "previous"

    if func_name == "generate_isone_dates":
        return generate_isone_dates(today, use_prev_month=use_previous)
    elif func_name == "generate_ercot_dates":
        return generate_ercot_dates(today, use_minus_3_weeks=use_previous)
    else:
        raise ValueError(f"Unsupported function for ISO: {iso_code}")

def should_update_today(iso_code, latest_date_str):
    config = ISO_CONFIG.get(iso_code)
    if not config:
        return False
    today = date.today()
    if config["update_freq"] == "weekly":
        expected = get_latest_friday(today)
    elif config["update_freq"] == "monthly":
        expected = get_latest_month_end(today)
    else:
        return False
    return latest_date_str != expected.strftime("%Y-%m-%d")

def login(email="anwar@truelightenergy.com", password="anwar@truelightenergy.com"):
    url = "https://truepriceenergy.com/login"
    response = requests.post(url, params={"email": email, "password": password}, verify=False)
    return eval(response.text)["access_token"]

def get_data_df(token, start_date, end_date, op_day, offset, curve, iso, strip, history, typ):
    url = "https://truepriceenergy.com/get_data"
    query = {
        "start": start_date,
        "end": end_date,
        "operating_day": op_day,
        "curve_type": curve,
        "iso": iso,
        "strip": strip,
        "history": history,
        "type": typ
    }
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, params=query, headers=headers, verify=False)
    return pd.read_csv(StringIO(response.content.decode('utf-8')), header=None)

def clean_forecast_df(df):
    #if df.shape[0] < 10:
     #   raise ValueError("Dataframe has fewer than 10 rows. Possibly malformed or empty response.")
      #  return pd.DataFrame()
    meta_cols = df.iloc[7, 2:].tolist()
    fixed_cols = df.iloc[9, :2].tolist()
    full_cols = fixed_cols + meta_cols
    df_clean = df.iloc[10:].copy()
    df_clean.columns = full_cols
    df_clean.reset_index(drop=True, inplace=True)
    df_clean[fixed_cols[0]] = pd.to_datetime(df_clean[fixed_cols[0]], errors='coerce')
    df_clean[full_cols[1]] = pd.to_datetime(df_clean[full_cols[1]], errors='coerce', utc=True)
    for col in df_clean.columns[2:]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    return df_clean

def analyze_forecast(df1, df2):
    df1['Curve Start Month'] = pd.to_datetime(df1['Curve Start Month'], utc=True)
    df2['Curve Start Month'] = pd.to_datetime(df2['Curve Start Month'], utc=True)
    df1.set_index('Curve Start Month', inplace=True)
    df2.set_index('Curve Start Month', inplace=True)
    df1.index += pd.DateOffset(months=1)

    diff = df2[df2.columns[1:]] - df1[df1.columns[1:]]
    pct = ((diff / df1[df1.columns[1:]]) * 100).round(2)
    diff.reset_index(inplace=True)
    pct.reset_index(inplace=True)

    def filter_range(df, start, end):
        return df[(df['Curve Start Month'] >= start) & (df['Curve Start Month'] <= end)]

    def summarize(diff_df, pct_df, label):
        merged = diff_df.copy()
        for col in diff_df.columns[1:]:
            merged[col + " (%)"] = pct_df.set_index("Curve Start Month").loc[merged["Curve Start Month"], col].values
        # prompt = f"""
        # Act as an energy market editor. Summarize this {label} forecast in one high-level sentence combining trends:
        # - Market direction
        # - Notable cost category changes
        # - Concise tone

        # Data:
        # {merged.round(2).to_string(index=False)}
        # """
        prompt = f"""
        You are an expert energy analyst. Write **A bullet point** summarizing trends from the following data. **Do not add any introductions** like "Here is the summary" — only the 3 points.

        Focus on:
        - Market direction (up/down/mixed)
        - Big increases/decreases in categories

        Data:
        {merged.round(2).to_string(index=False)}
        """


        return get_groq_summary(prompt)

    tz = timezone.utc
    summaries = {
        "Prompt Month": summarize(filter_range(diff, datetime(2025,5,1,tzinfo=tz), datetime(2025,5,1,tzinfo=tz)), pct, "Prompt Month"),
        "Q2": summarize(filter_range(diff, datetime(2025,4,1,tzinfo=tz), datetime(2025,6,30,tzinfo=tz)), pct, "Q2"),
        "H1": summarize(filter_range(diff, datetime(2025,1,1,tzinfo=tz), datetime(2025,6,30,tzinfo=tz)), pct, "H1"),
        "Year 2025": summarize(filter_range(diff, datetime(2025,1,1,tzinfo=tz), datetime(2025,12,31,tzinfo=tz)), pct, "Year 2025"),
        "Summer": summarize(filter_range(diff, datetime(2025,6,1,tzinfo=tz), datetime(2025,9,30,tzinfo=tz)), pct, "Summer")
    }
    return summaries, df1, df2

def get_data_folder():
    path = pathlib.Path("data")
    path.mkdir(exist_ok=True)
    return path

def get_iso_folder(iso):
    folder = get_data_folder() / iso.lower()
    folder.mkdir(exist_ok=True)
    return folder

def get_latest_data_date(iso):
    folder = get_iso_folder(iso)
    dates = [f for f in folder.iterdir() if f.is_dir()]
    if not dates:
        return None
    return sorted(dates, reverse=True)[0].name

def save_data(iso, df1, df2, summaries):
    folder = get_iso_folder(iso) / get_latest_friday().strftime("%Y-%m-%d")
    folder.mkdir(exist_ok=True)
    df1.reset_index().to_csv(folder / "df1.csv", index=False)
    df2.reset_index().to_csv(folder / "df2.csv", index=False)
    with open(folder / "summaries.json", "w") as f:
        json.dump(summaries, f)
    return folder

def load_data(iso, date_str=None):
    folder = get_iso_folder(iso)
    if date_str is None:
        date_str = get_latest_data_date(iso)
    if not date_str:
        return None, None, None
    path = folder / date_str
    try:
        df1 = pd.read_csv(path / "df1.csv").set_index(pd.to_datetime(pd.read_csv(path / "df1.csv")["Curve Start Month"]))
        df2 = pd.read_csv(path / "df2.csv").set_index(pd.to_datetime(pd.read_csv(path / "df2.csv")["Curve Start Month"]))
        with open(path / "summaries.json") as f:
            summaries = json.load(f)
        return df1, df2, summaries
    except Exception as e:
        st.error(f"Error loading: {e}")
        return None, None, None

def run_pipeline(iso):
    token = login()
    start2, end2, op2, _ = generate_dates(iso, version="current")
    start1, end1, op1, _ = generate_dates(iso, version="previous")
    df2 = clean_forecast_df(get_data_df(token, start2, end2, op2, op2, "nonenergy", iso, "standardized", False, "csv"))
    df1 = clean_forecast_df(get_data_df(token, start1, end1, op1, op1, "nonenergy", iso, "standardized", False, "csv"))
    summaries, df1_idx, df2_idx = analyze_forecast(df1, df2)
    save_data(iso, df1_idx, df2_idx, summaries)
    return df1, df2, summaries

def display_summaries(summaries):
    cols = st.columns(3)
    for i, (k, v) in enumerate(summaries.items()):
        # Split into individual bullet points if possible
        bullets = [point.strip() for point in v.split("•") if point.strip()]
        bullet_html = "".join(f"<li>{b}</li>" for b in bullets)

        with cols[i % 3]:
            st.markdown(f"""
            <div style='background-color: var(--secondary-background-color); padding: 15px; border-radius: 10px;'>
                <h4>{k}</h4>
                <ul style='padding-left: 1.2em;'>{bullet_html}</ul>
            </div>
            """, unsafe_allow_html=True)


# Streamlit App
st.title("Non-Energy NewsLetter")

tabs = st.tabs(["ERCOT", "NYISO", "ISONE", "MISO", "PJM"])
for tab_name, tab in zip(["ercot", "nyiso", "isone", "miso", "pjm"], tabs):
    with tab:
        st.subheader(f"🔍 {tab_name.upper()} Forecast Analysis")
        latest_date = get_latest_data_date(tab_name)
        auto_refresh = should_update_today(tab_name, latest_date)

        if auto_refresh:
            with st.spinner(f"Fetching {tab_name.upper()} data..."):
                try:
                    df1, df2, summaries = run_pipeline(tab_name)
                    st.header("📰 Newsletter Summary")
                    display_summaries(summaries)
                except Exception as e:
                    st.warning(f"Pipeline failed: {e}")
                    st.info("Falling back to previously cached data...")
                    df1, df2, summaries = load_data(tab_name)
                    if summaries:
                        st.header("📰 Newsletter Summary (Fallback)")
                        display_summaries(summaries)
                    else:
                        st.error("No previous data found either.")
                        continue
        else:
            df1, df2, summaries = load_data(tab_name)
            if summaries:
                st.info(f"Showing cached results for {tab_name.upper()} ({latest_date})")
                st.header("📰 Newsletter Summary")
                display_summaries(summaries)
            else:
                st.warning(f"No data found for {tab_name.upper()}.")
                continue  # skip visualization if no data
        

        # === Visualization: Bar Chart Race + Heatmap ===
        try:
            df1 = df1.copy()
            df2 = df2.copy()
            df1.index = pd.to_datetime(df1.index)
            df2.index = pd.to_datetime(df2.index)
            cost_columns = df1.columns[1:]
            for col in cost_columns:
                df1[col] = pd.to_numeric(df1[col], errors='coerce')
                df2[col] = pd.to_numeric(df2[col], errors='coerce')
            # Percentage change
            pct_df = ((df2[cost_columns] - df1[cost_columns]) / df1[cost_columns]) * 100
            pct_df.reset_index(inplace=True)

            # Top 10 components
            avg_change = pct_df[cost_columns].mean().sort_values(ascending=False)
            top_components = avg_change.head(10).index.tolist()

            # Bar Chart Race
            combined_df = pd.concat([df1[cost_columns], df2[cost_columns]])
            combined_df['Month'] = list(df1.index) + list(df2.index)
            melted_df = combined_df.melt(id_vars='Month', var_name='Component', value_name='Forecast')

            bar_fig = px.bar(
                melted_df[melted_df['Component'].isin(top_components)],
                x='Forecast', y='Component', color='Component',
                animation_frame='Month',
                orientation='h',
                title='📊 Top 10 Component Forecasts Over Time'
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            # Heatmap
            heatmap_fig = px.imshow(
                pct_df.set_index('Curve Start Month')[cost_columns].T,
                aspect='auto',
                color_continuous_scale='RdBu',
                labels=dict(x="Forecast Month", y="Cost Component", color="% Change"),
                title="🔺 Heatmap of Forecast Change Intensity"
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)

        except Exception as e:
            st.warning(f"Visualization error: {e}")

# for tab_name, tab in zip(["ercot", "nyiso", "isone", "miso", "pjm"], tabs):
#     with tab:
#         st.subheader(f"🔍 {tab_name.upper()} Forecast Analysis")
#         latest_date = get_latest_data_date(tab_name)
#         auto_refresh = should_update_today(tab_name, latest_date)

#         if auto_refresh:
#             with st.spinner(f"Fetching {tab_name.upper()} data..."):
#                 try:
#                     df1, df2, summaries = run_pipeline(tab_name)
#                     st.header("📰 Newsletter Summary")
#                     display_summaries(summaries)
#                 except Exception as e:
#                     st.error(f"Pipeline error: {e}")
#         else:
#             df1, df2, summaries = load_data(tab_name)
#             if summaries:
#                 st.info(f"Showing cached results for {tab_name.upper()} ({latest_date})")
#                 st.header("📰 Newsletter Summary")
#                 display_summaries(summaries)
#             else:
#                 st.warning(f"No data found for {tab_name.upper()}.")
# --- Footer Section ---
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
            <p style='color: white; margin-top: 1rem;'>&copy; 2025 TRUELight Energy</p>
        </div>
    </div>
""", unsafe_allow_html=True)

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


from eroct_data import get_ercot_data
from isone_data import get_isone_data
from miso_data import get_miso_data
from pjm_date import get_pjm_data

st.set_page_config(page_title="Forecast Comparison", layout="wide")




st.image("Images/Company.png", width=300)
st.title("TrueLight Non-Energy News Letter")

#########################################
groq_api_key = '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'

groq_client = Groq(api_key=groq_api_key)

def get_groq_summary(prompt):
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()
def summarize(diff_df, pct_df, label):
    merged = diff_df.copy()
    for col in diff_df.columns[1:]:
        merged[col + " (%)"] = pct_df.set_index("Curve Start Month").loc[merged["Curve Start Month"], col].values
        prompt = f"""
        You are an expert energy analyst. Write **3 crisp bullet points** summarizing trends from the following data. **Do not add any introductions** like "Here is the summary" — only the 3 points.

        Focus on:
        - Market direction (up/down/mixed)
        - Big increases/decreases in categories
        - Keep it professional, direct, and data-driven

        Data:
        {merged.round(2).to_string(index=False)}
        """


    return get_groq_summary(prompt)


########################################
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
def render_forecast_visuals(df1, df2, iso_name="ISO"):
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
        pct_df.rename(columns={"index": "Curve Start Month"}, inplace=True)

        # Top 10 movers
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
            title=f'📊 Top 10 {iso_name.upper()} Forecast Changes Over Time'
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        # Heatmap
        heatmap_fig = px.imshow(
            pct_df.set_index('Curve Start Month')[cost_columns].T,
            aspect='auto',
            color_continuous_scale='RdBu',
            labels=dict(x="Forecast Month", y="Cost Component", color="% Change"),
            title=f"🔺 {iso_name.upper()} Heatmap of Forecast Change Intensity"
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)

    except Exception as e:
        st.warning(f"[{iso_name.upper()}] Visualization error: {e}")
########################################



def filter_timeframe(df, start_date, end_date):
    mask = (df['Curve Start Month'] >= start_date) & (df['Curve Start Month'] <= end_date)
    return df.loc[mask]

def calculations(df1,df2,f_date,l_date):
    df1['Curve Start Month'] = pd.to_datetime(df1['Curve Start Month'], utc=True)
    df2['Curve Start Month'] = pd.to_datetime(df2['Curve Start Month'], utc=True)

    # Step 2: Convert 'Curve Start Month' to UTC datetime
    df1['Curve Start Month'] = pd.to_datetime(df1['Curve Start Month'], utc=True)
    df2['Curve Start Month'] = pd.to_datetime(df2['Curve Start Month'], utc=True)

    # Step 3: Identify cost columns (excluding the first two)
    cost_columns = df1.columns[2:]
    for col in cost_columns:
        df1[col] = pd.to_numeric(df1[col], errors='coerce')
        df2[col] = pd.to_numeric(df2[col], errors='coerce')

    # Step 5: Align indices by shifting NEERCOT-1 forward one month
    df1.set_index('Curve Start Month', inplace=True)
    df2.set_index('Curve Start Month', inplace=True)
    df1.index = df1.index + pd.DateOffset(months=1)

    # Step 6: Calculate differences
    df_diff = df2[cost_columns] - df1[cost_columns]
    df_pct = ((df_diff / df1[cost_columns]) * 100).round(2)

    # Step 7: Reset index for filtering
    df_diff.reset_index(inplace=True)
    df_pct.reset_index(inplace=True)

    prompt_month = filter_timeframe(df_diff, f_date, l_date)
    q2 = filter_timeframe(df_diff, datetime(2025, 4, 1, tzinfo=timezone.utc), datetime(2025, 6, 30, tzinfo=timezone.utc))
    h1 = filter_timeframe(df_diff, datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2025, 6, 30, tzinfo=timezone.utc))
    y2025 = filter_timeframe(df_diff, datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2025, 12, 31, tzinfo=timezone.utc))
    summer = filter_timeframe(df_diff, datetime(2025, 6, 1, tzinfo=timezone.utc), datetime(2025, 9, 30, tzinfo=timezone.utc))
    summaries = {
        "Prompt Month": summarize(prompt_month, df_pct, "Prompt Month"),
        "Q2": summarize(q2, df_pct, "Q2"),
        "H1": summarize(h1, df_pct, "H1"),
        "Year 2025": summarize(y2025, df_pct, "Year 2025"),
        "Summer": summarize(summer, df_pct, "Summer")
    }
    return summaries

# ---- Tab UI ----
tabs = st.tabs(["ERCOT", "NYISO", "ISONE", "MISO", "PJM"])
for tab_name, tab in zip(["ercot", "nyiso", "isone", "miso", "pjm"], tabs):
    with tab:
        st.subheader(f"🔍 {tab_name.upper()} Forecast Analysis")

        if tab_name == 'ercot':
            with st.spinner(f"Fetching ERCOT data..."):
                try:
                    result = get_ercot_data()
                    summaries = calculations(result['previous_df'], result['current_df'], result['f_date'], result['l_date'])
                    st.header("📰 Newsletter Summary")
                    display_summaries(summaries)
                    render_forecast_visuals(result['previous_df'], result['current_df'], iso_name=tab_name)
                except Exception as e:
                    st.error(f"Failed to fetch ERCOT data: {e}")
        elif tab_name == 'isone':
            with st.spinner("Fetching ISONE data..."):
                try:
                    result = get_isone_data()
                    summaries = calculations(result['previous_df'], result['current_df'], result['f_date'], result['l_date']) 
                    st.header("📰 Newsletter Summary")
                    display_summaries(summaries)
                    render_forecast_visuals(result['previous_df'], result['current_df'], iso_name=tab_name)

                except Exception as e:
                    st.error(f"Failed to fetch ERCOT data: {e}")
        elif tab_name == 'miso':
            with st.spinner("Fetching MISO data..."):
                try:
                    result = get_miso_data()
                    summaries = calculations(result['previous_df'], result['current_df'], result['f_date'], result['l_date']) 
                    st.header("📰 Newsletter Summary")
                    display_summaries(summaries)
                    render_forecast_visuals(result['previous_df'], result['current_df'], iso_name=tab_name)

                except Exception as e:
                    st.error(f"Failed to fetch ERCOT data: {e}")
        elif tab_name == 'pjm':
            with st.spinner("Fetching PJM data..."):
                try:
                    result = get_pjm_data()
                    summaries = calculations(result['previous_df'], result['current_df'], result['f_date'], result['l_date']) 
                    st.header("📰 Newsletter Summary")
                    display_summaries(summaries)
                    render_forecast_visuals(result['previous_df'], result['current_df'], iso_name=tab_name)

                except Exception as e:
                    st.error(f"Failed to fetch ERCOT data: {e}")
        else:
            st.info(f"{tab_name.upper()} support coming soon.")
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

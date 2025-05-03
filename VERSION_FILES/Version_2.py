import streamlit as st
import pandas as pd
import os
import plotly.express as px
import json
import openai
from datetime import datetime, timezone
from dotenv import load_dotenv

# 🔑 Set OpenAI API key
load_dotenv()  # Loads .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

# 🔍 Forecast Analysis Function
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
        You are an energy market analyst. Compare April vs March forecast for {name}.
        Focus on changes, trends, and insights.
        Data:
        {merged.round(3).to_string(index=False)}
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']

    summaries = {
        "Prompt Month": generate_summary(prompt_month, df_pct, "Prompt Month"),
        "Q2": generate_summary(q2, df_pct, "Q2"),
        "H1": generate_summary(h1, df_pct, "H1"),
        "Year 2025": generate_summary(y2025, df_pct, "Year 2025"),
        "Summer": generate_summary(summer, df_pct, "Summer")
    }

    os.makedirs("Plots", exist_ok=True)
    os.makedirs("Forecasts", exist_ok=True)

    # Prepare for Bar Chart Race
    df1_copy = df1[cost_columns].copy()
    df1_copy['Month'] = df1.index
    df2_copy = df2[cost_columns].copy()
    df2_copy['Month'] = df2.index

    bar_race_df = pd.concat([df1_copy, df2_copy], ignore_index=True)
    bar_race_df_melted = bar_race_df.melt(id_vars='Month', var_name='Component', value_name='Forecast')
    bar_race_df_melted["MonthStr"] = bar_race_df_melted["Month"].dt.strftime("%Y-%m")

    avg_change = df_pct[cost_columns].mean().sort_values(ascending=False)
    top_components = avg_change.head(10).index.tolist()

    bar_race_fig = px.bar(
        bar_race_df_melted[bar_race_df_melted['Component'].isin(top_components)],
        x='Forecast', y='Component', color='Component',
        animation_frame='MonthStr',  # ✅ FIXED
        orientation='h',
        title='Bar Chart Race: Top 10 Components Over Time'
    )
    bar_race_fig.write_html("Plots/Bar_Chart_Race.html")

    # Heatmap
    heatmap_data = df_pct.set_index("Curve Start Month")[cost_columns].T
    heatmap_fig = px.imshow(
        heatmap_data,
        aspect='auto',
        color_continuous_scale='RdBu',
        labels=dict(x="Forecast Month", y="Cost Component", color="% Change"),
        title="Heatmap: Change Intensity Over Time"
    )
    heatmap_fig.write_html("Plots/Heatmap.html")

    # Save summaries
    with open("Forecasts/forecast_summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)

    with open("structured_forecast_report.txt", "w") as out_file:
        headers = {
            "Prompt Month": "🗓️ Prompt Month (May 2025)",
            "Q2": "📆 Q2 (Apr – Jun 2025)",
            "H1": "🕐 H1 (Jan – Jun 2025)",
            "Year 2025": "📅 Year 2025",
            "Summer": "☀️ Summer (Jun – Sep 2025)"
        }
        out_file.write("📊 Forecast Summary Report\n\n")
        for key, summary in summaries.items():
            out_file.write(f"{headers.get(key, key)}:\n{summary}\n\n")

    return summaries

# ========================
# 🌐 Streamlit App Layout
# ========================

st.set_page_config(page_title="Forecast Comparison", layout="wide")
st.title("📊 Non-Energy Cost Forecast Comparison")

if not hasattr(st, 'tabs'):
    st.warning("Please upgrade Streamlit to version 1.10 or above to use st.tabs.")
else:
    tabs = st.tabs(["ERCOT", "NYISO", "ISONE"])

    for tab_name, tab in zip(["ERCOT", "NYISO", "ISONE"], tabs):
        with tab:
            st.subheader(f"🔍 {tab_name} Forecast Analysis")
            uploaded_march = st.file_uploader(f"📥 Upload {tab_name} March Forecast", type=["xlsx", "csv"], key=f"{tab_name}_march")
            uploaded_april = st.file_uploader(f"📥 Upload {tab_name} April Forecast", type=["xlsx", "csv"], key=f"{tab_name}_april")
            run_analysis = st.button("▶️ Run Analysis", key=f"{tab_name}_run")

            if run_analysis and uploaded_march and uploaded_april:
                df1 = pd.read_excel(uploaded_march) if uploaded_march.name.endswith("xlsx") else pd.read_csv(uploaded_march)
                df2 = pd.read_excel(uploaded_april) if uploaded_april.name.endswith("xlsx") else pd.read_csv(uploaded_april)

                summaries = analyze_forecast(df1, df2)

                st.success("✅ Analysis complete!")
                st.header("📝 GPT Summary Report")
                for key, text in summaries.items():
                    with st.expander(f"🔹 {key}"):
                        st.markdown(text)

                st.header("📈 Forecast Visualizations")

                st.subheader("📊 Bar Chart: Avg. Cost Change")
                with open("Plots/Bar_Chart_Race.html", "r") as f:
                    st.components.v1.html(f.read(), height=600, scrolling=True)

                st.subheader("🌡️ Heatmap: % Change Intensity")
                with open("Plots/Heatmap.html", "r") as f:
                    st.components.v1.html(f.read(), height=600, scrolling=True)

                with open("structured_forecast_report.txt", "rb") as f:
                    st.download_button("⬇️ Download Summary Report", f, file_name=f"{tab_name}_Forecast_Summary.txt")
            else:
                st.info("📤 Please upload both files and click 'Run Analysis' to begin.")

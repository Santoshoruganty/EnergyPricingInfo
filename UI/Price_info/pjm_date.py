import requests
import json
from datetime import date, datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import urllib3

urllib3.disable_warnings()

# 🔐 Login to API
def login(email="anwar@truelightenergy.com", password="anwar@truelightenergy.com"):
    url = "https://truepriceenergy.com/login"
    response = requests.post(url, params={"email": email, "password": password}, verify=False)
    return eval(response.text)["access_token"]

# 📅 Get last day of any given month
def get_month_end(d):
    return (d.replace(day=1) + relativedelta(months=1)) - timedelta(days=1)

# 📊 Pivot flat JSON list into DataFrame
def pivot_pjm_json(json_data):
    df = pd.DataFrame(json_data)
    df['Curve Start Month'] = pd.to_datetime(df['Curve Start Month'])
    df['Curve Update Date'] = pd.to_datetime(df['Curve Update Date'])
    pivot_df = df.pivot_table(
        index='Curve Start Month',
        columns='Cost Component',
        values='Data',
        aggfunc='first'
    ).reset_index()
    pivot_df['Curve Update Date'] = df['Curve Update Date'].iloc[0]
    cols = ['Curve Start Month', 'Curve Update Date'] + sorted([col for col in pivot_df.columns if col not in ['Curve Start Month', 'Curve Update Date']])
    return pivot_df[cols]

# 🌐 Fetch forecast JSON from TruePrice API
def fetch_pjm_data(operating_date):
    if isinstance(operating_date, str):
        operating_date = datetime.strptime(operating_date, "%Y-%m-%d").date()

    op_date_str = operating_date.strftime("%Y-%m-%d")
    next_month = operating_date + relativedelta(months=1)
    start_date = date(next_month.year, next_month.month, 1).strftime("%Y-%m-%d")
    end_date = date(2030, next_month.month, 1).strftime("%Y-%m-%d")

    url = "https://truepriceenergy.com/get_data"
    query = {
        "start": start_date,
        "end": end_date,
        "operating_day": op_date_str,
        "curve_type": "nonenergy",
        "iso": "pjm",
        "strip": "standardized",
        "history": False,
        "type": "json"
    }

    token = login()
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.get(url, params=query, headers=headers, verify=False)
        text = response.text.strip()

        print(f"[DEBUG] Querying for {op_date_str} returned status {response.status_code}")
        print(f"[DEBUG] Response: {text[:300]}")

        if response.status_code != 200 or text.lower().startswith("unable to fetch"):
            print(f"⚠️ No data returned for {op_date_str}")
            return None

        return json.loads(text)
    except Exception as e:
        print(f"Error fetching PJM data: {e}")
        return None

# 🧠 Master function to compute correct op dates, fetch data, and return DataFrames + metadata
def get_pjm_data():
    today = date.today()
    today_month_end = get_month_end(today)

    if today == today_month_end:
        current_op = today
        previous_op = get_month_end(today - relativedelta(months=1))
    else:
        current_op = get_month_end(today - relativedelta(months=1))
        previous_op = get_month_end(today - relativedelta(months=2))

    current_json = fetch_pjm_data(current_op)
    previous_json = fetch_pjm_data(previous_op)

    if not current_json:
        print("⚠️ Fallback: shifting one month earlier")
        current_op = previous_op
        previous_op = get_month_end(current_op - relativedelta(months=1))
        current_json = fetch_pjm_data(current_op)
        previous_json = fetch_pjm_data(previous_op)

    current_df = pivot_pjm_json(current_json) if current_json else None
    previous_df = pivot_pjm_json(previous_json) if previous_json else None

    # Prompt window: current_op + 1 month
    prompt_start = datetime(current_op.year, current_op.month, 1, tzinfo=timezone.utc) + relativedelta(months=1)
    prompt_end = prompt_start + relativedelta(day=31)

    return {
        "current_df": current_df,
        "previous_df": previous_df,
        "current_op_date": current_op.strftime("%Y-%m-%d"),
        "previous_op_date": previous_op.strftime("%Y-%m-%d"),
        "f_date": prompt_start,
        "l_date": prompt_end
    }

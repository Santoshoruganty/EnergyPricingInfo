# ercot.py

import requests
import json
from datetime import date, timedelta, datetime,timezone
from dateutil.relativedelta import relativedelta
import pandas as pd
import urllib3 

urllib3.disable_warnings()
def filterdate():

    today = datetime.now(timezone.utc)
    first_of_next_month = datetime(today.year, today.month, 1, tzinfo=timezone.utc) + relativedelta(months=1)
    last_of_next_month = first_of_next_month + relativedelta(day=31)
    return [first_of_next_month,last_of_next_month]
def login(email="anwar@truelightenergy.com", password="anwar@truelightenergy.com"):
    url = "https://truepriceenergy.com/login"
    response = requests.post(url, params={"email": email, "password": password}, verify=False)
    return eval(response.text)["access_token"]

def get_latest_friday(input_date):
    offset = (input_date.weekday() - 4) % 7
    return input_date - timedelta(days=offset)

def pivot_ercot_json(json_data):
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

def fetch_data(operating_date):
    if isinstance(operating_date, str):
        operating_date = datetime.strptime(operating_date, "%Y-%m-%d").date()
    op_date_str = operating_date.strftime("%Y-%m-%d")
    operating_month = operating_date.month
    next_month = operating_month + 1
    year = operating_date.year
    if next_month > 12:
        next_month = 1
        year += 1
    start_date = date(year, next_month, 1).strftime("%Y-%m-%d")
    end_date = date(2030, next_month, 1).strftime("%Y-%m-%d")
    url = "https://truepriceenergy.com/get_data"
    query = {
        "start": start_date,
        "end": end_date,
        "operating_day": op_date_str,
        "curve_type": "nonenergy",
        "iso": "ercot",
        "strip": "standardized",
        "history": False,
        "type": "json"
    }
    token = login()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, params=query, headers=headers, verify=False)
        return json.loads(response.text)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def get_ercot_data():
    today = date.today()
    is_friday = today.weekday() == 4
    latest_friday = get_latest_friday(today)
    if is_friday:
        try:
            current_test = fetch_data(today)
            if current_test:
                current_op = today
                prev_op = today - timedelta(weeks=3)
            else:
                current_op = latest_friday
                prev_op = current_op - timedelta(weeks=3)
        except:
            current_op = latest_friday - timedelta(weeks=1)
            prev_op = current_op - timedelta(weeks=3)
    else:
        current_op = latest_friday
        prev_op = current_op - timedelta(weeks=4)

    current_json = fetch_data(current_op)
    previous_json = fetch_data(prev_op)

    current_df = pivot_ercot_json(current_json) if current_json else None
    previous_df = pivot_ercot_json(previous_json) if previous_json else None
    filter_dates=filterdate()
    return {
        "current_df": current_df,
        "previous_df": previous_df,
        "current_op_date": current_op.strftime("%Y-%m-%d"),
        "previous_op_date": prev_op.strftime("%Y-%m-%d"),
        'f_date': filter_dates[0],
        'l_date':filter_dates[1]
    }

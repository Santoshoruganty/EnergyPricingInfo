import requests
import json
from datetime import date, datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import urllib3

urllib3.disable_warnings()

def login(email="anwar@truelightenergy.com", password="anwar@truelightenergy.com"):
    url = "https://truepriceenergy.com/login"
    response = requests.post(url, params={"email": email, "password": password}, verify=False)
    return eval(response.text)["access_token"]

def get_month_end(d):
    return (d.replace(day=1) + relativedelta(months=1)) - timedelta(days=1)

def get_last_friday(d):
    return d - timedelta(days=(d.weekday() - 4) % 7)

def pivot_nyiso_json(json_data):
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
    cols = ['Curve Start Month', 'Curve Update Date'] + sorted(
        [col for col in pivot_df.columns if col not in ['Curve Start Month', 'Curve Update Date']]
    )
    return pivot_df[cols]

def fetch_nyiso_data(operating_date):
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
        "iso": "nyiso",
        "strip": "standardized",
        "history": False,
        "type": "json"
    }

    token = login()
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.get(url, params=query, headers=headers, verify=False)
        text = response.text.strip()

        print(f"[DEBUG] NYISO query for {op_date_str} -> {response.status_code}")
        if response.status_code != 200 or text.lower().startswith("unable to fetch"):
            print("⚠️ NYISO returned no data.")
            return None

        return json.loads(text)
    except Exception as e:
        print(f"Error fetching NYISO data: {e}")
        return None

def get_nyiso_data():
    today = date.today()
    latest_friday = get_last_friday(today)

    current_op = latest_friday
    prev_op = current_op - timedelta(weeks=5)

    current_json = fetch_nyiso_data(current_op)
    if not current_json:
        print("⚠️ NYISO fallback to previous Friday")
        current_op = get_last_friday(today - timedelta(days=1))
        prev_op = current_op - timedelta(weeks=5)
        current_json = fetch_nyiso_data(current_op)

    previous_json = fetch_nyiso_data(prev_op)

    current_df = pivot_nyiso_json(current_json) if current_json else None
    previous_df = pivot_nyiso_json(previous_json) if previous_json else None

    prompt_start = datetime(current_op.year, current_op.month, 1, tzinfo=timezone.utc) + relativedelta(months=1)
    prompt_end = (prompt_start.replace(day=1) + relativedelta(months=1)) - timedelta(days=1)

    print(current_op.strftime("%Y-%m-%d"), prev_op.strftime("%Y-%m-%d"))
    return {
        "success": current_df is not None and previous_df is not None,
        "current_df": current_df,
        "previous_df": previous_df,
        "current_op_date": current_op.strftime("%Y-%m-%d"),
        "previous_op_date": prev_op.strftime("%Y-%m-%d"),
        "f_date": prompt_start,
        "l_date": prompt_end
    }

# def get_nyiso_data():
#     today = date.today()
#     is_friday = today.weekday() == 4
#     latest_friday = get_last_friday(today)

#     if is_friday:
#         try:
#             current_test = fetch_nyiso_data(today)
#             if current_test:
#                 current_op = today
#                 prev_op = today - timedelta(weeks=3)
#             else:
#                 current_op = latest_friday
#                 prev_op = current_op - timedelta(weeks=3)
#         except:
#             current_op = latest_friday - timedelta(weeks=1)
#             prev_op = current_op - timedelta(weeks=3)
#     else:
#         current_op = latest_friday
#         prev_op = current_op - timedelta(weeks=5)

#     current_json = fetch_nyiso_data(current_op)
#     previous_json = fetch_nyiso_data(prev_op)

#     # 🔁 Fallback to earlier Friday if current_json is still None
#     if not current_json:
#         print("⚠️ NYISO fallback to previous Friday")
#         current_op = get_last_friday(today - timedelta(days=1))
#         prev_op = current_op - timedelta(weeks=4)
#         current_json = fetch_nyiso_data(current_op)
#         previous_json = fetch_nyiso_data(prev_op)

#     current_df = pivot_nyiso_json(current_json) if current_json else None
#     previous_df = pivot_nyiso_json(previous_json) if previous_json else None
    
#     # if current_df is not None:
#     #     print("[DEBUG] NYISO Current DF Head:")
#     #     print(current_df.head())
#     #     previous_df.to_csv('xyz.csv')

#     # if previous_df is not None:
#     #     print("[DEBUG] NYISO Previous DF Head:")
#     #     print(previous_df.head())
#     #     previous_df.to_csv('abc.csv')
        
    
    
#     prompt_start = datetime(current_op.year, current_op.month, 1, tzinfo=timezone.utc) + relativedelta(months=1)
#     prompt_end = prompt_start + relativedelta(day=31)
    
#     print(current_op.strftime("%Y-%m-%d"),prev_op.strftime("%Y-%m-%d"))
#     return {
#         "success": current_df is not None and previous_df is not None,
#         "current_df": current_df,
#         "previous_df": previous_df,
#         "current_op_date": current_op.strftime("%Y-%m-%d"),
#         "previous_op_date": prev_op.strftime("%Y-%m-%d"),
#         "f_date": prompt_start,
#         "l_date": prompt_end
#     }

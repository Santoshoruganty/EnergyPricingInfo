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

from miso_data import get_miso_data


result = get_miso_data()
print(result.keys())
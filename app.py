import os
import streamlit as st
import re
import datetime
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('anime.csv')

st.write(df.head(5))
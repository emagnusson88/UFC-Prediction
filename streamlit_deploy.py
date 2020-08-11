import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

st.title("UFC Prediction")
st.markdown(
"""
Using past MMA fight and fighter data to predict the outcome of future bouts
""")

predictions = pd.read_csv(".\data\predictions.csv")

#st.write(predictions)

st.table(predictions)

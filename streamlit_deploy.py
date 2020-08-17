import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
from PIL import Image

image = Image.open("./archive/ufc_logo.png")

st.image(image,use_column_width=True, output_format='PNG')

st.title("UFC Fight Prediction")

st.subheader('Using past MMA fight and fighter data to predict the outcome of future bouts')

predictions = st.cache(pd.read_csv)("./data/predictions.csv")

#st.sidebar.markdown('Raw predictions')
st.table(predictions)

# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An example of showing geographic data."""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

DATE_TIME = "date/time"
DATA_URL = (
    "http://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz"
)

st.title("Uber Pickups in New York City")
st.markdown(
"""
This is a demo of a Streamlit app that shows the Uber pickups
geographical distribution in New York City. Use the slider
to pick a specific hour and look at how the charts change.
[See source code](https://github.com/streamlit/demo-uber-nyc-pickups/blob/master/app.py)
""")

@st.cache(persist=True)
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis="columns", inplace=True)
    data[DATE_TIME] = pd.to_datetime(data[DATE_TIME])
    return data

#data = load_data(10000)

data = pd.read_csv('predictions.csv')


"""hour = st.slider('hour', 0, 23, 10)
data = data[data[DATE_TIME].dt.hour == hour]

if st.checkbox('Show Raw Data'):
    'Raw Data at %sh' % hour, data

'## Geo Data at %sh' % hour
st.map(data)

st.subheader("Geo data between %i:00 and %i:00" % (hour, (hour + 1) % 24))
midpoint = (np.average(data["lat"]), np.average(data["lon"]))

st.write(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state={
        "latitude": midpoint[0],
        "longitude": midpoint[1],
        "zoom": 11,
        "pitch": 50,
    },
    layers=[
        pdk.Layer(
            "HexagonLayer",
            data=data,
            get_position=["lon", "lat"],
            radius=100,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
        ),
    ],
))"""

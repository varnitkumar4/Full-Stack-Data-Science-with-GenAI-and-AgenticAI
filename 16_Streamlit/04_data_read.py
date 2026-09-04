import streamlit as st
import pandas as pd
st.title("Chai sales dasboard")

file = st.file_uploader("Upload your csv file", type=['csv'])
df = pd.read_csv(file)
if file:
    # df = pd.read_csv(file)
    st.subheader("Data Preview")
    st.dataframe(df)

if file:
    st.subheader('Summry Stats')
    st.write(df.describe())

if file:
    cities = df["City"].unique()
    selected_city = st.selectbox('Filter by cities',cities)
    filterd_data = df[df["City"] == selected_city]
    st.dataframe(filterd_data)
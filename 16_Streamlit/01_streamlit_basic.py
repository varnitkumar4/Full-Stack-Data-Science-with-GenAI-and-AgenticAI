import streamlit as st

st.title("Hello Varnit")
st.subheader("start streamlit")
st.text("Welcome to the first app")
st.write("How are you")


code = st.selectbox("Select your favorite Language",['java','python','c','cpp','HTML','css','SQL'])

st.write(f'Your favorite language is  **{code}**')
st.success(f'you choose your favorite language ***{code}***')
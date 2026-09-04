import streamlit as st


code = st.number_input("write a number to check even or odd",step=1)

if code % 2 == 0:
    st.write("even")
else:
    st.write("odd")

code1 = st.number_input("write a number to check number is divided bu 4 or not",step=1)

if code1 % 4 == 0:
    st.write("number is divided by 4")
else:
    st.write("not divided bt 4")
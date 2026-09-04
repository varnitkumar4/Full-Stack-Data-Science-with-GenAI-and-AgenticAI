import streamlit as st

# 1. Add a title to your app
st.title("My first streamlit app created by Varnit")

# 2. Add some text
st.write("Welcome this app calculate the square of a number.")

# 3. create a intreractive slider
st.header("Select a Number")
number = st.slider("Pick a number", 1,20,5) # min , max , default

# 4. Calculate and display the result
st.subheader("Result")

squared_number = number ** 2

st.write(f"The square of **{number}** is **{squared_number}**.")
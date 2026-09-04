import streamlit as st

st.title("Chai Test Poll")

col1 , col2 =st.columns(2)

with col1:
    st.header("Masala chai")
    st.image("https://imgs.search.brave.com/dLEim4IS7KEQ1tVrfahxWkssvdh0KxuD9r5v2O7z4xg/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9pLnBp/bmltZy5jb20vb3Jp/Z2luYWxzL2QyL2Y0/LzRlL2QyZjQ0ZTkx/NDk1MzgzNGNkNDc4/NGY4NDk3ZjhiZWU2/LmpwZw",width=200) 
    vote1 = st.button("vote Masala chai")

with col2:
    st.header("Adrak chai")
    st.image("https://imgs.search.brave.com/TOU4WgyjCJafjMNmoJPixzJfsG3U7zvIKjHHldIVy0w/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9uYXZ2/YXlkLmNvbS9jZG4v/c2hvcC9maWxlcy9B/ZHJha0VsYWljaGlD/aGFpXzQ4ZjE0Y2Ri/LTRhODYtNDE0YS05/MjYwLTkzMmZmNDYx/ZjVjOC53ZWJwP2Ny/b3A9Y2VudGVyJmhl/aWdodD0yMDQ4JnY9/MTc2ODYzMzUzNSZ3/aWR0aD0yMDQ4",width=200)
    vote2 = st.button("vote Adrak chai")

if vote1:
    st.success("Thanks for voting Masala chai")
elif vote2:
    st.success("Thanks for voting Adrak chai")

name = st.sidebar.text_input("Enter your name")
tea = st.sidebar.selectbox("choose your chai",["Masla",'Adrak','kesar'])

st.write(f"Welcome {name} and your {tea} chai is getting ready")

with st.expander("show chai making instruction"):
    st.write("""
        1. Boil water with tea leaves
        2. add milk and spices
        3. serve hot
""")

st.markdown('### Welcome to chai app')
st.markdown('> Blockcode')
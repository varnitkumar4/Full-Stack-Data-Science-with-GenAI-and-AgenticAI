import streamlit as st 

st.title("Game maker app")

if st.button("Game make"):
    st.success("your game is maked")

add = st.checkbox("add some feature in your game")
if add:
    st.write("feature added in your game")

game_type = st.radio("pick your game :",['pubg','snake','suduko','chees','ludo'])

st.write(f'select game is **{game_type}**')

favorite = st.selectbox("select our favorite game ",['pubg','snake','suduko','chees','ludo'])
st.write(f"your favorite game is **{favorite}**")

dificalti = st.slider("Dificalti lavel",1,10,2)
st.write(f"your choose dificalti is **{dificalti}**")

level = st.number_input("which leve you want to play", min_value=1,max_value=20,step=1)
st.write(f"your choose **{level}** level to play a game")

name = st.text_input("enter your name ")
if name:
    st.write(f'welcome ***{name}*** in **{favorite}** world')

dob = st.date_input("Enter our date of birth")
st.write(f'your dob is **{dob}**')
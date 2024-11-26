import streamlit as st

st.checkbox('yes')
st.button('Click')

pilihan = st.radio(
    'pick your gender',
    options=['Male', 'Female']
)
st.write(f'Kamu memilih: {pilihan}')

st.selectbox('Male', 'Female')

mark = st.slider.sele("Pick a mark", "Bad", "good", "Excellent")
import numpy as np
import altair as alt
import pandas as pd
import streamlit as st

st.header('st.write')
st.write('Hello, *World!* :sunglasses:')
st.write(1234)

df = pd.DataFrame({
    'First Column' : [1, 2, 3, 4],
    'Second Column' : [10, 20, 30, 40]
})
st.write(df)

st.write('Below is a DataFrame:', df, 'Above is a dataframe.')

df2 = pd.DataFrame(
    np.random.randn(200, 3),
    columns=['a', 'b', 'c'])
c = alt.Chart(df2).mark_circle().encode(
    x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
st.write(c)
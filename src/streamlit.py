from backend import get_prediction, get_true_rating, load_models, sentiment
import streamlit as st

st.title('Classification of film reviews')

text = st.text_input(label='Enter feedback', value='')

if text == '':
    pass
else:
    rating = get_prediction(text)
    true_rating = get_true_rating(rating)
    st.markdown(f'Evaluating your feedback: **{rating}**')
    st.markdown(f'This is **{sentiment(true_rating)}**')
            


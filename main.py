import streamlit as st
import langchain_helper as lch
import textwrap

st.title("You Tube Assistant")

with st.sidebar:
    with st.form(key="my_form"):
        youtube_url = st.sidebar.text_area(
            label = " What is youtube vidoe URL",
            max_chars = 50
            
        )
        query = st.sidebar.text_area(

            label = "Ask me about the video?",
            max_chars = 800,
            key = "query"
        )

        submit_buttion = st.form_submit_button(label = 'Submit')

if query and youtube_url:
    db = lch.create_vector_db(youtube_url)
    response = lch.get_response_from_query(db, query) 

    st.subheader("Anwser:")
    st.text(textwrap.fill(response, width=80))
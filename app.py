import streamlit as st

pages = [
        st.Page("./pages/Home.py"),
        st.Page("./pages/Individual_Stock_Analysis.py"),
        st.Page("./pages/Portfolio_Analysis.py"),
]
pg = st.navigation(pages)
pg.run()
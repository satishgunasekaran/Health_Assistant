import streamlit as st

# Initialize the app and define the pages

# Set the app title
st.set_page_config(page_title="Health Bot", page_icon="🩺")

st.title("Health Bot Information")
st.write("""
    This Health Bot provides general health advice. Please remember:
    - This chatbot does not replace professional medical advice.
    - Consult a healthcare provider for any serious concerns.
    """)

# Run the app

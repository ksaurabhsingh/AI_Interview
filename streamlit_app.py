import streamlit as st
import requests
import time

st.title("AI Interviewer")

st.markdown("""
This application conducts AI-driven voice interviews. Enter your question below, and the system will:
1. Convert the question into speech.
2. Signal when to start speaking (beep sound).
3. Record and transcribe the candidate's response within 5 seconds.
4. Evaluate the response using AI.
""")

question = st.text_input("Enter your interview question:")

if st.button("Ask Question"):
    if not question:
        st.error("Please enter a question.")
    else:
        st.info("You will have **5 seconds to speak** after the beep sound.")
        st.write("The system will start processing now...")
        time.sleep(2)  

        with st.spinner("Processing... Please wait."):
           
            try:
                response = requests.post("http://127.0.0.1:5000/ask_question", json={"question": question}, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    st.success("Processing completed. Check the results below.")
                    st.write("### Question:", data["question"])
                    st.write("### Candidate's Answer:", data["answer"])
                    st.write("### Evaluation:", data["evaluation"])
                else:
                    st.error(f"Error: {response.json().get('error', 'Unknown error')}")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {str(e)}")

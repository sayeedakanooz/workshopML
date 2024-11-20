# Q&A Chatbot
import cohere
import streamlit as st

# Initialize Cohere client
COHERE_API_KEY = "hcfT4NcRx0u4GyVHRZx2ZoKpRwgS5uIrZPAZ3YcR"  # Replace with your API key
co = cohere.Client(COHERE_API_KEY)

# Function to load Cohere model and get a response
def get_cohere_response(question):
    response = co.generate(
        model="command-xlarge-nightly",  # You can adjust the model as needed
        prompt=question,
        max_tokens=300,  # Adjust max_tokens as needed
        temperature=0.5  # Similar to OpenAI's temperature for creativity
    )
    return response.generations[0].text.strip()

# Initialize Streamlit app
st.set_page_config(page_title="Q&A Demo")

st.header("Langchain Application with Cohere")

input_text = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

# If ask button is clicked
if submit:
    if input_text:
        st.subheader("The Response is")
        response = get_cohere_response(input_text)
        st.write(response)
    else:
        st.warning("Please enter a question!")



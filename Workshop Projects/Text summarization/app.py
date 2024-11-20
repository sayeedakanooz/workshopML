import streamlit as st
import cohere

# Initialize Cohere API client with your API key
api_key = 'hcfT4NcRx0u4GyVHRZx2ZoKpRwgS5uIrZPAZ3YcR'  # Replace this with your actual Cohere API key
co = cohere.Client(api_key)

# Function to summarize text
def summarize_text(text: str) -> str:
    response = co.generate(
        model='command-xlarge-nightly',  # Use 'large' model, change if needed
        prompt=f"Summarize the following text:\n\n{text}",
        max_tokens=1500,  # Adjust length of summary
        temperature=0.7,  # Adjust creativity of the response
        stop_sequences=["\n"]
    )
    summary = response.generations[0].text.strip()
    return summary

# Streamlit app
def main():
    st.title("Text Summarizer with Cohere")

    # User input for text
    text_input = st.text_area("Enter Text to Summarize", height=300)

    if st.button("Summarize"):
        if text_input:
            # Call the summarization function and display the result
            summary = summarize_text(text_input)
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Please enter some text to summarize.")

if __name__ == "__main__":
    main()

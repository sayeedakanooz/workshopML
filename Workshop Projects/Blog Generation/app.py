import streamlit as st
import cohere

# Initialize Cohere client
COHERE_API_KEY = "hcfT4NcRx0u4GyVHRZx2ZoKpRwgS5uIrZPAZ3YcR"  # Replace with your Cohere API key
co = cohere.Client(COHERE_API_KEY)

# Function to get response from Cohere API
def get_cohere_response(input_text, no_words, blog_style):
    # Prompt template
    prompt = f"""
    Write a blog for {blog_style} job profile on the topic "{input_text}" within {no_words} words.
    """
    
    # Generate response using Cohere API
    response = co.generate(
        model="command-xlarge-nightly",  # You can adjust the model based on your needs
        prompt=prompt,
        max_tokens=int(no_words),  # Ensuring the word limit
        temperature=0.7  # Adjust temperature for creativity
    )
    
    # Extract and return the generated text
    return response.generations[0].text.strip()

# Streamlit app configuration
st.set_page_config(page_title="Generate Blogs",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

# Input fields
input_text = st.text_input("Enter the Blog Topic")

# Two-column layout for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('Number of Words')
with col2:
    blog_style = st.selectbox(
        'Writing the blog for',
        ('Researchers', 'Data Scientist', 'Common People'), index=0)

# Submit button
submit = st.button("Generate")

# Generate and display the blog
if submit:
    if input_text and no_words.isdigit():
        st.subheader("Generated Blog")
        response = get_cohere_response(input_text, no_words, blog_style)
        st.write(response)
    else:
        st.warning("Please enter a valid blog topic and a numeric word count!")

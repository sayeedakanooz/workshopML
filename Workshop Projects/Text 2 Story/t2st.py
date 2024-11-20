import streamlit as st
import cohere

# Initialize Cohere Client
COHERE_API_KEY = "hcfT4NcRx0u4GyVHRZx2ZoKpRwgS5uIrZPAZ3YcR"  # Replace with your actual API key
co = cohere.Client(COHERE_API_KEY)

# Function to generate a long story
def generate_story(prompt, length="long"):
    try:
        # Use Cohere's generate endpoint
        response = co.generate(
            model="command-xlarge-nightly",  # Use a suitable Cohere model
            prompt=prompt,
            max_tokens=1000,  # Adjust for story length (1000 tokens for long stories)
            temperature=0.8,  # Creativity level
            k=0,  # No top-k sampling
            p=0.9,  # Top-p sampling for diversity
            frequency_penalty=0.2,  # Penalize frequent tokens (removed presence_penalty)
            stop_sequences=["--"],  # Define optional stop sequences
        )
        return response.generations[0].text
    except Exception as e:
        return f"Error generating story: {e}"


# Streamlit UI
st.title("Long Story Generator")
st.write("Generate a detailed story from a simple prompt!")

# User input
prompt = st.text_area("Enter a basic story prompt:", "Once upon a time in a distant galaxy,")
length = st.selectbox("Select story length:", ["short", "medium", "long"], index=2)

# Generate story
if st.button("Generate Story"):
    st.write("Generating story...")
    story = generate_story(prompt, length)
    st.subheader("Generated Story")
    st.write(story)

# Save story to a text file
if st.button("Save Story"):
    with open("generated_story.txt", "w") as f:
        f.write(story)
    st.success("Story saved as 'generated_story.txt'")

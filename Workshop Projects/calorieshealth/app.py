import streamlit as st
import cohere
from PIL import Image

# Initialize Cohere client
COHERE_API_KEY = "hcfT4NcRx0u4GyVHRZx2ZoKpRwgS5uIrZPAZ3YcR"  # Replace with your Cohere API key
co = cohere.Client(COHERE_API_KEY)

# Function to process image (optional placeholder for image handling)
def process_image(uploaded_file):
    # Placeholder: You can integrate an image-processing library or API here
    # For now, we'll return a placeholder message.
    return "Processed image data (details of food items)."

# Function to generate a response from Cohere
def get_cohere_response(prompt):
    response = co.generate(
        model="command-xlarge-nightly",
        prompt=prompt,
        max_tokens=300,  # Adjust as needed
        temperature=0.7  # Adjust creativity
    )
    return response.generations[0].text.strip()

# Streamlit App Configuration
st.set_page_config(page_title="Cohere Nutrition Analysis", page_icon="🥗")

st.header("Cohere Nutrition Analysis")

# Text input
input_prompt = st.text_input("Describe what you want analyzed:", key="input")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Submit button
submit = st.button("Analyze Food Nutrition")

# Predefined prompt template
nutrition_prompt_template = """
You are a nutrition expert. Based on the following description, calculate the total calories and provide a breakdown of each food item's calorie count in the format:

1. Item 1 - X calories
2. Item 2 - Y calories
---
Description: {description}
"""

if submit:
    if input_prompt:
        # Placeholder image processing
        processed_image_data = ""
        if uploaded_file:
            processed_image_data = process_image(uploaded_file)
        
        # Combine text and image information (if available)
        final_prompt = nutrition_prompt_template.format(
            description=f"{input_prompt}. {processed_image_data}"
        )
        
        # Get Cohere response
        response = get_cohere_response(final_prompt)
        
        # Display the result
        st.subheader("The Response is:")
        st.write(response)
    else:
        st.warning("Please enter a description or upload an image.")

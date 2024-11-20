import os
import cv2
import numpy as np
import streamlit as st
from diffusers import StableDiffusionPipeline
from gtts import gTTS
import torch
from PIL import Image

# Initialize Stable Diffusion Pipeline
def setup_pipeline():
    # Use the pre-trained Stable Diffusion model from Hugging Face
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

# Generate a story (can use a language model API if needed)
def generate_story(prompt):
    # For simplicity, generate a very basic story based on the prompt
    story = f"The adventure begins when {prompt}. Along the way, they encounter several challenges and surprises."
    return story

# Generate images for the story scenes
def generate_images(pipe, story, num_frames=10):
    images = []
    for i in range(num_frames):
        # Modify prompt to simulate scenes for different frames
        scene_prompt = f"{story} Scene {i+1}"
        image = pipe(scene_prompt).images[0]  # Generate one image per scene
        images.append(image)
    return images

# Convert images to a video
def create_video_from_images(images, output_path, frame_rate=1):
    if len(images) == 0:
        raise ValueError("No images to create a video.")

    # Get image dimensions
    width, height = images[0].size
    size = (width, height)

    # Initialize video writer
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, size)

    for image in images:
        # Convert PIL image to numpy array
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()

# Convert text to speech for narration
def text_to_speech(story, output_audio_path="audio.mp3"):
    tts = gTTS(text=story, lang="en", slow=False)
    tts.save(output_audio_path)

# Streamlit UI
st.title("Text-to-Video Generator")
st.write("Generate a video from your prompt. A story will be generated and visualized with images!")

# Input: Text prompt for the story
prompt = st.text_input("Enter a prompt for the video:", "A brave knight sets off on a journey")

# Options
num_frames = st.slider("Number of frames:", min_value=5, max_value=20, value=10)
frame_rate = st.slider("Frame rate (fps):", min_value=1, max_value=10, value=2)

# Generate video
if st.button("Generate Video"):
    st.write("Generating story...")
    story = generate_story(prompt)
    
    st.write("Setting up image generation pipeline...")
    pipe = setup_pipeline()

    st.write("Generating images for scenes...")
    images = generate_images(pipe, story, num_frames)

    st.write("Creating video from images...")
    video_path = "generated_video.mp4"
    create_video_from_images(images, video_path, frame_rate)

    st.write("Generating narration for the video...")
    audio_path = "audio.mp3"
    text_to_speech(story, audio_path)

    # Display results
    st.success("Video and audio created successfully!")
    st.video(video_path)
    
    # Option to download the video and audio
    with open(video_path, "rb") as video_file:
        st.download_button("Download Video", data=video_file, file_name="generated_video.mp4", mime="video/mp4")
    
    with open(audio_path, "rb") as audio_file:
        st.download_button("Download Audio", data=audio_file, file_name="audio.mp3", mime="audio/mp3")

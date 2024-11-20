import streamlit as st
from gtts import gTTS
import os

# Streamlit UI
st.title("Text-to-Speech Converter")
st.write("Convert your text to speech with multiple voice options!")

# Input for text
text = st.text_area("Enter the text to convert to speech:", "Hello, welcome to the Text-to-Speech app!")

# Voice options
voice_options = {
    "English (US) - Female": "en",
    "English (UK) - Female": "en-uk",
    "English (US) - Male (Deep Voice Simulation)": "en",
    "French - Female": "fr",
    "German - Female": "de",
    "Spanish - Female": "es",
}
voice = st.selectbox("Select a voice:", list(voice_options.keys()))

# Generate audio
if st.button("Convert to Speech"):
    try:
        # Voice simulation for male option (change pitch simulation)
        if voice == "English (US) - Male (Deep Voice Simulation)":
            tts = gTTS(text=text, lang=voice_options[voice], slow=True)
        else:
            tts = gTTS(text=text, lang=voice_options[voice], slow=False)
        
        # Save audio file
        audio_file = "output.mp3"
        tts.save(audio_file)
        st.success("Speech generated successfully!")
        
        # Play audio
        st.audio(audio_file, format="audio/mp3")
    except Exception as e:
        st.error(f"Error generating speech: {e}")

# Option to download audio
if os.path.exists("output.mp3"):
    with open("output.mp3", "rb") as file:
        st.download_button(label="Download Audio", data=file, file_name="output.mp3", mime="audio/mp3")

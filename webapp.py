# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:30:21 2023

@author: Parthipan R
"""
import streamlit as st
import os
import numpy as np
import librosa
import tensorflow as tf
from keras.models import load_model
# import matplotlib.pyplot as plt

model = load_model('./audio_classifier_LSTM_model_v1.h5')

def pad_audio_if_needed(audio, n_fft):
    if len(audio) < n_fft:
        padding = n_fft - len(audio)
        audio = np.pad(audio, pad_width=(0, padding), mode='constant')
    return audio

def pad_or_truncate(array, max_length):
        if array.shape[1] < max_length:
            padding = max_length - array.shape[1]
            array = np.pad(array, pad_width=((0, 0), (0, padding)), mode='constant')
        else:
            array = array[:, :max_length]
        return array
    
def extract_features(file_path, max_time_steps=109, SAMPLE_RATE=16000, N_MELS=128):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    audio = pad_audio_if_needed(audio, n_fft=512)
    
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=SAMPLE_RATE)
    tonnetz = librosa.feature.tonnetz(y=audio, sr=SAMPLE_RATE)

    # Pad or truncate each feature to have the same width
    mel_spectrogram = pad_or_truncate(mel_spectrogram, max_time_steps)
    mfccs = pad_or_truncate(mfccs, max_time_steps)
    chroma = pad_or_truncate(chroma, max_time_steps)
    spectral_contrast = pad_or_truncate(spectral_contrast, max_time_steps)
    tonnetz = pad_or_truncate(tonnetz, max_time_steps)
    
    features = np.concatenate((mel_spectrogram, mfccs, chroma, spectral_contrast, tonnetz), axis=0)
    return features

def classify_audio(audio_file):
    # Load audio file
    extracted_features = np.array(extract_features(file_path=audio_file))
    processed_audio = np.expand_dims(extracted_features, axis=0)
    prediction = model.predict(processed_audio)
    prediction = np.argmax(prediction, axis=1)
    if prediction == 1:
        return "Human Voice"
    else:
        return "AI Generated Voice"

# classify_audio('../audio-deepfake-detection-main/TestEvaluation/LA_E_4785445.flac')


# Streamlit app
# st.title('Human vs AI Voice Classifier')

# uploaded_file = st.file_uploader("Choose an audio file...", type=['wav', 'mp3', 'flac'])

# if uploaded_file is not None:
#     # Call the classify_audio function
#     result = classify_audio(uploaded_file)
#     st.write(f'The audio file is classified as: {result}')

# Init Session state 
if 'upload_history' not in st.session_state:
    st.session_state['upload_history'] = []

# Layout and styling
st.set_page_config(page_title='API_Project', layout='wide')
st.title('Silicon vs Vocal Cords')

# Using columns for layout
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader("Upload an audio file...", type=['wav', 'mp3', 'flac'])
    if uploaded_file is not None:
        with st.spinner('Processing...'):
            # Call your classify_audio function here
            result = classify_audio(uploaded_file)
            st.success(f'The audio file is classified as: {result}')
            
            # # Display the corresponding image
            # if result == "Human Voice":
            #     st.image('./imgs/human.png', caption='Human Voice')
            # else:
            #     st.image('./imgs/robot.png', caption='AI Generated Voice')
            
            # Update the history
            st.session_state['upload_history'].append((uploaded_file.name, result))
            
    # Display the history
    # st.markdown("**History**")
    # for filename, result in reversed(st.session_state['upload_history']):
    #     st.text(f"File: {filename} : {result}")
        
    # Display the history
    # st.markdown("## Classification History")
    # for filename, result in reversed(st.session_state['upload_history']):
    #     # st.markdown(f"**File:** {filename}", unsafe_allow_html=True)
    #     if result == "Human Voice":
    #         st.markdown(f'**File:** {filename} <button type="button" style="color:white; background-color:green; padding:5px 10px; border-radius:5px; border:none;">Human</button>', unsafe_allow_html=True)
    #     else:
    #         st.markdown(f'**File:** {filename} <button type="button" style="color:white; background-color:red; padding:5px 10px; border-radius:5px; border:none;">AI</button>', unsafe_allow_html=True)
    if True:
        with st.expander("Classification History"):
            for filename, result in reversed(st.session_state['upload_history']):
                if result == "Human Voice":
                    st.markdown(f'**File:** {filename} <button type="button" style="color:white; background-color:green; padding:5px 10px; border-radius:5px; border:none;">Human</button>', unsafe_allow_html=True)
                else:
                    st.markdown(f'**File:** {filename} <button type="button" style="color:white; background-color:red; padding:5px 10px; border-radius:5px; border:none;">AI</button>', unsafe_allow_html=True)


    with col2:
        st.markdown("## Instructions")
        st.markdown("""
            - Upload an audio file in WAV, MP3, or FLAC format.
            - The classifier will determine if the audio is human or AI-generated.
            - Results will be displayed immediately after processing.
        """)

# Additional features
if st.button('Show More Info'):
    with st.expander("See explanation"):
        st.write("""
            The program will extract the audio features like Mel Spectrogram, MFCCs, Tonnetz, Chroma and Spectral Contrast 
            from the audio file provides and run a classifier model on top it to classify the audio as Human or AI generated.
            
            The latest model accuracy is around 95%.
        """)

# Custom CSS Styling
st.markdown("""
    <style>
    .main {
    }
    </style>
    """, unsafe_allow_html=True)

# Footer
footer_html = """
<div style='position: fixed; bottom: 0; width: 100%; text-align: center; font-size: 12px;'>
    <hr style='border-color: #F0F2F6;'>
    <p>Developed by 🧑‍💻 Parthipan Ramakrishnan</p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)

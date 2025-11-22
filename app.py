import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# --- Load models ---
VIDEO_MODEL_PATH = 'Video_model/Video-CNN_model3_21-11-25_11-18-56_face.keras'
AUDIO_MODEL_PATH = 'Audio-CNN 1D_model2.h5'

video_model = load_model(VIDEO_MODEL_PATH)
audio_model = load_model(AUDIO_MODEL_PATH)

# --- Fusion Model ---
video_input = video_model.input
video_embedding = video_model.layers[-2].output

audio_input = Input(shape=audio_model.input_shape[1:], name='audio_input')
x = audio_input
for layer in audio_model.layers[1:-1]:  # skip input & output
    x = layer(x)
audio_embedding = x

combined = Concatenate(name='fusion_concat')([video_embedding, audio_embedding])
x = BatchNormalization(name='fusion_bn1')(combined)
x = Dense(512, kernel_regularizer=l2(1e-4), name='fusion_dense1')(x)
x = LeakyReLU(name='fusion_leakyrelu1')(x)
x = Dropout(0.4, name='fusion_dropout1')(x)
x = BatchNormalization(name='fusion_bn2')(x)
x = Dense(256, kernel_regularizer=l2(1e-4), name='fusion_dense2')(x)
x = LeakyReLU(name='fusion_leakyrelu2')(x)
x = Dropout(0.3, name='fusion_dropout2')(x)
x = BatchNormalization(name='fusion_bn3')(x)
x = Dense(128, kernel_regularizer=l2(1e-4), name='fusion_dense3')(x)
x = LeakyReLU(name='fusion_leakyrelu3')(x)
x = Dropout(0.2, name='fusion_dropout3')(x)
output = Dense(7, activation='softmax', name='fusion_output')(x)

fusion_model = Model(inputs=[video_input, audio_input], outputs=output)
fusion_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

# --- Streamlit UI ---
st.set_page_config(page_title='Emotion Recognition Fusion App', layout='centered')
st.title('Emotion Recognition Fusion App')
st.write('Upload a video and audio file to predict emotion using the combined model.')

video_file = st.file_uploader('Upload video file', type=['mp4', 'avi', 'mov'])
audio_file = st.file_uploader('Upload audio file', type=['wav', 'mp3'])

def preprocess_video(file):
    # Replace with actual video preprocessing
    return np.random.rand(1, *video_model.input_shape[1:])

def preprocess_audio(file):
    # Replace with actual audio preprocessing
    return np.random.rand(1, *audio_model.input_shape[1:])

emotion_labels = [
    ("Neutral", "Take a deep breath and enjoy the moment! 😊"),
    ("Calm", "Stay calm and keep shining! 🌿"),
    ("Happy", "Keep smiling, the world needs your joy! 😄"),
    ("Sad", "It's okay to feel sad. Remember, brighter days are ahead. 💙"),
    ("Angry", "Take a pause, breathe, and let go. You got this! 😌"),
    ("Fearful", "You are stronger than you think. Stay brave! 💪"),
    ("Disgust", "Let go of what bothers you. You deserve peace! 🌸")
]

if st.button('Predict'):
    if video_file is not None and audio_file is not None:
        X_video = preprocess_video(video_file)
        X_audio = preprocess_audio(audio_file)
        pred = fusion_model.predict([X_video, X_audio])
        emotion_class = int(np.argmax(pred, axis=1)[0])
        label, comfort = emotion_labels[emotion_class]
        st.success(f'Predicted emotion: {label}')
        st.markdown(f'<div style="font-size:1.3em;">{comfort}</div>', unsafe_allow_html=True)
    else:
        st.warning('Please upload both video and audio files.')

st.info('Note: Replace the preprocess_video and preprocess_audio functions with actual preprocessing.')

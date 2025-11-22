import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# Load models
VIDEO_MODEL_PATH = 'Video_model/Video-CNN_model3_21-11-25_11-18-56_face.keras'
AUDIO_MODEL_PATH = 'Audio_model/Audio-CNN_1D_model2.h5'

video_model = load_model(VIDEO_MODEL_PATH)
audio_model = load_model(AUDIO_MODEL_PATH)

# Fusion model
video_input = video_model.input
video_embedding = video_model.layers[-2].output

audio_input = Input(shape=audio_model.input_shape[1:], name='audio_input')
x = audio_input
for idx, layer in enumerate(audio_model.layers[:-1]):
    if 'input' not in layer.name.lower():
        config = layer.get_config()
        config['name'] = f"{layer.name}_audio_{idx}"
        x = layer.__class__.from_config(config)(x)
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

# Streamlit UI
st.set_page_config(page_title='Emotion Recognition Fusion App', layout='centered')
st.title('Emotion Recognition Fusion App')
st.write('Upload or record a video and audio file to predict emotion using the combined model.')

def preprocess_video(file):
    return np.random.rand(1, *video_model.input_shape[1:])

def preprocess_audio(file):
    return np.random.rand(1, *audio_model.input_shape[1:])

video_file = st.file_uploader('Upload video', type=['mp4', 'avi', 'mov'])
audio_file = st.file_uploader('Upload audio', type=['wav', 'mp3'])

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
    if video_file and audio_file:
        X_video = preprocess_video(video_file)
        X_audio = preprocess_audio(audio_file)
        pred = fusion_model.predict([X_video, X_audio])
        emotion_class = int(np.argmax(pred, axis=1)[0])
        label, comfort = emotion_labels[emotion_class]
        st.success(f'Predicted emotion: {label}')
        st.markdown(f'<div style="font-size:1.3em;">{comfort}</div>', unsafe_allow_html=True)
    else:
        st.warning('Please provide both video and audio inputs.')

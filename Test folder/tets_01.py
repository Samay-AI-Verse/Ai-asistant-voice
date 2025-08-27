import streamlit as st
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import io
import soundfile as sf
from groq import Groq
from a4f_local import A4F
import datetime
import os
from dotenv import load_dotenv
import logging
import langdetect
import time
import asyncio
import threading
import wave
import urllib.error
import json
import base64
import queue

# ---------- CONFIG ----------
MEMORY_FILE = "shakti_memory.json"
logging.basicConfig(filename='shakti.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- LOAD API ----------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check for API key
if not GROQ_API_KEY:
    st.error("Please set your GROQ_API_KEY in a .env file.")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)
# NOTE: The `A4F` client is a local library. Ensure it's in your environment.
tts_client = A4F()

# ---------- SPEECH RECOGNIZER ----------
recognizer = sr.Recognizer()
recognizer.energy_threshold = 150
recognizer.dynamic_energy_threshold = True

# ---------- CONTEXT ----------
last_command_time = 0
context_timeout = 20
preferred_language = "en"
conversation_history = []

supported_languages = {
    "en": "en-US",
    "hi": "hi-IN",
    "mr": "mr-IN"
}

# ---------- MEMORY ----------
def save_memory():
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.conversation_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Memory save error: {e}")

def load_memory():
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            st.session_state.conversation_history = json.load(f)
    except FileNotFoundError:
        st.session_state.conversation_history = []
    except Exception as e:
        logging.error(f"Memory load error: {e}")
        st.session_state.conversation_history = []

# ---------- UTILITIES ----------
def play_chime():
    try:
        with wave.open("chime.wav", "rb") as wf:
            data = wf.readframes(wf.getnframes())
            samplerate = wf.getframerate()
        sd.play(np.frombuffer(data, dtype=np.int16), samplerate)
        sd.wait()
    except Exception as e:
        logging.error(f"Chime error: {e}")
        sd.play(np.zeros(800), 16000)  # Silent fallback
        sd.wait()

def calibrate_microphone(duration=2, fs=16000, chunk_sec=0.3):
    chunk_frames = int(chunk_sec * fs)
    samples = []
    with sd.InputStream(samplerate=fs, channels=1, dtype='int16', blocksize=chunk_frames) as stream:
        print("🎧 Calibrating mic...")
        for _ in range(int(duration / chunk_sec)):
            chunk, _ = stream.read(chunk_frames)
            samples.extend(chunk.flatten())
    noise_level = np.mean(np.abs(samples)) / 32767.0
    threshold = max(noise_level * 3.0, 0.01)
    print(f"✅ Calibrated threshold: {threshold:.5f}")
    return threshold

# ---------- CAPTURE AUDIO ----------
def capture_audio(fs=16000, chunk_sec=0.3, energy_threshold=0.01,
                 silence_sec=1.5, max_phrase_sec=20):
    chunk_frames = int(chunk_sec * fs)
    max_silence_chunks = int(silence_sec / chunk_sec)
    max_phrase_chunks = int(max_phrase_sec / chunk_sec)
    audio_chunks = []
    silence_chunks = 0
    speech_started = False
    try:
        with sd.InputStream(samplerate=fs, channels=1, dtype='int16',
                            blocksize=chunk_frames) as stream:
            print("🎙 Waiting for speech...")
            while len(audio_chunks) < max_phrase_chunks:
                chunk, overflowed = stream.read(chunk_frames)
                if overflowed:
                    logging.error("Audio overflow detected")
                energy = np.mean(np.abs(chunk.flatten())) / 32767.0
                if energy > energy_threshold:
                    if not speech_started:
                        print("🎤 Speech detected!")
                    speech_started = True
                    audio_chunks.append(chunk)
                    silence_chunks = 0
                elif speech_started:
                    silence_chunks += 1
                    if silence_chunks >= max_silence_chunks:
                        print("⏹ End of speech detected")
                        break
        if not audio_chunks:
            return None, "en"
        recording = np.concatenate(audio_chunks, axis=0)
        audio_data = np.squeeze(recording)
        return sr.AudioData(audio_data.tobytes(), sample_rate=fs, sample_width=2), "en"
    except Exception as e:
        logging.error(f"Recording error: {e}")
        return None, "en"

# ---------- LISTEN ----------
def listen_to_user():
    threading.Thread(target=play_chime).start()
    audio, detected_lang = capture_audio()
    if not audio:
        return "", detected_lang
    text = ""
    try:
        text = recognizer.recognize_google(audio, language=supported_languages[st.session_state.preferred_language])
        try:
            detected_lang = langdetect.detect(text)
            if detected_lang not in supported_languages:
                detected_lang = st.session_state.preferred_language
        except:
            detected_lang = st.session_state.preferred_language
        text_lower = text.lower()
        st.session_state.last_command_time = time.time()
        st.session_state.conversation_history.append({"role": "user", "content": text_lower})
        save_memory()
        return text_lower, detected_lang
    except (sr.UnknownValueError, urllib.error.URLError, ConnectionResetError):
        return "unrecognized", detected_lang
    except Exception as e:
        logging.error(f"STT error: {e}")
        return "error", detected_lang

# ---------- ASK GROQ ----------
def ask_groq(prompt, lang):
    try:
        lang_prompt = f"Respond in {lang.upper()} language."
        system_prompt = f"""
        You are Shakti, a witty, culturally resonant AI assistant inspired by Indian heritage. 
        {lang_prompt} Keep responses concise, conversational, under 2 sentences, 
        with a charming tone. Address the user as 'ji' in Hindi/Marathi or 'sir/madam' in English.
        """
        if "time" in prompt.lower():
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            response = f"The time is {current_time}, sir." if lang == "en" else f"समय है {current_time}, जी।" if lang == "hi" else f"वेळ आहे {current_time}, साहेब।"
            st.session_state.conversation_history.append({"role": "assistant", "content": response})
            save_memory()
            return response
        messages = [{"role": "system", "content": system_prompt}] + st.session_state.conversation_history[-5:] + [{"role": "user", "content": prompt}]
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            max_tokens=70
        )
        response = chat_completion.choices[0].message.content
        st.session_state.conversation_history.append({"role": "assistant", "content": response})
        save_memory()
        return response
    except Exception as e:
        logging.error(f"Groq API error: {e}")
        response = "Sorry, I couldn't process that now." if lang == "en" else "क्षमा करें, मैं अभी प्रोसेस नहीं कर सकी, जी।" if lang == "hi" else "क्षमा करा, मी आता प्रक्रिया करू शकत नाही, साहेब।"
        st.session_state.conversation_history.append({"role": "assistant", "content": response})
        save_memory()
        return response

# ---------- SPEAK ----------
def speak_in_memory(text, lang, voice="shimmer"):
    try:
        audio_bytes = tts_client.audio.speech.create(
            model="tts-1",
            input=text,
            voice=voice
        )
        audio_stream = io.BytesIO(audio_bytes)
        data, samplerate = sf.read(audio_stream, dtype="float32")
        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        logging.error(f"TTS error: {e}")

# ---------- STREAMLIT APP ----------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Custom CSS for the glowing circle
def add_css():
    st.markdown("""
    <style>
    @keyframes pulse {
        0% { box-shadow: 0 0 10px #4d94ff, 0 0 20px #4d94ff, 0 0 30px #4d94ff; transform: scale(1); }
        50% { box-shadow: 0 0 20px #99c2ff, 0 0 40px #99c2ff, 0 0 60px #99c2ff; transform: scale(1.05); }
        100% { box-shadow: 0 0 10px #4d94ff, 0 0 20px #4d94ff, 0 0 30px #4d94ff; transform: scale(1); }
    }
    .glowing-circle {
        width: 250px;
        height: 250px;
        background-color: rgba(77, 148, 255, 0.2);
        border: 2px solid #4d94ff;
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        animation: pulse 2s infinite ease-in-out;
    }
    .glowing-circle h1 {
        font-size: 5rem;
        color: #fff;
    }
    </style>
    """, unsafe_allow_html=True)

# State management for the main loop
if 'listening' not in st.session_state:
    st.session_state.listening = False
if 'last_command_time' not in st.session_state:
    st.session_state.last_command_time = time.time()
if 'preferred_language' not in st.session_state:
    st.session_state.preferred_language = "en"
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'message_queue' not in st.session_state:
    st.session_state.message_queue = queue.Queue()

def start_listening_thread():
    if not st.session_state.listening:
        st.session_state.listening = True
        st.session_state.message_queue.put("start")
        thread = threading.Thread(target=main_thread_loop)
        thread.start()

def main_thread_loop():
    try:
        while st.session_state.listening:
            if not st.session_state.message_queue.empty():
                command = st.session_state.message_queue.get()
                if command == "stop":
                    break

            if time.time() - st.session_state.last_command_time > 60:
                calibrate_microphone()
                st.session_state.last_command_time = time.time()
            
            user_input, detected_lang = listen_to_user()
            
            if user_input:
                if user_input.lower() in ["exit", "quit", "stop", "bye"]:
                    speak_in_memory("Goodbye, until next time.", detected_lang)
                    st.session_state.listening = False
                    break
                
                ai_response = ask_groq(user_input, detected_lang)
                st.session_state.message_queue.put(f"User: {user_input}\nShakti: {ai_response}")
                speak_in_memory(ai_response, detected_lang)
            
            time.sleep(0.1)
    except Exception as e:
        logging.error(f"Main loop thread error: {e}")
        st.session_state.listening = False
    finally:
        st.session_state.listening = False
        st.session_state.message_queue.put("stop")

# Main Streamlit UI
st.set_page_config(layout="wide")

# This will only work if you have a dark background image locally
# set_background('dark_background.png') 
add_css()

# Use a container to center the content
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<div class='glowing-circle'><h1>Hi</h1></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    if not st.session_state.listening:
        if st.button("Start Listening"):
            st.session_state.listening = True
            st.session_state.message_queue.put("start")
            st.rerun()  # Corrected line
    else:
        st.write("Listening... Say 'exit' to stop.")
        if st.button("Stop Listening"):
            st.session_state.listening = False
            st.session_state.message_queue.put("stop")
            st.rerun()  # Corrected line

    # Display the conversation history
    st.markdown("---")
    st.subheader("Conversation History")
    for chat in st.session_state.conversation_history:
        role = chat["role"].capitalize()
        content = chat["content"]
        st.write(f"**{role}:** {content}")

    # Check the queue for new messages and update the UI
    while not st.session_state.message_queue.empty():
        new_message = st.session_state.message_queue.get()
        if new_message.startswith("User:"):
            pass
        elif new_message == "stop":
            st.stop()


# Initial setup
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    load_memory()
    calibrate_microphone()
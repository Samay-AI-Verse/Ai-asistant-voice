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

# ---------- CONFIG ----------
MEMORY_FILE = "shakti_memory.json"
logging.basicConfig(filename='shakti.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- LOAD API ----------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
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
            json.dump(conversation_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Memory save error: {e}")

def load_memory():
    global conversation_history
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            conversation_history = json.load(f)
    except FileNotFoundError:
        conversation_history = []

# ---------- UTILITIES ----------
def play_chime():
    try:
        with wave.open("chime.wav", "rb") as wf:
            data = wf.readframes(wf.getnframes())
            samplerate = wf.getframerate()
        sd.play(np.frombuffer(data, dtype=np.int16), samplerate)
        sd.wait()
    except Exception:
        sd.play(np.zeros(800), 16000)  # Silent fallback
        sd.wait()

def calibrate_microphone(duration=2, fs=16000, chunk_sec=0.3):
    """
    Listens for background noise and sets dynamic threshold.
    """
    chunk_frames = int(chunk_sec * fs)
    samples = []

    with sd.InputStream(samplerate=fs, channels=1, dtype='int16', blocksize=chunk_frames) as stream:
        print("🎧 Calibrating mic... stay silent")
        for _ in range(int(duration / chunk_sec)):
            chunk, _ = stream.read(chunk_frames)
            samples.extend(chunk.flatten())

    noise_level = np.mean(np.abs(samples)) / 32767.0
    threshold = max(noise_level * 3.0, 0.01)  # noise × factor
    print(f"✅ Calibrated threshold: {threshold:.5f}")
    return threshold

# ---------- CAPTURE AUDIO ----------
async def capture_audio(fs=16000, chunk_sec=0.3, energy_threshold=0.01,
                        silence_sec=1.5, max_phrase_sec=20):
    """
    Captures speech until user finishes speaking.
    - fs: sample rate
    - chunk_sec: duration per chunk
    - energy_threshold: min energy to count as speech
    - silence_sec: silence duration before cutting off
    - max_phrase_sec: hard limit on one utterance
    """
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
async def listen_to_user():
    global last_command_time, preferred_language
    threading.Thread(target=play_chime).start()

    audio, detected_lang = await capture_audio()
    if not audio:
        return "", detected_lang

    text = ""
    try:
        text = recognizer.recognize_google(audio, language=supported_languages[preferred_language])
        try:
            detected_lang = langdetect.detect(text)
            if detected_lang not in supported_languages:
                detected_lang = preferred_language
        except:
            detected_lang = preferred_language

        text_lower = text.lower()
        last_command_time = time.time()

        conversation_history.append({"role": "user", "content": text_lower})
        save_memory()
        return text_lower, detected_lang

    except (sr.UnknownValueError, urllib.error.URLError, ConnectionResetError):
        return "unrecognized", detected_lang
    except Exception as e:
        logging.error(f"STT error: {e}")
        return "error", detected_lang

# ---------- ASK GROQ ----------
async def ask_groq(prompt, lang):
    global conversation_history, preferred_language
    try:
        lang_prompt = f"Respond in {lang.upper()} language."
        system_prompt = f"""
        You are Shakti, a witty, culturally resonant AI assistant inspired by Indian heritage. 
        {lang_prompt} Keep responses concise, conversational, under 2 sentences, 
        with a charming tone. Address the user as 'ji' in Hindi/Marathi or 'sir/madam' in English.
        """

        # Local intents
        if "time" in prompt.lower():
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            response = f"The time is {current_time}, sir." if lang == "en" else f"समय है {current_time}, जी।" if lang == "hi" else f"वेळ आहे {current_time}, साहेब।"
            conversation_history.append({"role": "assistant", "content": response})
            save_memory()
            return response

        # Build context for LLM
        messages = [{"role": "system", "content": system_prompt}] + conversation_history[-5:] + [{"role": "user", "content": prompt}]
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            max_tokens=70
        )
        response = chat_completion.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": response})
        save_memory()
        return response

    except Exception as e:
        logging.error(f"Groq API error: {e}")
        response = "Sorry, I couldn't process that now." if lang == "en" else "क्षमा करें, मैं अभी प्रोसेस नहीं कर सकी, जी।" if lang == "hi" else "क्षमा करा, मी आता प्रक्रिया करू शकत नाही, साहेब।"
        conversation_history.append({"role": "assistant", "content": response})
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

# ---------- MAIN ----------
async def main():
    print("🤖 Shakti ready. Speak in English, Hindi, or Marathi. Say 'exit' to quit.")
    load_memory()
    calibrate_microphone()
    speak_in_memory("Shakti online, ready to assist.", "en")

    global last_command_time
    last_command_time = time.time()

    while True:
        try:
            # Recalibrate mic every 60 sec
            if time.time() - last_command_time > 60:
                calibrate_microphone()
                last_command_time = time.time()

            user_input, detected_lang = await listen_to_user()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "stop", "bye"]:
                print("👋 Shutting down...")
                speak_in_memory("Goodbye, until next time.", detected_lang)
                break

            ai_response = await ask_groq(user_input, detected_lang)
            print(f"🤖 Shakti: {ai_response}")
            speak_in_memory(ai_response, detected_lang)

        except Exception as e:
            logging.error(f"Main loop error: {e}")
            speak_in_memory("Apologies, something went wrong. Please try again.", "en")

if __name__ == "__main__":
    asyncio.run(main())

import sounddevice as sd
import numpy as np
import io
import soundfile as sf
from groq import Groq
from a4f_local import A4F
import datetime
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)
tts_client = A4F()

# 🎙 Listen and use Groq Whisper for STT
def listen_to_user(duration=10, fs=16000):
    print("🎙 Listening...")
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
        sd.wait()
        audio_data = np.squeeze(recording)

        # Save to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            sf.write(temp_audio.name, audio_data, fs)
            temp_audio_path = temp_audio.name

        # Send to Groq Whisper for transcription
        with open(temp_audio_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(temp_audio_path, f.read()),
                model="whisper-large-v3"
            )

        text = transcription.text
        print("🗣 You said:", text)

        # Clean up temp file
        os.remove(temp_audio_path)

        return text
    except Exception as e:
        print(f"⚠️ STT error: {e}")
        return ""

# 🤖 Ask Groq AI
def ask_groq(prompt):
    try:
        system_prompt = """You are a helpful voice assistant. Keep your responses brief, conversational, and under 2 sentences. 
        Speak like a real person, not a textbook. Use simple language."""

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            max_tokens=100
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"❌ Groq API error: {e}")
        return "Sorry, I couldn't process that."

# 🔊 Speak + Save audio + Save text
def speak_and_save(text, voice="shimmer"):
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

        filename = f"assistant_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        with open(filename, "wb") as f:
            f.write(audio_bytes)
        print(f"💾 Voice response saved as {filename}")

        with open("assistant_responses.txt", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now()}] {text}\n")
        print("📝 Text response saved to assistant_responses.txt")

    except Exception as e:
        print(f"❌ TTS error: {e}")

# 🚀 Main loop
def main():
    print("🤖 Personal Voice Assistant started. Say 'exit' to quit.")
    while True:
        user_input = listen_to_user()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "stop", "bye", "goodbye"]:
            print("👋 Goodbye!")
            speak_and_save("Goodbye! Have a great day.")
            break

        ai_response = ask_groq(user_input)
        print("🤖 Assistant:", ai_response)

        speak_and_save(ai_response)

if __name__ == "__main__":
    main()

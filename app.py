import os
from flask import Flask, request, send_file
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from dotenv import load_dotenv
import whisper
from gtts import gTTS
from io import BytesIO
from groq import Groq
import requests

# -----------------------------
# CONFIG
# -----------------------------
load_dotenv()
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_WHATSAPP = "whatsapp:+14155238886"  # Twilio sandbox number
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BASE_URL = os.getenv("BASE_URL")

client = Client(TWILIO_SID, TWILIO_AUTH)
groq_client = Groq(api_key=GROQ_API_KEY)

# Load Whisper model
whisper_model = whisper.load_model("tiny")  # lightweight for speed

app = Flask(__name__)

# -----------------------------
# WHATSAPP WEBHOOK
# -----------------------------
@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.values.get("Body", "").strip()
    media_url = request.values.get("MediaUrl0")
    media_type = request.values.get("MediaContentType0", "")

    resp = MessagingResponse()
    reply_text = "ക്ഷമിക്കണം, ഞാൻ അത് മനസ്സിലാക്കിയില്ല."

    try:
        # -----------------------------
        # 1️⃣ Text message
        # -----------------------------
        if incoming_msg:
            reply_text = process_llm(incoming_msg)

        # -----------------------------
        # 2️⃣ Voice message
        # -----------------------------
        elif media_url and "audio" in media_type:
            audio_bytes = requests.get(media_url).content
            with open("audio.ogg", "wb") as f:
                f.write(audio_bytes)
            result = whisper_model.transcribe("audio.ogg")
            user_text = result["text"]
            reply_text = process_llm(user_text)

        # -----------------------------
        # 3️⃣ Image message
        # -----------------------------
        elif media_url and "image" in media_type:
            reply_text = analyze_image(media_url)

        # -----------------------------
        # 4️⃣ Send text reply
        # -----------------------------
        resp.message(reply_text)

        # -----------------------------
        # 5️⃣ Send Malayalam audio reply
        # -----------------------------
        send_audio_tts(reply_text, request.values.get("From"))

    except Exception as e:
        print("Error:", e)
        resp.message("ക്ഷമിക്കണം! പിശക് സംഭവിച്ചു.")

    return str(resp)

# -----------------------------
# LLM Processing
# -----------------------------
def process_llm(user_text):
    prompt = f"You are a Kerala farming assistant. User said: {user_text}. Reply in Malayalam."
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant"
    )
    return chat_completion.choices[0].message.content.strip()

# -----------------------------
# Image Analysis (placeholder)
# -----------------------------
def analyze_image(image_url):
    return "ചിത്രം പരിശോധിച്ചു. ഇലയിൽ രോഗ ലക്ഷണങ്ങൾ കാണുന്നു, ജൈവ കീടനാശിനി ഉപയോഗിക്കുക."

# -----------------------------
# Generate TTS audio
# -----------------------------
def send_audio_tts(text, to):
    tts = gTTS(text=text, lang="ml")
    file_path = "reply.mp3"
    tts.save(file_path)

    client.messages.create(
        from_=TWILIO_WHATSAPP,
        to=to,
        body="🔊 Malayalam audio reply",
        media_url=[f"{BASE_URL}/reply.mp3"]
    )

# -----------------------------
# Serve TTS file
# -----------------------------
@app.route("/reply.mp3")
def serve_audio():
    return send_file("reply.mp3", mimetype="audio/mpeg")

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

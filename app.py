import os
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from dotenv import load_dotenv
import requests
import whisper
from gtts import gTTS
from io import BytesIO
from groq import Groq

# -----------------------------
# CONFIG
# -----------------------------
load_dotenv()
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_WHATSAPP = "whatsapp:+14155238886"   # Twilio Sandbox number
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Client(TWILIO_SID, TWILIO_AUTH)
groq_client = Groq(api_key=GROQ_API_KEY)

# Whisper tiny (fast) for Malayalam/English
whisper_model = whisper.load_model("tiny")

app = Flask(__name__)

# -----------------------------
# WEBHOOK - MAIN BOT
# -----------------------------
@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.values.get("Body", "").strip()
    media_url = request.values.get("MediaUrl0")
    media_type = request.values.get("MediaContentType0", "")

    resp = MessagingResponse()
    reply_text = "ക്ഷമിക്കണം, ഞാൻ അത് മനസ്സിലാക്കിയില്ല."

    try:
        # 1️⃣ Handle Text
        if incoming_msg:
            reply_text = process_llm(incoming_msg, lang="ml")

        # 2️⃣ Handle Audio
        elif media_url and "audio" in media_type:
            audio_bytes = requests.get(media_url).content
            with open("audio.ogg", "wb") as f:
                f.write(audio_bytes)
            result = whisper_model.transcribe("audio.ogg")
            user_text = result["text"].strip()
            reply_text = process_llm(user_text, lang="ml")

        # 3️⃣ Handle Image
        elif media_url and "image" in media_type:
            reply_text = analyze_image(media_url)

        # Send text reply
        resp.message(reply_text)

        # Send audio reply (Malayalam TTS)
        send_audio_tts(reply_text, request.values.get("From"))

    except Exception as e:
        print("Error:", e)
        resp.message("ക്ഷമിക്കണം! പിശക് സംഭവിച്ചു.")

    return str(resp)

# -----------------------------
# LLM Processing
# -----------------------------
def process_llm(user_text, lang="ml"):
    """Send text to Groq LLM and get Malayalam reply"""
    prompt = f"""
    You are a Kerala farming assistant. 
    User said: {user_text}
    Reply in Malayalam only, short and clear.
    """

    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant"
    )
    return chat_completion.choices[0].message.content.strip()

# -----------------------------
# Image Analysis (Vision LLM)
# -----------------------------
def analyze_image(image_url):
    """Analyze crop/disease photo using Vision model"""
    # Placeholder: you can connect Gemini/GPT-4o-mini Vision API here
    # For now, return dummy
    return "ചിത്രം പരിശോധിച്ചു. ഇലയിൽ രോഗ ലക്ഷണങ്ങൾ കാണുന്നു, ജൈവ കീടനാശിനി ഉപയോഗിക്കുക."

# -----------------------------
# Send Audio Back
# -----------------------------
def send_audio_tts(text, to):
    tts = gTTS(text=text, lang="ml")
    file_path = "reply.mp3"
    tts.save(file_path)

    client.messages.create(
        from_=TWILIO_WHATSAPP,
        to=to,
        body="🔊 ഓഡിയോ മറുപടി",
        media_url=[f"https://<your-server>/{file_path}"]  # must serve file via Flask
    )

# -----------------------------
# Serve Audio File
# -----------------------------
@app.route("/reply.mp3")
def serve_audio():
    return open("reply.mp3", "rb").read(), 200, {"Content-Type": "audio/mpeg"}

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

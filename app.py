import os
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq client
client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)

@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    """Handle incoming WhatsApp messages from Twilio Sandbox"""
    incoming_msg = request.form.get("Body")
    sender = request.form.get("From")  # e.g., whatsapp:+919876543210
    print(f"📩 Message from {sender}: {incoming_msg}")

    # Call Groq LLM
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # you can use mixtral, gemma etc.
        messages=[
            {"role": "system", "content": "You are a helpful AI WhatsApp assistant."},
            {"role": "user", "content": incoming_msg}
        ]
    )

    ai_reply = response.choices[0].message.content.strip()

    # Build Twilio WhatsApp response
    twilio_resp = MessagingResponse()
    twilio_resp.message(ai_reply)

    return str(twilio_resp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

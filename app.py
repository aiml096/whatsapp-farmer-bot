from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import requests
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms
from torchvision import models
import os
from groq import Groq   # or use OpenAI if preferred

app = Flask(__name__)

# -----------------------------
# Twilio Setup
# -----------------------------
ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
WHATSAPP_NUMBER = "whatsapp:+14155238886"  # Twilio Sandbox

client = Client(ACCOUNT_SID, AUTH_TOKEN)

# -----------------------------
# LLM Setup (Groq)
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm_client = Groq(api_key=GROQ_API_KEY)

# -----------------------------
# Image Model Setup (ResNet placeholder)
# -----------------------------
model = models.resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# WhatsApp Webhook
# -----------------------------
@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.form.get("Body", "")
    sender = request.form.get("From")
    num_media = int(request.form.get("NumMedia", 0))

    resp = MessagingResponse()
    reply = resp.message()

    if num_media > 0:
        # Farmer sent a photo
        media_url = request.form.get("MediaUrl0")
        file_type = request.form.get("MediaContentType0")

        if "image" in file_type:
            raw_result = analyze_plant_image(media_url)
            explanation = llm_explain(raw_result, "ml")
            reply.body(f"🌱 {explanation}")
        else:
            reply.body("❌ Only plant photos are supported for analysis.")
    else:
        # Farmer sent text
        explanation = llm_explain(incoming_msg, "text")
        reply.body(explanation)

    return str(resp)

# -----------------------------
# Image Analyzer
# -----------------------------
def analyze_plant_image(url):
    """Download and classify crop image"""
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_t = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_t)
            _, predicted = torch.max(output, 1)

        return f"Detected crop condition ID: {predicted.item()}"
    except Exception as e:
        return f"Error analyzing image: {e}"

# -----------------------------
# LLM Explanation
# -----------------------------
def llm_explain(user_input, mode="text"):
    try:
        if mode == "ml":
            prompt = (
                f"You are a Kerala farming assistant.\n"
                f"A plant analysis result is: {user_input}.\n"
                f"Explain this in simple Malayalam for a farmer, "
                f"with advice on what to do if it is a disease."
            )
        else:
            prompt = (
                f"You are a helpful Malayalam farming assistant. "
                f"Answer the farmer’s question clearly in Malayalam.\n"
                f"Farmer: {user_input}"
            )

        chat_completion = llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return chat_completion.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ LLM error: {e}"

# -----------------------------
# Proactive Message (Optional)
# -----------------------------
@app.route("/send", methods=["GET"])
def send_message():
    to = request.args.get("to")
    msg = request.args.get("msg", "🌾 Greetings from Kerala Farmer Bot!")

    message = client.messages.create(
        from_=WHATSAPP_NUMBER,
        to=to,
        body=msg
    )
    return {"status": "sent", "sid": message.sid}


if __name__ == "__main__":
    app.run(port=5000, debug=True)

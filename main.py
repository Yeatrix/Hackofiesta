from speech_recognition import Recognizer, AudioFile
import requests
import os
import base64

# Import the LLMModule from llm.py
from llm import LLMModule
from chat_reference import * 
from tts import * 

from elevenlabs.client import ElevenLabs
from elevenlabs import stream
from dotenv import load_dotenv

load_dotenv()

client = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)

# Initialize the LLM module
llm_module = LLMModule()

def text_to_speech(text, output_file):
    """Convert text to speech using Deepgram TTS and save as MP3."""
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    TTS_API_URL = "https://api.deepgram.com/v1/speak?model=aura-orpheus-en"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "text": text
    }
    response = requests.post(TTS_API_URL, headers=headers, json=payload, stream=True)
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        return True
    else:
        print(f"Deepgram TTS Error: {response.json()}")
        return False

def transcribe_audio(audio_file_path):
    """Transcribe audio using the SpeechRecognition library."""
    recognizer = Recognizer()
    with AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return ""

def main():
    print("Welcome to Telemedicine-Kiosk")

    file_path = "sample_dialogue.txt"
    chunks = preprocess_text(file_path)
    store_in_vector_db(chunks)

    # Initialize conversation history
    with open("prompt_template.txt", "r", encoding="utf-8") as file:
        text_content = file.read()

    messages = [
        ("system", text_content)
    ]

    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chat. Goodbye!")
            break
        
        context = retrieve_relevant_context(user_input)
        prompt = generate_prompt(user_input, context)
        user_input+=prompt

        messages.append(("user", user_input))
        
        # Generate AI response using LLMModule
        ai_response = llm_module.generate_response(messages)

        if ai_response:
            messages.append(("ai", ai_response))
            speak(client, ai_response)

        else:
            print('Telemedicine-Kiosk: Sorry, I could not generate a response.')

if __name__ == "__main__":
    main()

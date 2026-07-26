# nao_client.py (Runs on the Robot or via NAOqi environment)
import os
import time
import wave
import requests

# CONFIGURATION: Change this to your laptop's real local IP address on your Wi-Fi network
SERVER_IP = "127.0.0.1"  # Use "127.0.0.1" if testing on the same PC, or e.g., "192.168.1.5" for the robot
SERVER_URL = "http://{}:8000/ask".format(SERVER_IP)

AUDIO_FILE = "nao_input.wav"

def record_audio_from_mic(output_filename, duration=5):
    """
    Records audio cleanly from the microphone using sounddevice and built-in wave.
    No SciPy or PyAudio compilation required!
    """
    import sounddevice as sd
    import numpy as np
    import wave

    fs = 16000  # 16kHz is ideal for Whisper
    print("🤖 NAO: Listening... Speak now.")
    
    # Record audio into a numpy array (16-bit integers)
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until the recording timer finishes
    
    print("🤖 NAO: Recording stopped. Processing...")
    
    # Save the raw numpy binary data directly using Python's built-in wave library
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # 16-bit audio is exactly 2 bytes
        wf.setframerate(fs)
        wf.writeframes(recording.tobytes())


def send_audio_to_brain(filename):
    """Sends the audio file to the laptop over Wi-Fi and gets the text response."""
    print("🤖 NAO: Sending audio to laptop brain via Wi-Fi...")
    try:
        with open(filename, 'rb') as f:
            files = {'audio': (filename, f, 'audio/wav')}
            response = requests.post(SERVER_URL, files=files)
            
        if response.status_code == 200:
            return response.json()
        else:
            print("❌ Error: Server returned status code {}".format(response.status_code))
            return None
    except Exception as e:
        print("❌ Network Connection Error: {}".format(e))
        return None

def nao_speak(text):
    """
    Commands NAO to say the text out loud.
    """
    print("🤖 NAO Speaking: '{}'".format(text))
    try:
        # This is the official Aldebaran/SoftBank framework wrapper
        from naoqi import ALProxy
        tts = ALProxy("ALTextToSpeech", "127.0.0.1", 9559) # Connects to local robot core
        tts.say(text)
    except ImportError:
        # Fallback if you are running a test on your laptop without the robot connected
        print("[Simulation Mode: NAOqi SDK not detected. Cannot speak physically.]")

def main():
    # 1. Record the human's question
    record_audio_from_mic(AUDIO_FILE, duration=5)
    
    # 2. Transcribe and query via the laptop RAG server
    result = send_audio_to_brain(AUDIO_FILE)
    
    if result:
        print("📝 Heard Question: {}".format(result.get("question")))
        # 3. Speak the synthesized answer from the PDFs
        nao_speak(result.get("answer"))
    else:
        nao_speak("I am sorry, I am having trouble connecting to my server brain.")

    # Clean up the local audio file on the robot
    if os.path.exists(AUDIO_FILE):
        os.remove(AUDIO_FILE)

if __name__ == "__main__":
    main()

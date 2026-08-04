# body_client/nao_client.py (Run on your Windows laptop, talks to NAO over qi)
#
# Flow:
#   1. Connect to NAO via qi
#   2. Fetch and speak the fixed intro from the brain server
#   3. Loop: record mic -> send audio to brain server -> speak the answer
#      -> repeat, since the brain server's prompt already appends a
#      follow-up question to each answer

import os
import time
import requests
import qi

# ---- CONFIG: edit these for your setup ----
NAO_IP = "10.5.17.101"
NAO_PORT = 9561
SERVER_IP = "127.0.0.1"  # laptop running rag_server.py -- 127.0.0.1 if same machine
SERVER_URL = f"http://{SERVER_IP}:8000"
AUDIO_FILE = "nao_input.wav"
LISTEN_SECONDS = 5
# ---------------------------------------------


def record_audio_from_mic(output_filename, duration=LISTEN_SECONDS):
    """Records from the laptop's own microphone (not NAO's) using sounddevice."""
    import sounddevice as sd
    import wave

    fs = 16000  # 16kHz is what Whisper expects
    print("Listening... speak now.")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    print("Recording stopped. Sending to brain server...")

    with wave.open(output_filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(recording.tobytes())


def get_intro():
    """Fetches the fixed introduction text from the brain server."""
    try:
        response = requests.get(f"{SERVER_URL}/intro")
        return response.json().get("answer")
    except Exception as e:
        print(f"Could not reach brain server for intro: {e}")
        return "Hi, I'm NAO. I'm having trouble reaching my brain server right now."


def ask_brain(filename):
    """Sends recorded audio to the brain server, gets back question + answer text."""
    try:
        with open(filename, "rb") as f:
            files = {"audio": (filename, f, "audio/wav")}
            response = requests.post(f"{SERVER_URL}/ask", files=files)
        if response.status_code == 200:
            return response.json()
        print(f"Server returned status {response.status_code}")
        return None
    except Exception as e:
        print(f"Network error talking to brain server: {e}")
        return None


def main():
    print("Connecting to NAO...")
    print("Connecting to NAO...")
    tts = None
    try:
        session = qi.Session()
        session.connect(f"tcp://{NAO_IP}:{NAO_PORT}")
        tts = session.service("ALTextToSpeech")

        # Optional but recommended: stop Autonomous Life from interrupting speech
        try:
            life = session.service("ALAutonomousLife")
            if life.getState() != "disabled":
                life.setState("disabled")
        except Exception:
            pass  # not critical if this service isn't available

        print("Connected to NAO. Starting conversation.\n")
    except RuntimeError as e:
        print(f"Could not reach NAO ({e})")
        print("Running in TEXT-ONLY mode -- everything else will be tested normally,\n"
              "answers will just print here instead of NAO speaking them.\n")

    def speak(text):
        """Speaks via NAO if connected, otherwise just prints -- lets you
        test the full mic -> Whisper -> RAG -> Gemini pipeline without
        the robot physically present."""
        print(f"NAO: {text}")
        if tts is not None:
            tts.say(text)

    intro = get_intro()
    speak(intro)

    while True:
        record_audio_from_mic(AUDIO_FILE)
        result = ask_brain(AUDIO_FILE)

        if os.path.exists(AUDIO_FILE):
            os.remove(AUDIO_FILE)

        if not result:
            speak("Sorry, I'm having trouble reaching my brain right now.")
            continue

        question = result.get("question", "")
        answer = result.get("answer", "")

        print(f"Heard: {question}")

        if answer:
            speak(answer)


if __name__ == "__main__":
    main()
import sounddevice as sd
import numpy as np
import time
from datetime import datetime
import os
import json
import telebot
import winsound

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
CONFIG_PATH = os.path.join(ROOT_DIR, "telegram_config.json")

# Load Telegram config
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

BOT_TOKEN = cfg["bot_token"]
CHAT_ID = cfg["chat_id"]

bot = telebot.TeleBot(BOT_TOKEN)

SAMPLE_RATE = 44100      # mic sample rate
DURATION = 0.7           # seconds per chunk
LOUD_THRESHOLD = 0.04    # tune this for sensitivity (lower = more sensitive)
COOLDOWN = 12            # seconds between alerts
last_alert = 0


def play_alarm():
    try:
        for _ in range(2):
            winsound.Beep(1500, 400)
            time.sleep(0.1)
    except Exception as e:
        print("[ALARM ERROR]", e)


def send_audio_alert(volume):
    global last_alert
    now = time.time()

    if now - last_alert < COOLDOWN:
        return

    last_alert = now

    ts = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    msg = f"🔊 POSSIBLE DISTRESS SOUND DETECTED\nTime: {ts}\nVolume Spike: {volume:.4f}"

    try:
        bot.send_message(CHAT_ID, msg)
        print("[AUDIO ALERT SENT]", msg)
        play_alarm()
    except Exception as e:
        print("[TELEGRAM AUDIO ERROR]", e)


def start_audio_monitor():
    print("🎧 Audio distress monitor started (Ctrl+C to stop)...")
    while True:
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()

        volume = float(np.linalg.norm(audio) / len(audio))
        print("Volume:", volume)

        if volume > LOUD_THRESHOLD:
            print("🔊 LOUD NOISE / POSSIBLE SCREAM DETECTED")
            send_audio_alert(volume)


if __name__ == "__main__":
    start_audio_monitor()

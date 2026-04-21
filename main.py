from autobahn.twisted.component import Component, run
from twisted.internet.defer import inlineCallbacks
from twisted.internet import reactor
from autobahn.twisted.util import sleep
from google import genai
import os
import time
import threading
from tts import text_to_speech
from alpha_mini_rug.speech_to_text import SpeechToText
from hint_model.session import SessionManager
from hint_model.model import ACTION_LABELS

# ─────────────────────────────────────────
# CONFIGURATION — edit these before each session
# ─────────────────────────────────────────

LANGUAGE = "en"         # "en" for English (testing), "nl" for dutch (real sessions)
PATIENT_ID = "..."      # therapist types patients first name here
TRAINING_MODE = True    # True = therapist present, False = home practice

# Words to describe in order - therapist fills this in before each session
SESSION_WORDS = ["apple", "chair", "bicycle", "umbrella", "telephone"]

SILENCE_THRESHOLD_SECONDS = 3   # seconds of silence before chcking if done
MAX_TURN_SECONDS = 25           # maximum seconds to wait for patient response
HOME_CHECK_INTERVAL = 2         # seconds between Gemini "DONE or WAITING" checks
HINT_WAITING_THRESHOLD = 5      # seconds of WAITING before the LSTM hint check

# ────────────────────────────────────────

audio_processor = SpeechToText()
audio_processor.silence

import numpy as np
import sounddevice as sd
import whisper
import torch
import threading
import queue
import time

# Audio settings
SAMPLE_RATE = 16000       # Whisper expects 16kHz audio
WINDOW_SECONDS = 3        # How many seconds of audio per processing chunk
STEP_SECONDS = 1          # How often we process a new chunk (sliding window)
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS   # = 48000 samples
STEP_SAMPLES = SAMPLE_RATE * STEP_SECONDS       # = 16000 samples

# Filled pause words in both Dutch and English
FILLED_PAUSES = {"uh", "um", "eh", "hmm", "uhm", "erm", "euh", "aa", "uhh"}

# Silence detection — if audio volume is below this, we consider it silence
SILENCE_THRESHOLD = 0.01

class FeatureExtractor:
    def __init__(self, model_size="base", language="en"):
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"FeatureExtractor using device: {self.device}")

        # Load the Whisper model
        self.model = whisper.load_model(model_size).to(self.device)
        self.language = language # "en" for English, "nl" for Dutch

        # Buffer to hold the last few seconds of audio for processing
        self.audio_buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        
        # Queue to hold incoming audio chunks from the microphone
        self.audio_queue = queue.Queue()

        self.last_speech_time = time.time()

        self.latest_features = None

        self.running = False
    
    def _recoding_callback(self, indata, frames, time_info, status):
        """
        This funcion is called automatically by sounddevice every time a new chunk of audio is recorded from your microphone.
        It runs in a separate bachground thread.
        """
        # We take the first channel
        # we copy it to avoid data overwrite issues
        audio_chunk = indata[:, 0].copy()

        self.audio_queue.put(audio_chunk)

    def _processing_loop(self):
        """
        Runs in a background thread. Continuously reads audio from the queue,
        updates the rolling buffer, and extracts features.
        """

        accumulated = np.array([], dtype=np.float32)

        while self.running:
            try:
                # wait up to 0.1 seconds for new audio
                chunk = self.audio_queue.get(timeout=0.1)
                accumulated = np.append(accumulated, chunk)

                if len(accumulated) >= STEP_SAMPLES:
                    step = accumulated[:STEP_SAMPLES]
                    accumulated = accumulated[STEP_SAMPLES:]

                    # Slide the buffer: drop pldest 1 second, append newest 1 second
                    self.audio_buffer = np.roll(self.audio_buffer, -STEP_SAMPLES)
                    self.audio_buffer[-STEP_SAMPLES:] = step

                    # Now extract features from the current 3-second window
                    self.latest_features = self._extract_features(self.audio_buffer)

            except queue.Empty:
                continue # No audio yet, just keep waiting

    def _extract_features(self, audio):
        """
        Takes a 3-second audio window and returns a 1D numpy vector.
        This vector is what the LSTM will receive at each timestep.
        """
        # Feature 1 & 2: Silence duration and speech rate
        # Calculate the volume (rms: root mean square) of the audio
        rms = np.sqrt(np.mean(audio**2))
        is_speech = rms > SILENCE_THRESHOLD

        if is_speech:
            self.last_speech_time = time.time()
            silence_duration = 0.0
        else:
            silence_duration = time.time() - self.last_speech_time
        
        # whipser needs audio padded to 30 seconds
        audio_tensor = torch.from_numpy(whisper.pad_or_trim(audio)).to(self.device)

        # Compute log-mel spectrogram (what whisper actually sees)
        mel = whisper.log_mel_spectrogram(audio_tensor).unsqueeze(0)

        # Run whispers encoder to get acoustic embeddings
        with torch.no_grad():
            encoder_output = self.model.encoder(mel)

        # Encoder output is a 3D tensor: (batch x time x features)
        # take mean across time dimension to get a single vector
        embeddings = encoder_output.mean(dim=1).squeeze().cpu().numpy()

        # Whipser transcirprion for text based features
        result = self.model.transcribe(audio, language=self.language, fp16=False)
        text = result["text"].lower().strip()
        words = text.split()

        # Feature 3: Filled pause count
        pause_count = sum(1 for w in words if w in FILLED_PAUSES)

        # Feature 4: Speech rate (words per second)
        speech_rate = len(words) / max(WINDOW_SECONDS - silence_duration, 0.1)

        # Feature 5: Incomplete word (broken speech)
        common_short_words = {"i", "you", "he", "she", "it", "we", "they", "my", "your", "his", "her", "its", "our", "their", "a", "is", "it", "in", "on", "at", "to", "an", "of", "de", "het", "een"}
        last_word_incomplete = 0.0
        if words and len(words[-1]) <= 2 and words[-1] not in common_short_words:
            last_word_incomplete = 1.0

        # Combine all features into a single vector
        scalar_features = np.array([
            silence_duration,       # How long the patient has been silent
            float(pause_count),     # Number of filled pauses
            speech_rate,            # Words per second
            last_word_incomplete,   # 1.0 if speech seems cut off
            float(is_speech)],      # 1.0 if currently speaking
            dtype=np.float32)
        
        # Concatenate features with Whipser embeddings
        feature_vector = np.concatenate([scalar_features, embeddings])
        
        return feature_vector, text
    
    def start(self):
        """Starts the audio recording and processing threads."""
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

        # Start recording audio from the microphone
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=STEP_SAMPLES, callback=self._recoding_callback)
        self.stream.start()
        print("FeatureExtractor started - listening")
        
    def stop(self):
        """Stops the audio recording and processing threads."""
        self.running = False
        #self.processing_thread.join()
        self.stream.stop()
        self.stream.close()
        print("FeatureExtractor stopped")
    
    def get_latest_features(self):
        """Returns the most recently extracted feature vector and transcription text."""
        if self.latest_features is None:
            return None, None
        return self.latest_features
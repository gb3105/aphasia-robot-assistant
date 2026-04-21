import time
import threading
import numpy as np
from hint_model.features import FeatureExtractor
from hint_model.trainer import HintTrainer
from hint_model.therapist_ui import TherapistUI
from hint_model.model import ACTION_LABELS

class SessionManager:
    def __init__(self, patient_id, language="en", model_size="base", training_mode=True):
        self.patient_id = patient_id
        self.training_mode = training_mode

        # Initialize the feature extractor and trainer
        self.feature_extractor = FeatureExtractor(model_size=model_size, language=language)
        self.trainer = HintTrainer(patient_id=patient_id)

        # Initialize the therapist UI if in training mode, at home there is no keyboard presses
        if training_mode:
            self.therapist_ui = TherapistUI(self.trainer)
        else:
            self.therapist_ui = None
        
        # track if session is active
        self.active = False

        # Background thread that feeds features into the trainer every second
        self._tick_thread = None

    def start(self):
        """
        Loads the patients model, starts the feature extractor, 
        starts the keyboard listener (in in training mode) 
        and begins tick loop.
        """
        print(f"\n[Session] starting session for patient {self.patient_id}")
        print(f"[Session] Mode: {'Training' if self.training_mode else 'HOME PRACTICE'}")

        # Load the patient's model (if it exists)
        self.trainer.load_model()

        # Start listening on the microphone
        self.feature_extractor.start()

        # Start the therapist UI if in training mode
        if self.therapist_ui:
            self.therapist_ui.start()
        
        self.active = True
        # Start the background thread to feed features into the trainer every second
        self._tick_thread = threading.Thread(target=self._feature_feed_loop, daemon=True)
        self._tick_thread.start()

        print("[Session] Session started.\n")

    def _feature_feed_loop(self):
        """Background thread that runs every second to get features and feed them into the trainer."""
        while self.active:
            time.sleep(1.0)
            result = self.feature_extractor.get_latest_features()
            if result [0] is not None:
                features, _ = result
                
                self.trainer.update_buffer(features)
    
    def tick(self):
        """
        Called by the main conversation loop after each robot/patient interaction.
        In training mode#:
            - checks if the therapist pressed a key
            - if yes, returns the action index so the main loop can trigger the hint
            - the label is already stored in trainer by TherapistUI
        In home mode:
            - runs the model autonomously to decide wether to intervene
            - returns an action index if the model predicts help is needed
            - returns None if the model predicts do_nothing
        Returns:
            action_index (int): index of the action to take, or None for no action
        """

        if self.training_mode:
            # Check if the therapist has pressed a key to trigger a hint
            action_index = self.therapist_ui.get_pending_action()
            return action_index
        else:
            # In home mode, run the model to predict if we should intervene
            sequence = self.trainer.get_current_sequence()
            if sequence is None:
                return None
            
            self.trainer.model.eval()
            action_index, _, probs = self.trainer.model.select_action(sequence)

            # Only act if the model is confident (probability > 0.6) and action is not do_nothing
            confidence = probs[0][action_index].item()
            if action_index != 0 and confidence > 0.6:
                print(f"[Session] Model predicts action '{ACTION_LABELS[action_index]}' with confidence {confidence:.2f}.")
                return action_index
        
            return None
    
    def robot_speaking(self, is_speaking):
        """
        Call this to mute/unmute the feature extractor when the robot speaks.
        Prevents the robots own voice from being picked up as patient speech.

        is_speaking (bool): True if the robot starts speaking, False when it stops
        """
        if is_speaking:
            self.feature_extractor.robot_speaking_start()
        else:
            self.feature_extractor.robot_speaking_end()
    
    def end(self, save=True):
        """
        Ends the session cleanly:
        1. Stops the microphone and keyboard listener
        2. Trains on examples collected this session (if in training mode)
        3. Saves the updated model weights

        save: set to False to discard this session's learning
        """
        print("\n[Session] Ending session...")
        self.active = False

        # Stop the feature extractor and therapist UI
        self.feature_extractor.stop()
        if self.therapist_ui:
            self.therapist_ui.stop()

        # Train the model on examples collected this session
        if self.training_mode and save:
            n_examples = len(self.trainer.session_examples)
            if n_examples > 0:
                print(f"[Session] Training model on {n_examples} examples collected this session...")
                self.trainer.train_on_session(epochs=20)
                self.trainer.save_model()
            else:
                print("[Session] No examples collected this session, skipping training.")
        
        print(f"[Session] Session ended for patient '{self.patient_id}'.")
    
    def get_last_transcript(self):
        """
        Returns the most recent Whisper transcript.
        Used by main loop to pass patient speech to LLM.
        """
        result = self.feature_extractor.get_latest_features()
        if result[0] is not None:
            _, text = result
            return text
        return ""


if __name__ == "__main__":
    # Test training mode
    session = SessionManager(
        patient_id="test_patient",
        language="en",
        model_size="base",
        training_mode=True
    )

    session.start()

    print("Session running for 20 seconds.")
    print("Press 1, 2, or 3 to simulate therapist button presses.\n")

    for i in range(20):
        time.sleep(1)

        # Check for therapist input
        action = session.tick()
        if action is not None:
            print(f"[Test] tick() returned action: '{ACTION_LABELS[action]}'")
            print(f"[Test] → Robot would now give hint: {ACTION_LABELS[action]}")

        # Print transcript every 5 seconds
        if (i + 1) % 5 == 0:
            transcript = session.get_last_transcript()
            print(f"[Test] Latest transcript: '{transcript}'")

    session.end(save=False)  # save=False so we don't overwrite test_patient.pt
    print("\nTest complete.")
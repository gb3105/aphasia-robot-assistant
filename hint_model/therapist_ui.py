from pynput import keyboard
import threading
from hint_model.model import ACTION_LABELS


# maps keyboard keys to natching indices at ACTION_LABELS
KEY_MAP = {
    "1": 1,  # hint_sentence
    "2": 2,  # first_letter
    "3": 3,  # give_sound
}

HINT_TEMPLATES = {
    1: "hint_sentence",   # LLM generates a sentence with a blank
    2: "first_letter",    # we extract the first letter from the target word
    3: "give_sound",      # we extract the starting sound
}

class TherapistUI:
    def __init__(self, trainer):
        """
        trainer: a HintTrainer instance
        we call record_label() on it when a key is pressed
        """
        self.trainer = trainer

        # will hold the last triggered action
        self.pending_action = None

        # lock prevents two threads from reading/ writing pending_action to prebent race conditions
        self.lock = threading.Lock()

        # pynput listener starts automatically
        self.listener = None
        self.running = False
    
    def _on_press(self, key):
        """
        Called automatically by pynput when a key is pressed.
        Runs in the pynput background thread.
        """
        try:
            char = key.char
            if char in KEY_MAP:
                action_index = KEY_MAP[char]

                # record the label in the trainer, which will save it for training and also trigger the cooldown
                success = self.trainer.record_label(action_index)
                if success:
                    # use the lock to safely update the pending action
                    with self.lock:
                        self.pending_action = action_index
                print(f"[TherapistUI] Key '{char}' pressed. Recorded action: {ACTION_LABELS[action_index]}.")

        except AttributeError:
            # special keys (like shift) will cause this error, we can ignore them
            pass
    
    def start(self):
        """Starts the pynput listener to listen for key presses in background."""
        self.running = True
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()
        print("TherapistUI started. Press 1 for hint_sentence, 2 for first_letter, 3 for give_sound.")
    
    def stop(self):
        """Stops the pynput listener."""
        if self.listener:
            self.listener.stop()
        self.running = False
        print("TherapistUI stopped.")
    
    def get_pending_action(self):
        """
        Returns the pending action index if the therapist pressed a key, then clears it.
        Returns None if no key is pressed since last check.
        This is called by the main loop every second to check for hints.
        """
        with self.lock:
            action = self.pending_action
            self.pending_action = None
        return action



if __name__ == "__main__":
    import time

    # We need a mock trainer for testing — just an object with record_label()
    class MockTrainer:
        def record_label(self, action_index):
            print(f"[MockTrainer] record_label called with index {action_index}")
            return True  # always succeeds in mock

    mock_trainer = MockTrainer()
    ui = TherapistUI(trainer=mock_trainer)
    ui.start()

    print("\nPress keys 1, 2, or 3 to test. Running for 15 seconds...")
    for i in range(15):
        time.sleep(1)
        action = ui.get_pending_action()
        if action is not None:
            print(f"[Test] Main loop received pending action: '{ACTION_LABELS[action]}'")

    ui.stop()
    print("Test complete.")


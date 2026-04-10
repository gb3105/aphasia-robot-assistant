import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import time
from hint_model.model import HintPolicyNetwork, INPUT_SIZE, SEQUENCE_LENGTH, ACTION_LABELS, get_device

class HintTrainer:
    def __init__(self, patient_id, learning_rate=1e-3):
        self.patient_id = patient_id
        self.device = get_device()

        # Initialize the policy network and optimizer
        self.model = HintPolicyNetwork().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.criterion = nn.CrossEntropyLoss()

        self.feature_buffer = deque(maxlen=SEQUENCE_LENGTH)

        self.session_examples = [] # List of (feature_sequence, action_label) tuples for this session

        # Cooldown to prevent double-pressing
        self.cooldown_seconds = 8
        self.last_hint_time = 0

    def update_buffer(self, feature_vector):
        """
        Called every second with the last feature vector from features.py
        Keeps the rolling window of the last SEQUENCE_LENGTH timesteps up to date.
        feature_vector: numpy array of shape (INPUT_SIZE,)
        """
        self.feature_buffer.append(feature_vector)

    def get_current_sequence(self):
        """
        Returns the current sequence of features as a tensor of shape (1, SEQUENCE_LENGTH, INPUT_SIZE)
        If we don't have enough history yet, it pads with zeros.
        """
        n_real = len(self.feature_buffer)
        if n_real == 0:
            return None
        
        n_pad = SEQUENCE_LENGTH - n_real
        sequence = list(self.feature_buffer)

        if n_pad > 0:
            padding = [np.zeros(INPUT_SIZE, dtype=np.float32)] * n_pad
            sequence = padding + sequence

        sequence_array = np.stack(sequence, axis=0) # (SEQUENCE_LENGTH, INPUT_SIZE)
        tensor = torch.from_numpy(sequence_array).unsqueeze(0).to(self.device) # (1, SEQUENCE_LENGTH, INPUT_SIZE)
        return tensor
        
    def record_label(self, action_index):
        """
        Called when the therapist presses a hint button.
        action_index: int, index of the action in ACTION_LABELS
        Returns True if the label was recorded, False if it was on cooldown.
        """
        # Cooldown check to prevent double presses
        current_time = time.time()
        if current_time - self.last_hint_time < self.cooldown_seconds:
            remaining = self.cooldown_seconds - (current_time - self.last_hint_time)
            print(f"[Trainer] Hint on cooldown. Please wait {remaining:.1f} seconds before pressing again.")
            return False
        
        sequence = self.get_current_sequence()
        if sequence is None:
            print(f"[Trainer] Not enough audio collected yet.")
            return False

        self.session_examples.append((sequence.squeeze(0).cpu().numpy(), action_index))
        self.last_hint_time = current_time
        print(f"[Trainer] Recorded example for action '{ACTION_LABELS[action_index]}' | Total examples: {len(self.session_examples)}")
        return True
    
    def train_on_session(self, epochs=20):
        """
        Runs supervised training on all examples collected this session.
        Called once at the end of a therapy session.

        epochs: how many times we pass over all examples.
                20 is conservative -- safe for small datasets
        """
        if len(self.session_examples) == 0:
            print("[Trainer] No examples collected this session. Skipping training.")
            return
        
        print(f"[Trainer] Starting training on {len(self.session_examples)} examples for {epochs} epochs...")

        # Convert stored examples into tensors
        sequences = torch.tensor(np.stack([ex[0] for ex in self.session_examples]), dtype=torch.float32).to(self.device) # (N, SEQUENCE_LENGTH, INPUT_SIZE)
        labels = torch.tensor([ex[1] for ex in self.session_examples], dtype=torch.long).to(self.device) # (N,)

        # Put model in training mode
        self.model.train()

        for epoch in range(epochs):
            self.optimizer.zero_grad() # zero out gradients from previous step
            logits = self.model(sequences) # forward pass: get logits for all examples at once
            loss = self.criterion(logits, labels) # compute loss: how wrong are our predictions vs true labels
            loss.backward() # backpropagate to compute gradients for all parameters
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # clip gradients to prevent explosion
            
            self.optimizer.step()

            if (epoch+1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

        self.model.eval() # switch to eval mode to disable dropout, etc.
        print("[Trainer] Training complete. Model updated with new examples.")

        self.session_examples = [] # clear examples for next session
    
    def save_model(self, path="hint_model/patients"):
        """
        Saves the current model weights to a file named after the patient..
        Creates a directory if it doesn't exist.
        """
        import os
        os.makedirs(path, exist_ok=True)
        filepath = f"{path}/{self.patient_id}.pt"

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "patient_id": self.patient_id
            }, filepath)
        
        print(f"[Trainer] Model saved to {filepath}")
    
    def load_model(self, path="hint_model/patients"):
        """
        Loads model weights for this patient if a saved file exists.
        If no file exists (first session), starts from scratch.
        """
        filepath = f"{path}/{self.patient_id}.pt"
        import os
        if not os.path.exists(filepath):
            print(f"[Trainer] No saved model found for patient '{self.patient_id}'. Starting fresh.")
            return
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.model.eval() # set to eval mode by default after loading
        print(f"[Trainer] Loaded model for patient '{self.patient_id}' from {filepath}")







if __name__ == "__main__":
    trainer = HintTrainer(patient_id="test_patient")

    # Simulate 6 seconds of feature vectors coming in
    print("Simulating 6 timesteps of features...")
    for i in range(6):
        fake_features = np.random.randn(INPUT_SIZE).astype(np.float32)
        trainer.update_buffer(fake_features)
        print(f"  Timestep {i+1}: buffer size = {len(trainer.feature_buffer)}")

    # Simulate therapist pressing button 1 (hint_sentence)
    print("\nTherapist presses button 1 (hint_sentence)...")
    trainer.record_label(action_index=1)

    # Simulate therapist pressing button 2 (first_letter)
    print("Therapist presses button 2 immediately (should be in cooldown)...")
    trainer.record_label(action_index=2)

    # Train on collected examples
    trainer.train_on_session(epochs=10)

    # Save and reload
    trainer.save_model()
    trainer.load_model()
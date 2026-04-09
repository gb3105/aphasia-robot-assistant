import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Number of sclar features we defined
N_SCALAR_FEATURES = 5

# fixed by whispers achitecture
WHISPER_EMBEDDING_SIZE = 512

# Total input size to the network
INPUT_SIZE = N_SCALAR_FEATURES + WHISPER_EMBEDDING_SIZE

# How many timestamps the LSTM looks back at once
SEQUENCE_LENGTH = 5 # 5 seconds of the History

# The 4 possible actions the robot can take
ACTION_LABELS =["do_nothing", "hint_sentence", "first_letter", "give_sound"]
N_ACTIONS = len(ACTION_LABELS) 

class HintPolicyNetwork(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, mlp_hidden=128, lstm_hidden=128, n_actions=N_ACTIONS, dropout=0.3):
        """
        input_size: size of each featurer vector (517)
        mlp_hidden: size of the compressed representation after MLP
        lstm_hidden: number of units in the LSTM
        n_actions: number of possible actions (4)
        dropout: randomly zeros out neurons during training to prevent overfitting
        """
        super(HintPolicyNetwork).__init__()

        # Feature Compressor: MLP to reduce 517 features down to 128
        self.feature_mlp = nn.Sequential(
            nn.Linear(input_size, mlp_hidden), # 517 -> 128
            nn.ReLU(), # replaces negatives with 0
            nn.Dropout(dropout), # randomly drop 30% of neurons during training
            nn.Linear(mlp_hidden, mlp_hidden), # 128 -> 128
            nn.ReLU(),
        )

        # LSTM to process sequences of compressed features
        self.lstm = nn.LSTM(
            input_size=mlp_hidden, 
            hidden_size=lstm_hidden, 
            num_layers=2, # stack 2 LSTM ontop of each other for more capacity
            batch_first=True, # input shape: (batch, sequence, features)
            dropout=dropout # droput between the 2 LSTM layers
        )

        # Policy head: takes LSTM output and produces action probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden//2), # 128 -> 64
            nn.ReLU(),
            nn.Linear(lstm_hidden//2, n_actions), # 64 -> 4
        )

    def forward(self, x):
        """
        x: tensor of shape (batch_size, sequence_length, input_size)
        returns: logits of shape (batch_size, n_actions)
        """
        batch_size, seq_len, _ = x.shape

        # Compress each timesteps feature with the MLP
        x = x.view(batch_size * seq_len, -1) # (batch*seq_len, input_size
        x = self.feature_mlp(x) # (batch*seq_len, mlp_hidden)

        # Reshape back to (batch, seq_len, mlp_hidden) for LSTM
        x = x.view(batch_size, seq_len, -1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        last_timestep = lstm_out[:, -1, :]

        # Map to action scores
        logits = self.policy_head(last_timestep)
        return logits
    
    def get_action_probs(self, x):
        """
        x: tensor of shape (batch_size, sequence_length, input_size)
        returns: probabilities of shape (batch_size, n_actions)
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def select_action(self, x):
        """
        x: tensor of shape (1, sequence_length, input_size) for a single example
        returns: action index (0 to n_actions-1)
        """
        probs = self.get_action_probs(x) # (1, n_actions)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample() # randomly sample an action according to the probabilities
        log_prob = dist.log_prob(action) # log probability of the selected action (useful for training)
        return action.item(), log_prob, probs
    
    
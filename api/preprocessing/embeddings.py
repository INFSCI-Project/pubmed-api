import numpy as np

import torch
import transformers
from transformers import AutoTokenizer, AutoModel


class BioBertEmbedding:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "dmis-lab/biobert-base-cased-v1.1")
        self.model = AutoModel.from_pretrained(
            "dmis-lab/biobert-base-cased-v1.1", output_attentions=True)

    def generate_embedding(self, doc):
        inputs = self.tokenizer(doc, return_tensors="pt",
                                truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Shape: [batch_size, num_heads, seq_len, seq_len]
        attention_scores = torch.softmax(outputs.attentions[-1], dim=-1)
        average_attention = attention_scores.mean(
            dim=1)  # Average across attention heads
        average_attention = average_attention.mean(
            dim=2)  # Average across tokens
        # Shape: (batch_size, seq_length, 1)
        average_attention_expanded = average_attention.unsqueeze(-1)

        weighted_hidden_states = average_attention_expanded * \
            outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)
        weighted_hidden_states = weighted_hidden_states.mean(
            dim=1).squeeze(dim=0)  # Shape: (batch_size, hidden_size)

        # embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        # embedding = embedding/np.linalg.norm(embedding)
        return weighted_hidden_states/np.linalg.norm(weighted_hidden_states)

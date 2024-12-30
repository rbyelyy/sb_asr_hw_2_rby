from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer

# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved
# I have added BPE support but decided to avoid changing class name


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []

        # Move tensors to CPU
        predictions = log_probs.cpu()
        lengths = log_probs_length.cpu()

        # Get argmax predictions
        predictions = torch.argmax(predictions, dim=-1).numpy()

        for pred_sequence, length, target_text in zip(predictions, lengths, text):
            # Normalize target text
            target_text = self.text_encoder.normalize_text(target_text)

            # Get prediction using appropriate decoding
            pred_sequence = pred_sequence[:length]  # Trim to actual length
            pred_text = self.text_encoder.ctc_decode(pred_sequence)

            # For BPE tokenization, ensure we normalize the predicted text too
            if getattr(self.text_encoder, "use_bpe", False):
                pred_text = self.text_encoder.normalize_text(pred_text)

            # Calculate WER
            current_wer = calc_wer(target_text, pred_text)
            wers.append(current_wer)

        return sum(wers) / len(wers)

    def _log_probability_distribution(self, log_probs):
        """Helper method to log probability distributions"""
        probs = torch.exp(log_probs[:10])  # First 10 timesteps
        sum_probs = probs.sum(dim=1)
        print(f"Probability sums: {sum_probs}")

        # Log top-3 predictions for first few timesteps
        values, indices = probs.topk(3, dim=1)
        for t in range(min(3, probs.size(0))):
            print(f"\nTimestep {t}:")
            for prob, idx in zip(values[t], indices[t]):
                if getattr(self.text_encoder, "use_bpe", False):
                    token = self.text_encoder.tokenizer.decode([idx])
                else:
                    token = self.text_encoder.ind2char[idx.item()]
                print(f"  Token: {token}, Prob: {prob:.4f}")

    def _log_example(self, target: str, prediction: str, wer: float):
        """Helper method to log example predictions"""
        print("\nExample prediction:")
        print(f"Target    : {target}")
        print(f"Predicted : {prediction}")
        print(f"WER       : {wer:.4f}")
        print(f"Using BPE : {getattr(self.text_encoder, 'use_bpe', False)}")

        # Log word-level differences
        target_words = target.split()
        pred_words = prediction.split()
        print("\nWord-level comparison:")
        print(f"Target words  : {target_words}")
        print(f"Predicted words: {pred_words}")

        # Calculate word-level differences
        correct_words = sum(1 for t, p in zip(target_words, pred_words) if t == p)
        print(f"Correct words: {correct_words} / {len(target_words)}")

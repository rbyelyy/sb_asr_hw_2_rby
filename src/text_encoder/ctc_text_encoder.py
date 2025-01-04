import re
from string import ascii_lowercase

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class CTCTextEncoder:
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize input text - static method that can be called without instance

        Args:
            text: Input text string
        Returns:
            Normalized text string
        """
        # Convert to lowercase and remove all characters except letters and spaces
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text

    def __init__(
        self,
        alphabet=None,
        lm_path=None,
        use_bpe=True,
        beam_width=200,
        lm_weight=0.1,
        length_penalty=0.3,
        **kwargs,
    ):
        print("\nInitializing CTCTextEncoder:")
        print(f"use_bpe: {use_bpe}")
        print(f"lm_path: {lm_path}")
        print(f"beam_width: {beam_width}")
        print(f"lm_weight: {lm_weight}")

        self.subword_penalty = 0.8
        self.top_k_tokens = 50

        if use_bpe:
            print("BPE parameters:")
            print(f"  Subword penalty: {self.subword_penalty}")
            print(f"  Top-k tokens: {self.top_k_tokens}")

        # Add debug counter
        self._score_debug_counter = 0

        # Basic setup
        self.EMPTY_TOK = ""
        self.use_bpe = use_bpe

        # Initialize tokenizer if using BPE
        if self.use_bpe:
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                vocab_dict = dict(self.tokenizer.get_vocab())

                # Truncate vocabulary if needed
                max_vocab_size = 2000
                token_ids = list(vocab_dict.items())[:max_vocab_size]

                # Create new vocabulary
                self.vocab = [self.EMPTY_TOK] + [item[0] for item in token_ids]

                # Update tokenizer dictionaries
                self.tokenizer.get_vocab = lambda: dict(token_ids)

                print(
                    f"Initialized BPE tokenizer with vocabulary size: {len(self.vocab)}"
                )

            except Exception as e:
                print(f"Failed to load BPE tokenizer: {e}")
                print("Full error:", e.__class__.__name__)
                print("Attempting to debug tokenizer...")
                print("Available tokenizer attributes:", dir(self.tokenizer))
                self.use_bpe = False
                self.tokenizer = None
        else:
            self.tokenizer = None

        # Fall back to character-level if not using BPE
        if not self.use_bpe:
            self.alphabet = (
                list(ascii_lowercase + " ") if alphabet is None else alphabet
            )
            self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        # Create index mappings
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        # Beam search parameters
        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.length_penalty = length_penalty

        # Language model setup
        self.lm = None if lm_path is None else self._load_lm(lm_path)

    def encode(self, text) -> torch.Tensor:
        """Encode text to tensor of indices"""
        if self.use_bpe:
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            return torch.tensor(encoded)
        else:
            # Use character-level encoding
            text = self.normalize_text(text)
            try:
                return torch.tensor([self.char2ind[char] for char in text])
            except KeyError as e:
                unknown_chars = set(char for char in text if char not in self.char2ind)
                raise ValueError(f"Unknown characters: {unknown_chars}") from e

    def decode(self, indices) -> str:
        """Raw decoding without CTC"""
        if self.use_bpe:
            # Use BPE tokenizer
            if isinstance(indices, torch.Tensor):
                indices = indices.tolist()
            return self.tokenizer.decode(indices)
        else:
            # Use character-level decoding
            return "".join([self.ind2char[int(ind)] for ind in indices]).strip()

    def ctc_decode(self, log_probs, beam_width=None, lm_weight=None) -> str:
        """
        CTC decoding with beam search and LM integration
        Args:
            log_probs: Log probabilities (T x V) or indices
            beam_width: Override default beam width
            lm_weight: Override default LM weight
        """
        # Handle case when we get indices instead of log probs
        if len(log_probs.shape) == 1 or (
            isinstance(log_probs, np.ndarray) and log_probs.ndim == 1
        ):
            return self._greedy_decode(log_probs)

        beam_width = beam_width or self.beam_width
        lm_weight = lm_weight or self.lm_weight

        if self.use_bpe:
            return self._bpe_beam_search(log_probs, beam_width, lm_weight)
        else:
            return self._char_beam_search(log_probs, beam_width, lm_weight)

    def _greedy_decode(self, indices) -> str:
        """Your existing greedy decoding logic"""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()

        if self.use_bpe:
            collapsed = []
            previous = None
            for idx in indices:
                if idx != 0 and idx != previous:
                    collapsed.append(idx)
                    previous = idx
            return self.tokenizer.decode(collapsed)
        else:
            text = []
            previous = None
            for idx in indices:
                char = self.ind2char[int(idx)]
                if char != self.EMPTY_TOK and char != previous:
                    text.append(char)
                    previous = char
            return "".join(text)

    def _char_beam_search(self, log_probs, beam_width, lm_weight) -> str:
        """Character-level beam search with LM"""
        beam = [([], 0.0)]  # (prefix, score)

        for t in range(log_probs.shape[0]):
            new_beam = {}

            for prefix, score in beam:
                for c in range(len(self.vocab)):
                    new_prefix = prefix + [c]
                    acoustic_score = score + log_probs[t, c]

                    # Add language model score if available
                    if self.lm is not None:
                        partial_text = "".join(
                            [self.ind2char[idx] for idx in new_prefix]
                        )
                        lm_score = self._get_lm_score(partial_text)
                        combined_score = acoustic_score + lm_weight * lm_score
                    else:
                        combined_score = acoustic_score

                    # Apply length penalty
                    if len(new_prefix) > 0:
                        combined_score /= len(new_prefix) ** self.length_penalty

                    # Collapse CTC tokens
                    collapsed = self._ctc_collapse(new_prefix)
                    new_beam[collapsed] = max(
                        new_beam.get(collapsed, float("-inf")), combined_score
                    )

            # Keep top candidates
            beam = sorted(
                [(list(prefix), score) for prefix, score in new_beam.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:beam_width]

        # Return best hypothesis
        best_prefix = beam[0][0] if beam else []
        return "".join([self.ind2char[idx] for idx in best_prefix])

    def _bpe_beam_search(self, log_probs, beam_width, lm_weight) -> str:
        """Improved BPE-level beam search with subword handling"""
        beam = [([], 0.0)]  # (prefix, score)

        top_k = 50

        for t in range(log_probs.shape[0]):
            new_beam = {}

            for prefix, score in beam:
                context_score = self._get_context_score(prefix)

                probs, indices = torch.topk(log_probs[t], top_k)

                for prob, idx in zip(probs, indices):
                    new_prefix = prefix + [idx.item()]
                    acoustic_score = score + prob.item()

                    if self.lm is not None:
                        if self._is_subword_token(idx.item()):
                            lm_score = context_score
                        else:
                            partial_text = self.tokenizer.decode(new_prefix)
                            lm_score = self._get_lm_score(partial_text)
                        combined_score = acoustic_score + lm_weight * lm_score
                    else:
                        combined_score = acoustic_score

                    if len(new_prefix) > 0:
                        word_count = len(self.tokenizer.decode(new_prefix).split())
                        combined_score /= max(1, word_count) ** self.length_penalty

                    # save in beam
                    collapsed = tuple(new_prefix)
                    new_beam[collapsed] = max(
                        new_beam.get(collapsed, float("-inf")), combined_score
                    )

            # Keep top candidates
            beam = sorted(
                [(list(prefix), score) for prefix, score in new_beam.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:beam_width]

        # normalization
        best_prefix = beam[0][0] if beam else []
        result = self.tokenizer.decode(best_prefix)
        return self.normalize_text(result)

    def __len__(self):
        """Return vocabulary size"""
        return len(self.vocab)

    def __str__(self):
        return f"CTCTextEncoder(vocab_size={len(self)}, beam_width={self.beam_width})"

    def _load_lm(self, path):
        """
        Load language model from binary file or use GPT2
        """
        try:
            import kenlm

            if path.endswith(".binary") or path.endswith(".bin"):
                print(f"\nLoading KenLM model from {path}")
                model = kenlm.Model(path)
                print("KenLM model loaded successfully!")
                print(f"KenLM model order: {model.order}")
                # Test the model
                test_text = "hello world"
                score = model.score(test_text)
                print(f"Test scoring with '{test_text}': {score}")
                return model
            else:
                print("\nNo .binary or .bin extension, falling back to GPT2...")
                from transformers import GPT2LMHeadModel

                return GPT2LMHeadModel.from_pretrained("gpt2").eval()
        except Exception as e:
            print(f"\nFailed to load language model: {e}")
            return None

    def _get_lm_score(self, text: str) -> float:
        """
        Get language model score for text
        """
        if self.lm is None:
            return 0.0

        try:
            if isinstance(self.lm, GPT2LMHeadModel):
                print("Using GPT2 for scoring")
                inputs = self.tokenizer(text, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.lm(**inputs)
                    return -outputs.loss.item()
            else:
                # KenLM scoring
                score = self.lm.score(text)
                self._score_debug_counter += 1
                if self._score_debug_counter % 100 == 0:  # Print every 100th score
                    print("\nKenLM scoring example:")
                    print(f"Text: {text}")
                    print(f"Score: {score}")
                return score
        except Exception as e:
            print(f"Error in LM scoring: {e}")
            print(f"Problematic text: '{text}'")
            return 0.0

    def _get_context_score(self, prefix):
        """Get context-aware LM score for BPE tokens"""
        if not prefix:
            return 0.0
        if self.use_bpe:
            context = self.tokenizer.decode(prefix)
        else:
            context = "".join([self.ind2char[idx] for idx in prefix])
        return self._get_lm_score(context)

    def _is_subword_token(self, token_id):
        """Check if token is a subword unit"""
        if not self.use_bpe:
            return False
        token = self.tokenizer.decode([token_id])
        return token.startswith("##") or not token.startswith(" ")

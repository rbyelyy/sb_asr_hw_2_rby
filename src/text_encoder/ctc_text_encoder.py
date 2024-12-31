import re
from string import ascii_lowercase

import torch
from transformers import GPT2Tokenizer

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


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
        # Basic setup
        self.EMPTY_TOK = ""
        self.use_bpe = use_bpe

        # Initialize tokenizer if using BPE
        if self.use_bpe:
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                self.vocab = [self.EMPTY_TOK] + list(range(self.tokenizer.vocab_size))

                # Truncate tokenizer's vocabulary
                max_vocab_size = 2000
                token_ids = list(self.tokenizer.vocab.items())[:max_vocab_size]
                self.tokenizer.vocab = dict(token_ids)
                self.tokenizer.encoder = dict(token_ids)
                self.tokenizer.decoder = {v: k for k, v in token_ids}

                print(
                    f"Initialized BPE tokenizer with vocabulary size: {len(self.vocab)}"
                )
            except Exception as e:
                print(
                    f"Failed to load BPE tokenizer: {e}. Falling back to character-level tokenization."
                )
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
            # Use BPE tokenizer
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

    def ctc_decode(self, indices) -> str:
        """CTC decoding with token collapsing"""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()

        if self.use_bpe:
            # For BPE tokens, collapse repeats but keep EMPTY_TOK handling
            collapsed = []
            previous = None
            for idx in indices:
                if idx != 0 and idx != previous:  # 0 is EMPTY_TOK
                    collapsed.append(idx)
                    previous = idx
            return self.tokenizer.decode(collapsed)
        else:
            # Character-level CTC decoding
            text = []
            previous = None
            for idx in indices:
                char = self.ind2char[int(idx)]
                if char != self.EMPTY_TOK and char != previous:
                    text.append(char)
                    previous = char
            return "".join(text)

    def __len__(self):
        """Return vocabulary size"""
        return len(self.vocab)

    def __str__(self):
        return f"CTCTextEncoder(vocab_size={len(self)}, beam_width={self.beam_width})"

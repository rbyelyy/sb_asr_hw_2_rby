from torch import Tensor
from torch.nn import CTCLoss


class CTCLossWrapper(CTCLoss):
    def forward(
        self, log_probs, log_probs_length, text_encoded, text_encoded_length, **batch
    ) -> Tensor:
        # Ensure text_encoded is 2D (batch, sequence)
        if text_encoded.dim() == 3:
            text_encoded = text_encoded.squeeze(1)

        # Move tensors to the same device
        text_encoded = text_encoded.to(log_probs.device)
        text_encoded_length = text_encoded_length.to(log_probs.device)
        log_probs_length = log_probs_length.to(log_probs.device)

        # Transpose log_probs for CTC loss (time, batch, vocab)
        log_probs_t = log_probs.transpose(0, 1)

        # Compute CTC loss
        loss = super().forward(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=log_probs_length,
            target_lengths=text_encoded_length,
        )

        return {"loss": loss}

from torch import nn


class BaselineModel(nn.Module):
    def __init__(self, n_feats=128, n_tokens=28, fc_hidden=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=64 * n_feats,  # 64 channels * 128 freq bins
            hidden_size=fc_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(fc_hidden * 2, n_tokens)
        nn.init.kaiming_normal_(self.fc.weight)
        self.fc.bias.data.fill_(-1.0)  # Slight bias against blank token

    def forward(self, spectrogram, spectrogram_length, **batch):
        # Input shape: [B, 1, 128, T]
        x = self.conv(spectrogram)  # Output: [B, 64, 128, T]

        batch_size = x.size(0)
        time_steps = x.size(-1)

        # Reshape to [B, T, 64*128]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, time_steps, -1)

        # Pack sequence
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, spectrogram_length.cpu(), batch_first=True, enforce_sorted=False
        )

        x, _ = self.lstm(packed_x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        log_probs = nn.functional.log_softmax(self.fc(x), dim=-1)
        return {"log_probs": log_probs, "log_probs_length": spectrogram_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return input_lengths  # we don't reduce time dimension here

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info

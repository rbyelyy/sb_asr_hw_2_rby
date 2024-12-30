import logging

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in dataset items.

    Args:
        dataset_items (list[dict]): List of dataset samples
    Returns:
        dict[str, Tensor]: Batched data
    """
    try:
        result_batch = {}

        # Get lengths first
        audio_lengths = []
        spec_lengths = []
        text_lengths = []

        for item in dataset_items:
            # Audio length (handle both 1D and 2D inputs)
            audio = item["audio"]
            if audio.dim() == 2:
                audio_lengths.append(audio.shape[1])
            else:
                audio_lengths.append(audio.shape[0])

            spec_lengths.append(item["spectrogram"].shape[2])
            text_lengths.append(item["text_encoded"].shape[0])

        # Add lengths to batch
        result_batch["audio_length"] = torch.tensor(audio_lengths)
        result_batch["spectrogram_length"] = torch.tensor(spec_lengths)
        result_batch["text_encoded_length"] = torch.tensor(text_lengths)

        # Process each field
        for key in dataset_items[0].keys():
            values = [item[key] for item in dataset_items]

            if key == "audio":
                # Convert all audio to 1D tensor before padding
                processed_values = []
                for audio in values:
                    if audio.dim() == 2:
                        audio = audio.squeeze(0)
                    processed_values.append(audio)

                # Pad the sequences
                padded = pad_sequence(processed_values, batch_first=True)
                # Add channel dimension back
                result_batch[key] = padded.unsqueeze(1)

            elif key == "spectrogram":
                # Pad spectrograms in time dimension
                max_spec_len = max(spec_lengths)
                padded = []
                for spec in values:
                    if spec.shape[2] < max_spec_len:
                        padding_size = max_spec_len - spec.shape[2]
                        padded.append(torch.nn.functional.pad(spec, (0, padding_size)))
                    else:
                        padded.append(spec)
                result_batch[key] = torch.stack(padded)

            elif key == "text_encoded":
                # Handle text encoding padding
                processed_values = []
                for text_enc in values:
                    if text_enc.dim() == 2:
                        text_enc = text_enc.squeeze(0)
                    processed_values.append(text_enc)

                padded = pad_sequence(
                    processed_values, batch_first=True, padding_value=0
                )
                result_batch[key] = padded

            elif key in ["text", "audio_path"]:
                # Keep strings as lists
                result_batch[key] = values

        return result_batch

    except Exception as e:
        # Log detailed information about the batch
        logging.error(f"Error in collate_fn: {str(e)}")
        logging.error("Batch information:")
        for i, item in enumerate(dataset_items):
            logging.error(f"\nItem {i}:")
            for k, v in item.items():
                if isinstance(v, torch.Tensor):
                    logging.error(f"{k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    logging.error(f"{k}: type={type(v)}")
        raise

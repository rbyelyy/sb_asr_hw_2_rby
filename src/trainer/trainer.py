from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Process batch with memory optimizations and mixed precision training.
        """
        try:
            # Initialize gradient scaler for mixed precision training
            if not hasattr(self, "scaler"):
                self.scaler = torch.cuda.amp.GradScaler()

            # Move batch to device efficiently
            batch = self.move_batch_to_device(batch)

            # Transform batch with memory optimization
            torch.cuda.empty_cache()  # Clear cache before processing
            batch = self.transform_batch(batch)

            metric_funcs = self.metrics["inference"]
            if self.is_train:
                metric_funcs = self.metrics["train"]
                self.optimizer.zero_grad(
                    set_to_none=True
                )  # More efficient than zero_grad()

            # Use mixed precision for forward pass
            with torch.cuda.amp.autocast():
                # In your trainer code, where you call the forward method:
                outputs = self.model(**batch)
                batch.update(outputs)
                all_losses = self.criterion(**batch)
                batch.update(all_losses)

            if self.is_train:
                # Use gradient scaling for mixed precision training
                self.scaler.scale(batch["loss"]).backward()

                # Gradient clipping with scaled gradients
                if hasattr(self, "clip_grad_norm"):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=10.0,  # TODO clipping was set to 1 by default
                    )

                # Optimizer step with gradient scaling
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            # Update metrics efficiently
            with torch.no_grad():
                for loss_name in self.config.writer.loss_names:
                    metrics.update(loss_name, batch[loss_name].item())

                for met in metric_funcs:
                    metrics.update(met.name, met(**batch))

            return batch

        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                print("\nOOM in batch processing. Batch sizes:")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f"{k}: {v.shape}")
                raise RuntimeError("OOM in batch processing")
            raise e

    def move_batch_to_device(self, batch):
        """
        Efficiently move batch to device with memory optimization
        """
        device = next(self.model.parameters()).device
        processed_batch = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # Move tensor to device and convert to half precision if possible
                if value.dtype == torch.float32:
                    processed_batch[key] = value.to(device, non_blocking=True)
                else:
                    processed_batch[key] = value.to(device)
            else:
                processed_batch[key] = value

        return processed_batch

    def enable_memory_optimizations(self):
        """
        Enable various memory optimizations for the model
        """
        # Enable gradient checkpointing if available
        if hasattr(self.model, "encoder"):
            self.model.encoder.gradient_checkpointing_enable()

        # Enable efficient memory usage for backward pass
        torch.backends.cudnn.benchmark = True

        # Set up gradient clipping
        self.clip_grad_norm = True

        # Optional: Use parameter sharing if applicable
        if hasattr(self.model, "decoder") and hasattr(self.model, "embedding"):
            self.model.decoder.weight = self.model.embedding.weight

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

        # DEBUG
        if "original_spectrogram" in batch:
            self.writer.add_image(
                "original_spectrogram", batch["original_spectrogram"][0]
            )

        # Log augmented versions
        if "spectrogram" in batch:
            self.writer.add_image("augmented_spectrogram", batch["spectrogram"][0])

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_predictions(
        self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, **batch
    ):
        # TODO add beam search
        # Note: by improving text encoder and metrics design
        # this logging can also be improved significantly

        argmax_inds = log_probs.cpu().argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.cpu().numpy())
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
        tuples = list(zip(argmax_texts, text, argmax_texts_raw, audio_path))

        rows = {}
        for pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
            target = self.text_encoder.normalize_text(target)
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100

            rows[Path(audio_path).name] = {
                "target": target,
                "raw prediction": raw_pred,
                "predictions": pred,
                "wer": wer,
                "cer": cer,
            }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )

import json

import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        text_encoder,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            text_encoder (CTCTextEncoder): text encoder.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        self.text_encoder = text_encoder

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def move_batch_to_device(self, batch):
        """
        Move batch tensors to device with ASR-specific handling
        """
        # Get device tensors from config or use ASR defaults
        device_tensors = self.cfg_trainer.get(
            "device_tensors", ["spectrogram", "text_encoded"]
        )

        # If old config has 'data_object', replace with ASR tensors
        if "data_object" in device_tensors:
            device_tensors.remove("data_object")
            if "spectrogram" not in device_tensors:
                device_tensors.append("spectrogram")
            if "text_encoded" not in device_tensors:
                device_tensors.append("text_encoded")

        # Move tensors to device
        for tensor_name in device_tensors:
            if tensor_name in batch and isinstance(batch[tensor_name], torch.Tensor):
                batch[tensor_name] = batch[tensor_name].to(self.device)

        return batch

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        try:
            batch = self.move_batch_to_device(batch)
            torch.cuda.empty_cache()
            batch = self.transform_batch(batch)

            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                batch.update(outputs)

            batch_size = batch["spectrogram"].shape[0]
            current_id = batch_idx * batch_size

            for i in range(batch_size):
                # Get log probabilities and convert to indices like in training
                log_probs = batch["log_probs"][i].cpu()
                log_probs_length = batch["log_probs_length"][i].cpu()

                # Convert to indices using argmax (like in training)
                predictions = torch.argmax(log_probs, dim=-1).numpy()
                target_text = batch["text"][i]

                # Now use ctc_decode the same way as in training
                pred_text = self.text_encoder.ctc_decode(predictions[:log_probs_length])

                # Print comparison
                print(f"\ntarget_text - {target_text}")
                print(f"predicted_text - {pred_text}")

                output = {
                    "predicted_text": pred_text,
                    "target_text": target_text,
                    "audio_path": batch["audio_path"][i],
                }

                if self.save_path is not None:
                    save_path = self.save_path / part / f"output_{current_id + i}.json"
                    with open(save_path, "w") as f:
                        json.dump(output, f, indent=2)

            if metrics is not None:
                with torch.no_grad():
                    for met in self.metrics["inference"]:
                        metrics.update(met.name, met(**batch))

            return batch

        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            print("Batch contents:")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"{k}: shape={v.shape}, dtype={v.dtype}")
            raise e

    # def process_batch(self, batch_idx, batch, metrics, part):
    #     """
    #     Process batch for inference with memory optimizations
    #     """
    #     try:
    #         # Move batch to device efficiently
    #         batch = self.move_batch_to_device(batch)
    #
    #         # Clear cache and transform batch
    #         torch.cuda.empty_cache()
    #         batch = self.transform_batch(batch)
    #
    #         # Use mixed precision for forward pass
    #         with torch.cuda.amp.autocast():
    #             outputs = self.model(**batch)
    #             batch.update(outputs)
    #
    #         # Update metrics efficiently
    #         if metrics is not None:
    #             with torch.no_grad():
    #                 for met in self.metrics["inference"]:
    #                     metrics.update(met.name, met(**batch))
    #
    #         # Save ASR predictions
    #         batch_size = batch["spectrogram"].shape[0]
    #         current_id = batch_idx * batch_size
    #
    #         for i in range(batch_size):
    #             # Get predictions
    #             log_probs = batch["log_probs"][i].cpu()
    #             log_probs_length = batch["log_probs_length"][i].cpu()
    #
    #             # Get ground truth
    #             target_text = batch["text"][i]
    #
    #             # Decode predictions
    #             pred_text = self.text_encoder.ctc_decode(
    #                 log_probs[:log_probs_length]
    #             )
    #
    #             output = {
    #                 "predicted_text": pred_text,
    #                 "target_text": target_text,
    #                 "audio_path": batch["audio_path"][i],
    #                 "metrics": {
    #                     met.name: met(
    #                         log_probs=log_probs.unsqueeze(0),
    #                         log_probs_length=torch.tensor([log_probs_length]),
    #                         text=[target_text]
    #                     )
    #                     for met in self.metrics["inference"]
    #                 } if self.metrics else {}
    #             }
    #
    #             if self.save_path is not None:
    #                 # Save as JSON for better readability
    #                 import json
    #                 save_path = self.save_path / part / f"output_{current_id + i}.json"
    #                 with open(save_path, 'w') as f:
    #                     json.dump(output, f, indent=2)
    #
    #         return batch
    #
    #     except RuntimeError as e:
    #         if "out of memory" in str(e):
    #             torch.cuda.empty_cache()
    #             print(f"\nOOM in batch processing. Batch sizes:")
    #             for k, v in batch.items():
    #                 if isinstance(v, torch.Tensor):
    #                     print(f"{k}: {v.shape}")
    #             raise RuntimeError("OOM in batch processing")
    #         raise e

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result()

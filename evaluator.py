from typing import NamedTuple, List, Any, Optional, Dict
from dataclasses import dataclass
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np

from schedulers import Scheduler, LRSchedule
from models import Prober, build_mlp
from configs import ConfigBase

from dataset import WallDataset
from normalizer import Normalizer


@dataclass
class ProbingConfig(ConfigBase):
    probe_targets: str = "locations"
    lr: float = 0.0002
    epochs: int = 20
    schedule: LRSchedule = LRSchedule.Cosine
    sample_timesteps: int = 30
    prober_arch: str = "256"


class ProbeResult(NamedTuple):
    model: torch.nn.Module
    average_eval_loss: float
    eval_losses_per_step: List[float]
    plots: List[Any]


default_config = ProbingConfig()
   

def location_losses(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert pred.shape == target.shape, f"Pred shape: {pred.shape}, Target shape: {target.shape}"
    mse = (pred - target).pow(2).mean(dim=0)
    return mse


class ProbingEvaluator:
    def __init__(
        self,
        device: str,
        model: torch.nn.Module,
        probe_train_ds,
        probe_val_ds: dict,
        config: ProbingConfig = default_config,
        quick_debug: bool = False,
    ):
        self.device = device
        self.config = config

        self.model = model.to(device)
        self.model.eval()

        self.quick_debug = quick_debug

        self.ds = probe_train_ds
        self.val_ds = probe_val_ds

        self.normalizer = Normalizer()

    def train_pred_prober(self):
        """
        Probes whether the predicted embeddings capture the future locations
        """
        repr_dim = self.model.repr_dim
        dataset = self.ds
        model = self.model

        config = self.config
        epochs = config.epochs

        if self.quick_debug:
            epochs = 1
        test_batch = next(iter(dataset))

        prober_output_shape = getattr(test_batch, "locations")[0, 0].shape
        prober = Prober(
            embedding=repr_dim,
            arch=config.prober_arch,
            output_shape=prober_output_shape,
        ).to(self.device)

        all_parameters = []
        all_parameters += list(prober.parameters())

        optimizer_pred_prober = torch.optim.Adam(all_parameters, config.lr)

        step = 0

        batch_size = dataset.batch_size
        batch_steps = None

        scheduler = Scheduler(
            schedule=self.config.schedule,
            base_lr=config.lr,
            data_loader=dataset,
            epochs=epochs,
            optimizer=optimizer_pred_prober,
            batch_steps=batch_steps,
            batch_size=batch_size,
        )

        for epoch in tqdm(range(epochs), desc=f"Probe prediction epochs"):
            for batch in tqdm(dataset, desc="Probe prediction step"):
                ################################################################################
                # Forward pass through your model
                pred_encs, _ = model(states=batch.states, actions=batch.actions)
                pred_encs = pred_encs.detach()  # Shape: [B, T, D]
                pred_encs = pred_encs.permute(1, 0, 2)  # [T,B,D]
 
                # Ensure pred_encs and target have matching sequence lengths
                n_steps_pred = pred_encs.shape[0]
                n_steps_target = batch.locations.shape[1]
                n_steps = min(n_steps_pred, n_steps_target)

                # Slice pred_encs and target to have the same sequence length
                pred_encs = pred_encs[:n_steps, :, :]  # Shape: [T, B, D]
                target = batch.locations.to(self.device)[:, :n_steps, :]  # Shape: [B, T, 2]
                target = self.normalizer.normalize_location(target)

                bs = pred_encs.shape[1]

                # Reshape pred_encs to [T*B, D]
                pred_encs_reshaped = pred_encs.reshape(-1, pred_encs.shape[-1])  # [T*B, D]
                # Apply prober
                pred_locs = prober(pred_encs_reshaped)  # [T*B, 2]
                # Reshape pred_locs back to [T, B, 2]
                pred_locs = pred_locs.view(n_steps, bs, -1)

                # Transpose pred_locs to match target shape [B, T, 2]
                pred_locs = pred_locs.permute(1, 0, 2)  # [B, T, 2]

                if (
                    config.sample_timesteps is not None
                    and config.sample_timesteps < n_steps
                ):
                    # Randomly sample timesteps to avoid OOM
                    indices = torch.randperm(n_steps, dtype=torch.long)[: config.sample_timesteps]
                    pred_locs = pred_locs[:, indices, :]  # [B, sample_timesteps, 2]
                    target = target[:, indices, :]  # [B, sample_timesteps, 2]

                # Ensure pred_locs and target have the same shape
                assert pred_locs.shape == target.shape, f"Pred shape: {pred_locs.shape}, Target shape: {target.shape}"

                losses = location_losses(pred_locs, target)
                per_probe_loss = losses.mean()

                if step % 100 == 0:
                    print(f"normalized pred locations loss {per_probe_loss.item()}")

                optimizer_pred_prober.zero_grad()
                per_probe_loss.backward()
                optimizer_pred_prober.step()

                lr = scheduler.adjust_learning_rate(step)

                step += 1

                if self.quick_debug and step > 2:
                    break

        return prober

    @torch.no_grad()
    def evaluate_all(
        self,
        prober,
    ):
        """
        Evaluates on all the different validation datasets
        """
        avg_losses = {}

        for prefix, val_ds in self.val_ds.items():
            avg_losses[prefix] = self.evaluate_pred_prober(
                prober=prober,
                val_ds=val_ds,
                prefix=prefix,
            )

        return avg_losses

    @torch.no_grad()
    def evaluate_pred_prober(
        self,
        prober,
        val_ds,
        prefix="",
    ):
        quick_debug = self.quick_debug
        config = self.config

        model = self.model
        probing_losses = []
        prober.eval()

        for idx, batch in enumerate(tqdm(val_ds, desc="Eval probe pred")):
            ################################################################################
            # Forward pass through your model
            pred_encs, _ = model(states=batch.states, actions=batch.actions)
            pred_encs = pred_encs.detach()  # Shape: [B, T, D]
            pred_encs = pred_encs.permute(1, 0, 2)  # [T,B,D]

            # Ensure pred_encs and target have matching sequence lengths
            n_steps_pred = pred_encs.shape[0]
            n_steps_target = batch.locations.shape[1]
            n_steps = min(n_steps_pred, n_steps_target)

            # Slice pred_encs and target to have the same sequence length
            pred_encs = pred_encs[:n_steps, :, :]  # Shape: [T, B, D]
            target = batch.locations.to(self.device)[:, :n_steps, :]  # Shape: [B, T, 2]
            target = self.normalizer.normalize_location(target)

            bs = pred_encs.shape[1]

            # Reshape pred_encs to [T*B, D]
            pred_encs_reshaped = pred_encs.reshape(-1, pred_encs.shape[-1])  # [T*B, D]
            # Apply prober
            pred_locs = prober(pred_encs_reshaped)  # [T*B, 2]
            # Reshape pred_locs back to [T, B, 2]
            pred_locs = pred_locs.view(n_steps, bs, -1)

            # Transpose pred_locs to match target shape [B, T, 2]
            pred_locs = pred_locs.permute(1, 0, 2)  # [B, T, 2]

            # Ensure pred_locs and target have the same shape
            assert pred_locs.shape == target.shape, f"Pred shape: {pred_locs.shape}, Target shape: {target.shape}"

            losses = location_losses(pred_locs, target)
            print(losses)
            probing_losses.append(losses.cpu())

            if self.quick_debug and idx > 2:
                break

        losses_t = torch.stack(probing_losses, dim=0).mean(dim=0)
        losses_t = self.normalizer.unnormalize_mse(losses_t)

        losses_t = losses_t.mean(dim=-1)
        average_eval_loss = losses_t.mean().item()

        return average_eval_loss






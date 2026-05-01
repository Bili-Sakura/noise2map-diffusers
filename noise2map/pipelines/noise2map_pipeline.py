from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.utils import BaseOutput

from ..model import Noise2Map


ImageInput = Union[torch.Tensor, np.ndarray, Image.Image]


@dataclass
class Noise2MapPipelineOutput(BaseOutput):
    logits: Union[torch.Tensor, np.ndarray]
    predictions: Union[torch.Tensor, np.ndarray]


def _as_list(images: Union[ImageInput, Sequence[ImageInput]]) -> list[ImageInput]:
    if isinstance(images, (list, tuple)):
        return list(images)
    return [images]


def _ensure_batch_channels(
    tensor: torch.Tensor, expected_channels: int, name: str
) -> torch.Tensor:
    if tensor.ndim == 3:
        if tensor.shape[0] == expected_channels:
            tensor = tensor.unsqueeze(0)
        elif tensor.shape[-1] == expected_channels:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(
                f"{name} must have {expected_channels} channels, got shape {tuple(tensor.shape)}."
            )
    elif tensor.ndim == 4:
        if tensor.shape[1] == expected_channels:
            pass
        elif tensor.shape[-1] == expected_channels:
            tensor = tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                f"{name} must have {expected_channels} channels, got shape {tuple(tensor.shape)}."
            )
    else:
        raise ValueError(f"{name} must be 3D or 4D tensor, got shape {tuple(tensor.shape)}.")
    return tensor


def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.float()
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    if tensor.min() >= 0.0 and tensor.max() <= 1.0:
        tensor = tensor * 2.0 - 1.0
    return tensor


def _prepare_images(
    images: Union[ImageInput, Sequence[ImageInput]],
    expected_channels: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    tensors = []
    for image in _as_list(images):
        if isinstance(image, torch.Tensor):
            tensor = image
        elif isinstance(image, (np.ndarray, Image.Image)):
            tensor = TF.to_tensor(image)
        else:
            raise TypeError(
                f"{name} must be a torch.Tensor, numpy array, PIL image, or a list of those."
            )
        tensor = _ensure_batch_channels(tensor, expected_channels, name)
        tensor = _normalize_tensor(tensor)
        tensors.append(tensor)
    batch = torch.cat(tensors, dim=0)
    return batch.to(device=device, dtype=dtype)


class Noise2MapBasePipeline(DiffusionPipeline):
    model: Noise2Map
    scheduler: DDIMScheduler

    def __init__(self, model: Noise2Map, scheduler: Optional[DDIMScheduler] = None) -> None:
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler or DDIMScheduler())

    def _get_inference_timestep(self, timestep: Optional[int]) -> int:
        if timestep is not None:
            return int(timestep)
        return int(self.scheduler.config.num_train_timesteps - 1)

    def _get_dtype(self) -> torch.dtype:
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32


class Noise2MapSemanticSegmentationPipeline(Noise2MapBasePipeline):
    """Diffusers-style pipeline for Noise2Map semantic segmentation."""

    @torch.no_grad()
    def __call__(
        self,
        image: Union[ImageInput, Sequence[ImageInput]],
        *,
        timestep: Optional[int] = None,
        output_type: str = "torch",
        return_dict: bool = True,
    ) -> Noise2MapPipelineOutput | Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        self.model.eval()
        device = self._execution_device
        dtype = self._get_dtype()

        image_tensor = _prepare_images(image, expected_channels=3, device=device, dtype=dtype, name="image")
        t = self._get_inference_timestep(timestep)
        timesteps = torch.full((image_tensor.shape[0],), t, device=device, dtype=torch.long)
        structured_noise = image_tensor
        x_noisy = self.scheduler.add_noise(image_tensor, structured_noise, timesteps)

        logits = self.model(x_noisy, timesteps)
        predictions = torch.argmax(logits, dim=1)

        if output_type == "numpy":
            logits = logits.cpu().numpy()
            predictions = predictions.cpu().numpy()
        elif output_type != "torch":
            raise ValueError("output_type must be 'torch' or 'numpy'.")

        if not return_dict:
            return logits, predictions
        return Noise2MapPipelineOutput(logits=logits, predictions=predictions)


class Noise2MapChangeDetectionPipeline(Noise2MapBasePipeline):
    """Diffusers-style pipeline for Noise2Map change detection."""

    @torch.no_grad()
    def __call__(
        self,
        pre_image: Union[ImageInput, Sequence[ImageInput]],
        post_image: Union[ImageInput, Sequence[ImageInput]],
        *,
        timestep: Optional[int] = None,
        output_type: str = "torch",
        return_dict: bool = True,
    ) -> Noise2MapPipelineOutput | Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        self.model.eval()
        device = self._execution_device
        dtype = self._get_dtype()

        pre = _prepare_images(pre_image, expected_channels=3, device=device, dtype=dtype, name="pre_image")
        post = _prepare_images(post_image, expected_channels=3, device=device, dtype=dtype, name="post_image")
        if pre.shape != post.shape:
            raise ValueError(
                f"pre_image and post_image must have the same shape, got {tuple(pre.shape)} and {tuple(post.shape)}."
            )

        x = torch.cat([pre, post], dim=1)
        structured_noise = torch.cat([post, pre], dim=1)
        t = self._get_inference_timestep(timestep)
        timesteps = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
        x_noisy = self.scheduler.add_noise(x, structured_noise, timesteps)

        logits = self.model(x_noisy, timesteps)
        predictions = torch.argmax(logits, dim=1)

        if output_type == "numpy":
            logits = logits.cpu().numpy()
            predictions = predictions.cpu().numpy()
        elif output_type != "torch":
            raise ValueError("output_type must be 'torch' or 'numpy'.")

        if not return_dict:
            return logits, predictions
        return Noise2MapPipelineOutput(logits=logits, predictions=predictions)

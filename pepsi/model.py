import wandb
import torch
import random
from tqdm import tqdm
from pepsi.consts import *
from typing import Optional
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from point_e.models.download import load_checkpoint
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config


class PEPSI(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        log_wandb: bool,
        batch_size: int,
        copy_prob: float,
        copy_prompt: str,
        dev: torch.device,
        cond_drop_prob: float,
        init_val_data: bool = True,
        val_dataloader: Optional[DataLoader] = None,
    ):
        super().__init__()
        self.lr = lr
        self.dev = dev
        self.log_wandb = log_wandb
        self.copy_prob = copy_prob
        self.batch_size = batch_size
        self.copy_prompt = copy_prompt
        self._init_model(cond_drop_prob)
        if init_val_data and val_dataloader is not None:
            self._init_val_data(val_dataloader)

    def _init_model(self, cond_drop_prob: float):
        self.diffusion = diffusion_from_config(DIFFUSION_CONFIGS[MODEL_NAME])
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[UPSAMPLE])
        config = MODEL_CONFIGS[MODEL_NAME]
        config[COND_DROP_PROB] = cond_drop_prob
        self.model = model_from_config(config, self.dev)
        self.model.load_state_dict(load_checkpoint(MODEL_NAME, self.dev))
        self.model.create_control_layers()
        upsampler_model = model_from_config(MODEL_CONFIGS[UPSAMPLE], self.dev)
        upsampler_model.eval()
        upsampler_model.load_state_dict(load_checkpoint(UPSAMPLE, self.dev))
        self.sampler = PointCloudSampler(
            device=self.dev,
            guidance_scale=[3.0, 0.0],
            aux_channels=["R", "G", "B"],
            model_kwargs_key_filter=(TEXTS, ""),
            models=[self.model, upsampler_model],
            num_points=[N_PTS_ENCODE, N_PTS_SAMPLE - N_PTS_ENCODE],
            diffusions=[self.diffusion, upsampler_diffusion],
        )

    def _init_val_data(self, val_dataloader: DataLoader):
        if not self.log_wandb:
            return
        assert len(val_dataloader) == 1
        batch = next(iter(val_dataloader))
        log_data = {}
        parts, splits, prompts, source_latents, target_latents = (
            batch[PARTS],
            batch[SPLITS],
            batch[PROMPTS],
            batch[SOURCE_LATENTS],
            batch[TARGET_LATENTS],
        )
        for part, split, prompt, source_latent, target_latent in tqdm(
            zip(
                parts,
                splits,
                prompts,
                source_latents,
                target_latents,
            ),
            total=len(prompts),
            desc="Initializing val data",
        ):
            source_key, target_key = SOURCE + "_" + split, TARGET + "_" + split
            log_data.setdefault(source_key, [])
            log_data.setdefault(target_key, [])
            source_samples = self.sampler.sample_batch(
                batch_size=1,
                model_kwargs={},
                prev_samples=source_latent.unsqueeze(0),
            )
            target_samples = self.sampler.sample_batch(
                batch_size=1,
                model_kwargs={},
                prev_samples=target_latent.unsqueeze(0),
            )
            log_data[source_key].append(self._plot(source_samples, prompt, part))
            log_data[target_key].append(self._plot(target_samples, prompt, part))
            log_data[source_key].append(self._plot(target_samples, self.copy_prompt))
            log_data[target_key].append(self._plot(target_samples, self.copy_prompt))
        wandb.log(log_data, step=None)

    def _plot(
        self, samples: torch.Tensor, prompt: str, part: Optional[str] = None
    ) -> wandb.Image:
        pc = self.sampler.output_to_point_clouds(samples)[0]
        caption = prompt.replace(" ", "_")
        if part is not None:
            caption += "_" + part
        img = wandb.Image(pc.render(), caption=caption)
        return img

    def _sample_t(self) -> torch.Tensor:
        return (
            torch.tensor(
                random.sample(range(len(self.diffusion.betas)), self.batch_size)
            )
            .to(self.dev)
            .detach()
        )

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam((self.parameters()), lr=self.lr)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        prompts, source_latents, target_latents = (
            batch[PROMPTS],
            batch[SOURCE_LATENTS],
            batch[TARGET_LATENTS],
        )
        x_start = target_latents
        if random.random() < self.copy_prob:
            texts = [self.copy_prompt] * len(prompts)
            guidance = target_latents
        else:
            texts = prompts
            guidance = source_latents
        terms = self.diffusion.training_losses(
            x_start=x_start,
            model=self.model,
            t=self._sample_t(),
            model_kwargs={TEXTS: texts, GUIDANCE: guidance},
        )
        loss = terms[LOSS].mean()
        if self.log_wandb:
            wandb.log({LOSS: loss.item()}, step=None)
        return loss

    def _sample(self, prompt: str, guidance: torch.Tensor) -> torch.Tensor:
        return self.sampler.sample_batch(
            batch_size=1,
            guidances=[guidance, None],
            model_kwargs={TEXTS: [prompt]},
        )

    def validation_step(self, batch: dict, batch_idx: int):
        if not self.log_wandb:
            return
        assert batch_idx == 0
        log_data = {}
        parts, splits, prompts, source_latents, target_latents = (
            batch[PARTS],
            batch[SPLITS],
            batch[PROMPTS],
            batch[SOURCE_LATENTS],
            batch[TARGET_LATENTS],
        )
        with torch.no_grad():
            for part, split, prompt, source_latent, target_latent in tqdm(
                zip(parts, splits, prompts, source_latents, target_latents),
                total=len(prompts),
                desc="Testing val data",
            ):
                key = OUTPUT + "_" + split
                log_data.setdefault(key, [])
                samples = self._sample(prompt, source_latent)
                log_data[key].append(self._plot(samples, prompt, part))
                samples = self._sample(self.copy_prompt, target_latent)
                log_data[key].append(self._plot(samples, self.copy_prompt))
        wandb.log(log_data, step=None)

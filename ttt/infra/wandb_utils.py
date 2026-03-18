import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import jax
import numpy as np
from hydra.core.hydra_config import HydraConfig

from ttt.utils.jax_utils import master_log

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class WandbLogger:
    """
    Handle initialization and logging of a W&B run.

    Every launch creates a new W&B run. Runs sharing the same ``run_name``
    are grouped together via the W&B *group* field so they can be compared
    side-by-side.
    """

    def __init__(
        self,
        entity: str,
        project: str,
        run_name: str,
        log_dir: Path,
        wandb_key: str,
        logging_process: int,
        config: dict = None,
        enabled: bool = True,
    ):
        import wandb
        from wandb.sdk.wandb_settings import Settings

        self.wandb = wandb
        self.is_master = jax.process_index() == logging_process
        self.entity = entity
        self.project = project
        self.run_name = run_name
        self.enabled = enabled
        self.run = None
        self.log_dir = log_dir

        self.wandb_settings = Settings(
            api_key=wandb_key,
            entity=self.entity,
            project=self.project,
        )

        if self.is_master and self.enabled:
            os.environ["WANDB_API_KEY"] = wandb_key
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
            display_name = f"{self.run_name}-{timestamp}"
            config["overrides"] = list(HydraConfig.get().overrides.task) + list(HydraConfig.get().overrides.hydra)
            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                group=self.run_name,
                name=display_name,
                config=config,
                settings=self.wandb_settings,
            )
            master_log(logger, f"Initialized new run: {self.run.name} (ID: {self.run.id}, group: {self.run_name})")

    def log(self, metrics: dict, step: int):
        """
        Log metrics at given step.
        """
        if self.is_master and self.enabled:
            self.wandb.log(metrics, step=step)

    def save(self, path: str | Path, base_path: str | Path = "./"):
        """
        Save *any* file to wandb.
        """
        if self.is_master and self.enabled:
            self.wandb.save(path, base_path=base_path)

    def log_token_nll_loss(self, token_nll_loss: np.ndarray, step: int, k: str):
        """
        Log token-wise nll loss.
        """
        if not self.is_master or not self.enabled:
            return
        import wandb

        if token_nll_loss.ndim == 1:
            token_nll_loss = token_nll_loss[np.newaxis, ...]

        for r, row in enumerate(token_nll_loss):
            table = wandb.Table(data=[(step, i, tloss) for i, tloss in enumerate(row)], columns=["step", "token", "token_nll_loss"])
            self.log(
                {
                    f"{k}/token_nll_loss(before gs {r})": wandb.plot.line(
                        table, "token", "token_nll_loss", "step", title=f"Token NLL loss at step {step} (before gs {r})", split_table=True
                    )
                },
                step,
            )

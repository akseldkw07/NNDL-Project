import os
import typing as t
from datetime import datetime

import wandb

from nndl_model.base_model import BaseModel
from nndl_model.constants import ROOT_DIR


def start_wandb_run(
    model: BaseModel,  # your nn.Module with BaseModel.name()
    entity: str = "nndl-project-F25",  # your team/org slug
    project: str | None = "Multihead-Classification-Competition",  # the shared project
    mode: t.Literal["online", "offline", "disabled"] = "online",  # "online" | "offline" | "disabled"
    job_type: str = "train",  # "train" | "eval" | etc.
    tags: list[str] | None = None,
    config: dict[str, t.Any] | str | None = None,
):
    group = model.name()
    run_name = f"{group}__{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb_dir = os.getenv("WANDB_OUTPUT_DIR") or ROOT_DIR / "wandb_logs"
    config = {"model_summary": model._summary()} if config is None else config

    run = wandb.init(
        entity=entity,
        project=project,
        mode=mode,
        group=group,  # collapsible group in the UI
        name=run_name,  # readable run name
        job_type=job_type,
        tags=tags or [],
        config=config,
        reinit="finish_previous",
        dir=wandb_dir,
    )
    return run

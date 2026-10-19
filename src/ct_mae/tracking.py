"""W&B logging is optional for local diagnostics and mandatory for production Kaggle runs.

API reference: https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_init.py
"""

import os
from pathlib import Path

from ct_mae.config import config_digest


class RunTracker:
    def __init__(self, config: dict, stage: str, directory: Path, *, run_id: str | None = None,
                 smoke: bool = False):
        self.run = None
        self.id, self.url = run_id, None
        self.artifact_name = f"ct-mae-{config_digest(config)[:12]}-{stage}-resume"
        settings = config["wandb"]
        if settings["mode"] == "disabled":
            return
        import wandb

        if settings["mode"] == "online" and not os.environ.get("WANDB_API_KEY"):
            try:
                from kaggle_secrets import UserSecretsClient
                os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret(
                    settings["secret_label"],
                )
            except Exception:
                raise RuntimeError(
                    f"Enable the {settings['secret_label']} Kaggle Secret before training"
                ) from None
        self.run = wandb.init(
            project=settings["project"], entity=settings["entity"], mode=settings["mode"],
            id=run_id, resume="must" if run_id and settings["mode"] == "online" else None,
            name=f"{'smoke-' if smoke else ''}{stage}-seed{config['seed']}",
            group=f"{'smoke-' if smoke else ''}ct-mae-seed{config['seed']}",
            job_type=stage, config=config, dir=str(directory),
            settings=wandb.Settings(init_timeout=60),
        )
        self.id, self.url = self.run.id, self.run.url
        self.run.define_metric("epoch")
        for prefix in ("train/*", "val/*", "resource/*"):
            self.run.define_metric(prefix, step_metric="epoch")
        self.run.log({"preflight/ok": 1})

    def log_epoch(self, epoch: int, train: dict, validation: dict, resources: dict) -> None:
        if self.run is None:
            return
        metrics = {"epoch": epoch}
        for prefix, values in (("train", train), ("val", validation), ("resource", resources)):
            metrics.update({f"{prefix}/{key}": value for key, value in values.items()
                            if isinstance(value, (float, int))})
        self.run.log(metrics)

    def image(self, path: Path, key: str, caption: str) -> None:
        if self.run is not None:
            import wandb
            self.run.log({key: wandb.Image(str(path), caption=caption)})

    def checkpoint(self, directory: Path, epoch: int) -> None:
        if self.run is None:
            return
        import wandb
        artifact = wandb.Artifact(self.artifact_name, type="checkpoint",
                                 metadata={"completed_epoch": epoch})
        for filename in ("last.pt", "best.pt", "run.json"):
            artifact.add_file(str(directory / filename), name=filename)
        self.run.log_artifact(artifact, aliases=["latest", f"epoch-{epoch}"]).wait()

    def finish(self, status: str, summary: dict | None = None) -> None:
        if self.run is not None:
            self.run.summary["status"] = status
            if summary:
                self.run.summary.update(summary)
            self.run.finish(exit_code=1 if status == "failed" else 0)

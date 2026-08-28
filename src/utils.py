"""Utilidades sin dependencias de Streamlit."""

from __future__ import annotations

import os
import random
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Replica la política de semillas y determinismo del notebook."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_run_id(prefix: str = "web") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}_{uuid4().hex[:8]}"


def safe_torch_save(payload, destination: str | os.PathLike[str]) -> Path:
    """Guarda primero localmente y publica el checkpoint con reemplazo atómico."""
    import torch

    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    local_tmp_path: Path | None = None
    destination_tmp = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )
    try:
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as handle:
            local_tmp_path = Path(handle.name)
        torch.save(payload, local_tmp_path)
        shutil.copyfile(local_tmp_path, destination_tmp)
        os.replace(destination_tmp, destination)
    finally:
        if local_tmp_path is not None:
            local_tmp_path.unlink(missing_ok=True)
        destination_tmp.unlink(missing_ok=True)
    return destination

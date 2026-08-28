"""Loss y métricas con las mismas fórmulas del notebook científico."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


criterion = nn.SmoothL1Loss(beta=1.0)


@torch.no_grad()
def mae(pred, target):
    return torch.mean(torch.abs(pred - target))


@torch.no_grad()
def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))


@torch.no_grad()
def r2_score(pred, target):
    y = target
    yhat = pred
    y_mean = torch.mean(y)
    ss_res = torch.sum((y - yhat) ** 2)
    ss_tot = torch.sum((y - y_mean) ** 2).clamp_min(1e-6)
    return 1.0 - ss_res / ss_tot


def _match_prediction_to_target(pred, target, model_name="model"):
    if pred.ndim != target.ndim:
        raise ValueError(f"{model_name}: pred ndim {pred.ndim} != target ndim {target.ndim}")

    if pred.shape[0] != target.shape[0] or pred.shape[1] != target.shape[1]:
        raise ValueError(f"{model_name}: pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")

    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)

    if pred.shape != target.shape:
        raise ValueError(f"{model_name}: pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")

    return pred


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        pred = model(x)
        pred = _match_prediction_to_target(pred, y, model.__class__.__name__)

        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate_global(model, loader, device, return_loss=True):
    model.eval()

    abs_sum = 0.0
    sq_sum = 0.0
    loss_sum = 0.0
    batches = 0
    n = 0

    preds_all = []
    targets_all = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        pred = _match_prediction_to_target(pred, y, model.__class__.__name__)
        diff = pred - y

        abs_sum += torch.sum(torch.abs(diff)).item()
        sq_sum += torch.sum(diff ** 2).item()
        n += y.numel()

        if return_loss:
            loss_sum += criterion(pred, y).item()
            batches += 1

        preds_all.append(pred.flatten().cpu())
        targets_all.append(y.flatten().cpu())

    if n == 0:
        raise ValueError("No se puede evaluar un DataLoader vacío.")

    mae_value = abs_sum / n
    rmse_value = (sq_sum / n) ** 0.5

    preds_all = torch.cat(preds_all)
    targets_all = torch.cat(targets_all)

    y_mean = torch.mean(targets_all)
    ss_res = torch.sum((targets_all - preds_all) ** 2)
    ss_tot = torch.sum((targets_all - y_mean) ** 2).clamp_min(1e-6)
    r2_value = 1.0 - (ss_res / ss_tot)

    metrics = {
        "MAE": float(mae_value),
        "RMSE": float(rmse_value),
        "R2": float(r2_value),
    }
    if return_loss:
        metrics["loss"] = float(loss_sum / max(1, batches))
    return metrics

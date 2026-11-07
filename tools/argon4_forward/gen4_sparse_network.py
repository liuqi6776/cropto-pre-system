"""Trainable local TabNet-style expert, never an original-author reproduction.

Architecture reference: Google TabNet (Arik/Pfister), arxiv.org/abs/1908.07442.
Entmax-1.5 equations: Peters/Niculae/Martins, arxiv.org/abs/1905.05702;
author reference implementation: github.com/deep-spin/entmax (MIT).
This independent implementation uses the explicitly registered local choices:
four decision steps, train-only screening/scaling, shared FC weights with
separate GhostBN statistics, and an Entmax-1.5 rather than sparsemax selector.
No data discovery, network access, production integration, or order capability.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import os
from pathlib import Path
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F

MODEL_VERSION = "LOCAL_ADAPTATION_SPARSE_TABNET_ENTMAX15_V1"
CONFIG = {
    "steps": 4, "gamma": 1.5, "n_d": 8, "n_a": 8,
    "shared_glu_layers": 2, "independent_glu_layers": 2,
    "virtual_batch_size": 128, "bn_momentum": 0.02, "bn_epsilon": 1e-5,
    "correlation_limit": 0.88, "constant_tolerance": 1e-12,
    "lambda_sparse": 1e-4, "learning_rate": 0.01,
    "batch_size": 1024, "max_epochs": 20, "patience": 5,
    "early_stopping_min_delta": 0.0,
    "gradient_clip": 1.0, "prediction_batch_size": 4096,
    "seeds": [42, 456, 789], "threads": 2,
    "training_device_policy": "CUDA_IF_AVAILABLE_ELSE_CPU", "training_dtype": "float32",
    "early_stopping_dtype": "float32", "early_stopping_device": "SAME_AS_TRAINING",
    "prediction_device": "cpu", "prediction_dtype": "float64", "prediction_output": "SIGMOID_UP_PROBABILITY",
    "allow_tf32": False, "cudnn_benchmark": False, "deterministic_algorithms": "REQUIRED_FAIL_IF_UNSUPPORTED",
    "singleton_training_batch": "BORROW_ONE_FROM_PREVIOUS_OR_MERGE_IF_SIZE_TWO_NO_DROPPED_ROWS",
    "early_stopping_tie": "KEEP_EARLIEST_BEST_EPOCH",
}


class SparseNetworkError(ValueError):
    pass


class _Entmax15(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, dim):
        dim %= logits.ndim
        centered = (logits - logits.amax(dim=dim, keepdim=True)) * 0.5
        ordered = centered.sort(dim=dim, descending=True).values
        shape = [1] * logits.ndim
        shape[dim] = logits.shape[dim]
        ranks = torch.arange(1, logits.shape[dim] + 1, dtype=logits.dtype, device=logits.device).reshape(shape)
        means = ordered.cumsum(dim) / ranks
        variance_sums = ordered.square().cumsum(dim) - ranks * means.square()
        thresholds = means - ((1 - variance_sums) / ranks).clamp_min(0).sqrt()
        support = (ordered >= thresholds).sum(dim=dim, keepdim=True)
        threshold = thresholds.gather(dim, support - 1)
        probability = (centered - threshold).clamp_min(0).square()
        ctx.dim = dim
        ctx.save_for_backward(probability)
        return probability

    @staticmethod
    def backward(ctx, upstream):
        (probability,) = ctx.saved_tensors
        root = probability.sqrt()
        weighted = upstream * root
        average = weighted.sum(dim=ctx.dim, keepdim=True) / root.sum(dim=ctx.dim, keepdim=True)
        return weighted - average * root, None


def entmax15(logits, dim=-1):
    """Exact sparse probability mapping, with its support-aware Jacobian."""
    if not logits.is_floating_point() or logits.ndim == 0 or logits.shape[dim] == 0:
        raise SparseNetworkError("nonempty floating tensor required")
    return _Entmax15.apply(logits, dim)


class GhostBatchNorm(nn.Module):
    def __init__(self, width, config):
        super().__init__()
        self.virtual_batch_size = int(config["virtual_batch_size"])
        self.bn = nn.BatchNorm1d(width, momentum=config["bn_momentum"], eps=config["bn_epsilon"])

    def forward(self, value):
        if not self.training:
            return self.bn(value)
        if value.shape[0] < 2:
            raise SparseNetworkError("training batch must contain at least two rows")
        # Equal-sized chunks avoid the singleton tail of split(virtual_size).
        count = min(math.ceil(value.shape[0] / self.virtual_batch_size), value.shape[0] // 2)
        chunks = torch.tensor_split(value, count, dim=0)
        return torch.cat([self.bn(chunk) for chunk in chunks], dim=0)


class GLU(nn.Module):
    def __init__(self, linear, width, config):
        super().__init__()
        self.linear = linear
        self.norm = GhostBatchNorm(2 * width, config)

    def forward(self, value):
        activation, gate = self.norm(self.linear(value)).chunk(2, dim=1)
        return activation * torch.sigmoid(gate)


class FeatureTransformer(nn.Module):
    def __init__(self, shared, width, config):
        super().__init__()
        # Linear modules are the very same objects at every step. Each use has
        # its own GhostBN: this choice is recorded as a local adaptation.
        self.shared = nn.ModuleList([GLU(layer, width, config) for layer in shared])
        self.independent = nn.ModuleList([
            GLU(nn.Linear(width, 2 * width, bias=False), width, config)
            for _ in range(config["independent_glu_layers"])
        ])

    def forward(self, value):
        output = self.shared[0](value)
        for layer in list(self.shared)[1:] + list(self.independent):
            output = (output + layer(output)) * math.sqrt(0.5)
        return output


class AttentiveTransformer(nn.Module):
    def __init__(self, feature_count, config):
        super().__init__()
        self.linear = nn.Linear(config["n_a"], feature_count, bias=False)
        self.norm = GhostBatchNorm(feature_count, config)

    def forward(self, attention, prior):
        return entmax15(prior * self.norm(self.linear(attention)), dim=1)


class SparseTabNet(nn.Module):
    def __init__(self, feature_count, config=None):
        super().__init__()
        self.config = _configuration(config)
        c = self.config
        width = c["n_d"] + c["n_a"]
        self.input_norm = nn.BatchNorm1d(feature_count, momentum=c["bn_momentum"], eps=c["bn_epsilon"])
        self.shared_linear = nn.ModuleList([
            nn.Linear(feature_count if index == 0 else width, 2 * width, bias=False)
            for index in range(c["shared_glu_layers"])
        ])
        self.initial = FeatureTransformer(self.shared_linear, width, c)
        self.transforms = nn.ModuleList([FeatureTransformer(self.shared_linear, width, c) for _ in range(c["steps"])])
        self.attention = nn.ModuleList([AttentiveTransformer(feature_count, c) for _ in range(c["steps"])])
        self.head = nn.Linear(c["n_d"], 1, bias=False)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def forward(self, features, return_masks=False):
        original = self.input_norm(features)
        attention = self.initial(original)[:, self.config["n_d"]:]
        prior = torch.ones_like(original)
        decision = original.new_zeros((len(original), self.config["n_d"]))
        masks = []
        for selector, transform in zip(self.attention, self.transforms):
            mask = selector(attention, prior)
            prior = prior * (self.config["gamma"] - mask)
            # Every step masks the same normalized input, never a prior mask's output.
            processed = transform(mask * original)
            decision = decision + F.relu(processed[:, :self.config["n_d"]])
            attention = processed[:, self.config["n_d"]:]
            masks.append(mask)
        stacked = torch.stack(masks)
        negative_entropy = (stacked * stacked.clamp_min(1e-15).log()).sum(dim=-1).mean()
        logits = self.head(decision).squeeze(1)
        if return_masks:
            return logits, negative_entropy, stacked
        return logits, negative_entropy


def _configuration(overrides):
    result = {key: list(value) if isinstance(value, list) else value for key, value in CONFIG.items()}
    if overrides is not None:
        unknown = set(overrides) - set(result)
        if unknown:
            raise SparseNetworkError("unknown configuration fields")
        result.update(overrides)
    for key in ("steps", "n_d", "n_a", "shared_glu_layers", "independent_glu_layers", "virtual_batch_size",
                "batch_size", "max_epochs", "patience", "prediction_batch_size", "threads"):
        if type(result[key]) is not int or result[key] <= 0:
            raise SparseNetworkError("positive integer configuration required: " + key)
    if result["shared_glu_layers"] != 2 or result["independent_glu_layers"] != 2:
        raise SparseNetworkError("local architecture requires two shared and two independent GLUs")
    if result["batch_size"] < 2 or result["virtual_batch_size"] < 2 or result["threads"] > 4:
        raise SparseNetworkError("invalid batch or CPU thread configuration")
    for key in ("gamma", "bn_momentum", "bn_epsilon", "correlation_limit", "constant_tolerance", "lambda_sparse", "learning_rate", "gradient_clip"):
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise SparseNetworkError("positive finite configuration required: " + key)
    if result["gamma"] <= 1 or not 0 < result["correlation_limit"] < 1 or result["bn_momentum"] > 1:
        raise SparseNetworkError("invalid prior, correlation, or BN setting")
    if result["early_stopping_min_delta"] != 0:
        raise SparseNetworkError("strict zero-min-delta validation early stopping is frozen")
    for key in ("training_device_policy", "training_dtype", "early_stopping_dtype", "early_stopping_device", "prediction_device",
                "prediction_dtype", "prediction_output", "allow_tf32", "cudnn_benchmark", "deterministic_algorithms",
                "singleton_training_batch", "early_stopping_tie"):
        if result[key] != CONFIG[key]:
            raise SparseNetworkError("frozen numerical policy cannot be overridden: " + key)
    return result


def _matrix(value, name, feature_names=None):
    if hasattr(value, "columns") and feature_names is not None and list(value.columns) != list(feature_names):
        raise SparseNetworkError(name + " ordered feature schema mismatch")
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] == 0 or not np.isfinite(result).all():
        raise SparseNetworkError(name + " must be a finite feature matrix")
    return result


def _labels(value, rows):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (rows,) or not np.isfinite(result).all() or not np.isin(result, [0, 1]).all():
        raise SparseNetworkError("one observed binary label per row required")
    return result.astype(np.float32)


def select_train_features(x_train, feature_names, correlation_limit=0.88, constant_tolerance=1e-12):
    """Greedy order-preserving selection; no validation/test input is accepted."""
    x = _matrix(x_train, "training", feature_names)
    names = list(feature_names)
    if len(names) != x.shape[1] or len(set(names)) != len(names) or any(not isinstance(s, str) for s in names):
        raise SparseNetworkError("unique complete ordered feature names required")
    centered = x - x.mean(axis=0)
    deviation = np.sqrt(np.mean(centered ** 2, axis=0))
    candidates = np.flatnonzero(deviation > constant_tolerance).tolist()
    if not candidates:
        raise SparseNetworkError("all training features are constant")
    standardized = centered[:, candidates] / deviation[candidates]
    correlation = standardized.T @ standardized / len(x)
    kept_positions, rejected = [], []
    for index, original_index in enumerate(candidates):
        conflicts = [other for other in kept_positions if abs(correlation[index, other]) >= correlation_limit]
        if conflicts:
            other = conflicts[0]
            rejected.append({"feature": names[original_index], "kept_feature": names[candidates[other]],
                             "absolute_correlation": float(abs(correlation[index, other]))})
        else:
            kept_positions.append(index)
    selected = [candidates[index] for index in kept_positions]
    return selected, {"fit_scope": "TRAIN_ONLY", "training_rows": len(x), "correlation_limit": correlation_limit,
                      "selected_feature_names": [names[index] for index in selected],
                      "dropped_constant_names": [names[i] for i in range(len(names)) if i not in candidates],
                      "dropped_correlated": rejected}


def _training_slices(size, batch_size):
    endpoints = list(range(0, size, batch_size)) + [size]
    if len(endpoints) > 2 and endpoints[-1] - endpoints[-2] == 1:
        if endpoints[-2] - endpoints[-3] >= 3:
            endpoints[-2] -= 1  # previous batch loses one, final has two.
        else:
            endpoints.pop(-2)  # tiny test batches: merge 2 + 1 rather than create 1 + 2.
    return [slice(left, right) for left, right in zip(endpoints, endpoints[1:])]


@contextmanager
def _training_environment(seed, device, threads):
    previous_threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warning_only = torch.is_deterministic_algorithms_warn_only_enabled()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    workspace_previous = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    try:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            yield
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warning_only)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        if workspace_previous is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def _cpu_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


@contextmanager
def _cpu_threads(count):
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(count)
        yield
    finally:
        torch.set_num_threads(previous)


def _validation_bce(model, x, y, batch_size):
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            logits, _ = model(x[start:start + batch_size])
            total += float(F.binary_cross_entropy_with_logits(logits, y[start:start + batch_size], reduction="sum").item())
    return total / len(x)


def fit_sparse_expert(x_train, y_binary, x_val, y_val, seed, feature_names, *, device=None, config=None):
    """Return (safe CPU state, metadata). Early stopping sees validation BCE only."""
    started = time.perf_counter()
    c = _configuration(config)
    if type(seed) is not int:
        raise SparseNetworkError("integer seed required")
    names = list(feature_names)
    train = _matrix(x_train, "training", names)
    validation = _matrix(x_val, "validation", names)
    if len(train) < 2 or len(validation) < 1 or train.shape[1] != validation.shape[1]:
        raise SparseNetworkError("training and validation shapes invalid")
    y = _labels(y_binary, len(train))
    vy = _labels(y_val, len(validation))
    if len(np.unique(y)) != 2:
        raise SparseNetworkError("both training directions required")
    selected, selection = select_train_features(train, names, c["correlation_limit"], c["constant_tolerance"])
    scaler = StandardScaler().fit(train[:, selected])
    tx = np.ascontiguousarray(scaler.transform(train[:, selected]), dtype=np.float32)
    vx = np.ascontiguousarray(scaler.transform(validation[:, selected]), dtype=np.float32)
    chosen = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if chosen.type not in ("cpu", "cuda"):
        raise SparseNetworkError("only CPU or CUDA devices are supported")
    history, best_state, best_epoch, stale = [], None, 0, 0
    best_bce = math.inf
    gradient_seen = set()
    gradient_audited = False
    gradient_max = {"attention": 0.0, "shared_glu": 0.0, "independent_glu": 0.0}
    with _training_environment(seed, chosen, c["threads"]):
        model = SparseTabNet(len(selected), c).to(chosen)
        initial = {name: tensor.detach().cpu().clone() for name, tensor in model.named_parameters()}
        x = torch.from_numpy(tx).to(chosen)
        targets = torch.from_numpy(y).to(chosen)
        v = torch.from_numpy(vx).to(chosen)
        vtargets = torch.from_numpy(vy).to(chosen)
        optimizer = torch.optim.Adam(model.parameters(), lr=c["learning_rate"])
        initial_bce = _validation_bce(model, v, vtargets, c["prediction_batch_size"])
        rng = np.random.default_rng(seed)
        for epoch in range(1, c["max_epochs"] + 1):
            model.train()
            order = torch.as_tensor(rng.permutation(len(x)), dtype=torch.long, device=chosen)
            train_bce = negative_entropy_total = 0.0
            for batch in _training_slices(len(x), c["batch_size"]):
                ids = order[batch]
                optimizer.zero_grad(set_to_none=True)
                logits, negative_entropy = model(x[ids])
                bce = F.binary_cross_entropy_with_logits(logits, targets[ids])
                loss = bce - c["lambda_sparse"] * negative_entropy
                if not torch.isfinite(loss):
                    raise SparseNetworkError("nonfinite training objective")
                loss.backward()
                if not gradient_audited:
                    for name, parameter in model.named_parameters():
                        if parameter.grad is None:
                            continue
                        gradient_seen.add(name)
                        group = "attention" if name.startswith("attention.") else "shared_glu" if name.startswith("shared_linear.") else "independent_glu" if ".independent." in name else None
                        if group:
                            gradient_max[group] = max(gradient_max[group], float(parameter.grad.detach().abs().max().item()))
                    gradient_audited = True
                nn.utils.clip_grad_norm_(model.parameters(), c["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                train_bce += float(bce.item()) * len(ids)
                negative_entropy_total += float(negative_entropy.item()) * len(ids)
            validation_bce = _validation_bce(model, v, vtargets, c["prediction_batch_size"])
            if not math.isfinite(validation_bce):
                raise SparseNetworkError("nonfinite validation objective")
            history.append({"epoch": epoch, "train_bce": train_bce / len(x), "validation_bce": validation_bce,
                            "train_mask_negative_entropy": negative_entropy_total / len(x)})
            if validation_bce < best_bce:
                best_bce, best_epoch, stale = validation_bce, epoch, 0
                best_state = _cpu_state(model)
            else:
                stale += 1
                if stale >= c["patience"]:
                    break
        assert best_state is not None
        trainable = list(initial)
        parameter_changes = {name: float((best_state[name] - initial[name]).abs().max().item()) for name in trainable}
    state = {"format_version": 1, "model_version": MODEL_VERSION, "classification": "LOCAL_ADAPTATION_NOT_AUTHOR_REPRODUCTION",
             "input_feature_names": [str(name) for name in names], "selected_indices": selected, "selected_feature_names": [str(names[i]) for i in selected],
             "scaler_mean": torch.tensor(scaler.mean_.copy(), dtype=torch.float64),
             "scaler_scale": torch.tensor(scaler.scale_.copy(), dtype=torch.float64),
             "config": c, "seed": seed, "best_epoch": best_epoch, "model_state_dict": best_state}
    metadata = {"model_version": MODEL_VERSION, "classification": state["classification"], "seed": seed, "config": c,
                "training_rows": len(train), "validation_rows": len(validation), "selection": selection,
                "input_feature_count": len(names), "selected_feature_count": len(selected), "best_epoch": best_epoch,
                "epochs_trained": len(history), "initial_validation_bce": initial_bce, "best_validation_bce": best_bce,
                "history": history, "early_stopping_metric": "VALIDATION_BCE_ONLY", "test_data_used": False,
                "mask_regularizer": "BCE - lambda_sparse * mean(sum(mask * log(mask)))",
                "shared_weight_policy": "Same two FC parameter objects across initial splitter and four decision transformers; separate GhostBN statistics per use",
                "trainable_parameter_tensors": len(trainable), "parameters_receiving_gradients": len(gradient_seen),
                "gradient_audit_scope": "FIRST_TRAINING_BATCH_ONLY; best-state parameter deltas also retained",
                "parameter_names_without_gradient": sorted(set(trainable) - gradient_seen), "gradient_max_by_group": gradient_max,
                "best_parameter_max_absolute_change": parameter_changes, "training_device": str(chosen),
                "training_dtype": "float32", "inference_dtype": "float64", "torch_version": str(torch.__version__),
                "deterministic_algorithms": True, "elapsed_seconds": time.perf_counter() - started}
    return state, metadata


def _validate_state(state):
    def safe(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" or not torch.isfinite(value).all():
                raise SparseNetworkError("only finite CPU tensors may be persisted")
        elif type(value) is dict:
            if any(type(key) is not str for key in value):
                raise SparseNetworkError("state dictionary keys must be strings")
            for child in value.values():
                safe(child)
        elif type(value) in (list, tuple):
            for child in value:
                safe(child)
        elif type(value) in (str, int, float, bool) or value is None:
            if type(value) is float and not math.isfinite(value):
                raise SparseNetworkError("nonfinite state scalar")
        else:
            raise SparseNetworkError("state contains an unsafe non-tensor object")
    safe(state)
    if not isinstance(state, dict) or state.get("format_version") != 1 or state.get("model_version") != MODEL_VERSION:
        raise SparseNetworkError("unrecognized local sparse model state")
    names, selected = state["input_feature_names"], state["selected_indices"]
    if (not isinstance(names, list) or len(set(names)) != len(names) or not names
            or not isinstance(selected, list) or not selected or sorted(set(selected)) != selected
            or any(type(i) is not int or not 0 <= i < len(names) for i in selected)
            or state["selected_feature_names"] != [names[i] for i in selected]):
        raise SparseNetworkError("invalid persisted feature selection")
    for key in ("scaler_mean", "scaler_scale"):
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.shape != (len(selected),) or not torch.isfinite(value).all():
            raise SparseNetworkError("invalid CPU scaler state")
    if not (state["scaler_scale"] > 0).all():
        raise SparseNetworkError("invalid scaler scale")
    for value in state["model_state_dict"].values():
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or not torch.isfinite(value).all():
            raise SparseNetworkError("invalid CPU model tensor")
    _configuration(state["config"])


def predict_sparse_expert(state, x, *, feature_names=None, batch_size=4096, device="cpu"):
    """UP probability; eval-only frozen normalization, float64 batch parity."""
    _validate_state(state)
    names = state["input_feature_names"]
    if feature_names is not None and list(feature_names) != names:
        raise SparseNetworkError("prediction ordered feature schema mismatch")
    values = _matrix(x, "prediction", names)
    if values.shape[1] != len(names) or type(batch_size) is not int or batch_size < 1:
        raise SparseNetworkError("prediction shape or batch size invalid")
    if not len(values):
        return np.empty(0, dtype=np.float64)
    values = (values[:, state["selected_indices"]] - state["scaler_mean"].numpy()) / state["scaler_scale"].numpy()
    chosen = torch.device(device)
    if chosen.type != "cpu":
        raise SparseNetworkError("public inference is fixed to CPU float64")
    # Model initialization must not perturb another training run's random state.
    with torch.random.fork_rng(devices=[]):
        model = SparseTabNet(len(state["selected_indices"]), state["config"])
    model.load_state_dict(state["model_state_dict"], strict=True)
    model = model.to(device=chosen, dtype=torch.float64).eval()
    outputs = []
    with torch.inference_mode(), _cpu_threads(state["config"]["threads"]):
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(np.ascontiguousarray(values[start:start + batch_size]), dtype=torch.float64, device=chosen)
            logits, _ = model(batch)
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    probability = np.concatenate(outputs)
    if not np.isfinite(probability).all():
        raise SparseNetworkError("nonfinite prediction")
    return probability


def save_sparse_expert(state, path):
    """Write our CPU-only tensor/basic-container package; no arbitrary object graph."""
    _validate_state(state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_sparse_expert(path):
    """Only PyTorch's restricted weights loader; callers still must pin file SHA."""
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    _validate_state(state)
    return state

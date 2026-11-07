"""Offline mathematical and learning tests; no production account or data."""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="isolated research torch dependency; separately executed in the pinned torch environment")

from tools.argon4_forward import gen4_sparse_network as network


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def synthetic(rows=768, seed=18):
    rng = np.random.default_rng(seed)
    independent = rng.normal(size=(rows, 5))
    x = np.column_stack([independent, independent[:, 0], np.ones(rows)])
    y = (independent[:, 0] + 0.3 * independent[:, 1] > 0).astype(np.float32)
    return x, y, ["x" + str(i) for i in range(x.shape[1])]


def test_entmax_known_value_sparse_simplex_and_translation():
    x = torch.tensor([[-2.0, 0.0, 0.5], [10.0, 0.0, -10.0]], dtype=torch.float64)
    p = network.entmax15(x)
    assert torch.all(p >= 0)
    torch.testing.assert_close(p.sum(1), torch.ones(2, dtype=torch.float64), rtol=0, atol=1e-14)
    assert p[0, 0] == 0 and torch.equal(p[1], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(p[0, 1:], torch.tensor([0.3260073637, 0.6739926363], dtype=torch.float64), rtol=0, atol=1e-9)
    torch.testing.assert_close(network.entmax15(x + 50), p, rtol=0, atol=1e-14)
    torch.testing.assert_close(network.entmax15(x.T, dim=0), p.T, rtol=0, atol=1e-14)


def test_entmax_custom_backward_matches_finite_differences():
    torch.manual_seed(22)
    x = (torch.randn(3, 6, dtype=torch.float64) * 0.4).requires_grad_()
    assert torch.autograd.gradcheck(network.entmax15, (x,), eps=1e-6, atol=1e-6, rtol=1e-4)
    sparse = torch.tensor([-4.0, 0.0, 0.4], dtype=torch.float64, requires_grad=True)
    network.entmax15(sparse).dot(torch.tensor([1.0, 2.0, -3.0], dtype=torch.float64)).backward()
    assert sparse.grad[0] == 0
    assert abs(sparse.grad.sum().item()) < 1e-14


def test_true_shared_glu_parameters_and_independent_batch_statistics():
    model = network.SparseTabNet(7)
    assert len(model.transforms) == len(model.attention) == 4
    for transformer in [model.initial, *model.transforms]:
        assert len(transformer.shared) == len(transformer.independent) == 2
        for i in range(2):
            assert transformer.shared[i].linear is model.shared_linear[i]
    assert model.initial.shared[0].norm.bn is not model.transforms[0].shared[0].norm.bn
    assert len(list(model.parameters())) == len({id(p) for p in model.parameters()})


def test_masks_use_same_original_input_and_prior_product_without_detach():
    torch.manual_seed(8)
    model = network.SparseTabNet(7).double().eval()
    x = torch.randn(12, 7, dtype=torch.float64)
    transform_inputs, priors = [], []
    handles = []
    for layer in model.transforms:
        handles.append(layer.register_forward_pre_hook(lambda module, args: transform_inputs.append(args[0])))
    for layer in model.attention:
        handles.append(layer.register_forward_pre_hook(lambda module, args: priors.append(args[1])))
    logits, negative_entropy, masks = model(x, return_masks=True)
    original = model.input_norm(x)
    running = torch.ones_like(original)
    for index in range(4):
        torch.testing.assert_close(priors[index], running, rtol=0, atol=1e-14)
        torch.testing.assert_close(transform_inputs[index], masks[index] * original, rtol=0, atol=1e-14)
        running = running * (1.5 - masks[index])
    assert priors[1].requires_grad and (priors[1] > 1).any()
    assert negative_entropy <= 0
    (logits.square().mean() - 1e-4 * negative_entropy).backward()
    assert model.attention[0].linear.weight.grad.abs().sum() > 0
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("rows,batch", [(3, 2), (5, 2), (17, 8), (129, 128), (1025, 1024)])
def test_singleton_training_tails_preserve_each_row(rows, batch):
    pieces = network._training_slices(rows, batch)
    assert all(piece.stop - piece.start >= 2 for piece in pieces)
    assert np.array_equal(np.concatenate([np.arange(rows)[piece] for piece in pieces]), np.arange(rows))
    bn = network.GhostBatchNorm(3, {**network.CONFIG, "virtual_batch_size": batch})
    output = bn(torch.randn(rows, 3))
    assert output.shape == (rows, 3) and torch.isfinite(output).all()


def test_train_only_greedy_screen_preserves_order_and_does_not_force_62():
    rng = np.random.default_rng(11)
    a, b = rng.normal(size=(2, 500))
    train = np.column_stack([a, a * 2, b, np.ones(500), -a])
    selected, detail = network.select_train_features(train, ["a", "duplicate", "b", "constant", "negative_duplicate"])
    assert selected == [0, 2]
    assert detail["selected_feature_names"] == ["a", "b"]
    assert detail["dropped_constant_names"] == ["constant"]
    assert detail["fit_scope"] == "TRAIN_ONLY"
    with pytest.raises(network.SparseNetworkError, match="constant"):
        network.select_train_features(np.ones((20, 3)), ["a", "b", "c"])


@pytest.fixture(scope="module")
def learned():
    x, y, names = synthetic()
    state, detail = network.fit_sparse_expert(x[:512], y[:512], x[512:], y[512:], 42, names, device="cpu",
        config={"max_epochs": 10, "patience": 5, "batch_size": 128, "virtual_batch_size": 64, "threads": 1})
    return x, y, names, state, detail


def test_actual_training_reaches_attention_glu_and_reduces_holdout_loss(learned):
    x, y, names, state, detail = learned
    assert detail["best_validation_bce"] < detail["initial_validation_bce"] - 0.08
    assert detail["parameters_receiving_gradients"] == detail["trainable_parameter_tensors"]
    assert detail["parameter_names_without_gradient"] == []
    assert all(value > 0 for value in detail["gradient_max_by_group"].values())
    changes = detail["best_parameter_max_absolute_change"]
    assert changes["attention.0.linear.weight"] > 0
    assert changes["shared_linear.0.weight"] > 0
    assert changes["transforms.0.independent.0.linear.weight"] > 0
    assert detail["best_epoch"] == min(detail["history"], key=lambda row: row["validation_bce"])["epoch"]
    assert detail["test_data_used"] is False
    np.testing.assert_allclose(state["scaler_mean"].numpy(), x[:512, :5].mean(0), rtol=0, atol=1e-14)
    np.testing.assert_allclose(state["scaler_scale"].numpy(), x[:512, :5].std(0), rtol=0, atol=1e-14)


def test_validation_cannot_change_feature_screen_or_scaler():
    x, y, names = synthetic(180)
    config = {"max_epochs": 1, "batch_size": 64, "threads": 1}
    first, _ = network.fit_sparse_expert(x[:128], y[:128], x[128:], y[128:], 42, names, device="cpu", config=config)
    second, _ = network.fit_sparse_expert(x[:128], y[:128], x[128:] * 100 + 30, y[128:], 42, names, device="cpu", config=config)
    assert first["selected_indices"] == second["selected_indices"]
    assert torch.equal(first["scaler_mean"], second["scaler_mean"])
    assert torch.equal(first["scaler_scale"], second["scaler_scale"])


def test_inference_batches_and_validation_z_selection_agree(learned):
    x, y, names, state, detail = learned
    values = x[512:559]
    bulk = network.predict_sparse_expert(state, values, feature_names=names, batch_size=4096)
    for size in (1, 7, 16, 128):
        chunked = network.predict_sparse_expert(state, values, batch_size=size)
        np.testing.assert_allclose(chunked, bulk, rtol=0, atol=1e-12)
        baseline_z = (bulk - bulk.mean()) / bulk.std()
        candidate_z = (chunked - bulk.mean()) / bulk.std()
        np.testing.assert_allclose(candidate_z, baseline_z, rtol=0, atol=1e-12)
        assert np.array_equal(abs(candidate_z) >= 1.1, abs(baseline_z) >= 1.1)
        assert np.array_equal(candidate_z > 0, baseline_z > 0)
    assert bulk.dtype == np.float64 and ((bulk >= 0) & (bulk <= 1)).all()
    before = {name: tensor.clone() for name, tensor in state["model_state_dict"].items()}
    network.predict_sparse_expert(state, values)
    assert all(torch.equal(tensor, state["model_state_dict"][name]) for name, tensor in before.items())


def test_safe_tensor_save_reload_and_explicit_schema(learned, tmp_path):
    x, y, names, state, detail = learned
    file = tmp_path / "local_sparse.pt"
    network.save_sparse_expert(state, file)
    loaded = network.load_sparse_expert(file)
    restricted = torch.load(file, weights_only=True, map_location="cpu")
    assert loaded["seed"] == restricted["seed"] == 42
    assert all(t.device.type == "cpu" for t in loaded["model_state_dict"].values())
    first = network.predict_sparse_expert(state, x[520:540])
    second = network.predict_sparse_expert(loaded, x[520:540])
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-12)
    with pytest.raises(network.SparseNetworkError, match="schema"):
        network.predict_sparse_expert(loaded, x[520:540], feature_names=list(reversed(names)))
    with pytest.raises(network.SparseNetworkError, match="CPU float64"):
        network.predict_sparse_expert(loaded, x[520:540], device="cuda")
    unsafe = copy.deepcopy(loaded)
    unsafe["unexpected_object"] = object()
    with pytest.raises(network.SparseNetworkError, match="unsafe"):
        network.save_sparse_expert(unsafe, tmp_path / "unsafe.pt")
    assert not (tmp_path / "unsafe.pt").exists()


def test_validation_does_not_update_bn_and_prediction_restores_cpu_rng(learned):
    x, y, names, state, detail = learned
    model = network.SparseTabNet(5)
    model.train()
    model(torch.randn(128, 5))
    before = {name: value.clone() for name, value in model.named_buffers()}
    network._validation_bce(model, torch.randn(73, 5) + 100, torch.zeros(73), 16)
    assert all(torch.equal(before[name], value) for name, value in model.named_buffers())
    rng = torch.random.get_rng_state().clone()
    network.predict_sparse_expert(state, x[512:521], batch_size=1)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_fixed_seed_repeated_training_reproduces_saved_state():
    x, y, names = synthetic(180)
    cfg = {"max_epochs": 2, "batch_size": 64, "threads": 1}
    one, _ = network.fit_sparse_expert(x[:128], y[:128], x[128:], y[128:], 456, names, device="cpu", config=cfg)
    two, _ = network.fit_sparse_expert(x[:128], y[:128], x[128:], y[128:], 456, names, device="cpu", config=cfg)
    assert one["best_epoch"] == two["best_epoch"]
    assert all(torch.equal(tensor, two["model_state_dict"][name]) for name, tensor in one["model_state_dict"].items())


@pytest.mark.parametrize("bad", ["nan", "wrong_width", "labels", "duplicate_names"])
def test_bad_inputs_fail_closed(bad):
    x, y, names = synthetic(40)
    train, val = x[:30].copy(), x[30:].copy()
    if bad == "nan":
        train[0, 0] = np.nan
    elif bad == "wrong_width":
        val = val[:, :-1]
    elif bad == "labels":
        y[0] = 0.5
    else:
        names[1] = names[0]
    with pytest.raises(network.SparseNetworkError):
        network.fit_sparse_expert(train, y[:30], val, y[30:], 42, names, device="cpu", config={"max_epochs": 1})


def test_early_stopping_equal_bce_keeps_first_epoch(monkeypatch):
    x, y, names = synthetic(80)
    monkeypatch.setattr(network, "_validation_bce", lambda *args: 0.7)
    state, detail = network.fit_sparse_expert(x[:64], y[:64], x[64:], y[64:], 42, names, device="cpu",
        config={"max_epochs": 10, "patience": 2, "batch_size": 32, "threads": 1})
    assert state["best_epoch"] == 1 and detail["epochs_trained"] == 3
    assert detail["best_validation_bce"] == 0.7

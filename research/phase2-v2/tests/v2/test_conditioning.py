from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _torch():
    return pytest.importorskip("torch")


def _model(conditioning: str, **overrides):
    from models.model_conditioning import AWNConditioned

    arguments = dict(
        num_classes=3,
        num_levels=1,
        in_channels=4,
        kernel_size=3,
        latent_dim=6,
        regu_details=0.01,
        regu_approx=0.01,
        conditioning=conditioning,
        num_snr_bins=4,
        snr_embedding_dim=2,
        snr_min_db=-20.0,
        snr_max_db=18.0,
    )
    arguments.update(overrides)
    return AWNConditioned(**arguments)


def _condition_inputs(conditioning: str, batch_size: int = 2):
    torch = _torch()
    if conditioning in {"M1", "M5", "M6"}:
        return {"snr_db": torch.tensor([-20.0, 18.0])[:batch_size]}
    if conditioning in {"M2", "M3", "M4", "M7"}:
        return {"snr_bin": torch.tensor([0, 3], dtype=torch.long)[:batch_size]}
    return {}


@pytest.mark.parametrize("conditioning", [f"M{i}" for i in range(8)])
def test_every_conditioner_smoke_returns_logits_and_regularizers(conditioning):
    torch = _torch()
    model = _model(conditioning).eval()
    inputs = torch.randn(2, 2, 32)

    with torch.no_grad():
        logits, regularizers = model(inputs, **_condition_inputs(conditioning))

    assert logits.shape == (2, 3)
    assert len(regularizers) == 1
    assert regularizers[0].ndim == 0


@pytest.mark.parametrize("conditioning", [f"M{i}" for i in range(8)])
def test_every_conditioner_exposes_pooled_features_and_inspection(conditioning):
    torch = _torch()
    model = _model(conditioning).eval()

    with torch.no_grad():
        pooled, regularizers = model.extract_pooled_features(torch.randn(2, 2, 32))

    assert pooled.shape == (2, 8)
    assert len(regularizers) == 1
    metadata = model.conditioner_metadata()
    assert metadata["id"] == conditioning
    assert metadata["pooled_feature_dim"] == 8
    assert isinstance(model.conditioner_parameters(), dict)


def test_m0_is_plain_awn_and_has_no_conditioner_parameters():
    torch = _torch()
    model = _model("M0")
    features = torch.randn(2, model.out_channels)

    assert model.apply_feature_conditioning(features) is features
    assert model.conditioner_parameters() == {}
    assert model.fc[0].in_features == model.out_channels


def test_m1_normalizes_scalar_snr_and_concatenates_at_pooled_feature_location():
    torch = _torch()
    model = _model("M1")
    encoded = model.encode_condition(snr_db=torch.tensor([-20.0, -1.0, 18.0]))
    features = torch.zeros(3, model.out_channels)

    assert torch.allclose(encoded[:, 0], torch.tensor([-1.0, 0.0, 1.0]))
    assert torch.equal(model.apply_feature_conditioning(features, snr_db=torch.tensor([-20.0, -1.0, 18.0])), torch.cat((features, encoded), dim=1))
    assert model.fc[0].in_features == model.out_channels + 1


def test_m2_uses_a_fixed_width_one_hot_snr_vector():
    torch = _torch()
    model = _model("M2")

    encoded = model.encode_condition(snr_bin=torch.tensor([0, 3]))

    assert encoded.shape == (2, 4)
    assert torch.equal(encoded, torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]))
    assert model.fc[0].in_features == model.out_channels + 4


def test_m3_uses_learned_embedding_and_documents_exact_concat_capacity():
    torch = _torch()
    from models.model_conditioning import AWNConditioned

    model = _model("M3")
    with torch.no_grad():
        model.snr_embedding.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]))

    assert torch.equal(model.encode_condition(snr_bin=torch.tensor([1, 3])), torch.tensor([[3.0, 4.0], [7.0, 8.0]]))
    assert model.fc[0].in_features == model.out_channels + 2
    assert "z-dependent additive terms" in AWNConditioned.__doc__
    assert "not arbitrary per-bin feature normals" in AWNConditioned.__doc__
    assert "num_classes" in AWNConditioned.__doc__


def test_m3_strictly_loads_phase2_awnsnr_checkpoint_layout_and_matches_output():
    torch = _torch()
    from models.model_conditioning import AWNConditioned
    from models.model_snr import AWNSNR

    arguments = dict(
        num_classes=3,
        num_levels=1,
        in_channels=4,
        kernel_size=3,
        latent_dim=6,
        regu_details=0.01,
        regu_approx=0.01,
        num_snr_bins=4,
    )
    legacy = AWNSNR(**arguments, snr_emb_dim=2).eval()
    unified = AWNConditioned(**arguments, conditioning="M3", snr_embedding_dim=2).eval()
    assert set(legacy.state_dict()) == set(unified.state_dict())
    unified.load_state_dict(legacy.state_dict(), strict=True)
    inputs = torch.randn(2, 2, 32)
    bins = torch.tensor([0, 3], dtype=torch.long)

    with torch.no_grad():
        legacy_logits, legacy_regularizers = legacy(inputs, bins)
        unified_logits, unified_regularizers = unified(inputs, snr_bin=bins)

    assert torch.equal(unified_logits, legacy_logits)
    assert all(torch.equal(new, old) for new, old in zip(unified_regularizers, legacy_regularizers))


def test_m4_adds_only_a_per_bin_class_bias_without_changing_features():
    torch = _torch()
    model = _model("M4").eval()
    features = torch.randn(2, model.out_channels)
    assert model.apply_feature_conditioning(features, snr_bin=torch.tensor([1, 3])) is features
    with torch.no_grad():
        for parameter in model.fc.parameters():
            parameter.zero_()
        model.logit_bias.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(4, 3))

    logits = model.classify_pooled(features, snr_bin=torch.tensor([1, 3]))

    assert torch.equal(logits, torch.tensor([[3.0, 4.0, 5.0], [9.0, 10.0, 11.0]]))
    metadata = model.conditioner_metadata()
    assert metadata["feature_interaction"] == "none"
    assert metadata["location"] == "class_logits"


@pytest.mark.parametrize("conditioning", ["M1", "M2", "M3", "M5", "M6"])
def test_feature_level_conditioners_report_attended_pooled_feature_location(conditioning):
    assert _model(conditioning).conditioner_metadata()["location"] == "attended_pooled_features"


def test_m5_applies_feature_dimensional_film_gamma_and_beta_from_normalized_snr():
    torch = _torch()
    model = _model("M5")
    with torch.no_grad():
        model.film.weight[: model.out_channels].fill_(0.5)
        model.film.bias[: model.out_channels].fill_(1.0)
        model.film.weight[model.out_channels :].fill_(0.25)
        model.film.bias[model.out_channels :].fill_(0.1)
    features = torch.ones(2, model.out_channels)

    actual = model.apply_feature_conditioning(features, snr_db=torch.tensor([-20.0, 18.0]))
    expected = torch.tensor([[0.35] * model.out_channels, [1.85] * model.out_channels])

    assert torch.allclose(actual, expected)
    assert model.conditioner_metadata()["modulation_dimension"] == model.out_channels


def test_m6_gate_is_twice_sigmoid_with_open_zero_two_range_and_feature_dimension():
    torch = _torch()
    model = _model("M6")
    with torch.no_grad():
        model.gate.weight.fill_(1.0)
        model.gate.bias.zero_()
    features = torch.ones(2, model.out_channels)

    actual = model.apply_feature_conditioning(features, snr_db=torch.tensor([-20.0, 18.0]))
    expected = 2.0 * torch.sigmoid(torch.tensor([[-1.0], [1.0]])).expand_as(features)

    assert torch.allclose(actual, expected)
    metadata = model.conditioner_metadata()
    assert metadata["gate_function"] == "2 * sigmoid(W z + b)"
    assert metadata["gate_range"] == "(0, 2)"
    assert metadata["modulation_dimension"] == model.out_channels


def test_m7_selects_one_complete_classifier_head_per_snr_bin():
    torch = _torch()
    model = _model("M7")
    with torch.no_grad():
        for index, head in enumerate(model.snr_heads):
            for parameter in head.parameters():
                parameter.zero_()
            head[-1].bias.fill_(float(index))
    features = torch.randn(2, model.out_channels)

    logits = model.classify_pooled(features, snr_bin=torch.tensor([1, 3]))

    assert torch.equal(logits, torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]))
    metadata = model.conditioner_metadata()
    assert metadata["shared_feature_extractor"] is True
    assert metadata["classifier_head_count"] == 4


def test_m7_calls_only_selected_heads_and_preserves_order_and_autograd():
    torch = _torch()
    model = _model("M7")
    called = []
    hooks = [head.register_forward_hook(lambda _module, _inputs, _output, index=index: called.append(index)) for index, head in enumerate(model.snr_heads)]
    with torch.no_grad():
        for index, head in enumerate(model.snr_heads):
            for parameter in head.parameters():
                parameter.zero_()
            head[-1].bias.fill_(float(index))
    features = torch.randn(4, model.out_channels, requires_grad=True)

    logits = model.classify_pooled(features, snr_bin=torch.tensor([3, 1, 3, 1]))
    logits.sum().backward()
    for hook in hooks:
        hook.remove()

    assert called == [1, 3]
    assert torch.equal(logits.detach()[:, 0], torch.tensor([3.0, 1.0, 3.0, 1.0]))
    assert features.grad is not None
    assert all(parameter.grad is not None for index in (1, 3) for parameter in model.snr_heads[index].parameters())
    assert all(parameter.grad is None for index in (0, 2) for parameter in model.snr_heads[index].parameters())


@pytest.mark.parametrize("conditioning", ["M1", "M2", "M3", "M4", "M5", "M6", "M7"])
def test_conditioned_models_require_their_declared_snr_input(conditioning):
    torch = _torch()
    model = _model(conditioning)
    with pytest.raises(ValueError, match="requires"):
        model(torch.randn(2, 2, 32))


def test_discrete_conditioners_reject_out_of_range_or_nonintegral_bins():
    torch = _torch()
    for conditioning in ("M2", "M3", "M4", "M7"):
        model = _model(conditioning)
        with pytest.raises((TypeError, ValueError), match="snr_bin"):
            model.encode_condition(snr_bin=torch.tensor([0.0, 1.0]))
        with pytest.raises(ValueError, match="snr_bin"):
            model.encode_condition(snr_bin=torch.tensor([0, 4]))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_classes", True, "num_classes"),
        ("num_classes", 3.5, "num_classes"),
        ("num_classes", 0, "num_classes"),
        ("num_levels", False, "num_levels"),
        ("num_levels", 1.0, "num_levels"),
        ("num_levels", 0, "num_levels"),
        ("in_channels", 4.5, "in_channels"),
        ("in_channels", 0, "in_channels"),
        ("kernel_size", 2, "kernel_size"),
        ("kernel_size", 0, "kernel_size"),
        ("latent_dim", True, "latent_dim"),
        ("latent_dim", 0, "latent_dim"),
        ("num_snr_bins", 4.5, "num_snr_bins"),
        ("num_snr_bins", 1, "num_snr_bins"),
        ("snr_embedding_dim", False, "snr_embedding_dim"),
        ("snr_embedding_dim", 0, "snr_embedding_dim"),
    ],
)
def test_constructor_rejects_invalid_integer_count_arguments(field, value, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _model("M0", **{field: value})


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [
        (float("nan"), 18.0),
        (-20.0, float("nan")),
        (float("-inf"), 18.0),
        (-20.0, float("inf")),
        (-20.0, -20.0),
        (19.0, 18.0),
        (False, 18.0),
    ],
)
def test_constructor_rejects_nonfinite_or_unordered_snr_bounds(minimum, maximum):
    with pytest.raises((TypeError, ValueError), match="snr_(min|max)_db|SNR bounds"):
        _model("M0", snr_min_db=minimum, snr_max_db=maximum)


def test_runtime_condition_validation_covers_dtype_shape_batch_range_and_device():
    torch = _torch()
    scalar = _model("M1")
    discrete = _model("M3")
    features = torch.randn(2, scalar.out_channels)
    with pytest.raises(TypeError, match="floating"):
        scalar.normalize_snr(torch.tensor([-20, 18], dtype=torch.long))
    with pytest.raises(ValueError, match="one-dimensional"):
        scalar.normalize_snr(torch.tensor([[-20.0], [18.0]]))
    with pytest.raises(ValueError, match="finite"):
        scalar.normalize_snr(torch.tensor([-20.0, float("nan")]))
    with pytest.raises(ValueError, match="range"):
        scalar.normalize_snr(torch.tensor([-22.0, 18.0]))
    with pytest.raises(ValueError, match="batch size"):
        scalar.apply_feature_conditioning(features, snr_db=torch.tensor([-20.0]))
    with pytest.raises(ValueError, match="batch size"):
        discrete.apply_feature_conditioning(features, snr_bin=torch.tensor([0]))
    with pytest.raises(ValueError, match="device"):
        scalar.apply_feature_conditioning(features, snr_db=torch.empty(2, device="meta"))
    with pytest.raises(ValueError, match="device"):
        discrete.apply_feature_conditioning(features, snr_bin=torch.empty(2, dtype=torch.long, device="meta"))


@pytest.mark.parametrize("conditioning", ["M5", "M6"])
def test_feature_modulators_reject_nontensor_snr_before_device_access(conditioning):
    torch = _torch()
    model = _model(conditioning)
    features = torch.randn(2, model.out_channels)
    for invalid_snr in ([-20.0, 18.0], -20.0):
        with pytest.raises(TypeError, match="snr_db must be a torch.Tensor"):
            model.apply_feature_conditioning(features, snr_db=invalid_snr)


@pytest.mark.parametrize("conditioning", [f"M{i}" for i in range(8)])
def test_forward_batch_is_one_runner_contract_for_true_snr_and_bins(conditioning):
    torch = _torch()
    model = _model(conditioning).eval()
    signals = torch.randn(2, 2, 32)
    snr_db = torch.tensor([-20.0, 18.0])
    snr_bin = torch.tensor([0, 3], dtype=torch.long)

    with torch.no_grad():
        logits, regularizers = model.forward_batch(signals, snr_db=snr_db, snr_bin=snr_bin)

    assert logits.shape == (2, 3)
    assert len(regularizers) == 1


def test_forward_batch_preserves_phase2_positional_calls_for_m0_and_m3():
    torch = _torch()
    signals = torch.randn(2, 2, 32)
    snr_db = torch.tensor([-20.0, 18.0])
    snr_bin = torch.tensor([0, 3], dtype=torch.long)
    m0 = _model("M0").eval()
    m3 = _model("M3").eval()
    with torch.no_grad():
        assert torch.equal(m0(signals)[0], m0.forward_batch(signals, snr_db=snr_db, snr_bin=snr_bin)[0])
        assert torch.equal(m3(signals, snr_bin)[0], m3.forward_batch(signals, snr_db=snr_db, snr_bin=snr_bin)[0])


def test_forward_batch_validates_both_runner_snr_tensors_before_backbone_execution():
    torch = _torch()
    model = _model("M0")
    called = []
    hook = model.conv1.register_forward_hook(lambda *_args: called.append(True))
    with pytest.raises(ValueError, match="batch size"):
        model.forward_batch(torch.randn(2, 2, 32), snr_db=torch.tensor([-20.0]), snr_bin=torch.tensor([0, 1]))
    with pytest.raises(TypeError, match="snr_bin"):
        model.forward_batch(torch.randn(2, 2, 32), snr_db=torch.tensor([-20.0, 18.0]), snr_bin=torch.tensor([0.0, 1.0]))
    hook.remove()
    assert called == []


def test_parameter_counts_are_exact_relative_to_equivalent_m0_capacity():
    baseline = _model("M0")
    baseline_total = sum(parameter.numel() for parameter in baseline.parameters())
    baseline_head = sum(parameter.numel() for parameter in baseline.fc.parameters())
    for conditioning in [f"M{i}" for i in range(8)]:
        model = _model(conditioning)
        counts = model.parameter_counts()
        actual_total = sum(parameter.numel() for parameter in model.parameters())
        assert counts == {
            "total": actual_total,
            "m0_equivalent": baseline_total,
            "additional_conditioner": actual_total - baseline_total,
        }
        assert model.conditioner_metadata()["parameter_counts"] == counts
    m7 = _model("M7")
    assert m7.parameter_counts()["additional_conditioner"] == (m7.num_snr_bins - 1) * baseline_head


def test_metadata_binds_snr_bounds_and_uses_config_canonical_conditioner_names():
    config = yaml.safe_load((ROOT / "configs/v2/phase3.yaml").read_text(encoding="utf-8"))
    for conditioning in [f"M{i}" for i in range(8)]:
        metadata = _model(conditioning).conditioner_metadata()
        assert metadata["snr_min_db"] == -20.0
        assert metadata["snr_max_db"] == 18.0
        assert metadata["conditioner"] == config["models"][conditioning]["conditioner"]
    assert config["snr"]["snr_min_db"] == -20
    assert config["snr"]["snr_max_db"] == 18
    assert config["runner_call"] == {
        "api": "forward_batch",
        "arguments": ["signals", "snr_db", "snr_bin"],
    }


def test_phase3_config_freezes_phase2_protocol_and_defines_m0_through_m7():
    phase2 = yaml.safe_load((ROOT / "configs/v2/phase2.yaml").read_text(encoding="utf-8"))
    phase3 = yaml.safe_load((ROOT / "configs/v2/phase3.yaml").read_text(encoding="utf-8"))
    for field in ("dataset", "split", "seeds", "device", "training", "architecture", "evaluation_batch_size"):
        assert phase3[field] == phase2[field]
    assert phase3["split"]["hash"] == "42450053b13189fdd4ca1ab859e26a5ff61c93b8cdb26bff48c76fde7b1f54a9"
    assert phase3["seeds"] == [2022, 2023, 2024, 2025, 2026]
    assert phase3["device"] == "cpu"
    assert phase3["preprocessing"] == {
        "input_layout": "NCT (I/Q channels, time)",
        "signal_dtype": "float32",
        "transforms": [],
    }
    assert set(phase3["models"]) == {f"M{i}" for i in range(8)}
    assert phase3["snr_bands"] == {
        "low": {"max_db": -8},
        "mid": {"min_db": -6, "max_db": -2},
        "high": {"min_db": 0},
    }
    assert phase3["models"]["M4"]["feature_interaction"] == "none"
    assert phase3["models"]["M6"] == {
        "conditioner": "feature_gate",
        "snr_representation": "normalized_scalar",
        "function": "2 * sigmoid(W z + b)",
        "range": "(0, 2)",
        "dimension": 128,
    }
    assert phase3["models"]["M7"]["head_count"] == 20

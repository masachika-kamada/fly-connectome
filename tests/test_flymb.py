"""Properties the experiments silently rely on.

Each randomisation control claims to preserve some structure and destroy the
rest; if one quietly preserved less than it says, every comparison built on it
would be wrong without any number looking odd. These run on small synthetic
matrices, so they need no downloaded data.
"""
import numpy as np
import pytest

from flymb import capacity, coding, odors, paths, wiring


@pytest.fixture
def W():
    rng = np.random.default_rng(0)
    w = (rng.random((30, 200)) < 0.08) * rng.integers(1, 20, size=(30, 200))
    return w.astype(np.float32)


def row_weights(W):
    return [sorted(r[r > 0].tolist()) for r in W]


def test_degree_swap_keeps_both_degrees_and_each_rows_weights(W):
    out = wiring.degree_preserving_swap(W, np.random.default_rng(1))
    assert ((out > 0).sum(axis=0) == (W > 0).sum(axis=0)).all()
    assert ((out > 0).sum(axis=1) == (W > 0).sum(axis=1)).all()
    assert row_weights(out) == row_weights(W)
    assert ((out > 0) != (W > 0)).any(), "nothing was rewired"


def test_pn_degree_kept_keeps_rows_only(W):
    out = wiring.pn_degree_kept(W, np.random.default_rng(1))
    assert row_weights(out) == row_weights(W)
    assert ((out > 0).sum(axis=0) != (W > 0).sum(axis=0)).any()


def test_kc_degree_kept_keeps_columns_only(W):
    out = wiring.kc_degree_kept(W, np.random.default_rng(1))
    assert ((out > 0).sum(axis=0) == (W > 0).sum(axis=0)).all()
    assert sorted(out[out > 0]) == sorted(W[W > 0])
    assert ((out > 0).sum(axis=1) != (W > 0).sum(axis=1)).any()


def test_uniform_keeps_edge_count_and_weight_pool(W):
    out = wiring.uniform_random(W, np.random.default_rng(1))
    assert sorted(out[out > 0]) == sorted(W[W > 0])


def test_hash_codes_keep_exactly_the_sparsity(W):
    x = np.random.default_rng(2).random((7, 30)).astype(np.float32)
    codes = coding.hash_codes(x, W)
    assert (codes.sum(axis=1) == round(coding.SPARSITY * 200)).all()


def test_auc_matches_pairwise_definition():
    rng = np.random.default_rng(3)
    trained = rng.integers(0, 6, size=40).astype(float)   # heavy ties on purpose
    novel = rng.integers(2, 9, size=55).astype(float)
    pairs = trained[:, None] - novel[None, :]
    expected = ((pairs < 0) + 0.5 * (pairs == 0)).mean()
    assert coding.auc(trained, novel) == pytest.approx(expected)


def test_depress_only_touches_active_cells():
    w = np.arange(1.0, 6.0)
    out = coding.depress(w, np.array([True, False, True, False, False]))
    assert out.tolist() == pytest.approx([0.15, 2.0, 0.45, 4.0, 5.0])
    assert w.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0], "input must not be modified"


META_PN = [{"bodyId": i, "type": t} for i, t in enumerate(
    ["DA1_lPN", "DA1_lPN", "DM1_lPN", "VP2_adPN", "M_vPNml50", "M_vPNml51"])]


def test_channels_group_glomeruli_and_split_multiglomerular():
    chan, n = odors.channel_map(META_PN)
    assert n == 5
    assert chan[0] == chan[1] and len(set(chan[2:])) == 4


def test_olfactory_channels_drop_multiglomerular_and_thermo_hygro():
    chan, n = odors.channel_map(META_PN)
    mask = odors.olfactory_channels(META_PN, chan, n)
    assert [bool(mask[c]) for c in chan] == [True, True, True, False, False, False]


def test_channel_mask_silences_without_changing_the_draws():
    chan, n = odors.channel_map(META_PN)
    mask = odors.olfactory_channels(META_PN, chan, n)
    full = odors.synthetic_odors(50, chan, n, np.random.default_rng(4))
    part = odors.synthetic_odors(50, chan, n, np.random.default_rng(4), channels=mask)
    silenced = ~mask[chan]
    assert (part[:, silenced] == 0).all()
    # Same draws underneath: the surviving channels differ only by normalisation.
    ratio = part[:, ~silenced] / np.where(full[:, ~silenced] > 0, full[:, ~silenced], np.nan)
    rows = ~np.isnan(ratio).all(axis=1)
    assert np.allclose(np.nanmin(ratio[rows], axis=1), np.nanmax(ratio[rows], axis=1), rtol=1e-5)


@pytest.mark.parametrize("instance, label", [
    ("MBON11(y1pedc>a/B)_R", "MBON11 γ1pedc>α/β"),
    ("MBON02(B2B'2a)_L", "MBON02 β2β'2a"),       # trailing a = anterior
    ("MBON12(y2a'1)_R", "MBON12 γ2α'1"),
    ("MBON16(a'3ap)_R", "MBON16 α'3ap"),
    ("MBON19(a2p3p)_R", "MBON19 α2p3p"),
    ("MBON22(calyx)_R", "MBON22 calyx"),
])
def test_greek_labels(instance, label):
    assert capacity.greek(instance) == label


def test_root_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYMB_ROOT", str(tmp_path))
    assert paths.find_root() == str(tmp_path)


def test_root_is_found_from_a_subdirectory(monkeypatch, tmp_path):
    monkeypatch.delenv("FLYMB_ROOT", raising=False)
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "a" / "b").mkdir(parents=True)
    assert paths.find_root(tmp_path / "a" / "b") == str(tmp_path)

import pytest

from scripts import recover_argon4_prepared_revision as cli


def test_recovery_command_defaults_to_readonly_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli.sys, "argv", ["recover_argon4_prepared_revision.py"])
    monkeypatch.setattr(cli.socket, "create_connection", lambda *a, **k: pytest.fail("readonly connected"))
    calls = []
    def check(runtime, **kwargs):
        calls.append(kwargs)
        assert kwargs["apply"] is False and kwargs["cutover_now"] is None
        return {"status": "VERIFIED_PREPARED_CUTOVER"}
    monkeypatch.setattr(cli, "recover_prepared_price_revision", check)
    assert cli.main() == 0 and len(calls) == 1
    assert not (tmp_path / "paper_runtime").exists()


def test_unapplied_command_is_separate_readonly_verification_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli.sys, "argv", ["recover_argon4_prepared_revision.py", "--abort-unapplied"])
    monkeypatch.setattr(cli.socket, "socket", lambda *a, **k: pytest.fail("readonly reserved network"))
    monkeypatch.setattr(cli, "recover_prepared_price_revision", lambda *a, **k: pytest.fail("wrong recovery mode"))
    calls = []
    def check(runtime, **kwargs):
        calls.append(kwargs)
        assert kwargs == {"apply": False, "abort_now": None}
        return {"status": "VERIFIED_UNAPPLIED_REVISION"}
    monkeypatch.setattr(cli, "abort_unapplied_price_revision", check)
    assert cli.main() == 0 and len(calls) == 1
    assert not (tmp_path / "paper_runtime").exists()


def test_unapplied_apply_requires_pause_before_touching_network_or_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli.sys, "argv", ["recover_argon4_prepared_revision.py", "--abort-unapplied", "--apply"])
    monkeypatch.setattr(cli.socket, "socket", lambda *a, **k: pytest.fail("missing pause reserved network"))
    monkeypatch.setattr(cli, "abort_unapplied_price_revision", lambda *a, **k: pytest.fail("missing pause aborted"))
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 2


def test_apply_requires_pause_before_touching_network_or_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli.sys, "argv", ["recover_argon4_prepared_revision.py", "--apply"])
    monkeypatch.setattr(cli.socket, "create_connection", lambda *a, **k: pytest.fail("missing pause connected"))
    monkeypatch.setattr(cli, "recover_prepared_price_revision", lambda *a, **k: pytest.fail("missing pause recovered"))
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 2


def test_apply_refuses_a_running_local_monitor(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli.sys, "argv", ["recover_argon4_prepared_revision.py", "--apply"])
    runtime = tmp_path / "paper_runtime/argon4_forward_20261105"
    runtime.mkdir(parents=True)
    (runtime / "verification_pause.json").write_text("{}")
    def occupied(*args):
        raise OSError("address already in use")
    monkeypatch.setattr(cli.socket, "socket", lambda *a, **k: SimpleNamespace(
        bind=occupied, setsockopt=lambda *a: None, close=lambda: None))
    monkeypatch.setattr(cli, "recover_prepared_price_revision", lambda *a, **k: pytest.fail("live writer recovered"))
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 2

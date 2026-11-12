"""Late diagnostics cannot change the current financial intent or its ordering."""
import importlib
from tests.test_infra121_idempotency import _paper_fixture


def _two_cycles(monkeypatch):
    paper, cfg = _paper_fixture(monkeypatch)
    first = paper.run_paper_cycle(cfg)['cycle_id']
    second = paper.run_paper_cycle(cfg)['cycle_id']
    return paper, cfg, importlib.import_module('core.commit_protocol'), first, second


def test_late_diagnostic_cannot_reopen_old_intent_after_newer_confirmation(isolated_storage, monkeypatch):
    paper, cfg, q, first, second = _two_cycles(monkeypatch)
    before = dict(q._gate())
    result = q.post_commit_failure(first, confirmed=True)
    assert dict(q._gate()) == before
    assert not q.gate_status()['blocked']
    assert q._gate()['last_cycle_id'] == second
    assert result['stale_diagnostic'] is True
    assert len(paper.read_paper_trades()) == 1


def test_old_reconciliation_and_diagnostic_never_release_new_pending_intent(isolated_storage, monkeypatch):
    paper, cfg, q, first, second = _two_cycles(monkeypatch)
    pending = q.begin_intent(paper.load_paper_state(cfg), [], [])
    before = dict(q._gate())
    q.post_commit_failure(first, confirmed=True)
    result = q.reconcile(first)
    assert result['outcome'] == 'UNKNOWN_COMMIT' and result['blocked']
    assert dict(q._gate()) == before and q.gate_status()['cycle_id'] == pending
    assert len(paper.read_paper_trades()) == 1


def test_historical_receipt_is_not_reconciled_as_current_intent(isolated_storage, monkeypatch):
    paper, cfg, q, first, second = _two_cycles(monkeypatch)
    before = dict(q._gate())
    result = q.reconcile(first)
    assert result['historical'] is True and result['current_cycle_id'] == second
    assert result['evidence'] == 'PREVIOUSLY_VERIFIED_RECEIPT_NOT_CURRENT_STATE'
    assert dict(q._gate()) == before and len(paper.read_paper_trades()) == 1


import pytest
from tests.infra123_tcp_proxy import is_commit, LocalPostgresProxy


@pytest.mark.parametrize('kind,payload,expected', [
    (b'Q', b'COMMIT\0', True), (b'Q', b' commit;\0', True),
    (b'P', b'\0COMMIT\0\0\0', True), (b'Q', b'ROLLBACK\0', False),
    (b'Q', b'SELECT 1\0', False), (b'P', b'\0SELECT 1\0', False),
    (b'Q', b'COMMIT PREPARED x\0', False), (b'p', b'opaque-auth-bytes', False),
])
def test_proxy_parser_recognizes_only_commit_frames(kind, payload, expected):
    assert is_commit(kind, payload) is expected


@pytest.mark.parametrize('host', ['0.0.0.0', 'localhost', 'remote.invalid'])
def test_proxy_rejects_non_loopback_target_without_connecting(host):
    with pytest.raises(ValueError):
        LocalPostgresProxy(host, 5432)
    from scripts.test_infra12_postgres import _loopback_only
    with pytest.raises(RuntimeError):
        _loopback_only('socket.connect', (None, (host, 5432)))


def test_historical_ui_receipt_is_not_a_current_financial_success(isolated_storage, monkeypatch):
    from tests.conftest import load_module
    from tests.test_ui_mutations import UI
    helper = load_module('core.ui_mutations')
    q = importlib.import_module('core.commit_protocol')
    ui = UI()
    ui.button = lambda *args, **kwargs: True
    monkeypatch.setattr(q,'gate_status',lambda: {'blocked':True,'cycle_id':'old','reason':''})
    monkeypatch.setattr(q,'reconcile',lambda *_: {'outcome':'COMMITTED','historical':True,
                                               'current_cycle_id':'new','resolution_recorded':True})
    helper.render_commit_control(ui=ui,allow_reconcile=True)
    assert any('histórico' in message for _,message in ui.messages)
    assert not any(kind=='success' for kind,_ in ui.messages)


@pytest.mark.parametrize('dsn',[
    'host=remote.invalid port=5432 dbname=atlas_infra123_fixture user=atlas_fixture',
    'host=127.0.0.1 port=5432 dbname=real_database user=atlas_fixture',
    'host=127.0.0.1 port=5432 dbname=atlas_infra123_fixture user=real_user',
])
def test_backup_refuses_non_synthetic_dsn_before_connecting(dsn):
    from scripts.local_synthetic_backup import local_params
    with pytest.raises(ValueError):
        local_params(dsn)

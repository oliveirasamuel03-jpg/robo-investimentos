"""Durable fencing and feedback without claiming local files prove PG COMMIT."""
import importlib

import pytest
from tests.conftest import load_module
from tests.test_infra121_idempotency import _paper_fixture
from tests.test_ui_mutations import UI


def _pending(monkeypatch):
    paper, cfg = _paper_fixture(monkeypatch)
    store = importlib.import_module('core.state_store')
    store.load_bot_state()
    state = paper.load_paper_state(cfg)
    protocol = importlib.import_module('core.commit_protocol')
    return protocol, store, paper, protocol.begin_intent(state, [], [])


def test_unknown_intent_survives_reload_and_heartbeat(isolated_storage, monkeypatch):
    q, store, paper, cycle_id = _pending(monkeypatch)
    store.update_worker_heartbeat('online')
    q = importlib.reload(q)
    assert q.gate_status()['cycle_id'] == cycle_id
    with pytest.raises(q.ReconciliationRequired):
        paper.run_paper_cycle()
    assert paper.read_paper_trades() == []


@pytest.mark.parametrize('action', ['start', 'reset', 'paper_reset', 'raw_gate', 'raw_finance'])
def test_common_actions_cannot_remove_pending_fence(isolated_storage, monkeypatch, action):
    q, store, paper, cycle_id = _pending(monkeypatch)
    p = importlib.import_module('core.persistence')
    stop = importlib.import_module('core.stop_control')
    snapshot = store.load_bot_state()
    if action == 'start':
        operation = lambda: stop.start_robot(snapshot)
    elif action == 'reset':
        operation = store.reset_state
    elif action == 'paper_reset':
        operation = paper.reset_paper_state
    elif action == 'raw_finance':
        operation = lambda: p.replace_json_rows('paper_trades', [], paper.PAPER_TRADES_FILE)
    else:
        operation = lambda: p.update_json_state(q.GATE, dict, q._GATE_PATH, lambda _: {'active_cycle_id': None})
    with pytest.raises(p.PersistenceConflictError):
        operation()
    assert q.gate_status()['cycle_id'] == cycle_id


def test_stop_is_separate_and_available_during_unknown(isolated_storage, monkeypatch):
    q, store, _, cycle_id = _pending(monkeypatch)
    stop = importlib.import_module('core.stop_control')
    receipt = stop.request_stop('synthetic-operator')
    assert receipt.persisted and not receipt.acknowledged
    assert stop.acknowledge_stop('synthetic-worker').acknowledged
    assert q.gate_status()['cycle_id'] == cycle_id
    assert store.load_bot_state()['bot_status'] == 'STOPPED'


def test_files_do_not_claim_postgres_rollback_or_unlock_on_timeout(isolated_storage, monkeypatch):
    q, _, _, cycle_id = _pending(monkeypatch)
    result = q.reconcile(cycle_id)
    assert result['outcome'] == 'UNKNOWN_COMMIT' and result['blocked']
    assert q.gate_status()['blocked']


@pytest.mark.parametrize('outcome', ['UNKNOWN_COMMIT', 'POST_COMMIT_DIAGNOSTIC_FAILURE', 'LOCAL_ONLY', 'NOT_COMMITTED'])
def test_financial_feedback_has_priority_over_legacy_boolean(isolated_storage, outcome):
    helper = load_module('core.ui_mutations')
    assert helper.classify_cycle_feedback({'committed': True, 'commit_outcome': outcome}) == outcome


def test_unknown_ui_never_reports_not_committed_or_retries(isolated_storage):
    helper = load_module('core.ui_mutations')
    q = importlib.import_module('core.commit_protocol')
    ui, attempts = UI(), []
    def fail():
        attempts.append(True)
        raise q.UnknownCommit('synthetic-cycle')
    result = helper.run_ui_mutation(fail, ui=ui, action='Ciclo PAPER')
    assert not result.confirmed and result.error_code == 'UNKNOWN_COMMIT'
    assert len(attempts) == 1
    assert any('UNKNOWN_COMMIT' in message for _, message in ui.messages)
    assert not any(kind == 'success' for kind, _ in ui.messages)


def test_read_failure_cannot_open_gate_or_reconcile(isolated_storage, monkeypatch):
    q, _, _, _ = _pending(monkeypatch)
    def fail():
        raise OSError('synthetic unavailable')
    monkeypatch.setattr(q, '_gate', fail)
    with pytest.raises(OSError):
        q.assert_execution_allowed()
    assert q.reconcile()['outcome'] == 'UNKNOWN_COMMIT'


def test_absence_of_selected_intent_is_not_proof_of_rollback(isolated_storage):
    q = importlib.import_module('core.commit_protocol')
    assert q.reconcile()['outcome'] == 'UNKNOWN_COMMIT'


def test_production_cannot_switch_to_unverified_local_files(isolated_storage, monkeypatch):
    q = importlib.import_module('core.commit_protocol')
    monkeypatch.setenv('APP_ENV', 'production')
    with pytest.raises(q.ReconciliationRequired):
        q.assert_execution_allowed()


@pytest.mark.parametrize('page', ['pages/1_Trader.py', 'pages/4_Controle_do_Bot.py'])
def test_actual_pages_show_unknown_separate_from_operator_stop(isolated_storage, monkeypatch, page):
    from datetime import datetime, timezone
    from streamlit.testing.v1 import AppTest
    from pathlib import Path
    monkeypatch.setenv('AUTH_REQUIRED', 'true')
    q, store, _, cycle_id = _pending(monkeypatch)
    users = load_module('core.auth.users_store')
    user = users.create_user('fixture-unknown-admin', 'synthetic-password', 'admin')
    root = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(root/page), default_timeout=30)
    at.session_state['auth_user'] = {**user, 'auth_mode': 'session'}
    at.session_state['auth_last_activity'] = datetime.now(timezone.utc).isoformat()
    at.run()
    assert not at.exception
    assert any('UNKNOWN_COMMIT' in message.value for message in at.error)
    next(b for b in at.button if b.label == 'Verificar reconciliação (sem operar)').click().run()
    assert not at.exception and q.gate_status()['blocked']
    assert store.load_bot_state()['bot_status'] != 'STOPPED'
    next(b for b in at.button if b.label == '⏹ Parar robô').click().run()
    assert not at.exception and store.load_bot_state()['bot_status'] == 'STOPPED'
    assert q.gate_status()['blocked']


def test_post_commit_label_requires_positive_evidence(isolated_storage, monkeypatch):
    q, _, _, cycle_id = _pending(monkeypatch)
    result = q.post_commit_failure(cycle_id)
    assert result['outcome'] == 'UNKNOWN_COMMIT' and result['committed'] is False
    assert q.gate_status()['blocked']

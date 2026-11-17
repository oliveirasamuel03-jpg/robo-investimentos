"""Durable PAPER intent fence and evidence-based reconciliation.

No financial retry is performed here. Receipts contain hashes/identities, not a
second ledger. PostgreSQL receipts commit atomically with the financial unit;
the pre-existing gate stays closed until a separately verified resolution.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from uuid import uuid4, UUID

import core.persistence as p
from core.config import RUNTIME_DIR, BOT_STATE_FILE, TRADER_ORDERS_FILE, TRADER_REPORTS_FILE

GATE = 'paper_commit_control'
QUARANTINE = 'paper_integrity_quarantine'
_QUARANTINE_PATH = RUNTIME_DIR / 'paper_integrity_quarantine.json'
RECEIPT_PREFIX = 'paper_cycle_receipt:'
_GATE_PATH = RUNTIME_DIR / 'paper_commit_control.json'
_PROTOCOL_WRITE = ContextVar('atlas_commit_protocol_write', default=False)
FINANCIAL_ROWS = {'paper_trades', 'table:trader_orders.csv', 'table:trade_reports.csv'}


class ReconciliationRequired(p.PersistenceConflictError):
    outcome = 'UNKNOWN_COMMIT'
    def __init__(self, cycle_id='', reason='RECONCILIATION_REQUIRED'):
        self.cycle_id, self.reason = cycle_id, reason
        super().__init__('Resultado financeiro desconhecido; reconciliação necessária. Nenhum retry financeiro autorizado.')


class UnknownCommit(ReconciliationRequired):
    pass


class PostCommitDiagnosticFailure(RuntimeError):
    outcome = 'POST_COMMIT_DIAGNOSTIC_FAILURE'
    committed = True
    def __init__(self, cycle_id):
        self.cycle_id = cycle_id
        super().__init__('COMMIT confirmado; verificação ou diagnóstico posterior falhou. Não repetir a operação financeira.')


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def _receipt_path(cycle_id):
    if not isinstance(cycle_id, str) or UUID(cycle_id).hex != cycle_id:
        raise ValueError('Invalid cycle identity')
    return RUNTIME_DIR / f'paper_cycle_receipt_{cycle_id}.json'


@contextmanager
def protocol_write():
    token = _PROTOCOL_WRITE.set(True)
    try:
        yield
    finally:
        _PROTOCOL_WRITE.reset(token)


@contextmanager
def _scope():
    with p.persistence_transaction(GATE):
        with (nullcontext() if p.database_enabled() else p._local_lock(_GATE_PATH)):
            yield


def _gate():
    return p.load_json_state(GATE, lambda: {'active_cycle_id': None}, _GATE_PATH)


def _quarantine():
    value = p.load_json_state(QUARANTINE, lambda: {'active': False, 'evidence': []}, _QUARANTINE_PATH)
    if (type(value.get('active')) is not bool or not isinstance(value.get('evidence'), list)
            or any(not isinstance(item, dict) or not isinstance(item.get('reason'), str)
                   for item in value['evidence'])):
        raise p.PersistenceDataError('Invalid integrity quarantine; execution remains blocked.')
    if not isinstance(value.get('history', []), list):
        raise p.PersistenceDataError('Invalid quarantine history; execution remains blocked.')
    if value['active']:
        if not value['evidence']:
            raise p.PersistenceDataError('Missing quarantine evidence; execution remains blocked.')
        identity = value.get('quarantine_id')
        if not isinstance(identity, str) or UUID(identity).hex != identity:
            raise p.PersistenceDataError('Invalid quarantine identity; execution remains blocked.')
    return value


def _quarantine_locked(reason, *, cycle_id=None, evidence=None):
    """Stage containment in the caller's fenced transaction, never alter intent.

    No success/commit claim is made here. An outer rollback also rolls this back;
    every new financial intent independently checks the last verified preimage.
    """
    current = _quarantine()
    item = {'reason': reason, 'cycle_id': cycle_id, 'source': evidence or {}}
    if not current.get('active'):
        if current.get('quarantine_id'):
            current.setdefault('history', []).append({key: deepcopy(value) for key, value in current.items()
                                                     if key != 'history'})
        current.pop('resolved_at', None)
        current.pop('resolution', None)
        current.update(active=True, quarantine_id=uuid4().hex, created_at=_now(),
                       cycle_id=cycle_id, evidence=[])
    if item not in current['evidence']:
        current['evidence'].append(item)
        with protocol_write():
            p.save_json_state(QUARANTINE, current, _QUARANTINE_PATH)
    return {'quarantine_id': current['quarantine_id'], 'blocked': True,
            'quarantine_persistence': 'STAGED_IN_CALLER_TRANSACTION'}


def quarantine_integrity(reason, *, cycle_id=None, evidence=None):
    """Persist then independently read back containment; never repair history."""
    nested = p._ACTIVE_CONNECTION.get() is not None
    try:
        with _scope():
            result = _quarantine_locked(reason, cycle_id=cycle_id, evidence=evidence)
        if nested:
            return result
        observed = _quarantine()
        confirmed = observed.get('active') and observed.get('quarantine_id') == result['quarantine_id']
        return {**result, 'quarantine_persistence': 'CONFIRMED' if confirmed else 'UNCONFIRMED'}
    except Exception:
        return {'blocked': True, 'quarantine_persistence': 'UNCONFIRMED'}


def _validate_committed_receipt(receipt, cycle_id, current, observed_rows):
    from paper_trading_engine import PAPER_STATE_FILE
    actual = _state('paper_state', PAPER_STATE_FILE)
    if actual.get('last_cycle_id') != cycle_id:
        raise ReconciliationRequired(cycle_id, 'MISSING_FINANCIAL_IDENTITY')
    if current['paper'] != receipt.get('expected_paper'):
        raise ReconciliationRequired(cycle_id, 'UNEXPECTED_FINANCIAL_STATE')
    if p.database_enabled():
        from paper_trading_engine import PAPER_TRADES_FILE
        actual_trades = observed_rows['paper_trades']
        actual_reports = observed_rows['table:trade_reports.csv']
        for rows, expected in ((actual_trades, receipt['planned_trades']),
                               (actual_reports, receipt['planned_reports'])):
            observed = [_hash({k: v for k, v in row.items() if k != 'cycle_id'})
                        for row in rows if row.get('cycle_id') == cycle_id]
            if observed != expected:
                raise ReconciliationRequired(cycle_id, 'MISSING_OR_DUPLICATE_EVENTS')
        cycle_trades = [t for t in actual_trades if t.get('cycle_id') == cycle_id]
        cycle_reports = [r for r in actual_reports if r.get('cycle_id') == cycle_id]
        sells = {t['trade_id']: t for t in cycle_trades if t.get('side') == 'SELL'}
        if len(cycle_reports) != len(sells):
            raise ReconciliationRequired(cycle_id, 'CLOSED_REPORT_COUNT_DIVERGENCE')
        for report in cycle_reports:
            sell = sells.get(report.get('trade_id'))
            if not sell or report.get('asset') != sell.get('asset'):
                raise ReconciliationRequired(cycle_id, 'CLOSED_REPORT_REFERENCE_DIVERGENCE')
            for field, event_field, tolerance in (('quantity', 'quantity', 1e-9),
                    ('exit_price', 'price', 1e-6), ('realized_pnl', 'realized_pnl', .01),
                    ('gross_exit_value', 'gross_value', .01)):
                if abs(float(report[field]) - float(sell[event_field])) > tolerance:
                    raise ReconciliationRequired(cycle_id, 'CLOSED_REPORT_AMOUNT_DIVERGENCE')
        if receipt.get('require_platform_sync'):
            platform = _state('bot_state', BOT_STATE_FILE)
            if any(abs(float(platform[k])-float(actual[k])) > .01 for k in ('cash','realized_pnl')):
                raise ReconciliationRequired(cycle_id, 'PLATFORM_BALANCE_DIVERGENCE')
            mirror = [v for v in platform.get('positions', []) if v.get('module') == 'TRADER']
            if len(mirror) != len(actual['positions']) or len({v['asset'] for v in mirror}) != len(mirror):
                raise ReconciliationRequired(cycle_id, 'PLATFORM_POSITION_DIVERGENCE')
            for position in mirror:
                source = actual['positions'].get(position['asset'])
                if not source or abs(float(position['qty'])-float(source['quantity'])) > 1e-9:
                    raise ReconciliationRequired(cycle_id, 'PLATFORM_QUANTITY_DIVERGENCE')
        from core.integrity_diagnostics import diagnose_paper_integrity
        if diagnose_paper_integrity(actual, actual_trades):
            raise ReconciliationRequired(cycle_id, 'FINANCIAL_INVARIANT_DIVERGENCE')


def _previous_evidence_error(gate, current, observed_rows):
    """Fresh proof at the writer fence; no counter/cache or financial retry."""
    previous_id = gate.get('last_cycle_id')
    if not previous_id:
        from paper_trading_engine import PAPER_STATE_FILE
        if _state('paper_state', PAPER_STATE_FILE).get('last_cycle_id'):
            return 'MISSING_PREVIOUS_CYCLE_CONTROL'
        return None
    try:
        receipt = _receipt(previous_id)
        if receipt.get('cycle_id') != previous_id or not receipt.get('verified_at'):
            return 'PREVIOUS_RECEIPT_NOT_VERIFIED'
        resolution = receipt.get('resolution')
        expected = receipt.get('after') if resolution == 'COMMITTED' else receipt.get('before')
        if resolution not in {'COMMITTED', 'NOT_COMMITTED'} or current != expected:
            return 'PREVIOUS_FINANCIAL_EVIDENCE_CHANGED'
        if resolution == 'COMMITTED':
            if receipt.get('status') != 'COMMITTED':
                return 'PREVIOUS_RECEIPT_STATUS_DIVERGENCE'
            _validate_committed_receipt(receipt, previous_id, current, observed_rows)
        elif receipt.get('status') not in {'PREPARED', 'NOT_COMMITTED'}:
            return 'PREVIOUS_RECEIPT_STATUS_DIVERGENCE'
    except Exception as exc:
        return getattr(exc, 'reason', 'PREVIOUS_RECEIPT_UNREADABLE')
    return None


def _receipt(cycle_id):
    path = _receipt_path(cycle_id)
    namespace = RECEIPT_PREFIX + cycle_id
    if p.database_enabled():
        with p._reader() as conn:
            row = conn.execute('SELECT payload_json FROM app_state WHERE namespace=%s', (namespace,)).fetchone()
        if not row:
            raise ValueError('Missing durable receipt')
        return p.StateSnapshot(p._decode_json(row[0], dict, 'cycle receipt'),
                               p._state_source(namespace, path), row[0])
    if not path.exists():
        raise ValueError('Missing durable receipt')
    return p.load_json_state(namespace, dict, path)


def gate_status():
    # Errors propagate: unreadable/inaccessible is never an open gate.
    gate = _gate()
    quarantine = _quarantine()
    cycle_id = gate.get('active_cycle_id')
    if quarantine.get('active'):
        return {'blocked': True, 'cycle_id': cycle_id or quarantine.get('cycle_id'),
                'outcome': 'UNKNOWN_COMMIT', 'reason': 'INTEGRITY_QUARANTINE',
                'quarantine_id': quarantine['quarantine_id']}
    return {'blocked': bool(cycle_id), 'cycle_id': cycle_id,
            'outcome': ('POST_COMMIT_DIAGNOSTIC_FAILURE' if gate.get('reason') == 'POST_COMMIT_DIAGNOSTIC_FAILURE'
                        else 'UNKNOWN_COMMIT') if cycle_id else gate.get('last_outcome', 'NONE'),
            'reason': gate.get('reason', '')}


def assert_execution_allowed():
    if _PROTOCOL_WRITE.get():
        return
    if not p.database_enabled() and os.getenv('APP_ENV', '').lower() == 'production':
        raise ReconciliationRequired(reason='POSTGRES_REQUIRED_FOR_PRODUCTION')
    status = gate_status()
    if status['blocked']:
        raise ReconciliationRequired(status['cycle_id'])
    # Also closes the restart gap if a previous containment write/read failed.
    # Guard callbacks may run inside a larger transaction; they never claim a
    # staged quarantine survived that caller's rollback.
    if p.database_enabled():
        failure = None
        with _scope():
            gate = _gate()
            if gate.get('active_cycle_id') or _quarantine().get('active'):
                raise ReconciliationRequired(gate.get('active_cycle_id'))
            if gate.get('last_cycle_id'):
                try:
                    observed = {}
                    current = manifest(_observed_rows=observed)
                    failure = _previous_evidence_error(gate, current, observed)
                except Exception:
                    failure = 'PREVIOUS_FINANCIAL_EVIDENCE_UNREADABLE'
                if failure:
                    _quarantine_locked('RECONCILIATION_DIVERGENCE', cycle_id=gate['last_cycle_id'],
                                       evidence={'reason': failure})
        if failure:
            raise ReconciliationRequired(gate['last_cycle_id'], failure)


def financial_projection(state):
    return {key: state.get(key) for key in ('cash', 'positions', 'realized_pnl', 'wallet_value')}


def guard_state_write(namespace, previous, candidate):
    if _PROTOCOL_WRITE.get():
        return
    if namespace in {GATE, QUARANTINE} or namespace.startswith(RECEIPT_PREFIX):
        raise p.PersistenceConflictError('Recibos financeiros só podem mudar pelo protocolo de reconciliação.')
    if namespace == 'paper_state' or (namespace == 'bot_state' and
            financial_projection(previous) != financial_projection(candidate)):
        assert_execution_allowed()


def guard_rows_write(namespace):
    if namespace in FINANCIAL_ROWS and not _PROTOCOL_WRITE.get():
        assert_execution_allowed()


def _state(namespace, path):
    if p.database_enabled():
        with p._reader() as conn:
            row = conn.execute('SELECT payload_json FROM app_state WHERE namespace=%s', (namespace,)).fetchone()
        if not row:
            raise ValueError('Missing financial namespace')
        return p._decode_json(row[0], dict, 'reconciliation state')
    return p._decode_json(path.read_text(), dict, 'reconciliation local state')


def manifest(*, _observed_rows=None):
    """Exact persisted snapshot, under the common writer fence during verification.

    Optional output shares the rows just decoded from this query with the same
    reconciliation call. The caller consumes it only in this verification; it
    is not cached or accepted as input on later calls. No database read/hash is
    skipped. Existing receipt hash format remains unchanged.
    """
    from paper_trading_engine import PAPER_STATE_FILE, PAPER_TRADES_FILE
    paper = _state('paper_state', PAPER_STATE_FILE)
    bot = _state('bot_state', BOT_STATE_FILE)
    result = {'paper': _hash(paper), 'platform': _hash(financial_projection(bot))}
    if p.database_enabled():
        with p._reader() as conn:
            for namespace in sorted(FINANCIAL_ROWS):
                rows = conn.execute('SELECT seq,row_json FROM app_rows WHERE namespace=%s ORDER BY seq',
                                    (namespace,)).fetchall()
                decoded = [[seq, p._decode_json(text, dict, 'reconciliation row')] for seq, text in rows]
                result[namespace] = {'count': len(rows), 'sha256': _hash(decoded)}
                if _observed_rows is not None:
                    _observed_rows[namespace] = [item[1] for item in decoded]
    else:
        for namespace, path in [('paper_trades', PAPER_TRADES_FILE),
                                ('table:trader_orders.csv', TRADER_ORDERS_FILE),
                                ('table:trade_reports.csv', TRADER_REPORTS_FILE)]:
            result[namespace] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else 'ABSENT'
    return result


def begin_intent(state, trades, reports, *, require_platform=False):
    """Persist a fence BEFORE attempting the financial transaction, once only."""
    if p._ACTIVE_CONNECTION.get() is not None:
        raise RuntimeError('Financial intent requires an independent durable transaction.')
    assert_execution_allowed()
    cycle_id = uuid4().hex
    failure = None
    with _scope(), protocol_write():
        gate = _gate()
        if _quarantine().get('active'):
            raise ReconciliationRequired(reason='INTEGRITY_QUARANTINE')
        if gate.get('active_cycle_id'):
            raise ReconciliationRequired(gate['active_cycle_id'])
        observed = {}
        before = manifest(_observed_rows=observed)
        failure = _previous_evidence_error(gate, before, observed) if p.database_enabled() else None
        if failure:
            _quarantine_locked('RECONCILIATION_DIVERGENCE', cycle_id=gate.get('last_cycle_id'),
                               evidence={'reason': failure})
        else:
            _create_intent(gate, cycle_id, before, state, trades, reports, require_platform)
    if failure:
        raise ReconciliationRequired(gate.get('last_cycle_id'), failure)
    # Any error from this commit/read remains conservative; no financial tx began.
    if _gate().get('active_cycle_id') != cycle_id:
        raise UnknownCommit(cycle_id, 'INTENT_VERIFICATION_FAILED')
    return cycle_id


def _create_intent(gate, cycle_id, before, state, trades, reports, require_platform):
    receipt = {'cycle_id': cycle_id, 'status': 'PREPARED', 'created_at': _now(),
               'backend': 'postgresql' if p.database_enabled() else 'local_files',
               'require_platform_sync': require_platform,
               'before': before, 'expected_paper': _hash({**state, 'last_cycle_id': cycle_id}),
               'planned_trades': [_hash(t) for t in trades],
               'planned_reports': [_hash(r) for r in reports], 'events': ['PREPARED']}
    p.save_json_state(RECEIPT_PREFIX + cycle_id, receipt, _receipt_path(cycle_id))
    gate.update(active_cycle_id=cycle_id, reason='AWAITING_VERIFIED_OUTCOME')
    p.save_json_state(GATE, gate, _GATE_PATH)

def reject_duplicate_events(trades, reports):
    from paper_trading_engine import PAPER_TRADES_FILE
    keys = [(t.get('trade_id'), t.get('side')) for t in trades if t.get('trade_id')]
    if len(keys) != len(set(keys)):
        raise p.PersistenceDataError('Duplicate event in financial draft')
    # An empty candidate set cannot duplicate an identity. Strict validation of
    # ALL histories still occurs in begin_intent, record_unit_result and reconcile.
    if keys:
        existing = p.load_json_rows('paper_trades', PAPER_TRADES_FILE)
        if set(keys) & {(t.get('trade_id'), t.get('side')) for t in existing}:
            raise p.PersistenceDataError('Trade identity already persisted; no overwrite or retry')
    report_ids = [r.get('trade_id') for r in reports if r.get('trade_id')]
    if len(report_ids) != len(set(report_ids)):
        raise p.PersistenceDataError('Duplicate closed report in draft')
    if report_ids:
        if p.database_enabled():
            rows = p.load_json_rows('table:trade_reports.csv', TRADER_REPORTS_FILE.with_suffix('.csv.json'))
        else:
            rows = p.read_table(TRADER_REPORTS_FILE).to_dict('records')
        if set(report_ids) & {r.get('trade_id') for r in rows}:
            raise p.PersistenceDataError('Closed report already persisted')


def record_unit_result(cycle_id, committed):
    """Called INSIDE the SAME transaction as cash, positions, trades and reports."""
    if _quarantine().get('active'):
        raise ReconciliationRequired(cycle_id, 'INTEGRITY_QUARANTINE')
    receipt = _receipt(cycle_id)
    gate = _gate()
    if gate.get('active_cycle_id') != cycle_id or receipt.get('status') != 'PREPARED':
        raise ReconciliationRequired(cycle_id, 'INTENT_CHANGED')
    receipt.update(status='COMMITTED' if committed else 'NOT_COMMITTED', after=manifest(),
                   recorded_at=_now())
    receipt['events'].append(receipt['status'])
    p.save_json_state(RECEIPT_PREFIX + cycle_id, receipt, _receipt_path(cycle_id))


def reconcile(cycle_id=None, *, release=True, release_quarantine=False, expected_quarantine_id=None):
    """No financial execution: fence old writers, inspect positive evidence, resolve.

    A missing receipt, mismatch or failed read leaves the pre-persisted gate closed.
    Local files cannot prove a PostgreSQL COMMIT or atomic multi-file rollback.
    """
    verified_outcome = None
    selected = cycle_id
    try:
        with _scope(), protocol_write():
            gate = _gate()
            quarantine = _quarantine()
            active = gate.get('active_cycle_id')
            cycle_id = cycle_id or active or quarantine.get('cycle_id')
            selected = cycle_id
            if not cycle_id:
                raise ReconciliationRequired(reason='NO_CYCLE_SELECTED_NO_COMMIT_EVIDENCE')
            if active and active != cycle_id:
                raise ReconciliationRequired(active, 'DIFFERENT_ACTIVE_INTENT')
            receipt = _receipt(cycle_id)
            if receipt.get('cycle_id') != cycle_id or not receipt.get('before'):
                raise ReconciliationRequired(cycle_id, 'MISSING_RECEIPT')
            if not active and gate.get('last_cycle_id') != cycle_id:
                # A resolved older receipt is historical evidence, not a claim
                # about today's manifest and never an authorization to clear it.
                if receipt.get('resolution') not in {'COMMITTED', 'NOT_COMMITTED', 'LOCAL_ONLY'} or not receipt.get('verified_at'):
                    raise ReconciliationRequired(cycle_id, 'HISTORICAL_RECEIPT_NOT_VERIFIED')
                return {'outcome': receipt['resolution'], 'blocked': bool(quarantine.get('active')), 'cycle_id': cycle_id,
                        'historical': True, 'current_cycle_id': gate.get('last_cycle_id'),
                        'evidence': 'PREVIOUSLY_VERIFIED_RECEIPT_NOT_CURRENT_STATE',
                        'resolution_recorded': True}
            observed_rows = {}
            current = manifest(_observed_rows=observed_rows)
            status = receipt.get('status')
            if status == 'COMMITTED' and current == receipt.get('after'):
                _validate_committed_receipt(receipt, cycle_id, current, observed_rows)
                outcome = 'COMMITTED' if p.database_enabled() else 'LOCAL_ONLY'
            elif status in {'PREPARED', 'NOT_COMMITTED'} and current == receipt.get('before'):
                # The common fence cannot be held until any previous financial
                # transaction has ended. PREPARED + exact preimage proves absence.
                if not p.database_enabled():
                    raise ReconciliationRequired(cycle_id, 'LOCAL_ROLLBACK_NOT_PROVABLE')
                outcome = 'NOT_COMMITTED'
            else:
                raise ReconciliationRequired(cycle_id, 'FINANCIAL_DIVERGENCE_OR_INSUFFICIENT_EVIDENCE')
            verified_outcome = outcome
            receipt.update(resolution=outcome, verified_at=_now())
            receipt['events'].append('VERIFIED_' + outcome)
            p.save_json_state(RECEIPT_PREFIX + cycle_id, receipt, _receipt_path(cycle_id))
            if release:
                gate.update(active_cycle_id=None, last_cycle_id=cycle_id, last_outcome=outcome,
                            reason='', verified_at=_now())
                p.save_json_state(GATE, gate, _GATE_PATH)
                if quarantine.get('active') and release_quarantine:
                    safe = (quarantine.get('quarantine_id') == expected_quarantine_id
                            and quarantine.get('cycle_id') == cycle_id
                            and all(item['reason'] == 'RECONCILIATION_DIVERGENCE'
                                    for item in quarantine['evidence']))
                    if safe:
                        quarantine.update(active=False, resolved_at=_now(), resolution=outcome)
                        p.save_json_state(QUARANTINE, quarantine, _QUARANTINE_PATH)
        return {'outcome': outcome, 'blocked': not release or bool(quarantine.get('active')), 'cycle_id': cycle_id,
                'evidence': 'FENCED_MANIFEST_AND_RECEIPT_MATCH', 'resolution_recorded': True}
    except Exception as exc:
        if verified_outcome is not None:
            # Financial evidence was observed under the fence. Bookkeeping
            # failure does not make the original COMMIT itself uncertain again.
            return {'outcome': verified_outcome, 'blocked': True, 'cycle_id': cycle_id,
                    'resolution_recorded': False, 'reason': 'RESOLUTION_BOOKKEEPING_UNCONFIRMED',
                    'evidence': 'FENCED_MANIFEST_AND_RECEIPT_MATCH'}
        reason = getattr(exc, 'reason', type(exc).__name__)
        containment = {'quarantine_persistence': 'UNCONFIRMED'}
        try:
            # An existing durable intent already blocks; never change B while
            # reporting a failed/stale verification of A.
            gate = _gate()
            if gate.get('active_cycle_id'):
                containment = {'quarantine_persistence': 'EXISTING_INTENT'}
            else:
                current_id = gate.get('last_cycle_id')
                containment = quarantine_integrity(
                    'RECONCILIATION_DIVERGENCE' if selected in {None, current_id} else 'HISTORICAL_RECEIPT_FAILURE',
                    cycle_id=current_id, evidence={'reason': reason, 'requested_cycle_id': selected})
        except Exception:
            pass
        return {'outcome': 'UNKNOWN_COMMIT', 'blocked': True, 'cycle_id': cycle_id,
                'reason': reason, **containment}


def post_commit_failure(cycle_id, *, confirmed=False):
    """Presentation failed after the financial commit was positively verified."""
    stale_diagnostic = False
    current_cycle_id = None
    try:
        with _scope(), protocol_write():
            gate = _gate()
            current_cycle_id = gate.get('active_cycle_id') or gate.get('last_cycle_id')
            stale_diagnostic = current_cycle_id not in {None, cycle_id}
            receipt = _receipt(cycle_id)
            if receipt.get('status') != 'COMMITTED':
                raise ReconciliationRequired(cycle_id)
            confirmed = True
            receipt['diagnostic_status'] = 'POST_COMMIT_DIAGNOSTIC_FAILURE'
            receipt['events'].append('LATE_DIAGNOSTIC' if stale_diagnostic else 'POST_COMMIT_DIAGNOSTIC_FAILURE')
            if stale_diagnostic:
                receipt['superseded_by'] = current_cycle_id
            p.save_json_state(RECEIPT_PREFIX + cycle_id, receipt, _receipt_path(cycle_id))
            # Retain a conservative fence until explicit reconciliation.
            if not stale_diagnostic and gate.get('active_cycle_id') in {None, cycle_id}:
                gate.update(active_cycle_id=cycle_id, reason='POST_COMMIT_DIAGNOSTIC_FAILURE')
                p.save_json_state(GATE, gate, _GATE_PATH)
    except Exception:
        # A caller with a positive server reply retains that evidence even if
        # diagnostic bookkeeping cannot be reached. Without evidence, stay UNKNOWN.
        pass
    if not confirmed:
        return {'outcome': 'UNKNOWN_COMMIT', 'committed': False, 'cycle_id': cycle_id}
    return {'outcome': 'POST_COMMIT_DIAGNOSTIC_FAILURE' if p.database_enabled() else 'LOCAL_ONLY',
            'committed': True, 'cycle_id': cycle_id, 'diagnostic_status': 'POST_COMMIT_DIAGNOSTIC_FAILURE',
            'stale_diagnostic': stale_diagnostic, 'current_cycle_id': current_cycle_id}

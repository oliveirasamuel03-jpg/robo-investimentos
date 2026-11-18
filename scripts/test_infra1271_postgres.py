"""Close the fixed 175-case suite without changing tests or prior evidence.

Runs the original runner from a detached, disposable worktree of the exact
authorized source. No existing database, DSN, image download or worker service
is accepted. Only the original runner owns/cleans its new PostgreSQL container.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '8bdb4bd2fe099ddc59af158f8a32f190fd6b84ab'
BRANCH = 'atlas/infra-1-2-7-1-fechamento-regressao'
EVIDENCE = Path('docs/retomada/evidencias')
MIN_FREE_BYTES = 256 * 1024 ** 2  # preserve the existing test reserve
CONTAINER_HEADROOM = 768 * 1024 ** 2
DOCKER = ['docker', '--host=unix:///var/run/docker.sock']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def expected_ids():
    data = subprocess.check_output(['git', 'show', SOURCE + ':' + str(EVIDENCE / 'infra127-checks.json')], cwd=ROOT)
    ids = set(json.loads(data)['final_postgres_case_results'])
    if len(ids) != 175:
        raise ValueError('Authorized source must identify exactly 175 PostgreSQL cases')
    return ids


def validate_junit(path, expected):
    root = ET.parse(path).getroot()
    suites = list(root.iter('testsuite'))
    cases = list(root.iter('testcase'))
    identities = [c.get('classname', '').replace('.', '/') + '.py::' + c.get('name', '') for c in cases]
    counts = {'tests': len(cases), 'passed': 0, 'failed': 0, 'errors': 0, 'skipped': 0}
    for case in cases:
        kind = ('errors' if case.find('error') is not None else 'failed' if case.find('failure') is not None
                else 'skipped' if case.find('skipped') is not None else 'passed')
        counts[kind] += 1
    missing, unexpected = sorted(expected - set(identities)), sorted(set(identities) - expected)
    duplicates = sorted(key for key, count in Counter(identities).items() if count > 1)
    issues = []
    if len(suites) != 1:
        issues.append('Expected one complete pytest suite')
    if missing or unexpected or duplicates:
        issues.append('Case identities differ from the authorized 175-case source')
    if counts['passed'] != len(expected) or counts['failed'] or counts['errors'] or counts['skipped']:
        issues.append('Every expected case must pass; skipped/error/failed cases cannot close regression')
    for field, actual in [('tests', counts['tests']), ('failures', counts['failed']),
                          ('errors', counts['errors']), ('skipped', counts['skipped'])]:
        if sum(int(s.get(field, '0')) for s in suites) != actual:
            issues.append('JUnit header mismatch: ' + field)
    return {'valid': not issues, **counts, 'missing_ids': missing, 'unexpected_ids': unexpected,
            'duplicate_ids': duplicates, 'issues': issues, 'case_ids': identities}


def evidence_snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in (root / EVIDENCE).rglob('*') if p.is_file()}


def capture_outputs(sandbox, before, destination):
    """Copy changed/new test artifacts; never change the source checkout."""
    changes = []
    after = evidence_snapshot(sandbox)
    for name, data in after.items():
        if before.get(name) == data:
            continue
        target = destination / Path(name).relative_to(EVIDENCE)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        changes.append({'source_path': name, 'output_path': str(target), 'sha256': sha(data),
                        'source_sha256': sha(before[name]) if name in before else None})
    if set(before) - set(after):
        raise RuntimeError('Test sandbox removed evidence; preserve sandbox for investigation')
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=('18', '16'), default='18')
    parser.add_argument('--alpine', action='store_true')
    args = parser.parse_args()
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() != BRANCH:
        raise SystemExit('Use only the authorized closure branch')
    expected = expected_ids()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = ROOT / EVIDENCE / 'infra1271' / f'pg{args.version}-{stamp}-{uuid.uuid4().hex[:8]}'
    output.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'LOCAL_SYNTHETIC_PAPER_ONLY', 'source_commit': SOURCE,
              'branch': BRANCH, 'expected_cases': len(expected), 'requested_version': args.version,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'NOT_EXECUTED',
              'min_free_bytes_unchanged': MIN_FREE_BYTES, 'broker_mode': 'paper', 'broker_provider': 'paper'}
    save(output / 'run.json', report)
    clean_env = {k: v for k, v in os.environ.items() if k not in {
        'DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_TLS', 'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH'}}
    image = f'postgres:{args.version}' + ('-alpine' if args.alpine else '')
    info = subprocess.run(DOCKER + ['image', 'inspect', image], env=clean_env, capture_output=True, text=True)
    free = shutil.disk_usage(ROOT).free
    report.update(free_bytes_before=free, image=image)
    if info.returncode:
        report.update(status='RESOURCE_BLOCKED', reason='Required local official image unavailable; no automatic pull or cleanup')
        save(output / 'run.json', report)
        print(json.dumps({'status': report['status'], 'evidence': str(output)}), flush=True)
        return 3
    image_info = json.loads(info.stdout)[0]
    required = MIN_FREE_BYTES + int(image_info['Size']) + CONTAINER_HEADROOM
    report.update(image_id=image_info['Id'], image_digests=image_info['RepoDigests'], required_free_bytes=required)
    if free < required:
        report.update(status='RESOURCE_BLOCKED', reason='Insufficient free disk for owned VFS container, data and unchanged reserve')
        save(output / 'run.json', report)
        print(json.dumps({'status': report['status'], 'evidence': str(output)}), flush=True)
        return 3
    prior_paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', SOURCE, str(EVIDENCE)],
                                          cwd=ROOT, text=True).splitlines()
    protected = {name: (ROOT / name).read_bytes() for name in prior_paths}
    outer = Path(tempfile.mkdtemp(prefix='atlas-infra1271-'))
    sandbox = outer / 'source'
    added = captured = False
    started = time.monotonic()
    try:
        subprocess.run(['git', 'worktree', 'add', '--detach', str(sandbox), SOURCE], cwd=ROOT,
                       check=True, capture_output=True)
        added = True
        before = evidence_snapshot(sandbox)
        command = [sys.executable, str(sandbox / 'scripts/test_infra127_postgres.py'), '--version', args.version,
                   '--junitxml=' + str(output / 'junit.xml')]
        if args.alpine:
            command.append('--alpine')
        report.update(status='RUNNING', runner_sha256=sha(Path(__file__).read_bytes()),
                      command=['python', 'scripts/test_infra127_postgres.py', '--version', args.version,
                               '--alpine' if args.alpine else '(standard image)', '--junitxml=junit.xml'],
                      collection_filter=None, sandbox_source_commit=SOURCE)
        save(output / 'run.json', report)
        print(json.dumps({'status': 'RUNNING', 'evidence': str(output), 'expected_cases': 175}), flush=True)
        with (output / 'pytest.txt').open('w') as log:
            result = subprocess.run(command, cwd=sandbox, stdout=log, stderr=subprocess.STDOUT)
        report['child_exit_code'] = result.returncode
        log = (output / 'pytest.txt').read_text()
        warning_counts = re.findall(r'(\d+) warnings', log)
        report['warnings'] = int(warning_counts[-1]) if warning_counts else 0
        server = next((json.loads(line) for line in log.splitlines() if line.startswith('{"server_version"')), {})
        report['observed_server'] = server
        verification = validate_junit(output / 'junit.xml', expected)
        report['verification'] = verification
        deselected = sum(int(n) for n in re.findall(r'(\d+) deselected', log))
        report['deselected'] = deselected
        good = (result.returncode == 0 and verification['valid'] and deselected == 0
                and str(server.get('server_version', '')).split('.')[0] == args.version
                and server.get('broker_mode') == server.get('broker_provider') == 'paper')
        report['status'] = 'PASSED' if good else 'FAILED'
    except Exception as error:
        report.update(status='FAILED', error_type=type(error).__name__)
    finally:
        try:
            if added:
                report['artifacts'] = capture_outputs(sandbox, before, output / 'artifacts')
            captured = True
        except Exception as error:
            report.update(status='FAILED', artifact_capture_error=type(error).__name__, retained_sandbox=str(sandbox))
        current = evidence_snapshot(ROOT)
        report['prior_evidence_unchanged'] = all(current.get(k) == v for k, v in protected.items())
        if not report['prior_evidence_unchanged']:
            report['status'] = 'FAILED'
        if added and captured:
            removed = subprocess.run(['git', 'worktree', 'remove', '--force', str(sandbox)], cwd=ROOT,
                                     capture_output=True)
            report['sandbox_removed'] = removed.returncode == 0
            if removed.returncode:
                report.update(status='FAILED', retained_sandbox=str(sandbox))
        if not any(outer.iterdir()):
            outer.rmdir()
        report.update(elapsed_seconds=time.monotonic() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        save(output / 'run.json', report)
    print(json.dumps({'status': report['status'], 'evidence': str(output)}), flush=True)
    return 0 if report['status'] == 'PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())

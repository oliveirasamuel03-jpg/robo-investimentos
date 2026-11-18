"""The closure wrapper must never certify an incomplete or failed suite."""
import xml.etree.ElementTree as ET

import pytest

from scripts.test_infra1271_postgres import capture_outputs, evidence_snapshot, validate_junit


EXPECTED = {'tests/test_synthetic.py::first', 'tests/test_synthetic.py::second'}


def junit(tmp_path):
    root = ET.Element('testsuites')
    suite = ET.SubElement(root, 'testsuite', tests='2', failures='0', errors='0', skipped='0')
    for name in ('first', 'second'):
        ET.SubElement(suite, 'testcase', classname='tests.test_synthetic', name=name)
    return root, suite, tmp_path / 'synthetic.xml'


def test_complete_unique_success_is_required(tmp_path):
    root, suite, path = junit(tmp_path)
    ET.ElementTree(root).write(path)
    result = validate_junit(path, EXPECTED)
    assert result['valid'] and result['passed'] == 2 and not result['issues']


@pytest.mark.parametrize('defect', ['missing', 'unexpected', 'duplicate', 'failure', 'error', 'skipped', 'header'])
def test_incomplete_or_failed_junit_cannot_be_certified(tmp_path, defect):
    root, suite, path = junit(tmp_path)
    if defect == 'missing':
        suite.remove(suite[1])
    elif defect == 'unexpected':
        suite[1].set('name', 'different')
    elif defect == 'duplicate':
        suite[1].set('name', 'first')
    elif defect == 'header':
        suite.set('tests', '175')
    else:
        ET.SubElement(suite[0], defect)
    ET.ElementTree(root).write(path)
    assert not validate_junit(path, EXPECTED)['valid']


def test_artifacts_are_copied_without_overwriting_original_evidence(tmp_path):
    root = tmp_path / 'sandbox'
    evidence = root / 'docs/retomada/evidencias'
    evidence.mkdir(parents=True)
    (evidence / 'previous.json').write_text('original')
    before = evidence_snapshot(root)
    (evidence / 'previous.json').write_text('new synthetic measurement')
    (evidence / 'new.json').write_text('new synthetic proof')
    outputs = tmp_path / 'captured'
    result = capture_outputs(root, before, outputs)
    assert len(result) == 2
    assert before['docs/retomada/evidencias/previous.json'] == b'original'
    assert (outputs / 'previous.json').read_text() == 'new synthetic measurement'
    assert (outputs / 'new.json').read_text() == 'new synthetic proof'

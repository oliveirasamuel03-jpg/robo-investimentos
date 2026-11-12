"""Prototype cannot be activated outside the explicit local PAPER runner."""
import pytest
from scripts.prototypes.infra125_snapshot import SnapshotPrototype


def test_snapshot_prototype_refuses_ordinary_runtime(monkeypatch):
    monkeypatch.delenv('ATLAS_INFRA12_LOCAL_RUN',raising=False)
    with pytest.raises(ValueError,match='disposable synthetic runner'):
        SnapshotPrototype('host=127.0.0.1 port=5432 dbname=atlas_infra123_fixture user=atlas_fixture')


@pytest.mark.parametrize('variable',['BROKER_MODE','BROKER_PROVIDER'])
def test_snapshot_prototype_requires_both_paper_settings(monkeypatch,variable):
    monkeypatch.setenv('ATLAS_INFRA12_LOCAL_RUN','synthetic-disposable-only')
    monkeypatch.setenv('BROKER_MODE','paper');monkeypatch.setenv('BROKER_PROVIDER','paper')
    monkeypatch.setenv(variable,'real')
    with pytest.raises(ValueError,match='PAPER only'):SnapshotPrototype('host=127.0.0.1')


@pytest.mark.parametrize('host',['remote.invalid','localhost','0.0.0.0'])
def test_snapshot_prototype_refuses_non_literal_loopback_before_connect(monkeypatch,host):
    monkeypatch.setenv('ATLAS_INFRA12_LOCAL_RUN','synthetic-disposable-only')
    monkeypatch.setenv('BROKER_MODE','paper');monkeypatch.setenv('BROKER_PROVIDER','paper')
    with pytest.raises(ValueError,match='synthetic local runner'):
        SnapshotPrototype(f'host={host} port=5432 dbname=atlas_infra123_fixture user=atlas_fixture')

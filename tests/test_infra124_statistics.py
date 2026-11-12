"""Benchmark reporting must not invent precision or count timeouts as samples."""
import pytest
from tests.test_infra124_postgres import statistics


@pytest.mark.parametrize('values,n,median,p95,maximum',[
    ([None],0,None,None,None),([1,2,3,4,5,6,7,8,9],9,5,None,9),
    (list(range(1,11)),10,5.5,10,10),(list(range(1,21)),20,10.5,19,20),
])
def test_benchmark_statistics_report_actual_samples(values,n,median,p95,maximum):
    result=statistics(values)
    assert result['n']==n and result['median']==median
    assert result['p95']==p95 and result['max']==maximum

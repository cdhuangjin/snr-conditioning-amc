import importlib.util
from pathlib import Path
import pytest


def module():
    path=Path(__file__).resolve().parents[2]/'scripts/v2/audit_phase7_dominance.py'
    spec=importlib.util.spec_from_file_location('dominance_audit',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def test_pool_uses_counts_not_average_of_bin_frequencies():
    rows=[{'n':10,'predicted_count':[9,1],'true_count':[5,5]}, {'n':90,'predicted_count':[9,81],'true_count':[45,45]}]
    result=module().pool_counts(rows)
    assert result['n']==100
    assert result['dominant_id']==1
    assert result['dominant_share']==.82
    assert result['true_frequency']==[.5,.5]


def test_pool_rejects_inconsistent_n():
    with pytest.raises(ValueError,match='count'):
        module().pool_counts([{'n':10,'predicted_count':[9,2],'true_count':[5,5]}])

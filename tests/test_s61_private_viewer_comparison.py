"""S61 must never turn experimental extra S60 joins into default physical truth."""
import json
import pytest
from scripts.research.bubble_chamber.s61_build_calibrated_private_showcase import (
    upgrade_html,integrate_payload,SCHEMA)


def test_private_viewer_requires_exact_audited_source():
    with pytest.raises(ValueError):integrate_payload('<html/>',{'graphs':[]})


def test_demo_rejects_fake_or_missing_GT_signals():
    html='<script id="private-data" type="application/json">'+json.dumps({
        'schema':'s57_machine_only_global_paths_PRIVATE_v1',
        'native_GT_count':1,'physical_auto_links':0,'tiles':[{}]*39})+'</script>'
    with pytest.raises(ValueError):integrate_payload(html,{'graphs':[]})


def test_require_original_baseline_not_modified():
    with pytest.raises(ValueError):upgrade_html('unrelated HTML')

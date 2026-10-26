"""Public viewer template code must not bundle any native H50Z photo data."""
from pathlib import Path
import json
import pytest
from scripts.research.bubble_chamber.s57_build_global_paths_viewer import build_template,_replace_exact


def test_s57_html_is_offline_and_has_no_private_data_in_public_template():
    html=build_template()
    assert '__INLINE_PAYLOAD__' in html
    assert html.count('showGlobalOnly')>=3
    assert html.count('showAllEdges')>=3
    assert 'native_relation_truth' in html or 'NOT_PHYSICS_GT' in html
    assert 'http://' not in html and 'https://' not in html and 'fetch(' not in html
    assert 'base64,' not in html


def test_exact_template_change_fails_closed():
    with pytest.raises(ValueError):_replace_exact('hello','NOT_FOUND','new')
    with pytest.raises(ValueError):_replace_exact('abc abc','abc','x')

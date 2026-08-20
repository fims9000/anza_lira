import json
import numpy as np

from trace_extraction.export import traces_to_geojson
from trace_extraction.graph import extract_trace_graph


def test_cracks_trace_geojson_uses_candidate_branch_and_border_contract() -> None:
    skeleton = np.zeros((20, 20), dtype=bool)
    skeleton[2, 1:12] = True
    graph = extract_trace_graph(skeleton, border_margin=5)
    payload = traces_to_geojson(
        graph.segments,
        source_image_id="section_010",
        patch_id="full_section_255x701",
        model="unet",
        seed=42,
    )
    json.dumps(payload, allow_nan=False)
    assert payload["type"] == "FeatureCollection"
    assert payload["features"][0]["properties"]["source_image_id"] == "section_010"
    assert payload["features"][0]["properties"]["border_truncated"] is True

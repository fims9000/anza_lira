from path_completion.realistic_synthetic import REALISTIC_PROTOCOL
from synthetic.crossing_trace_bench_v6 import generate_sample_v6


def test_v6_development_is_deterministic_and_test_locked():
    first = generate_sample_v6("development", 0, image_size=32)
    second = generate_sample_v6("development", 0, image_size=32)
    assert (first["image"] == second["image"]).all()
    try:
        generate_sample_v6("test", 0, image_size=32)
    except PermissionError as error:
        assert "LOCKED" in str(error)
    else:
        raise AssertionError("v6 test opened before development freeze")


def test_realistic_protocol_uses_predictions_and_no_expert():
    assert REALISTIC_PROTOCOL["endpoint_source"] == "predicted binary mask only"
    assert REALISTIC_PROTOCOL["v6_test"] == "LOCKED_UNOPENED"
    assert REALISTIC_PROTOCOL["expert"] == "FORBIDDEN"

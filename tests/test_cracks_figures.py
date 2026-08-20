from cracks_experiment.figures import _median_row, _save


def test_figure_export_writes_all_formats_and_median_is_deterministic(tmp_path) -> None:
    import csv
    import matplotlib.pyplot as plt

    table = tmp_path / "rows.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "metric"])
        writer.writeheader()
        writer.writerows([{"index": 2, "metric": 0.9}, {"index": 1, "metric": 0.1}, {"index": 3, "metric": 0.5}])
    assert _median_row(table, "metric")["index"] == "3"
    figure = plt.figure(figsize=(2, 1))
    plt.plot([0, 1], [0, 1], label="line")
    plt.legend(loc="upper left")
    _save(figure, tmp_path, 1)
    for suffix in ("png", "svg", "pdf"):
        assert (tmp_path / f"figure_1.{suffix}").stat().st_size > 0

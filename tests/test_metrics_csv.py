import csv

from jaxgcrl.utils.env import MetricsRecorder


def test_csv_header_grows(tmp_path):
    p = str(tmp_path / "m.csv")
    r = MetricsRecorder(100, ["eval/x", "training/y"], str(tmp_path), "t", mode="online", csv_path=p, render=False)
    r.write_csv(0, {"eval/x": 1.0})
    r.write_csv(10, {"eval/x": 2.0, "training/y": 3.0})
    r.write_csv(20, {"eval/x": 4.0, "training/y": 5.0})
    rows = list(csv.DictReader(open(p)))
    assert [r["step"] for r in rows] == ["0", "10", "20"]
    assert rows[0]["training/y"] == "" and rows[1]["training/y"] == "3.0" and rows[2]["eval/x"] == "4.0"

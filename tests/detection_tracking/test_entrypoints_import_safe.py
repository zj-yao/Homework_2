import importlib
import subprocess
import sys


def test_cli_modules_are_import_safe_without_loading_ultralytics():
    for name in list(sys.modules):
        if name.startswith("ultralytics"):
            del sys.modules[name]

    for module_name in [
        "src.detection_tracking.train_yolo",
        "src.detection_tracking.track_video",
        "src.detection_tracking.occlusion_analysis",
    ]:
        importlib.import_module(module_name)

    assert not any(name.startswith("ultralytics") for name in sys.modules)


def test_train_yolo_builds_kwargs_without_starting_training(tmp_path):
    from src.detection_tracking.train_yolo import build_train_kwargs

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("train: images/train\nval: images/val\nnames: [car]\nnc: 1\n", encoding="utf-8")

    kwargs = build_train_kwargs(
        data_yaml=data_yaml,
        epochs=3,
        imgsz=320,
        batch=2,
        project=tmp_path / "runs",
        name="smoke",
        device="cpu",
    )

    assert kwargs["data"] == str(data_yaml)
    assert kwargs["epochs"] == 3
    assert kwargs["imgsz"] == 320
    assert kwargs["batch"] == 2
    assert kwargs["project"] == str(tmp_path / "runs")
    assert kwargs["name"] == "smoke"
    assert kwargs["device"] == "cpu"


def test_prepare_data_module_help_runs_without_runpy_warning():
    result = subprocess.run(
        [sys.executable, "-m", "src.detection_tracking.prepare_data", "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert "RuntimeWarning" not in result.stderr

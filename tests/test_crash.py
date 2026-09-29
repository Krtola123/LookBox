import os
import sys
import threading

from lookbox import crash


def _boom():
    try:
        raise ValueError("kaboom 42")
    except ValueError:
        return sys.exc_info()


def _with_log_dir(tmp_path, fn):
    old = os.environ.get("LOOKBOX_LOG_DIR")
    os.environ["LOOKBOX_LOG_DIR"] = str(tmp_path / "logs")
    try:
        return fn()
    finally:
        if old is None:
            del os.environ["LOOKBOX_LOG_DIR"]
        else:
            os.environ["LOOKBOX_LOG_DIR"] = old


def test_report_is_written_with_traceback(tmp_path):
    def run():
        crash.write_report(*_boom(), where="test")
        text = open(crash.log_path(), encoding="utf-8").read()
        assert "ValueError: kaboom 42" in text and "_boom" in text and "test" in text
    _with_log_dir(tmp_path, run)


def test_ui_errors_show_once_per_burst_and_threads_never_show(tmp_path):
    shown = []

    def run():
        r = crash.ErrorReporter(show=lambda t, m: shown.append(m), quiet_seconds=60)
        et, ev, tb = _boom()
        r._report(et, ev, tb, where="", ui=True)
        r._report(et, ev, tb, where="", ui=True)  # same error again at once: logged, not re-shown
        r._report(et, ev, tb, where="thread x", ui=False)
        assert len(shown) == 1 and "kaboom 42" in shown[0] and crash.log_path() in shown[0]
        assert open(crash.log_path(), encoding="utf-8").read().count("kaboom 42") >= 3
    _with_log_dir(tmp_path, run)


def test_install_hooks_threads(tmp_path):
    def run():
        old_sys, old_thr = sys.excepthook, threading.excepthook
        try:
            crash.ErrorReporter().install()
            t = threading.Thread(target=lambda: (_ for _ in ()).throw(RuntimeError("in worker")), name="w1")
            t.start()
            t.join()
            assert "in worker" in open(crash.log_path(), encoding="utf-8").read()
        finally:
            sys.excepthook, threading.excepthook = old_sys, old_thr
            import faulthandler
            faulthandler.disable()
    _with_log_dir(tmp_path, run)


def test_data_folder_moves_over_from_the_old_name(tmp_path):
    import os
    import sys

    from lookbox import branding

    env = "LOCALAPPDATA" if sys.platform == "win32" else "XDG_DATA_HOME"
    old_env = os.environ.get(env)
    os.environ[env] = str(tmp_path)
    try:
        (tmp_path / "LookBox" / "models").mkdir(parents=True)
        (tmp_path / "LookBox" / "models" / "m.onnx").write_bytes(b"model")
        root = branding.data_root()
        assert root == str(tmp_path / "RRIPP") and (tmp_path / "RRIPP" / "models" / "m.onnx").exists()
        assert not (tmp_path / "LookBox").exists()
        assert branding.data_root() == root  # second time: nothing to do
    finally:
        if old_env is None:
            del os.environ[env]
        else:
            os.environ[env] = old_env


def test_old_project_files_still_open():
    from lookbox.core import serialize

    assert serialize.is_project("C:/x/a.RRIPP") and serialize.is_project("b.lookbox")
    assert not serialize.is_project("c.png") and serialize.EXTENSION == ".rripp"

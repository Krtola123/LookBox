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

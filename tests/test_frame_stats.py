from lookbox.ui.canvas.frame_stats import FrameStats


def test_frame_stats_needs_a_few_frames():
    f = FrameStats()
    f.add(0.004, 1.0)
    f.add(0.004, 1.016)
    assert f.summary() is None


def test_frame_stats_reports_p95_cost_and_rate():
    f = FrameStats()
    for i in range(61):
        f.add(0.005 if i % 20 else 0.050, 1.0 + i / 60.0)  # mostly 5 ms, a few 50 ms spikes
    text = f.summary(final=True)
    assert text.startswith("Last drag:")
    assert "60 fps shown" in text
    assert "SLOW" in text  # spikes land in the 95th percentile → honest verdict


def test_frame_stats_ok_and_window():
    f = FrameStats(window=10)
    for i in range(50):
        f.add(0.010, i / 30.0)
    assert len(f.costs) == 10
    assert f.summary().endswith("— OK")

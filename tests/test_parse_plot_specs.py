import time

from src import gradio_ui


def _elapsed_ok(func, *args, max_seconds=2, **kwargs):
    start = time.perf_counter()
    res = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    assert elapsed < max_seconds, f"Call took too long: {elapsed:.2f}s"
    return res


def test_parse_blank_and_simple_specs():
    # Get default plot params from the module helper
    _, _, default_plot = gradio_ui.get_default_params()

    # a) None / blank string -> []
    r = _elapsed_ok(gradio_ui.parse_plot_specs, None, default_plot)
    assert isinstance(r, list) and r == []

    r2 = _elapsed_ok(gradio_ui.parse_plot_specs, "", default_plot)
    assert isinstance(r2, list) and r2 == []

    # b) simple single-line key=value spec
    r3 = _elapsed_ok(gradio_ui.parse_plot_specs, "layers=DEFAULT", default_plot)
    assert isinstance(r3, list) and len(r3) == 1
    assert hasattr(r3[0], "plot_layers")

    # c) simple JSON spec
    r4 = _elapsed_ok(gradio_ui.parse_plot_specs, '{"layers":"ALL_COST"}', default_plot)
    assert isinstance(r4, list) and len(r4) == 1
    assert hasattr(r4[0], "plot_layers")


def test_parse_multiline_example():
    _, _, default_plot = gradio_ui.get_default_params()
    multi = """layers=DEFAULT
layers=DATA_SCATTER+OLS_PRED_LINEAR,x_max=200
{"layers":"ALL_COST","x_max":null}
"""
    r = _elapsed_ok(gradio_ui.parse_plot_specs, multi, default_plot, max_seconds=3)
    assert isinstance(r, list)
    # Expect three PlotParams entries
    assert len(r) == 3
    for p in r:
        assert hasattr(p, "plot_layers")

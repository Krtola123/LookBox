import os


def pytest_addoption(parser):
    parser.addoption("--update-golden", action="store_true",
                     help="Regenerate golden files instead of comparing (do this deliberately).")


def pytest_configure(config):
    if config.getoption("--update-golden"):
        os.environ["LOOKBOX_UPDATE_GOLDEN"] = "1"

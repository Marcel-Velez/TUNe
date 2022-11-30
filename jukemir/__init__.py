import pathlib
from os import environ as os_env
import os

LIB_DIR = pathlib.Path(__file__).resolve().parent

if "JUKEMIR_CACHE_DIR" in os_env:
    CACHE_DIR = pathlib.Path(os_env["JUKEMIR_CACHE_DIR"])
else:
    CACHE_DIR = pathlib.Path(os.getcwd())
CACHE_DIR = CACHE_DIR.resolve()

CACHE_DATASETS_DIR = pathlib.Path(CACHE_DIR, "data/processed")
CACHE_REPRESENTATIONS_DIR = pathlib.Path(CACHE_DIR, "data/representations")
CACHE_PROBES_DIR = pathlib.Path(CACHE_DIR, "data/probes")

# NOTE: This changes the test discovery pattern from "test*.py" (default) to "*test.py".
def load_tests(loader, standard_tests, pattern):
    package_tests = loader.discover(start_dir=LIB_DIR, pattern="*test.py")
    standard_tests.addTests(package_tests)
    return standard_tests

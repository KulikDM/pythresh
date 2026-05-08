import os
import sys
from collections import defaultdict

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

_progress = defaultdict(lambda: {"printed": False})
_printed_files = set()


def _parse_nodeid(nodeid):
    parts = nodeid.split("::")

    # File path → filename
    file_path = parts[0]
    file_name = os.path.basename(file_path)

    # Function name (no params)
    func_name = parts[-1].split("[")[0]

    return file_name, func_name


def pytest_runtest_logreport(report):
    if report.when != "call":
        return

    file_name, func_name = _parse_nodeid(report.nodeid)

    # Print file header once
    if file_name not in _printed_files:
        sys.stdout.write(f"\n\n{file_name}\n")
        sys.stdout.write("-" * len(file_name) + "\n")
        _printed_files.add(file_name)

    name = f"{file_name.replace('.py', '')}:{func_name}"

    # Print function header once
    if not _progress[name]["printed"]:
        sys.stdout.write(f"\n{name:<40}")
        _progress[name]["printed"] = True

    # Progress markers
    if report.passed:
        sys.stdout.write(".")
    elif report.failed:
        sys.stdout.write("F")
    elif report.skipped:
        sys.stdout.write("s")

    sys.stdout.flush()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    terminalreporter.write("\n\n")

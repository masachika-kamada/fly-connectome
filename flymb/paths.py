"""Every path in the project, resolved from the project you are working in.

Deliberately NOT resolved from the package location. flymb is installed
editable, so the package always lives in whichever checkout was installed - and
a second clone would then silently read and write the first one's data/. (That
is not hypothetical; it is exactly what happened before this was fixed.) A
non-editable install would be worse still, pointing at site-packages.

So the root is found by walking up from the working directory looking for
pyproject.toml. Run anything from anywhere inside a checkout and it operates on
that checkout. Outside one, it falls back to the installed package's own
repository, which keeps an ad-hoc `python -c "import flymb"` working.

FLYMB_ROOT overrides both, for the case where you want to point a run at a
different data directory on purpose.
"""
import os

_MARKER = "pyproject.toml"


def find_root(start=None):
    """Walk up from `start` (default: cwd) to the directory holding pyproject.toml."""
    env = os.environ.get("FLYMB_ROOT")
    if env:
        return os.path.abspath(env)
    here = os.path.abspath(start or os.getcwd())
    while True:
        if os.path.exists(os.path.join(here, _MARKER)):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ROOT = find_root()
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
WEB = os.path.join(ROOT, "web")
DIST = os.path.join(ROOT, "dist")

MB_NPZ = os.path.join(DATA, "mb_right.npz")
MB_META = os.path.join(DATA, "mb_meta.json")
ODOR_KC = os.path.join(DATA, "odor_kc.json")


def ensure(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

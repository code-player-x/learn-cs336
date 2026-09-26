"""CS336 teaching modules, with compatibility for Python's stdlib ``code`` API.

The historical package name shadows ``code.py`` when run from this repository.
Expose its public console API so pdb and interactive tooling keep working,
without changing existing imports such as ``code.model``.
"""

import importlib.util as _importlib_util
from pathlib import Path as _Path
import sysconfig as _sysconfig

_spec = _importlib_util.spec_from_file_location(
    '_cs336_stdlib_code', _Path(_sysconfig.get_path('stdlib')) / 'code.py'
)
_stdlib_code = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_stdlib_code)

__all__ = _stdlib_code.__all__
for _name in __all__:
    globals()[_name] = getattr(_stdlib_code, _name)

"""Read the corrected provider through the registered gateway, unmodified.

``src/data_access.py`` is one of the 140 protected files, and its ``_init_qlib``
hard-codes ``config['data']['qlib_provider_uri']``.  Editing either that file or
``config/config.yaml`` to reach the corrected bundle would break the identity D12
spends its whole preflight establishing, so the override lives here instead, as a
subclass that changes exactly one method.

Everything else -- the window marker, the sealed-path rule, the mandatory
``end_date`` -- is inherited untouched, because a repaired run must stay under the
same causality guards as the run it is compared against.

``qlib.init`` binds a process-wide singleton.  A process that has already read the
registered bundle would keep serving it no matter what this class says, so
``guard_process_provider`` refuses that case outright rather than letting a run
silently mix two providers.
"""
from __future__ import annotations

from pathlib import Path

from src.data_access import ResearchDataAccess
from src.d12_repo import D12Error


def guard_process_provider(uri):
    """Refuse to continue if this process already initialised a different provider."""
    import sys
    if 'qlib.config' not in sys.modules:
        return {'already_initialised': False, 'provider_uri': str(uri)}
    from qlib.config import C
    try:
        active = C.get('provider_uri', None)
    except Exception:  # pragma: no cover - qlib may not be configured yet
        active = None
    if not active:
        return {'already_initialised': False, 'provider_uri': str(uri)}
    resolved = {str(Path(str(v)).expanduser().resolve()) for v in
                (active.values() if isinstance(active, dict) else [active])}
    wanted = str(Path(uri).expanduser().resolve())
    if resolved and wanted not in resolved:
        raise D12Error(f'Qlib is already bound to {sorted(resolved)}; start a fresh process for {wanted}')
    return {'already_initialised': True, 'provider_uri': wanted}


class RepairedDataAccess(ResearchDataAccess):
    """``ResearchDataAccess`` served from a corrected provider built by D12."""

    def __init__(self, project_root, provider_uri, config_path=None):
        super().__init__(project_root, config_path)
        provider = Path(provider_uri).expanduser().resolve()
        for required in ['features', 'calendars/day.txt', 'instruments']:
            if not (provider / required).exists():
                raise D12Error(f'Corrected provider is missing {required}: {provider}')
        registered = (Path(self.config['data']['qlib_provider_uri']).expanduser())
        if registered.exists() and provider == registered.resolve():
            raise D12Error('RepairedDataAccess must not point back at the registered provider')
        self.provider_uri = str(provider)
        self.registered_provider_uri = str(registered)

    def _init_qlib(self) -> None:
        if self._qlib_ready:
            return
        guard_process_provider(self.provider_uri)
        import qlib
        qlib.init(provider_uri=self.provider_uri)
        self._qlib_ready = True


def repaired_config(config, *, processed_design_dir, processed_oos_dir):
    """A copy of the config whose processed inputs point at rebuilt caches.

    Only the two directories D11's ``run_research`` reads weights and tradability
    flags from move.  Windows, costs, constraints, the label and every model
    parameter stay exactly as registered, so the repaired R0 differs from the
    original in its inputs and in nothing else.
    """
    import copy
    repaired = copy.deepcopy(config)
    repaired['data']['processed_design_dir'] = str(processed_design_dir)
    repaired['d9']['v2']['processed_oos_dir'] = str(processed_oos_dir)
    return repaired

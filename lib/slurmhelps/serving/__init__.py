"""Preemptible Slurm model-serving controller (stdlib only).

The package is the P0 controller described in ``docs/serving-gateway.md``:

* :mod:`slurmhelps.serving.config` -- serving configuration and argv rendering;
* :mod:`slurmhelps.serving.state` -- atomic state + load-history store;
* :mod:`slurmhelps.serving.controller` -- submit / poll / wait / endpoint / down;
* :mod:`slurmhelps.serving.tunnel` -- ssh ``-L`` argv and a loopback TCP relay.

Nothing here reads a secret value: ``config.key_env`` is a *name* only, and no
API key is ever written to state or included in Slurm arguments.
"""

from __future__ import annotations

from .config import ServingConfig
from .controller import Controller
from .state import StateStore
from .tunnel import Relay, tunnel_argv

__all__ = ["Controller", "Relay", "ServingConfig", "StateStore", "tunnel_argv"]

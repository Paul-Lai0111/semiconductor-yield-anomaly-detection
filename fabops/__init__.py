"""Fab-operations layer on top of the frozen v3 study.

The v3 study (``smartfab``) answers "which supervised detector ranks failures
best under a 20% review budget?". This package asks the questions a fab
actually runs into after that:

* ``mspc``       - the industry-standard FDC baseline (PCA, Hotelling T^2, SPE)
                   with control limits and per-sensor contributions.
* ``drift``      - per-sensor distribution drift (PSI, KS, missingness).
* ``retraining`` - prequential simulation of frozen / periodic / drift-triggered
                   retraining under label delay.
* ``stability``  - how stable sensor rankings are across bootstraps and time.
* ``service``    - a deployable bundle, FastAPI batch-scoring API and dashboard.

Nothing here edits or re-selects the frozen v3 recipe; it is read from
``manufacturing_results/selection.json`` and verified by hash.
"""

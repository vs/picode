"""Shared pytest configuration for the whole test suite.

Must stay import-light: this module is loaded before any test module, which
is exactly what makes the MPS fallback setting below reliable.
"""

import os

# Apple's MPS backend has no `aten::grid_sampler_2d_backward`, so any test that
# backprops through an STN (trainer, GAN, PicoTrust v2, PicoGrain) dies without
# this fallback. Individual test modules used to set it themselves, which made
# the result depend on pytest's collection order — whichever module happened to
# initialise MPS first decided whether the rest of the suite passed. Setting it
# here applies it once, before any test module is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

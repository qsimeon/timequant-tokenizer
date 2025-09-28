"""HDF5 I/O utilities for tokenizer state persistence.

Provides functions to save/load tokenizer state to/from HDF5 files,
which is compatible with temporaldata's storage format.
"""

from typing import Union
from ..integrations.temporaldata_adapter import (
    save_tokenizer_to_hdf5,
    load_tokenizer_from_hdf5
)

# Re-export the functions for easier access
__all__ = ["save_tokenizer_to_hdf5", "load_tokenizer_from_hdf5"]
"""
registry.py

Purpose: Maps a filter's string name (as set in config.yaml) to its
actual class. This is the single place that knows about all available
filters — pipeline.py, benchmark.py, and the dashboard never need to
import filter classes directly; they just ask the registry for one by name.
"""

from src.filters.zscore_filter import ZScoreFilter
from src.filters.isolation_forest_filter import IsolationForestFilter
from src.filters.autoencoder_filter import AutoencoderFilter
from src.filters.snn_filter import SNNFilter


# Maps a plain string name -> the actual filter class (not yet instantiated)
FILTER_REGISTRY = {
    "zscore": ZScoreFilter,
    "isolation_forest": IsolationForestFilter,
    "autoencoder": AutoencoderFilter,
    "snn": SNNFilter,
}


def get_filter(name, **kwargs):
    """
    Looks up a filter by name and returns a ready-to-use instance.

    name:    string key, e.g. "zscore", "isolation_forest", "autoencoder", "snn"
    kwargs:  optional parameters passed through to the filter's constructor
             (e.g. window_size=50, threshold=3.0) — lets config.yaml override
             defaults without registry.py needing to know each filter's
             specific parameter names in advance.

    Raises a clear error if the name isn't recognized, instead of failing
    silently or crashing somewhere unrelated later.
    """
    filter_class = FILTER_REGISTRY.get(name)

    if filter_class is None:
        available = ", ".join(FILTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown filter name: '{name}'. Available filters: {available}"
        )

    return filter_class(**kwargs)


def list_available_filters():
    """
    Returns all registered filter names — used by the dashboard
    to populate the filter-selector dropdown dynamically, so it
    never goes out of sync with what's actually registered here.
    """
    return list(FILTER_REGISTRY.keys())

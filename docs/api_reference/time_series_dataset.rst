.. _time_series_dataset:

=================
TimeSeriesDataset
=================

This class extends a Python :code:`List` and represents multiple time series,
they can be univariate or multivariate (not implemented yet). Therefore, each
item in a TimeSeriesDataset has its own time index.

Thanks to its ability to handle multiple indices, this class provides a set of
method to go from raw data, with unknown characteristics (frequencies, start,
end, etc.), to clean data that is easy to process, model or analyze.

.. warning::
    The aim of a TimeSeriesDataset object is to be immutable

Constructor
-----------
.. currentmodule:: timeatlas.time_series_dataset

.. autosummary::
    :toctree:

    TimeSeriesDataset

Methods
-------
.. currentmodule:: timeatlas.time_series_dataset.TimeSeriesDataset

.. autosummary::
    :toctree:

    create
    append
    plot
    copy
    split_at
    split_in_chunks
    fill
    empty
    pad
    trim
    merge
    merge_by_label
    select_components_randomly
    select_components_by_percentage
    shuffle

Processing
----------
.. currentmodule:: timeatlas.time_series_dataset.TimeSeriesDataset

.. autosummary::
    :toctree:

    apply
    resample
    group_by
    interpolate
    normalize
    round
    sort
    regularize

Analysis
--------
.. currentmodule:: timeatlas.time_series_dataset.TimeSeriesDataset

.. autosummary::
    :toctree:

    min
    max
    mean
    median
    skewness
    kurtosis
    describe
    start
    end
    boundaries
    frequency
    time_deltas
    duration

I/O
---
.. currentmodule:: timeatlas.time_series_dataset.TimeSeriesDataset

.. autosummary::
    :toctree:

    to_text
    to_pickle
    to_array
    to_df

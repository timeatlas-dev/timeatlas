.. _time_series:

===========
Time Series
===========

This class represents a univariate time series. It allows for the storage of
time indexed values as well as metadata. The constructor guarantees the
monotonic increasing character of a time series.

.. warning::
    The aim of a TimeSeries object is to be immutable

Constructor
-----------
.. currentmodule:: timeatlas.time_series

.. autosummary::
    :toctree:

    TimeSeries

Methods
-------
.. currentmodule:: timeatlas.time_series.TimeSeries

.. autosummary::
    :toctree:

    create
    plot
    split_at
    split_in_chunks
    fill
    empty
    pad
    trim
    merge

Processing
----------
.. currentmodule:: timeatlas.time_series.TimeSeries

.. autosummary::
    :toctree:

    apply
    resample
    group_by
    interpolate
    normalize
    round
    sort

Analysis
--------
.. currentmodule:: timeatlas.time_series.TimeSeries

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
.. currentmodule:: timeatlas.time_series.TimeSeries

.. autosummary::
    :toctree:

    to_text
    to_array
    to_pickle
    to_darts
    to_df


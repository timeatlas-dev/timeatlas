.. _metadata:

========
Metadata
========

The :code:`Metadata` class extends the Python :code:`Dict` and allows for the
storage of any kind of metadata. The difference with a default :code:`Dict` is
that the current class recognizes the types (see :ref:`types`).

Constructor
-----------
.. currentmodule:: timeatlas.metadata

.. autosummary::
    :toctree:

    Metadata

Methods
-------
.. currentmodule:: timeatlas.metadata.Metadata

.. note::
    Here are the methods from :code:`Metadata` that differs (or have been
    overwritten) from a Python :code:`Dict`.

.. autosummary::
    :toctree:

    add
    to_json

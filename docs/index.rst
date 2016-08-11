=======
all2vec
=======

Library to store and process sets of high dimensional vectors

Installation
============

.. code::

    python -m pip install all2vec

Quickstart
==========

.. code:: python

    from all2vec import EntitySet
    import random

    f = 40  # num dimensions

    t = EntitySet(f)
    t.create_entity_type(entity_type_id=0, entity_type="entity0", ntrees=10,
                         metric="angular")
    t.create_entity_type(entity_type_id=1, entity_type="entity1", ntrees=10,
                         metric="euclidean")

    # populate entity 0 with 100 vectors
    for i in xrange(100):
        v = [random.gauss(0, 1) for z in xrange(f)]
        t.add_item(0, i, v)

    # populate entity 1 with 20 vectors
    for i in xrange(20):
        v = [random.gauss(0, 1) for z in xrange(f)]
        t.add_item(1, i, v)

    t.build()
    t.save("test_model_folder")

    # ...

    u = EntitySet.load("test_model_folder")

    # get 10 similar entities from entity1 that are similar to the first
    u.get_similar(
        entity_type="entity0",
        entity_id=0,
        match_type="entity1",
        num_similar=10,
        oversample=1,
        normalize=True
    )

Contents:

.. toctree::
   :maxdepth: 1

   api
   changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


from all2vec import EntitySet, FileGetter

def test_get_similar_vector():
    t = EntitySet(3)
    t.create_entity_type(entity_type_id=0, entity_type="type", ntrees=10,
                         metric="angular")
    t.add_item(0, 0, [0, 0, 1])
    t.add_item(0, 1, [0, 1, 0])
    t.add_item(0, 2, [1, 0, 0])
    t.build()

    sim = t.get_similar_vector([3, 2, 1], "type", 3, 1, False)
    assert [x['entity_id'] for x in sim] == [2, 1, 0]
    sim = t.get_similar_vector([1, 2, 3], "type", 3, 1.0, False)
    assert [x['entity_id'] for x in sim] == [0, 1, 2]
    sim = t.get_similar_vector([2, 0, 1], "type", 3, 2, True)
    assert [x['entity_id'] for x in sim] == [2, 0, 1]


def test_cross_entity_similar():
    t = EntitySet(3)
    t.create_entity_type(entity_type_id=0, entity_type="type0", ntrees=10,
                         metric="angular")
    t.create_entity_type(entity_type_id=1, entity_type="type1", ntrees=10,
                         metric="angular")


    t.add_item(0, 0, [1, 0, 1])
    t.add_item(0, 1, [0, 1, 1])
    t.add_item(1, 0, [0, 1, 0])
    t.add_item(1, 1, [1, 0, 0])
    t.build()

    sim = t.get_similar("type0", 0, "type1", 1, 1, False)
    assert sim[0]['entity_id'] == 1
    sim = t.get_similar("type0", 1, "type1", 1, 1, False)
    assert sim[0]['entity_id'] == 0
    

def test_cross_entity_scores():
    t = EntitySet(3)
    t.create_entity_type(entity_type_id=0, entity_type="type0", ntrees=10,
                         metric="angular")
    t.create_entity_type(entity_type_id=1, entity_type="type1", ntrees=10,
                         metric="angular")

    t.add_item(0, 0, [1, 2, 3])
    t.add_item(1, 0, [1, 0, 0])
    t.add_item(1, 1, [0, 1, 0])
    t.add_item(1, 2, [0, 0, 1])
    t.build()

    sim = t.get_scores('type0', 0, 'type1', [0,1,2])
    assert [x['score'] for x in sim] == [1.0, 2.0, 3.0]

def test_save_and_load(tmpdir):
    t = EntitySet(3)
    t.create_entity_type(entity_type_id=0, entity_type="type0", ntrees=10,
                         metric="angular")
    t.create_entity_type(entity_type_id=1, entity_type="type1", ntrees=10,
                         metric="angular")

    t.add_item(0, 0, [1, 2, 3])
    t.add_item(1, 0, [1, 0, 0])
    t.add_item(1, 1, [0, 1, 0])
    t.add_item(1, 2, [0, 0, 1])
    t.build()

    a_dir = str(tmpdir)
    t.save(a_dir)

    loaded = EntitySet.load(FileGetter(a_dir))

    sim = loaded.get_scores('type0', 0, 'type1', [0,1,2])
    assert [x['score'] for x in sim] == [1.0, 2.0, 3.0]
    assert loaded.get_size() == 2

def test_save_and_load_subset(tmpdir):
    t = EntitySet(3)
    t.create_entity_type(entity_type_id=0, entity_type="type0", ntrees=10,
                         metric="angular")
    t.create_entity_type(entity_type_id=1, entity_type="type1", ntrees=10,
                         metric="angular")

    t.add_item(0, 0, [0, 0, 1])
    t.add_item(0, 1, [0, 1, 0])
    t.add_item(0, 2, [1, 0, 0])
    t.build()

    a_dir = str(tmpdir)
    t.save(a_dir)

    loaded = EntitySet.load(FileGetter(a_dir), ['type0'])

    sim = loaded.get_similar_vector([3, 2, 1], "type0", 3, 1, False)
    assert [x['entity_id'] for x in sim] == [2, 1, 0]
    assert loaded.get_size() == 1

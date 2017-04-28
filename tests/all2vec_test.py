import boto3
import os
from os.path import basename, join

from moto import mock_s3
from all2vec import EntitySet, FileGetter, S3FileGetter

class TestAll2Vec:

    @mock_s3
    def __moto_setup(self):
        self.client = boto3.client('s3')

    
    def test_get_similar_vector(self):
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
    
    
    def test_cross_entity_similar(self):
        t = self._get_entity_set()
        sim = t.get_similar("type0", 0, "type1", 1, 1, False)
        assert sim[0]['entity_id'] == 1
        sim = t.get_similar("type0", 1, "type1", 1, 1, False)
        assert sim[0]['entity_id'] == 0
        
    
    def test_cross_entity_scores(self):
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
    
    def test_save_and_load(self, tmpdir):
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
    
    def test_save_and_load_subset(self, tmpdir):
        self._create_entity_set(tmpdir)
    
        loaded = EntitySet.load(FileGetter(str(tmpdir)), ['type0'])
    
        sim = loaded.get_similar_vector([3, 2, 1], "type0", 3, 1, False)
        assert [x['entity_id'] for x in sim] == [0, 1]
        assert loaded.get_size() == 1
    
    
    @mock_s3
    def test_load_from_s3(self, tmpdir):
        self.__moto_setup()
        bucket = 'test'
        key = 'all2vec'
        files = self._create_entity_set(tmpdir)
        self._save_to_s3(self.client, bucket, key, files)

        s3_getter = S3FileGetter(bucket, key, str(tmpdir), self.client)
        loaded = EntitySet.load(s3_getter)

        sim = loaded.get_similar("type0", 0, "type1", 1, 1, False)
        assert sim[0]['entity_id'] == 1
        sim = loaded.get_similar("type0", 1, "type1", 1, 1, False)
        assert sim[0]['entity_id'] == 0
    
    
    def _save_to_s3(self, client, b, k, files):
        client.create_bucket(Bucket=b)
        for f in files:
            name = basename(f)
            r = client.upload_file(f, b, join(k, name))


    def _get_entity_set(self):
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
        return t


    def _create_entity_set(self, tmpdir):
        t = self._get_entity_set()
        a_dir = str(tmpdir)
        return t.save(a_dir)

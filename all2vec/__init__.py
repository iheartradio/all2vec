"""Set of entity types storing high dimensional vectors."""

from collections import namedtuple
import logging
import os
import json

import boto3
import numpy as np
import dill
from annoy import AnnoyIndex

__all__ = ('EntitySet',)

logging.basicConfig(level=logging.INFO)

EntityVector = namedtuple("EntityVector", ["entity_id", "vector"])


class EntityType(object):
    """Convenience wrapper around Annoy.

    More generally a way to collect vectors within the same entity type and
    quickly find similar vectors.

    * Helps deal with non-contiguous ids through an id map.
    * Checks for 0 vectors before returning matches.
    """

    def __init__(self, nfactor, ntrees, metric='angular',
                 entity_type_id=None, entity_type=None):
        """Initialize EntityType."""
        # metadata
        self._nfactor = nfactor
        self._metric = metric
        # object is accessed using this id. e.g. 'user'
        self._entity_type = entity_type
        # data is loaded in using this id. This can be more compact than the
        # entity_type, depending on the data source
        self._entity_type_id = entity_type_id
        self._ntrees = ntrees

        # data
        self._ann_obj = AnnoyIndex(nfactor, metric)
        # maps entity id to internal representation of id
        self._ann_map = {}
        # maps internal representation of id to entity id
        self._ann_map_inv = {}
        self._nitems = 0

    def add_item(self, entity_id, factors):
        """Add item, populating id map."""
        if entity_id in self._ann_map:
            raise ValueError('Duplicate entity: type = {0}, id = {1}'.format(
                self._entity_type, entity_id))
        self._ann_obj.add_item(self._nitems, factors)
        self._ann_map[entity_id] = self._nitems
        self._nitems = self._nitems + 1

    def build(self, verbose=False):
        """Build annoy model, create invert dictionary for future lookups."""
        self._ann_obj.verbose(verbose)
        self._ann_obj.build(self._ntrees)
        # this is only necessary after build, so we'll create it here
        self._ann_map_inv = {v: k for k, v in self._ann_map.items()}

    def get_nns_by_vector(self, vec, n, search_k):
        """Get nearest neighbors from an input vector."""
        nns = self._ann_obj.get_nns_by_vector(vec, n, search_k)
        return [self._ann_map_inv[x] for x in nns]

    def get_item_vector(self, entity_id):
        """Get a vector for an entity."""
        if entity_id in self._ann_map:
            return self._ann_obj.get_item_vector(self._ann_map[entity_id])
        else:
            return []

    def __iter__(self):
        """Iterate over object, return (entity_id, vector) tuples."""
        return (EntityVector(
                    entity_id=entity_id,
                    vector=self.get_item_vector(entity_id)
                ) for entity_id in self._ann_map.keys())

    def get_nfactor(self):
        return self._nfactor

    def load(self, pkl, filepath):
        entity_type = pkl.get_entity_type(self._entity_type_id)
        self.__dict__ = entity_type.__dict__
        # initialize index
        self._ann_obj = AnnoyIndex(pkl.get_nfactor(), entity_type._metric)
        # mmap the file
        self._ann_obj.load(filepath)


class EntitySet(object):
    """Organize many EntityType instances for different entity types.

    Expose a way to find similarities and matches within and between entities.
    """

    def __init__(self, nfactor):
        """Initialize EntitySet."""
        self._is_built = False
        self._nfactor = nfactor
        self._annoy_objects = {}
        self._entity_id_map = {}

    def get_size(self):
        """Return size of set."""
        return len(self._annoy_objects)

    def create_entity_type(self, entity_type_id, entity_type,
                           ntrees, metric='angular'):
        """Create an entity type and populate its metadata."""
        self._annoy_objects[entity_type] = EntityType(
            self._nfactor, ntrees, metric, entity_type_id, entity_type)
        self._entity_id_map[entity_type_id] = entity_type

    def get_entity_type(self, entity_type_id):
        """Get entity-type for specified id."""
        entity_type = self._entity_id_map[entity_type_id]
        return self._annoy_objects[entity_type]

    def get_nfactor(self):
        """Get n-factor."""
        return self._nfactor

    def add_item(self, entity_type_id, entity_id, factors):
        """Wrap annoy_object add_item."""
        self._annoy_objects[self._entity_id_map[entity_type_id]].add_item(
            entity_id, factors)

    def build(self, parallel=False, verbose=False):
        """Loop through all annoy objects and build in order."""
        # todo: this could be parallelized
        if self._is_built:
            return

        for annoy_object in self._annoy_objects.values():
            logging.info("Starting build for entity {} - {}...".format(
                annoy_object._entity_type_id,
                annoy_object._entity_type,
            ))
            annoy_object.build(verbose)
            logging.info("Done build for entity {} - {}".format(
                annoy_object._entity_type_id,
                annoy_object._entity_type,
            ))
        self._is_built = True

    def get_vector(self, entity_type, entity_id):
        """Wrap annoy_object get_vector."""
        return self._annoy_objects[entity_type].get_item_vector(entity_id)

    @staticmethod
    def vec_score(vec1, vec2, normalize=False):
        """Return score between two vectors.

        Optionally normalize the vectors to euclidean length of 1.
        """
        if normalize:
            vec1 = vec1/np.linalg.norm(vec1)
            vec2 = vec2/np.linalg.norm(vec2)
        return np.inner(vec1, vec2)

    def get_similar_vector(self, match_vector, match_type, num_similar,
                           oversample, normalize):
        """Get similar items from an input vector."""
        if not match_vector:
            return []

        # search_k defaults to n * n_trees in Annoy - multiply by oversample
        # don't allow oversample to go below 1, this causes errors in Annoy
        if oversample < 1:
            oversample = 1
        search_k = int(num_similar * self._annoy_objects[match_type]._ntrees *
                       oversample)

        similar_items = self._annoy_objects[match_type].get_nns_by_vector(
            match_vector, num_similar, search_k)
        # compute inner products, and sort
        scores = self.get_scores_vector(
            match_vector, match_type, similar_items, normalize)
        scores = sorted(scores, key=lambda k: k['score'], reverse=True)
        return scores[:num_similar]

    def get_similar(self, entity_type, entity_id, match_type, num_similar,
                    oversample=1, normalize=False):
        """Get similar items from an item."""
        match_vector = self.get_vector(entity_type, entity_id)
        return self.get_similar_vector(
            match_vector, match_type, num_similar, oversample, normalize)

    def get_scores_vector(self, vector, match_type, match_id_array, normalize):
        """Score a vector and an array of matching entities."""
        if not vector:
            return []
        scores = []
        for i in match_id_array:
            match_vector = self.get_vector(match_type, i)
            if match_vector:
                scores.append({
                    'entity_id': i,
                    'score': self.vec_score(vector, match_vector, normalize)})
            else:
                scores.append({'entity_id': i, 'score': None})
        return scores

    def get_scores(self, entity_type, entity_id, match_type, match_id_array,
                   normalize=False):
        """Score an item and an array of matching entities."""
        vector = self._annoy_objects[entity_type].get_item_vector(entity_id)
        return self.get_scores_vector(
            vector, match_type, match_id_array, normalize)

    def get_similar_threshold(self, entity_type, entity_id, match_type,
                              threshold, n_try=10):
        """Get similar items within a score threshold to a query item."""
        scores = self.get_similar(
            entity_type, entity_id, match_type, n_try)
        if not scores:
            return scores
        if np.any([item["score"] < threshold for item in scores]):
            return [item for item in scores if item["score"] > threshold]
        else:
            return self.get_similar_threshold(
                entity_type, entity_id, match_type, threshold, n_try*10)

    def get_entity_types(self):
        """Helper for getting entity types object."""
        return [{
            'num_entities': etype._ann_obj.get_n_items(),
            'entity_type_id': etype._entity_type_id,
            'entity_type': etype._entity_type,
            'metric':  etype._metric,
            'num_trees': etype._ntrees
        } for etype in self._annoy_objects.values()]

    def save(self, folder):
        """Save object and return corresponding files."""
        if not os.path.exists(folder):
            os.makedirs(folder)
        files = []
        # annoy objects can't be pickled, so save these separately
        for k, v in self._annoy_objects.items():
            annoy_filepath = os.path.join(folder, '{}.ann'.format(k))
            v._ann_obj.save(annoy_filepath)
            files.append(annoy_filepath)
        pickle_filepath = os.path.join(folder, 'object.pickle')
        with open(pickle_filepath, 'wb') as handle:
            dill.dump(self, handle)
        files.append(pickle_filepath)

        # write entity types
        enttypes = self.get_entity_types()

        info_file = os.path.join(folder, 'entity_info.json')
        with open(info_file, 'w') as handle:
            json.dump(enttypes, handle)
        files.append(info_file)
        return files

    def load_entities(self, entities, file_getter):
        """Load underlying entities."""
        for k in entities:
            annoy_filepath = file_getter.get_file_path('{}.ann'.format(k))
            try:
                self._annoy_objects[k].load(self,
                                            annoy_filepath)
            except IOError as e:
                raise IOError(
                    "Error: cannot load file {0}, which was built "
                    "with the model. '{1}'".format(annoy_filepath, e)
                )

    @classmethod
    def load_pickle(cls, file_getter):
        """Load pickled EntitySet."""
        # grab pickle, replace the models with the mmapped saved annoy objects
        with file_getter.get_binary_file('object.pickle') as f:
            unpickled_class = dill.load(f)
        return unpickled_class

    @classmethod
    def load_entity_info(cls, file_getter):
        """Load entity info from file."""
        # Grab entity_info.json to ensure all the entity types exist and are
        # the right size
        with file_getter.get_file('entity_info.json') as f:
            enttype_info = json.load(f)
        return enttype_info

    @classmethod
    def check_load(cls, enttype_info, unpickled_class):
        """Check loaded info against unpickled class."""
        # Check that sizes match up - this can be used to protect against files
        # overwriting each other during a transfer, for example
        enttype_sizes = {
            enttype['entity_type']: enttype['num_entities']
            for enttype in enttype_info
        }
        for k, annoy_object in unpickled_class._annoy_objects.items():
            if enttype_sizes[annoy_object._entity_type] != annoy_object._nitems:  # NOQA
                raise ValueError(
                    'Entity type {0} should have size {1} '
                    'but actually has size {2}'.format(
                        annoy_object._entity_type,
                        annoy_object._nitems,
                        enttype_sizes[annoy_object._entity_type],
                    )
                )

        # Check that every entity type in entity_info.json also exists in the
        # loaded model
        for enttype in enttype_info:
            if enttype['entity_type'] not in unpickled_class._annoy_objects:
                raise ValueError(
                    'Entity type {0} exists in model_info.json '
                    'but was not loaded'.format(enttype['entity_type'])
                )

    @classmethod
    def load(cls, file_getter_or_folder, entities=None):
        """Load object.

        file_getter_or_folder -- a FileGetter or a folder string
        entities -- optional subset of entities to load
        """
        # to preserve backwards compatibility
        if isinstance(file_getter_or_folder, str):
            file_getter = FileGetter(file_getter_or_folder)
        else:
            file_getter = file_getter_or_folder

        unpickled_class = cls.load_pickle(file_getter)

        enttype_info = cls.load_entity_info(file_getter)
        if entities is None:
            entities = unpickled_class._annoy_objects
        else:
            # filter unwanted entities
            enttype_info = [v for v in enttype_info
                            if v['entity_type'] in entities]
            unpickled_class._annoy_objects = {
                k: unpickled_class._annoy_objects[k]
                for k in unpickled_class._annoy_objects
                if k in entities
            }

        # annoy objects can't be pickled, so load these after pickle is loaded
        unpickled_class.load_entities(entities, file_getter)

        cls.check_load(enttype_info, unpickled_class)
        return unpickled_class


class FileGetter(object):
    """Helper class used in EntitySet load methods."""

    def __init__(self, folder=None):
        self.folder = folder

    def get_file_path(self, file_name):
        return os.path.join(self.folder, file_name)

    def get_file(self, file_name):
        """Return file object."""
        return open(self.get_file_path(file_name))

    def get_binary_file(self, file_name):
        """Return binary file object."""
        return open(self.get_file_path(file_name), 'rb')


class SparkFileGetter(FileGetter):

    def __init__(self, sparkfiles):
        """Initialize SparkFileGetter.

        sparkfiles -- SparkFiles (imported from pyspark), must have
        sc.addPyFile(path + file) prior to SparkFiles.get(file).
        """
        self.sparkfiles = sparkfiles
        super(SparkFileGetter, self).__init__()

    def get_file_path(self, file_name):
        return self.sparkfiles.get(file_name)


class S3FileGetter(FileGetter):

    def __init__(self,
                 s3_bucket,
                 s3_prefix,
                 my_dir,
                 client=boto3.client('s3')):
        """Pull model files from S3 before loading.

        Assumes user has configured boto3.

        :param str s3_bucket:
        :param str s3_prefix: prefix of all model files in S3
        :param str my_dir: directory for saving files from S3
        :param client: must have or be able to access AWS credentials
        """
        self.bucket = s3_bucket
        self.prefix = s3_prefix
        self._client = client
        self.model_dir = my_dir

        super(S3FileGetter, self).__init__(folder=self.model_dir)

    def get_file_path(self, file_name):
        file_path = os.path.join(self.folder, file_name)
        self._client.download_file(self.bucket,
                                   os.path.join(self.prefix, file_name),
                                   file_path)
        return file_path

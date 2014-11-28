from __future__ import division
from collections import Counter
import json
import os
import math
import errno
import operator
import numpy as np


class TfidfCorpus(object):
    """Managing feature datasets in an inverted index. Useful for NLP and machine learning applications.
    Corpus takes form:

    To initialise a new TfidfCorpus instance:
    >>> corpus = TfidfCorpus()

    By default the corpus will save to 'tfidf_corpus/corpus.json'. You can specify an existing file to load
    or a specific save path as follows:
    >>> corpus = TfidfCorpus(corpus_path='path/to/corpus.json')

    Data Attributes:
        corpus_path (str): save/load path of the corpus
        document_list (list): list of strings indicating the documents stored in the corpus
        document_lengths (dict): sum of word frequencies contained in each document, takes the form:
            {
                "document1": int,
                "document2": int,
                ...
            }
        corpus (dict): dict of Counters that takes the form:
            {
                "term1": {
                    "document1": int,
                    "document2": int
                },
                "term2": {
                    "document1": int,
                    "document2": int,
                },
                ...
            }

    Methods:
        load_corpus
        get_corpus_path
        get_document_list
        add_document
        get_document
        delete_document
        append_document
        get_idf
        get_tfidf
        get_document_tfidfs
        get_top_terms
        build_feature_array
    """

    def __init__(self, corpus_path='corpus.json'):

        if not corpus_path.lower().endswith('.json'):
            raise Exception('corpus_path provided is not a valid JSON file.')
        make_path(corpus_path)

        self.corpus_path = corpus_path
        self.document_list = list()
        self.document_lengths = dict()
        self.corpus = dict()

        if os.path.isfile(corpus_path):
            self.load_corpus()

    def save_corpus(self):
        """Save the corpus to a JSON file at the path specified in self.corpus_path."""
        with open(self.corpus_path, 'wb') as save_file:
            json.dump(
                {
                    'document_list': self.document_list,
                    'document_lengths': self.document_lengths,
                    'corpus': self.corpus
                },
                save_file
            )

    def load_corpus(self):
        """Load the corpus from a JSON file. File path defined in self.corpus_path."""
        with open(self.corpus_path, 'rb') as load_file:
            data = json.load(load_file)

        try:
            self.document_list = data['document_list']
            self.document_lengths = data['document_lengths']
            self.corpus = data['corpus']

            # Make sure that frequency dicts in corpus are Counter objects
            for term in self.corpus.iterkeys():
                self.corpus[term] = Counter(self.corpus[term])
        except KeyError as err:
            print 'Provided file does not have expected structure'
            raise err

    def get_corpus_path(self):
        return self.corpus_path

    def set_corpus_path(self, path):
        if not path.lower().endswith('.json'):
            raise Exception('Corpus path must be a JSON file (.json extension).')
        self.corpus_path = path

    def get_document_list(self):
        return self.document_list

    def get_vocabulary(self):
        """Return the full list of terms in the corpus."""
        return self.corpus.keys()

    def get_document(self, document_name):
        """Retrieve a document from the corpus."""
        if document_name not in self.document_list:
            raise Exception("No document with name '{0}' found in corpus".format(document_name))
        return Counter({
            term: freqs[document_name] for term, freqs in self.corpus.iteritems() if freqs.get(document_name, 0)
        })

    def add_document(self, document, document_name):
        """Load a document into the corpus.

        :param document: takes the form {'term1': freq1, 'term2', freq2, ...}
        :param document_name: string which uniquely identifies the document
        """
        if document_name in self.document_list:
            print "Document with name '{0}' already exists in corpus." \
                  "Do you wish to replace it?".format(document_name)
            while True:
                replace_doc = raw_input("Response (y/n): ")
                if replace_doc in ['y', 'yes', 'ye']:
                    self.delete_document(document_name)
                    break
                elif replace_doc in ['n', 'no']:
                    return
                else:
                    print 'Could not interpret response. Try again.'

        for term, freq in document.iteritems():
            if not self.corpus.get(term, False):
                self.corpus[term] = Counter()

            self.corpus[term][document_name] = freq

        self.document_list.append(document_name)
        self.document_lengths[document_name] = sum(document.itervalues())

    def delete_document(self, document_name):
        """Delete a document from the corpus.

        :param document_name: string indicating document's name in the corpus - should exist in self.document_list
        """
        if document_name not in self.document_list:
            return
        [freqs.pop(document_name) for term, freqs in self.corpus.iteritems() if freqs.get(document_name, 0)]
        self.document_list.remove(document_name)
        self.document_lengths.pop(document_name)

    def append_document(self, document, document_name):
        """Add new counts to an existing document. If the document doesn't exist in the corpus then it is added.

        :param document: dict or Counter of word counts, e.g. {'i': 1, 'like': 2, 'cheese': 1}
        :param document_name: string indicating document's name in the corpus - should exist in self.document_list
        """
        if document_name not in self.document_list:
            self.add_document(document, document_name)
        else:
            for term, freq in document.iteritems():
                if not self.corpus.get(term, False):
                    self.corpus[term] = Counter()

                self.corpus[term][document_name] += freq

        self.document_lengths[document_name] += sum(document.itervalues())

    def get_idf(self, term):
        """Get inverse document frequency of a given term in the corpus."""
        num_documents = len(self.document_list)
        docs_containing_term = len(self.corpus[term])
        return math.log(num_documents / (1 + docs_containing_term))

    def get_tfidf(self, term, document_name):
        """Get tf-idf score given a term and document in the corpus."""
        tf = self.corpus[term].get(document_name, '') / self.document_lengths[document_name]
        idf = self.get_idf(term)
        return tf * idf

    def get_document_tfidfs(self, document_name, l2_norm=True):
        """Get tf-idf scores for all terms in a document.

        :param document_name: string indicating document's name in the corpus - should exist in self.document_list
        :param l2_norm: if True, applies Euclidean normalization to tf-idf scores of the document
        :return: Counter of tf-idf scores for each term
        """
        tfidfs = {
            term: self.get_tfidf(term, document_name) for term, freq in self.corpus.iteritems()
            if freq.get(document_name, '')
        }

        if l2_norm:
            normalization = np.linalg.norm(tfidfs.values(), axis=0)
            for key, value in tfidfs.items():
                tfidfs[key] = value / normalization

        return Counter(tfidfs)

    def get_top_terms(self, document_name, num_terms=30):
        """Get the top terms for a given document by tf-idf score.

        :param document_name: string indicating document's name in the corpus - should exist in self.document_list
        :param num_terms: number of top terms to return (30 by default)
        :return: dict of top terms and their corresponding tf-idf scores
        """
        tfidfs = self.get_document_tfidfs(document_name)
        sorted_tfidfs = sorted(tfidfs.items(), key=operator.itemgetter(1), reverse=True)
        return dict(sorted_tfidfs[:num_terms])

    def build_feature_array(self, tfidf=True):
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.feature_extraction.text import TfidfTransformer

        index = [self.get_document(document) for document in self.document_list]
        vectorizer = DictVectorizer()
        vectorizer.fit_transform(index)

        if tfidf:
            vectorizer = TfidfTransformer().fit_transform(vectorizer)

        return vectorizer.toarray()

    def count_words_from_list(self, document_name, word_list, normalize=True):
        document = self.get_document(document_name)
        word_counts = [document[word] for word in word_list]
        total_count = sum(word_counts)
        if normalize:
            total_count /= self.document_lengths[document_name]

        return total_count

    def get_mean_word_length(self, document_name, upper_limit=12):
        document = self.get_document(document_name)
        return sum([len(term) * freq for term, freq in document.iteritems()
                    if len(term) <= upper_limit]) / sum(document.itervalues())


def make_path(path):
    """Check if path exists. If it doesn't, create the necessary folders."""

    # Remove file name from path
    base_name = os.path.basename(path)
    if '.' in base_name:
        path = path[:-len(base_name)]

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    return path
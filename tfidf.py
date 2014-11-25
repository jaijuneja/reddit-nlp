from collections import Counter
import json
import os
import math


class TfIdfCorpus(object):
    """
    Corpus takes form:

    {
    "term1": {"document1": tf, "document2": tf},
    "term2": {"document1": tf, "document2": tf,
    }
    """

    def __init__(self, corpus_path='tfidf_corpus'):
        corpus_path += 'corpus.json' if corpus_path.endswith('/') else corpus_path
        corpus_path += '.json' if not corpus_path.endswith('.json') else corpus_path
        self.corpus_path = corpus_path

        self.num_docs = 0
        self.num_docs_per_term = dict()
        self.document_lengths = dict()
        self.document_list = list()
        self.corpus = dict()

        if os.path.isfile(corpus_path):
            self.load_corpus()

    def save_corpus(self):
        with open(self.corpus_path, 'wb') as save_file:
            json.dump(
                {
                    'num_docs': self.num_docs,
                    'num_docs_per_term': self.num_docs_per_term,
                    'document_lengths': self.document_lengths,
                    'document_list': self.document_list,
                    'corpus': self.corpus
                },
                save_file
            )

    def load_corpus(self):
        with open(self.corpus_path, 'rb') as load_file:
            data = json.load(load_file)

        try:
            self.num_docs = data['num_docs']
            self.num_docs_per_term = data['num_docs_per_term']
            self.document_lengths = data['document_lengths']
            self.document_list = data['document_list']
            self.corpus = data['corpus']
        except KeyError as err:
            print 'Provided file does not have expected structure'
            raise err

    def add_document(self, document, document_name):
        """
        Load a document into the corpus.
        :param document: takes the form {'term1': freq1, 'term2', freq2, ...}
        :param document_name: string which uniquely identifies the document
        :return:
        """
        if document_name in self.document_list:
            raise Exception("Document with name '{0}' already exists in corpus".format(document_name))

        for term, freq in document.iteritems():
            if not self.corpus.get(term, 1):
                self.corpus[term] = Counter()

            self.corpus[term][document_name] = freq

    def get_document(self, document_name):
        if document_name not in self.document_list:
            raise Exception("No document with name '{0}' found in corpus".format(document_name))
        return Counter({
            term: freqs[document_name] for term, freqs in self.corpus.iteritems() if freqs.get(document_name, 0)
        })

    def delete_document(self, document_name):
        if document_name not in self.document_list:
            return
        [freqs.pop(document_name) for term, freqs in self.corpus.iteritems() if freqs.get(document_name, 0)]

    def get_idf(self, term):
        num_documents = len(self.document_list)
        docs_containing_term = len(self.corpus[term])
        return math.log(num_documents / (1 + docs_containing_term))

    def get_tfidf(self, term, document_name):
        tf = self.corpus[term].get(document_name, '') / self.document_lengths[document_name]
        idf = self.get_idf(term)
        return tf * idf

    def get_document_tfidfs(self, document_name):
        return Counter({
            term: self.get_tfidf(term, document_name) for term, freq in self.corpus.iteritems()
            if freq.get(document_name, '')
        })

from collections import Counter
import json
import os
import math
import errno
import operator


class TfidfCorpus(object):
    """
    Corpus takes form:

    {
    "term1": {"document1": tf, "document2": tf},
    "term2": {"document1": tf, "document2": tf,
    }
    """

    def __init__(self, corpus_path='tfidf_corpus'):
        corpus_path = corpus_path + 'corpus.json' if corpus_path.endswith('/') else corpus_path
        corpus_path = corpus_path + '.json' if not corpus_path.endswith('.json') else corpus_path
        make_path(corpus_path)
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

        self.document_list.append(document_name)

        for term, freq in document.iteritems():
            if not self.corpus.get(term, False):
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
        self.document_list.remove(document_name)

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

    def get_top_terms(self, document_name, num_terms=30):
        tfidfs = self.get_document_tfidfs(document_name)
        sorted_tfidfs = sorted(tfidfs.items(), key=operator.itemgetter(1))
        return Counter(sorted_tfidfs[:num_terms])


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
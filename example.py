import config
from redditnlp.wordcount import RedditWordCounter
from redditnlp.tfidf import TfidfCorpus
import requests
import os
from collections import deque

if __name__ == '__main__':

    def get_subreddit_vocabularies():
        # Initialise Reddit word counter instance
        reddit_counter = RedditWordCounter(config.USERNAME)

        # Initialise tf-idf corpus instance
        corpus_path = os.path.join(config.SAVE_DIR, 'corpus.json')
        comment_corpus = TfidfCorpus(corpus_path)

        # Extract the vocabulary for each of the subreddits specified
        subreddit_queue = deque([subreddit for subreddit in config.SUBREDDITS])
        while len(subreddit_queue) > 0:
            subreddit = subreddit_queue.popleft()

            try:
                vocabulary = reddit_counter.subreddit_comments(subreddit, limit=config.COMMENTS_PER_SUBREDDIT)
            except requests.exceptions.HTTPError as err:
                print err
                # Add subreddit back into queue
                subreddit_queue.append(subreddit)
                continue

            comment_corpus.add_document(vocabulary, subreddit)
            comment_corpus.save_corpus()

        return comment_corpus, corpus_path

    def save_subreddit_top_terms(corpus):
        # Save the top terms for each subreddit in a text file
        save_path = os.path.join(config.SAVE_DIR, 'top_words.txt')
        for document in corpus.get_document_list():
            top_terms = corpus.get_top_terms(document)
            top_terms = sorted(top_terms.items(), key=lambda x: x[1], reverse=True)
            with open(save_path, 'ab') as f:
                f.write(document.encode('utf-8') + '\n' +
                        '\n'.join(['{0}, {1}'.format(term.encode('utf-8'), weight) for term, weight in top_terms])
                        + '\n\n')

        return save_path

    def get_swearword_counts(corpus):
        with open('redditnlp/words/swearwords_english.txt', 'rb') as f:
            swearwords = [word.strip('\n') for word in f.readlines()]

        swearword_counts = dict()
        for document in corpus.get_document_list():
            swearword_counts[document] = corpus.count_words_from_list(document, swearwords)
        return swearword_counts

    def get_vocabulary_sophistication(corpus):
        mean_word_lengths = dict()
        for document in corpus.get_document_list():
            mean_word_lengths[document] = corpus.get_mean_word_length(document)
        return mean_word_lengths

    comment_corpus, corpus_path = get_subreddit_vocabularies()
    print 'TF-IDF corpus saved to %s' % corpus_path
    #
    # top_terms_path = save_subreddit_top_terms(comment_corpus)
    # print 'Top terms saved to %s' % corpus_path

    # swearword_frequency = get_swearword_counts(comment_corpus)
    # print 'Normalized swearword frequency:'
    # for subreddit, frequency in swearword_frequency.items():
    #     print '%s, %s' % (subreddit, frequency)
    #
    # print '\nAverage word length by subreddit:'
    # word_lengths = get_vocabulary_sophistication(comment_corpus)
    # for subreddit, frequency in word_lengths.items():
    #     print '%s, %s' % (subreddit, frequency)
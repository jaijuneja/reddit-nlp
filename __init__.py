from redditcounter import RedditWordCounter
from tfidf import TfidfCorpus

if __name__ == '__main__':
    reddit_counter = RedditWordCounter()
    tfidf_corpus = TfidfCorpus('tfidf_corpus/')
    vocab = reddit_counter.subreddit_comments('funny')
    tfidf_corpus.add_document(vocab, 'funny')
    tfidf_corpus.save_corpus()
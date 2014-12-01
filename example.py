from redditnlp import RedditWordCounter, TfidfCorpus
import requests
import os
from collections import deque

######################
# SETTINGS (EDIT THIS)
######################

USERNAME = 'your_username'  # Change this to your username
SAVE_DIR = 'tfidf_corpus'
CORPUS_FILE = 'corpus.json'
COMMENTS_PER_SUBREDDIT = 1000
SUBREDDITS = [
    'funny', 'pics', 'AskReddit', 'todayilearned', 'worldnews',
    'science', 'blog', 'IAmA', 'videos', 'gaming',
    'movies', 'Music', 'aww', 'technology', 'bestof',
    'WTF', 'AdviceAnimals', 'news', 'gifs', 'askscience',
    'explainlikeimfive', 'EarthPorn', 'books', 'television', 'politics'
]

###########################
# VOCABULARY ANALYTICS DEMO
###########################


def get_subreddit_vocabularies():
    # Initialise Reddit word counter instance
    reddit_counter = RedditWordCounter(USERNAME)

    # Initialise tf-idf corpus instance
    corpus_path = os.path.join(SAVE_DIR, CORPUS_FILE)
    comment_corpus = TfidfCorpus(corpus_path)

    # Extract the vocabulary for each of the subreddits specified
    subreddit_queue = deque([subreddit for subreddit in SUBREDDITS])
    while len(subreddit_queue) > 0:
        subreddit = subreddit_queue.popleft()

        try:
            vocabulary = reddit_counter.subreddit_comments(subreddit, limit=COMMENTS_PER_SUBREDDIT)
        except requests.exceptions.HTTPError as err:
            print err
            # Add subreddit back into queue
            subreddit_queue.append(subreddit)
            continue

        comment_corpus.add_document(vocabulary, subreddit)
        comment_corpus.save()

    return comment_corpus, corpus_path


def save_subreddit_top_terms(corpus):
    # Save the top terms for each subreddit in a text file
    save_path = os.path.join(SAVE_DIR, 'top_words.txt')
    for document in corpus.get_document_list():
        top_terms = corpus.get_top_terms(document, num_terms=50)
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

# Extract their word counts
corpus, corpus_path = get_subreddit_vocabularies()
print 'TF-IDF corpus saved to %s' % corpus_path

# Get the top words by subreddit
top_terms_path = save_subreddit_top_terms(corpus)
print 'Top terms saved to %s' % corpus_path

# Get the swearword frequency
swearword_frequency = get_swearword_counts(corpus)
print 'Normalized swearword frequency:'
for subreddit, frequency in swearword_frequency.items():
    print '%s, %s' % (subreddit, frequency)

# Get the average word length
print '\nAverage word length by subreddit:'
word_lengths = get_vocabulary_sophistication(corpus)
for subreddit, frequency in word_lengths.items():
    print '%s, %s' % (subreddit, frequency)

#######################
# MACHINE LEARNING DEMO
#######################

# Collect the comments for a particular user and determine which subreddit their comments best match up with
counter = RedditWordCounter(USERNAME)
corpus = TfidfCorpus(os.path.join(SAVE_DIR, CORPUS_FILE))

user_comments = counter.user_comments('way_fairer')
corpus.train_classifier(classifier_type='LinearSVC', tfidf=True)
print corpus.classify_document(user_comments)
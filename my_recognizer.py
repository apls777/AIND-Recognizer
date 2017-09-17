import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    Xlengths = list(test_set.get_all_Xlengths().values())

    # get words' scores
    for test_X, test_lengths in Xlengths:
        scores = {}
        for word in models:
            model = models[word]
            if model is not None:
                try:
                    scores[word] = model.score(test_X, test_lengths)
                except ValueError:
                    continue

        probabilities.append(scores)

    # choose the best score for each word
    for scores in probabilities:
        guesses.append(max(scores, key=scores.get))

    return probabilities, guesses

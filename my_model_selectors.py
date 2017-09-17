import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        N, num_features = self.X.shape
        logN = np.log(N)

        best_score = best_model = None

        for num_components in range(self.min_n_components, self.max_n_components + 1):
            # get model
            model = self.base_model(num_components)
            if model is None:
                continue

            # get model score
            try:
                logL = model.score(self.X, self.lengths)
            except ValueError:
                continue

            # calculate BIC score
            p = (num_components ** 2) + 2 * num_features * num_components - 1
            score = -2 * logL + p * logN

            # update the best score
            if best_score is None or score < best_score:
                best_score = score
                best_model = model

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = best_model = None

        for num_components in range(self.min_n_components, self.max_n_components + 1):
            # get model
            model = self.base_model(num_components)
            if model is None:
                continue

            try:
                # get score for the current word
                logL = model.score(self.X, self.lengths)

                # sum scores of other words
                words_logL = 0
                for word in self.words:
                    if word != self.this_word:
                        X, lengths = self.hwords[word]
                        words_logL += model.score(X, lengths)
            except ValueError:
                continue

            # calculate DIC score
            score = logL - (1 / (len(self.words) - 1)) * words_logL

            # update the best score
            if best_score is None or score > best_score:
                best_score = score
                best_model = model

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = None
        best_num_components = 0

        # split the training data
        n_splits = min(3, len(self.sequences))
        if n_splits > 1:
            splits = list(KFold(n_splits).split(self.sequences))
        else:
            # use the same data for training and test
            splits = [([0], [0])]

        for num_components in range(self.min_n_components, self.max_n_components + 1):
            scores = []
            try:
                for cv_train_idx, cv_test_idx in splits:
                    # create a model using the training data
                    train_Xlengths = combine_sequences(cv_train_idx, self.sequences)
                    model = SelectorConstant({self.this_word: self.sequences}, {self.this_word: train_Xlengths},
                                             self.this_word, n_constant=num_components).select()
                    if model is None:
                        raise ValueError

                    # get the model score for the test data
                    test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                    logL = model.score(test_X, test_lengths)
                    scores.append(logL)
            except ValueError:
                continue

            score = sum(scores) / len(scores)

            # update the best score
            if best_score is None or score > best_score:
                best_score = score
                best_num_components = num_components

        # create a model with the best number of hidden states
        best_model = self.base_model(best_num_components)

        return best_model

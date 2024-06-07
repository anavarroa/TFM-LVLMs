from pathlib import Path
from collections import defaultdict
import pickle
import math
from copy import copy
import numpy as np
import os

def precook(s, n=4, out=False):
    """Takes a string as input and returns an object that can be given to either cook_refs or cook_test."""
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4):
    """Takes a list of reference sentences for a single segment and returns an object that encapsulates everything that BLEU needs to know about them."""
    return [precook(ref, n) for ref in refs]

def cook_test(test, n=4):
    """Takes a test sentence and returns an object that encapsulates everything that BLEU needs to know about it."""
    return precook(test, n, True)

class CiderScorer(object):
    """CIDEr scorer."""

    def save_df(self, df_name="corpus", path=None):
        """Save the idf computed in corpus mode."""
        if path:
            path = Path(path)
            if not path.exists():
                path = Path.home()
                print(f"the path provided is not valid. The df will be saved in {path}")
        else:
            path = Path.home()
            print(f"the path provided is not valid. The df will be saved in {path}")

        filename = Path(path, df_name + '.p')

        if len(self.document_frequency) > 0:
            with open(filename, "wb") as fp:
                df_idf = {
                    "ref_len": np.log(float(len(self.crefs))),
                    "df": self.document_frequency
                }
                pickle.dump(df_idf, fp)
                print(f"saved to {filename}")
        else:
            raise ValueError("document frequency not computed run 'compute_score'")

    def copy(self):
        """Copy the refs."""
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, df_mode=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"df.p")), test=None, refs=None, n=4, sigma=6.0):
        """Singular instance."""
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.ref_len = None
        self.df_mode = df_mode

        if self.df_mode != "corpus":
            with open(self.df_mode, 'rb') as fp:
                df = pickle.load(fp)
            self.document_frequency = df['df']
            self.ref_len = df['ref_len']
        self.cook_append(test, refs)

    def clear(self):
        self.crefs = []
        self.ctest = []

    def cook_append(self, test, refs):
        """Called by constructor and __iadd__ to avoid creating new instances."""
        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                self.ctest.append(cook_test(test))  # N.B.: -1
            else:
                self.ctest.append(None)  # lens of crefs and ctest have to match

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        """Add an instance (e.g., from another sentence)."""
        if type(other) is tuple:
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
        return self

    def compute_doc_freq(self):
        """Compute term frequency for reference data."""
        for refs in self.crefs:
            for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
                self.document_frequency[ngram] += 1

    def compute_cider(self):
        def counts2vec(cnts):
            """Function maps counts of ngram to vector of tfidf weights."""
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.items():
                df = np.log(max(1.0, self.document_frequency[ngram]))
                n = len(ngram) - 1
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                norm[n] += pow(vec[n][ngram], 2)
                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            """Compute the cosine similarity of two vectors."""
            delta = float(length_hyp - length_ref)
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                for (ngram, count) in vec_hyp[n].items():
                    val[n] += vec_hyp[n][ngram] * vec_ref[n][ngram]
                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n] * norm_ref[n])
                assert (not math.isnan(val[n]))
            return val

        if self.df_mode == "corpus":
            self.ref_len = np.log(float(len(self.crefs)))

        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            vec, norm, length = counts2vec(test)
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score)
            score_avg /= len(refs)
            score_avg *= 10.0
            scores.append(score_avg)
        return scores

    def compute_score(self, option=None, verbose=0):
        if self.df_mode == "corpus":
            self.document_frequency = defaultdict(float)
            self.compute_doc_freq()
            assert (len(self.ctest) >= max(self.document_frequency.values()))
        score = self.compute_cider()
        return np.mean(np.array(score)), np.array(score)

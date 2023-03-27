import numpy as np
import pandas as pd
from ipfn import ipfn
import random

# See https://pypi.org/project/ipfn/

def get_top_indices(a, want):
    flat_indices = np.argpartition(a.ravel(), -want)[-want:]
    flat_elements = a.ravel()[flat_indices]
    ans = []
    for i in flat_indices:
        indices = np.unravel_index(i, a.shape)
        ans.append(indices)
    return ans, flat_elements

class Survey:
    def __init__(self):
        pass

    def set_questions(self, questions):
        self.questions = questions
        self.nques = len(questions)
        self.aggregates = []
        self.aggregates_1d_lookup = {}
        self.dimensions = []
        self.checks = []
        self.reasons = []

    def set_choice_names(self, choice_names):
        self.choice_names = choice_names

    def add_1d_constraint(self, qid, dist, reason):
        nques = self.nques
        npa = np.array(dist)
        self.aggregates.append(npa)
        assert qid not in self.aggregates_1d_lookup
        self.aggregates_1d_lookup[qid] = npa
        self.dimensions.append([qid])
        self.checks.append(lambda a: np.apply_over_axes(np.sum, a, list(filter(lambda x: x != qid, range(nques)))))
        self.reasons.append(reason)

    # Given absolute value of bivariance marginal dist of qid1 and qid2.
    # Must be discreet, otherwise may break the whole assumption.
    def add_2d_constraint_abs(self, qid1, qid2, dist, reason):
        nques = self.nques
        npa = np.array(dist)
        assert npa.ndim == 2
        self.aggregates.append(npa)
        self.dimensions.append([qid1, qid2])
        self.checks.append(lambda a: np.apply_over_axes(np.sum, a, list(filter(lambda x: x not in [qid1, qid2], range(nques)))))
        self.reasons.append(reason)

    # Derive qid2 from qid1
    def add_2d_constraint(self, qid1, qid2, dist_rate, reason):
        nques = self.nques
        np_rate = np.array(dist_rate)
        assert qid1 in self.aggregates_1d_lookup
        assert np_rate.ndim == 2
        weights = self.aggregates_1d_lookup[qid1][:, np.newaxis]
        npa = np.multiply(np_rate, weights)
        self.aggregates.append(npa)
        self.dimensions.append([qid1, qid2])
        self.checks.append(lambda a: np.apply_over_axes(np.sum, a, list(filter(lambda x: x not in [qid1, qid2], range(nques)))))
        self.reasons.append(reason)

    def generate(self, ntop, count):
        m = np.random.randint(0, 100, tuple(self.questions))
        IPF = ipfn.ipfn(m, self.aggregates, self.dimensions, convergence_rate=1e-6)
        m = IPF.iteration()

        for (a, c, r) in zip(self.aggregates, self.checks, self.reasons):
            res = c(m).reshape(a.shape)
            tolerance = 0.05 * abs(a)
            if np.allclose(a, res, atol=tolerance):
                print("[OK] constraint meets: {}".format(r))
            else:
                print("[ERROR] constraint meets: {}. expected {} got {}".format(r, a, res))

        (indices, weights) = get_top_indices(m, ntop)
        print("The {} top choices for the survey are {} with weights {}".format(ntop, indices, weights))
        print("=== Begin to generate the faked survey ===")

        choices = random.choices(range(len(indices)), weights=weights, k=count)
        for c in choices:
            s = []
            for qid, rid in enumerate(indices[c]):
                s.append(self.choice_names[qid][rid])
            print(', '.join(s))

        print("=== Finish generation ===")

survey = Survey()
survey.set_questions([2, 5, 3])
survey.set_choice_names([
            ["Boy", "Girl"],
            ["0-15", "15-30", "30-45", "45-60", "60-+inf"],
            ["TAIHU", "LIANGXI", "Others"]
        ])
survey.add_1d_constraint(0, [50, 50], "Boys and girls count equal.")
survey.add_1d_constraint(1, [10, 20, 30, 20, 20], "Age range distribution: most people are aged 30-45.")
# add_2d_constraint_abs(0, 2, [[25, 5, 20], [10, 35, 5]], 
#                    """
#                    Boys prefer TAIHU new town while girls prefer LIANGXI new town.
#                    Boys also have affection for other districts in WUXI, while girls don't.
#                    """)
survey.add_2d_constraint(0, 2, [[0.5, 0.1, 0.4], [0.2, 0.7, 0.1]], 
                   """
                   Boys prefer TAIHU new town while girls prefer LIANGXI new town.
                   Boys also have affection for other districts in WUXI, while girls don't.
                   """)
survey.generate(5, 40)
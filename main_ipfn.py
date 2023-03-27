import numpy as np
import pandas as pd
from ipfn import ipfn
import random

# See https://pypi.org/project/ipfn/

def get_top_indices(a, want):
    flat_indices = np.argpartition(a, -want, axis=None)[-want:]
    ans = []
    for i in flat_indices:
        indices = np.unravel_index(i, a.shape)
        ans.append(indices)
    return ans

questions = [2, 5, 3]
choice_names = [
    ["Boy", "Girl"],
    ["0-15", "15-30", "30-45", "45-60", "60-+inf"],
    ["TAIHU", "LIANGXI", "Others"]
]
nques = len(questions)
aggregates = []
aggregates_1d_lookup = {}
dimensions = []
checks = []
reasons = []

def add_1d_constraint(qid, dist, reason):
    npa = np.array(dist)
    aggregates.append(npa)
    assert qid not in aggregates_1d_lookup
    aggregates_1d_lookup[qid] = npa
    dimensions.append([qid])
    checks.append(lambda a: np.apply_over_axes(np.sum, a, list(filter(lambda x: x != qid, range(nques)))))
    reasons.append(reason)

# Given absolute value of bivariance marginal dist of qid1 and qid2.
# Must be discreet, otherwise may break the whole assumption.
def add_2d_constraint_abs(qid1, qid2, dist, reason):
    npa = np.array(dist)
    assert npa.ndim == 2
    aggregates.append(npa)
    dimensions.append([qid1, qid2])
    checks.append(lambda a: np.apply_over_axes(np.sum, a, list(filter(lambda x: x not in [qid1, qid2], range(nques)))))
    reasons.append(reason)

# Derive qid2 from qid1
def add_2d_constraint(qid1, qid2, dist_rate, reason):
    np_rate = np.array(dist_rate)
    assert qid1 in aggregates_1d_lookup
    assert np_rate.ndim == 2
    weights = aggregates_1d_lookup[qid1][:, np.newaxis]
    npa = np.multiply(np_rate, weights)
    aggregates.append(npa)
    dimensions.append([qid1, qid2])
    checks.append(lambda a: np.apply_over_axes(np.sum, a, list(filter(lambda x: x not in [qid1, qid2], range(nques)))))
    reasons.append(reason)
    
add_1d_constraint(0, [50, 50], "Boys and girls count equal.")
add_1d_constraint(1, [10, 20, 30, 20, 20], "Age range distribution: most people are aged 30-45.")
# add_2d_constraint_abs(0, 2, [[25, 5, 20], [10, 35, 5]], 
#                    """
#                    Boys prefer TAIHU new town while girls prefer LIANGXI new town.
#                    Boys also have affection for other districts in WUXI, while girls don't.
#                    """)
add_2d_constraint(0, 2, [[0.5, 0.1, 0.4], [0.2, 0.7, 0.1]], 
                   """
                   Boys prefer TAIHU new town while girls prefer LIANGXI new town.
                   Boys also have affection for other districts in WUXI, while girls don't.
                   """)

m = np.random.randint(0, 100, tuple(questions))
IPF = ipfn.ipfn(m, aggregates, dimensions, convergence_rate=1e-6)
m = IPF.iteration()

for (a, c, r) in zip(aggregates, checks, reasons):
    res = c(m).reshape(a.shape)
    tolerance = 0.05 * abs(a)
    if np.allclose(a, res, atol=tolerance):
        print("[OK] contraint meets: {}".format(r))
    else:
        print("[ERROR] contraint meets: {}. expected {} got {}".format(r, a, res))

NTOP = 5
indices = get_top_indices(m, NTOP)
print("The {} top choices for the survey are {}".format(NTOP, indices))
print("=== Begin to generate the faked survey ===")


for i in range(20):
    c = random.randint(0, len(indices) - 1)
    s = []
    for qid, rid in enumerate(indices[c]):
        s.append(choice_names[qid][rid])
    print(', '.join(s))


print("=== Finish generation ===")
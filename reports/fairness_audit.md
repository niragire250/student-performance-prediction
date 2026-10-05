# Fairness Audit

Subgroup performance of the final tuned model on the held-out test set (n=79, overall accuracy 89.9%).

**Caution:** several subgroups below are small; treat percentage gaps for groups flagged `small_sample_warning` as indicative, not conclusive.


## By `sex`

| Group | n | Actual pass rate | Accuracy | Precision | Recall | F1 | Note |
|---|---|---|---|---|---|---|---|
| F | 47 | 63.8% | 91.5% | 93.3% | 93.3% | 93.3% |  |
| M | 32 | 71.9% | 87.5% | 100.0% | 82.6% | 90.5% |  |

Largest accuracy gap across reliably-sized `sex` groups: **4.0 percentage points**.

## By `address`

| Group | n | Actual pass rate | Accuracy | Precision | Recall | F1 | Note |
|---|---|---|---|---|---|---|---|
| R | 13 | 61.5% | 92.3% | 100.0% | 87.5% | 93.3% |  |
| U | 66 | 68.2% | 89.4% | 95.2% | 88.9% | 92.0% |  |

Largest accuracy gap across reliably-sized `address` groups: **2.9 percentage points**.

## By `school`

| Group | n | Actual pass rate | Accuracy | Precision | Recall | F1 | Note |
|---|---|---|---|---|---|---|---|
| GP | 71 | 69.0% | 88.7% | 95.6% | 87.8% | 91.5% |  |
| MS | 8 | 50.0% | 100.0% | 100.0% | 100.0% | 100.0% | small sample |

## Interpretation

This audit uses a single 79-student test split, so per-subgroup sample sizes are small and these figures should be treated as a first check, not a certified fairness guarantee. Before any real deployment, this audit should be repeated on a larger, more current dataset, ideally with repeated cross-validation rather than a single split, so subgroup estimates are less sensitive to which particular students happened to land in the test set.
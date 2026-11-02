# 03 - Imputation Strategy Diagnostics

## Why compare two imputers?

My original question was whether borrowing information from nearby observations
would help more than filling every gap with a column median. The old comparison
was exploratory: preprocessing and threshold choices were not controlled well
enough to establish a general KNN advantage. I retain both choices in the revised
experiment rather than treating KNN as a proven prerequisite.

For the direct comparison, `rf_median` and `rf_knn` share development folds, sensor
filtering, scaling, and Random Forest settings. Only the imputer changes. Their
mean fold AP values are the primary comparison; F1 at 0.5 answers a different,
threshold-specific question.

## The actual preprocessing order

Both pipelines use:

```text
training-fitted sensor filter
  -> training-fitted StandardScaler, preserving NaNs
  -> training-fitted median or KNN imputer
  -> classifier
```

Scaling comes before both imputers. In the KNN case this ensures that neighbor
selection uses standardized observed coordinates. The median branch follows the
same order for a controlled comparison. Validation rows are transformed using
training statistics; they do not supply replacement medians, scaling statistics,
or a new neighbor reference set.

## What each method estimates

| Choice | Implementation | Useful property | Limitation |
| --- | --- | --- | --- |
| Median | `SimpleImputer(strategy="median")` | Simple per-sensor estimate, resistant to extreme observed values | Ignores relationships with other sensors and can compress variation where values are missing |
| KNN | `KNNImputer(n_neighbors=5)` | Uses observed coordinates to choose training donors | Depends on the distance representation and can borrow from a different class or process regime |

The median is learned separately for each retained, scaled sensor. This is a
baseline, not a claim that missingness is random or that every observation should
be near a common operating point.

KNN uses the default NaN-aware Euclidean distance. For two rows with shared
observed coordinates, it rescales their summed squared differences for the number
of coordinates available. A donor can itself have missing entries; it must have
an observed value for the sensor being imputed. With default uniform weighting,
available nearby donors contribute equally. If distances cannot be defined for
the required feature, the implementation can fall back to its training-column
mean.

Neither labels nor timestamps enter donor selection. A nearby training row is not
necessarily a row with the same label, a nearby timestamp, or the same physical
failure mode. KNN can smooth an unusual observation just as median imputation can.

## Reading the comparison

A consistent AP difference on shared folds supports a predictive comparison under
this preprocessing policy. Fold variation, failure support, and runtime qualify
the result. A small development difference can be sampling noise, especially with
few failures per validation fold.

An identical PCA component count after two imputers would show only that one
cumulative-variance cutoff happened to select the same count. It would not prove
identical covariance, equal neighborhoods, or preservation of failure structure.
The revised candidate set includes KNN plus PCA; it does not include a second
median-plus-PCA classifier comparison.

## What I can claim now

Imputation is a modeling assumption that belongs inside each training fold. In
the original comparison, KNN/RF mean fold AP is **0.2007**, versus **0.1848** for
Median/RF. In the three supplementary comparisons, KNN trails Median in two of
three repeats. The measured advantage therefore depends on the partition; it does
not support the old blanket claim that KNN improves performance by 50%.
The [saved results](00_navigation.md#code-and-generated-evidence) and
[repeat table](13_development_diagnostics.md#repeated-comparisons) make that
qualification explicit.

# muspy_augment

Fast data augmentations for symbolic music.

### Notes
* The augmentation work on pairs of scores (i.e. 'source' and 'target') and ensures the equivalence between the scores is maintained.
* The augmentations assume that your music is 4/4 and without partial measures (such as pickup measures).
  For other time signatures the results may be invalid.
  This was enough for our use case, but we would be happy to accept contributions.
* We use our own fork of muspy which added several features and bugfixes.
* The augmentations adjust the given scores' `beats` and each track's `notes`

### Usage Examples

Using the object API:
```python
# Configure augmentations
# (if you want you can serialize this configuration,
# e.g. as part of your TrainingArguments)
augmentations = [
    # With a probability of 0.2, apply either CutMeasures
    # or RemoveMeasures (with probability of 0.5 each)
    EitherOr(0.2, CutMeasures(1), RemoveMeasures(1), 0.5),
    # Apply AddEmptyMeasures with probability of 0.25
    AddEmptyMeasures(0.25)
]
# At some point later, apply the augmentations
for augmentation in augmentations:
    score1, score2 = augmentation((score1, score2))
```

Using the functional API:
```python
def augment_phrase_pair(score1: Music, score2: Music):
    assert score1.resolution == score2.resolution
    applied_augmentations = []
    if torch.rand(1) < 0.4:
        if torch.rand(1) < 0.5:
            score1, score2, aug = cut_phrase(score1, score2)
            applied_augmentations.append(aug)
        else:
            non_tied_measures = get_common_non_tied_measures([score1, score2])
            score1, score2, aug = remove_some_measures(score1, score2, non_tied_measures)
            applied_augmentations.append(aug)

    if torch.rand(1) < 0.25:
        score1, score2, aug = add_empty_bars(score1, score2)
        applied_augmentations.append(aug)

    return score1, score2, applied_augmentations

```
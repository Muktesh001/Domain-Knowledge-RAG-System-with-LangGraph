# Common AI / ML Interview Questions

This note captures answers interviewers expect for frequent conceptual questions.

## What is the bias-variance tradeoff?
Bias is error from overly simple assumptions; variance is sensitivity to the training sample. High bias underfits; high variance overfits. Regularization, more data, and model capacity move the tradeoff.

## What is overfitting and how do you detect it?
The model fits training data but generalizes poorly. Detect via a held-out validation set, learning curves, and a growing train-vs-val gap. Mitigations: more data, simpler models, dropout, weight decay, early stopping, data augmentation.

## Explain precision vs recall.
Precision = TP / (TP + FP). Recall = TP / (TP + FN). A spam filter with high precision rarely flags good mail; high recall catches more spam. F1 is the harmonic mean when you need a single score.

## What is a confusion matrix?
A table of TP, FP, FN, TN used to compute those rates. Accuracy is misleading on imbalanced data.

## Gradient descent vs stochastic gradient descent?
Batch GD uses the full dataset per step (stable, slow). SGD uses one example (noisy, fast). Mini-batch is the practical default. Adam adds adaptive per-parameter rates.

## Train / validation / test
Train fits parameters, validation tunes hyperparameters and early stopping, test is touched once for an unbiased estimate. Peeking at test is leakage.

## Interview one-liner
Interviewers want crisp definitions plus when you would choose one metric or split over another—not a textbook dump.

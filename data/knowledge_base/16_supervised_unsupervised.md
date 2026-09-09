# Supervised Learning vs Unsupervised Learning

Interviewers still open with classical ML before jumping to LLMs. Supervised learning trains on labeled pairs (x, y): classification and regression. Unsupervised learning finds structure without labels: clustering, dimensionality reduction, density estimation.

## Semi-supervised and self-supervised
Semi-supervised uses a small labeled set plus a large unlabeled set. Self-supervised learning (the pretraining of Transformers) creates labels from the data itself, e.g. masked tokens or next-token prediction. That is why LLM pretraining is not "supervised in the classic sense" even though it uses a loss on targets.

## Clustering quick hits
k-means assumes spherical clusters and needs k. Hierarchical clustering builds a dendrogram. DBSCAN finds density-connected regions and can mark noise. For text, cluster in embedding space, not raw bag-of-words, when possible.

## Dimensionality reduction
PCA finds orthogonal directions of variance. t-SNE and UMAP are for visualization and can distort global distances. Do not use t-SNE coordinates as features for a production classifier without care.

## Connecting to RAG
Unsupervised embeddings and clustering help analyze a knowledge base (duplicate topics, coverage gaps). Supervised labels appear in evaluation sets and re-ranker training.

## Interview one-liner
Supervised learning predicts known targets; unsupervised finds structure; modern LLM pretraining is mostly self-supervised next-token or masked-token prediction.

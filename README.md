# Search GCN
This is like a fun implementation of the paper [Multilingual GCN paper](https://cdn.amazon.science/67/ce/e083779041c092772a2105748ec9/graph-based-multilingual-product-retrieval-in-e-commerce-search.pdf). I am going to cover my recent work on the search engines I studied in recent days, where we are going to create a product encoder and the query encoder part
Apart from this, for search to be good, I have realised we also need to have some sort of inverted indexing must be there and the best for this is BM25, but SPLADE(Sparse Lexical and Expansion Model) will expand the search for different products 
In this 2-tower model, for search to outperform for various use cases(including for e-commerce, movie platform, or anywhere where there is like search happens).

### Training
This also includes the training of both models; they also need some training on your dataset for good retrieval tasks(like infonce for contrastive learning in the embedding) --> done using LoRA fine-tuning
For SPLADE, we will do it with SPLADE loss

I may not know the actual starting point, but I will keep updating on this incrementally what part to update.
All of the blogs for those you can find there; I will keep updating about this, all of them into the [docs](rdxtreme.dev/search_anything)

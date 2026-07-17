# Search GCN

This is a fun implementation inspired by the paper **[Graph-Based Multilingual Product Retrieval in E-Commerce Search](https://cdn.amazon.science/67/ce/e083779041c092772a2105748ec9/graph-based-multilingual-product-retrieval-in-e-commerce-search.pdf)**.

In this series, I'll cover my recent work on search engines and the concepts I've been studying. The goal is to build both a **product encoder** and a **query encoder**.

Apart from that, I've realized that a good search engine also needs some form of **inverted indexing**. **BM25** is an excellent baseline for lexical retrieval, while **SPLADE (Sparse Lexical and Expansion Model)** expands lexical matching, allowing the search engine to retrieve more relevant products by handling vocabulary mismatches.

This two-tower architecture is designed to perform well across a variety of search use cases, including **e-commerce**, **movie platforms**, and, in general, any application where search plays an important role.

## Training

This project also covers training both encoders. To achieve good retrieval performance, the models need to be trained on your own dataset using **contrastive learning** (e.g., **InfoNCE loss**) for embedding learning. The dense encoders will be fine-tuned using **LoRA**.

For the sparse retriever, **SPLADE** will be trained using the appropriate **SPLADE loss**.

I may not know the perfect starting point yet, but I'll continue updating this project incrementally as I build and learn. I'll document every major step along the way.

You can find all the blogs and documentation at **rdxtreme.dev/search_anything**. I'll keep updating the docs as the project progresses.


```bibtex
@article{Lu2021,
  author = {Hanqing Lu and Youna Hu and Tong Zhao and Tony Wu and Yiwei Song and Bing Yin},
  title = {Graph-based Multilingual Product Retrieval in E-Commerce Search},
  year = {2021},
  url = {https://www.amazon.science/publications/graph-based-multilingual-product-retrieval-in-e-commerce-search}
}
```
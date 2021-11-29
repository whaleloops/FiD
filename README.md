This repository contains code for [facebook fid](https://github.com/facebookresearch/FiD):
- Fusion-in-Decoder models
- Distilling Knowledge from Reader to Retriever

## Dependencies

- Python 3
- [PyTorch](http://pytorch.org/) (currently tested on version 1.6.0)
- [Transformers](http://huggingface.co/transformers/) (**version 4.8.2**)
- [NumPy](http://www.numpy.org/)



# I. Fusion-in-Decoder
Performance of the my pretrained and [origianl pretrained models](https://github.com/facebookresearch/FiD):

<table>
  <tr><td>Mode size</td><td colspan="2">TriviaQA-selftrain10</td><td colspan="2">TriviaQA-original10</td><td colspan="2">TriviaQA-original100</td></tr>
  <tr><td></td><td>dev</td><td>test</td><td>dev</td><td>test</td><td>dev</td><td>test</td></tr>
  <tr><td>base</td><td>abc</td><td>abc</td><td>65.20</td><td>65.52</td><td>67.84</td><td>68.10</td></tr>
</table>




TODO: delete below



# II. Distilling knowledge from reader to retriever for question answering
This repository also contains code to train a retriever model following the method proposed in our paper: Distilling knowledge from reader to retriever for question answering. This code is heavily inspired by the [DPR codebase](https://github.com/facebookresearch/DPR) and reuses parts of it. The proposed method consists in several steps:

### 1. Obtain reader cross-attention scores
Assuming that we have already retrieved relevant passages for each question, the first step consists in generating cross-attention scores. This can be done using the option `--write_crossattention_scores` in [`test.py`](test.py). It saves the dataset with cross-attention scores in `checkpoint_dir/name/dataset_wscores.json`. To retrieve the initial set of passages for each question, different options can be considered, such as DPR or BM25.

```shell
python test.py \
        --model_path my_model_path \
        --eval_data data.json \
        --per_gpu_batch_size 4 \
        --n_context 100 \
        --name my_test \
        --checkpoint_dir checkpoint \
        --write_crossattention_scores \
```

### 2. Retriever training

[`train_retriever.py`](train_retriever.py) provides the code to train a retriever using the scores previously generated.

```shell
python train_retriever.py \
        --lr 1e-4 \
        --optim adamw \
        --scheduler linear \
        --train_data train_data.json \
        --eval_data eval_data.json \
        --n_context 100 \
        --total_steps 20000 \
        --scheduler_steps 30000 \
```


### 3. Knowldege source indexing

Then the trained retriever is used to index a knowldege source, Wikipedia in our case.

```shell
python3 generate_retriever_embedding.py \
        --model_path <model_dir> \ #directory
        --passages passages.tsv \ #.tsv file
        --output_path wikipedia_embeddings \
        --shard_id 0 \
        --num_shards 1 \
        --per_gpu_batch_size 500 \
```

### 4. Passage retrieval

After indexing, given an input query, passages can be efficiently retrieved:


```shell
python passage_retrieval.py \
    --model_path <model_dir> \
    --passages psgs_w100.tsv \
    --data_path data.json \
    --passages_embeddings "wikipedia_embeddings/wiki_*" \
    --output_path retrieved_data.json \
    --n-docs 100 \
```

We found that iterating the four steps here can improve performances, depending on the initial set of documents.


## References

[1] G. Izacard, E. Grave [*Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering*](https://arxiv.org/abs/2007.01282)

```bibtex
@misc{izacard2020leveraging,
      title={Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering},
      author={Gautier Izacard and Edouard Grave},
      year={2020},
      eprint={2007.01282},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

[2] G. Izacard, E. Grave [*Distilling Knowledge from Reader to Retriever for Question Answering*](https://arxiv.org/abs/2012.04584)

```bibtex
@misc{izacard2020distilling,
      title={Distilling Knowledge from Reader to Retriever for Question Answering},
      author={Gautier Izacard and Edouard Grave},
      year={2020},
      eprint={2012.04584},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

See the [LICENSE](LICENSE) file for more details.

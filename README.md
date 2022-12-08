# TyDiP: A Dataset for Politeness Classification in Nine Typologically Diverse Languages
This repo contains the code and data for the EMNLP 2022 findings paper TyDiP: A Dataset for Politeness Classification in Nine Typologically Diverse Languages which can be found [here](https://arxiv.org/abs/2211.16496).
## Data
The TyDiP dataset is licensed under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

The `data` folder contains the different files we release as part of the TyDiP dataset. The TyDiP dataset comprises of an English train set and English test set that are adapted from the Stanford Politeness Corpus, and test data in 9 more languages (Hindi, Korean, Spanish, Tamil, French, Vietnamese, Russian, Afrikaans, Hungarian) that we annotated.

```
data/
├── all
├── binary
└── unlabelled_train_sets
```
`data/all` consists of the complete train and test sets.  
`data/binary` is a filtered version of the above where sentences from the top and bottom 25 percentile of scores is only present. This is the data that we used for training and evaluation in the paper.  
`data/unlabelled_train_sets`

If you use the English train or test data, please cite the Stanford Politeness Dataset
```
@inproceedings{danescu-niculescu-mizil-etal-2013-computational,
    title = "A computational approach to politeness with application to social factors",
    author = "Danescu-Niculescu-Mizil, Cristian  and
      Sudhof, Moritz  and
      Jurafsky, Dan  and
      Leskovec, Jure  and
      Potts, Christopher",
    booktitle = "Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2013",
    address = "Sofia, Bulgaria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P13-1025",
    pages = "250--259",
}
```
If you use the test data from the 9 target languages, please cite our paper
```
@inproceedings{srinivasan-choi-2022-tydip,
    title = "TyDiP: A Dataset for Politeness Classification in Nine Typologically Diverse Languages",
    author = "Srinivasan, Anirudh  and
      Choi, Eunsol",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
}
```

## Code
`politeness_regresor.py` is used for training and evaluation of transformer models

To train a model
```
python politeness_regressor.py --train_file data/binary/en_train_binary.csv --test_file data/binary/en_test_binary.csv --model_save_location model.pt --pretrained_model xlm-roberta-large --gpus 1 --batch_size 4 --accumulate_grad_batches 8 --max_epochs 5 --checkpoint_callback False --logger False --precision 16 --train --test --binary --learning_rate 5e-6
```

To test this trained model on $lang
```
python politeness_regressor.py --test_file data/binary/${lang}_test_binary.csv --load_model model.pt --gpus 1 --batch_size 32 --test --binary
```

## Politeness Strategies
`strategies` contains the processed strategy lexicon for different languages. `strategies/learnt_strategies.xlsx` contains the human edited strategies for 4 langauges

## Annotation Interface
`annotation.html` contains the UI used for conducting data annotation
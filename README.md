# FigureSpanDetection
Implementation of feature-based CRF for "Evidence Fragment Detection" described in EACL 2021 paper [Scientific Discourse Tagging for Evidence Extraction](https://arxiv.org/abs/1909.04758).

This figure span tagger takes clauses from biomedical paragraphs and tags each clause with BIO tags. Under certain block based assumptions described in the paper, all sub figures semantically referred by each clause can be recovered. Based on these sub figure references, "evidence fragments", which are description of the experimental figures in the texts can be extracted easily.

## Requirements
* Scikit-Learn
* Scipy
* Numpy
* [sklearn_crfsuite](https://sklearn-crfsuite.readthedocs.io/en/latest/)

## Note
* The tagger is in `figureSpanTagger.ipynb`, which should be self-explanatory.
* `figureSpanTaggerApplication.ipynb` shows an example of predicting real-world figure spans.
* `util.py` includes many utility functions.
* `figureSpanExtractor.py` includes functions for extracting direct mentions of figure numbers. It is useful for the actual inference time. The only function you need to call is `extractFigureSpan(sentence)`.

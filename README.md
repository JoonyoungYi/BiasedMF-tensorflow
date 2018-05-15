# BiasedMF-tensorflow

* The tensorflow implementation of BiasedMF(Biased Matrix Factroization) from [](http://base.sjtu.edu.cn/~bjshen/2.pdf).
* This repository was forked from [UtsavManiar/Movie_Recommendation_Engine](https://github.com/UtsavManiar/Movie_Recommendation_Engine).
* I used MovieLens 100k dataset.

## Environment

* How to init
```
virtualenv .venv -p python3.6
. .venv/bin/activate
pip install -r requirements.txt
deactivate
```

* How to train
```
. .venv/bin/activate
python train.py
```

* How to run
```
. .venv/bin/activate
python run.py
```

# Word Embedding for Financial News
Pytorch implementation of skip-gram model(Word2Vec) and t-SNE visualization/ most similar word for specific financial news built in 5 days


### Requirement
* pytorch
* Numpy
* pandas
* matplotlib
* nltk
* tqdm
* sklearn

### Data Preprocessing

If the original data file is in .json, 
```python
python3 csvConverter.py
```
Change the following variable to the column title in your file:
```python
'Url', 'Title', 'Text'
```
After you have a csv file with at least column 'Text', run following command. It will do the basic data cleaning and tokenization of words and convert them to be trainable data
```python
python3 dataset.py
```

### Training
This will download model state_dict in your running directory
```python
python3 train.py
```

### Visualization and Most Similar Words
```python
python3 visualgensim.py
```
For similar words, you need to change these two variables`TOPN` and `WORDYOUWANT` to your choices



# Rumour-detection

## Requirements
```bash
python==3.8
torch==1.10.1
transformers==4.18.0
PyYAML==5.4.1
tqdm==4.62.3
pandas==1.2.5
numpy==1.21.2
scikit-learn==1.0.2
```

## Data Pre-processing
Through call *process_tweepy_data.py* and define the data type (i.e., train or dev or test or covid, pre-processe the tweet text including source tweets and replies and also extract statistic features from raw data.
```python
cd ./task1/pre-process
python process_tweepy_data.py -m train
python process_tweepy_data.py -m dev
python process_tweepy_data.py -m test
python process_tweepy_data.py -m covid
```
Call *process_data.py* to fill in missing data using KNN interpolation and do standard scaler on them.
```python
cd ./task1/pre-process
python process_data.py
```
The results data will be stored as .csv file in *res* file.
### 1. Pre-processing tweet text
- Expand abbreviations (e.g., don't -> do not)
- Remove punctuations, numbers, twitter ID, url
- Tokenization (i.e., convert sentences into lists of words)
- Stemming (i.e., remove any form of a word and change it into its root form)
- Convert emoticon into corresponding words using *emoji* package in Python
### 2. Extract Feature from raw data
- Tweet Content's Features
  -  The number of *question marks* in the tweet text
  - The number of *url* mentioned in the tweet text
  - The number of *twitter ID* mentioned in the tweet text
  - The *sentiment* of the tweet text, which is calculated by *TextBlob* in Python and then tagged with Positive (i.e., 1) or Negative(i.e., 0). 
- Tweet Statistical Features
  - *Reply Count*: The number of replies of the tweet 
  - *Favorite Count*: The number of twitters like the tweet
- User Features
  - Friends Count
  - Protected or not
  - Verified or not
  - User Engagement
  - Following Rate
  - Favourite Rate   
## Model training
### 1. BERTweet with only raw tweet text (*TextBERTweet*)
```python
cd ./task1/main_models
python train.py -m textbertweet -l 128  # Note: 128 is the longest for this model
```
### 2. BERTweet with raw tweet text and statistics features (*TextBERTweet-TF*)
```python
cd ./task1/main_models
python train.py -m textbertweet_tf -l 128  # Note: 128 is the longest for this model
```
### 3. Bert-Base with only pre-processed tweet text (*TextBERT*)
```python
cd ./task1/main_models
python train.py -m textbert -l 256 # Note: 512 is the longest for this model
```
### 4. Bert-Base with pre-processed tweet text and statistics features (*TextBERT-TF*)
```python
cd ./task1/main_models
python train.py -m textbert_tf -l 256 # Note: 512 is the longest for this model
```
## Model test
Define the model name (i.e., textbertweet / textbertweet_tf / textbert / textbert_tf), max length of bert, test data type (i.e., test or covid) and existing model path.
For example, predict rumour or non-rumour on test data with the existing bert-based with statistics model *bertcls_16.dat*.
```python 
cd ./task1/main_models
python test.py -m bert -l 256 -t textbertweet_tf -e bertcls_16.dat
```

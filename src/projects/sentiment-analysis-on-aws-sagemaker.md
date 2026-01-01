---
title: Sentiment Analysis on AWS SageMaker
category: Deep Learning
date: 2020-06-25
client: Udacity (Project)
description: Building an end-to-end sentiment analysis web application using PyTorch and AWS SageMaker to predict sentiment of movie reviews.
imageUrl: /images/sentiment-analysis.png
link: "https://github.com/shrikantnaidu/Sentiment-Analysis-on-AWS-SageMaker"
tags:
  - PyTorch
  - AWS SageMaker
  - NLP
  - LSTM
---

In this project, we construct a complete end-to-end project on SageMaker. Our goal is to have a simple web page which a user can use to enter a movie review. The web page will then send the review off to our deployed model which will predict the sentiment of the entered review.

### General Outline

Following is the general outline for SageMaker projects using a notebook instance:

1. Download or otherwise retrieve the data.
2. Process / Prepare the data.
3. Upload the processed data to S3.
4. Train a chosen model.
5. Test the trained model (typically using a batch transform job).
6. Deploy the trained model.
7. Use the deployed model.

For this project, we'll be following the steps in the general outline with some modifications. We will deploy and use the trained model a second time with custom inference code, and our newly deployed model will be used in the sentiment analysis web app.

---

### Step 1: Downloading the data

We will be using the [IMDb dataset](http://ai.stanford.edu/~amaas/data/sentiment/).

```python
%mkdir ../data
!wget -O ../data/aclImdb_v1.tar.gz http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -zxf ../data/aclImdb_v1.tar.gz -C ../data
```

**Output:**
```
Connecting to ai.stanford.edu (ai.stanford.edu)|171.64.68.10|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 84125825 (80M) [application/x-gzip]
Saving to: '../data/aclImdb_v1.tar.gz'

../data/aclImdb_v1. 100%[===================>]  80.23M  7.37MB/s    in 31s     

2020-06-23 13:30:16 (2.58 MB/s) - '../data/aclImdb_v1.tar.gz' saved [84125825/84125825]
```
    
---

### Step 2: Preparing and Processing the data

We'll start with some initial data processing. We will read in each of the reviews and combine them into a single input structure, then split the dataset into a training set and a testing set.

```python
import os
import glob

def read_imdb_data(data_dir='../data/aclImdb'):
    data = {}
    labels = {}
    
    for data_type in ['train', 'test']:
        data[data_type] = {}
        labels[data_type] = {}
        
        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []
            
            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)
            
            for f in files:
                with open(f) as review:
                    data[data_type][sentiment].append(review.read())
                    # Here we represent a positive review by '1' and a negative review by '0'
                    labels[data_type][sentiment].append(1 if sentiment == 'pos' else 0)
                    
            assert len(data[data_type][sentiment]) == len(labels[data_type][sentiment]), \
                    "{}/{} data size does not match labels size".format(data_type, sentiment)
                
    return data, labels

data, labels = read_imdb_data()
print("IMDB reviews: train = {} pos / {} neg, test = {} pos / {} neg".format(
            len(data['train']['pos']), len(data['train']['neg']),
            len(data['test']['pos']), len(data['test']['neg'])))
```

**Output:**
```
IMDB reviews: train = 12500 pos / 12500 neg, test = 12500 pos / 12500 neg
```    

Now we combine the positive and negative reviews and shuffle the resulting records.

```python
from sklearn.utils import shuffle

def prepare_imdb_data(data, labels):
    """Prepare training and test sets from IMDb movie reviews."""
    
    # Combine positive and negative reviews and labels
    data_train = data['train']['pos'] + data['train']['neg']
    data_test = data['test']['pos'] + data['test']['neg']
    labels_train = labels['train']['pos'] + labels['train']['neg']
    labels_test = labels['test']['pos'] + labels['test']['neg']
    
    # Shuffle reviews and corresponding labels within training and test sets
    data_train, labels_train = shuffle(data_train, labels_train)
    data_test, labels_test = shuffle(data_test, labels_test)
    
    # Return a unified training data, test data, training labels, test labels
    return data_train, data_test, labels_train, labels_test

train_X, test_X, train_y, test_y = prepare_imdb_data(data, labels)
print("IMDb reviews (combined): train = {}, test = {}".format(len(train_X), len(test_X)))
```

**Output:**
```
IMDb reviews (combined): train = 25000, test = 25000
```

---

The first step in processing the reviews is to make sure that any html tags that appear should be removed. In addition we wish to tokenize our input, that way words such as *entertained* and *entertaining* are considered the same with regard to sentiment analysis.

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

import re
from bs4 import BeautifulSoup

def review_to_words(review):
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()
    
    text = BeautifulSoup(review, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words] # stem
    
    return words
```

The method below applies the `review_to_words` method to each of the reviews in the training and testing datasets. It also caches the results for future runs.

```python
import pickle

cache_dir = os.path.join("../cache", "sentiment_analysis")  # where to store cache files
os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists

def preprocess_data(data_train, data_test, labels_train, labels_test,
                    cache_dir=cache_dir, cache_file="preprocessed_data.pkl"):
    """Convert each review to words; read from cache if available."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay
    
    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Preprocess training and test data to obtain words for each review
        words_train = [review_to_words(review) for review in data_train]
        words_test = [review_to_words(review) for review in data_test]
        
        # Write to cache file for future runs
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test,
                              labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                cache_data['words_test'], cache_data['labels_train'], cache_data['labels_test'])
    
    return words_train, words_test, labels_train, labels_test

# Preprocess data
train_X, test_X, train_y, test_y = preprocess_data(train_X, test_X, train_y, test_y)
```

**Output:**
```
Wrote preprocessed data to cache file: preprocessed_data.pkl
```

---

### Transforming the data

For the model we are going to construct we'll transform the data from its word representation to a bag-of-words feature representation. We will fix the size of our working vocabulary and only include the words that appear most frequently, combining all infrequent words into a single category labeled as `1`.

### Creating a word dictionary

We construct a way to map words that appear in the reviews to integers. Here we fix the size of our vocabulary to be `5000`.

```python
import numpy as np

def build_dict(data, vocab_size = 5000):
    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""
    
    word_count = {} # A dict storing the words that appear in the reviews along with how often they occur
    
    for each_review in data:
        for each_word in each_review:
            if each_word in word_count:
                word_count[each_word] += 1
            else:
                word_count[each_word] = 1
                
    sorted_words = sorted(word_count, key=word_count.get, reverse=True)
    
    word_dict = {} # This is what we are building, a dictionary that translates words into integers
    for idx, word in enumerate(sorted_words[:vocab_size - 2]): # The -2 is so that we save room for the 'no word'
        word_dict[word] = idx + 2                              # 'infrequent' labels
        
    return word_dict

word_dict = build_dict(train_X)

top_keywords = list(word_dict.keys())
print("Top 5 Keywords:", top_keywords[:5])
```

**Output:**
```
Top 5 Keywords: ['movi', 'film', 'one', 'like', 'time']
```

---

### Saving the `word_dict`

Later on when we construct an endpoint which processes a submitted review we will need to make use of the `word_dict` which we have created. We save it to a file for future use.

```python
data_dir = '../data/pytorch' # The folder we will use for storing data
if not os.path.exists(data_dir): # Make sure that the folder exists
    os.makedirs(data_dir)

with open(os.path.join(data_dir, 'word_dict.pkl'), "wb") as f:
    pickle.dump(word_dict, f)
```

### Transforming the reviews

Now that we have our word dictionary, we convert our reviews to their integer sequence representation, making sure to pad or truncate to a fixed length of `500`.

```python
def convert_and_pad(word_dict, sentence, pad=500):
    NOWORD = 0 # We will use 0 to represent the 'no word' category
    INFREQ = 1 # and we use 1 to represent the infrequent words, i.e., words not appearing in word_dict
    
    working_sentence = [NOWORD] * pad
    
    for word_index, word in enumerate(sentence[:pad]):
        if word in word_dict:
            working_sentence[word_index] = word_dict[word]
        else:
            working_sentence[word_index] = INFREQ
            
    return working_sentence, min(len(sentence), pad)

def convert_and_pad_data(word_dict, data, pad=500):
    result = []
    lengths = []
    
    for sentence in data:
        converted, leng = convert_and_pad(word_dict, sentence, pad)
        result.append(converted)
        lengths.append(leng)
        
    return np.array(result), np.array(lengths)

train_X, train_X_len = convert_and_pad_data(word_dict, train_X)
test_X, test_X_len = convert_and_pad_data(word_dict, test_X)
```

---

### Step 3: Upload the data to S3

We will need to upload the training dataset to S3 in order for our training code to access it.

#### Save the processed training dataset locally

Each row of the dataset has the form `label`, `length`, `review[500]` where `review[500]` is a sequence of `500` integers representing the words in the review, encoded using `word_dict`.

```python
import pandas as pd
    
pd.concat([pd.DataFrame(train_y), pd.DataFrame(train_X_len), pd.DataFrame(train_X)], axis=1) \
        .to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)
```

#### Uploading the training data

```python
import sagemaker

sagemaker_session = sagemaker.Session()

bucket = sagemaker_session.default_bucket()
prefix = 'sagemaker/sentiment_rnn'

role = sagemaker.get_execution_role()

input_data = sagemaker_session.upload_data(path=data_dir, bucket=bucket, key_prefix=prefix)
```

---

### Step 4: Build and Train the PyTorch Model

A model comprises of three objects in SageMaker:
- Model Artifacts
- Training Code
- Inference Code

We will start by implementing our own neural network in PyTorch along with a training script.

#### The LSTM Model

```python
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """
        Initialize the model by setting up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        out = out[lengths - 1, range(len(lengths))]
        return self.sig(out.squeeze())
```

---

#### Writing the training method

```python
def train(model, train_loader, epochs, optimizer, loss_fn, device):
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:         
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            # forward pass
            output = model.forward(batch_X)
            # calculate the batch loss
            loss = loss_fn(output, batch_y)
            # backpropagation
            loss.backward()
            # optimization
            optimizer.step()
            
            total_loss += loss.data.item()
        print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))
```

Testing the training method on a sample:

```python
import torch.optim as optim
from train.model import LSTMClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(32, 100, 5000).to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = torch.nn.BCELoss()

train(model, train_sample_dl, 5, optimizer, loss_fn, device)
```

**Output:**
```
Epoch: 1, BCELoss: 0.6932154417037963
Epoch: 2, BCELoss: 0.683722734451294
Epoch: 3, BCELoss: 0.675475811958313
Epoch: 4, BCELoss: 0.6661329388618469
Epoch: 5, BCELoss: 0.6543593883514405
```

---

#### Training the model on SageMaker

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(entry_point="train.py",
                    source_dir="train",
                    role=role,
                    framework_version='0.4.0',
                    train_instance_count=1,
                    train_instance_type='ml.p2.xlarge',
                    hyperparameters={
                        'epochs': 10,
                        'hidden_dim': 200,
                    })

estimator.fit({'training': input_data})
```

**Output:**
```
2020-06-23 14:42:13 Starting - Starting the training job...
2020-06-23 14:42:15 Starting - Launching requested ML instances.........
2020-06-23 14:43:45 Starting - Preparing the instances for training......
2020-06-23 14:45:06 Downloading - Downloading input data......
2020-06-23 14:46:08 Training - Training image download completed. Training in progress.
..
..
2020-06-23 14:49:42 Uploading - Uploading generated training model
2020-06-23 14:49:42 Completed - Training job completed
Epoch: 10, BCELoss: 0.2970385502795784
Training seconds: 276
Billable seconds: 276
```

---

### Step 5: Testing the model

We test the model by deploying it and sending the testing data to the deployed endpoint.

### Step 6: Deploy the model for testing

```python
# Deploy the trained model
predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
```

---

### Step 7 - Use the model for testing

```python
test_X = pd.concat([pd.DataFrame(test_X_len), pd.DataFrame(test_X)], axis=1)

# We split the data into chunks and send each chunk separately, accumulating the results.
def predict(data, rows=512):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = np.array([])
    for array in split_array:
        predictions = np.append(predictions, predictor.predict(array))
    
    return predictions

predictions = predict(test_X.values)
predictions = [round(num) for num in predictions]

from sklearn.metrics import accuracy_score
accuracy_score(test_y, predictions)
```

**Output:**
```
0.84772
```

---

#### More testing

Testing with a custom review:

```python
test_review = 'The simplest pleasures in life are the best, and this film is one of them. Combining a rather basic storyline of love and adventure this movie transcends the usual weekend fair with wit and unmitigated charm.'

# Convert test_review into a form usable by the model
test_review_data, test_review_len = convert_and_pad(word_dict, review_to_words(test_review))

# combine review length and data in one numpy array
test_data = np.array(test_review_data)
test_data = np.insert(test_data, 0, test_review_len)

# add empty batch dimension
test_data = test_data[None, :]

predictor.predict(test_data)
```

**Output:**
```
array(0.871142, dtype=float32)
```

Since the return value is close to `1`, we can be certain that the review we submitted is positive.

---

### Step 6 (again) - Deploy the model for the web app

Now that we know that our model is working, it's time to create some custom inference code so that we can send the model a review which has not been processed and have it determine the sentiment of the review.

#### Custom Inference Code

The custom inference functions:

```python
import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from model import LSTMClassifier
from utils import review_to_words, convert_and_pad

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(model_info['embedding_dim'], model_info['hidden_dim'], model_info['vocab_size'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved word_dict.
    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)

    model.to(device).eval()

    print("Done loading model.")
    return model

def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == 'text/plain':
        data = serialized_input_data.decode('utf-8')
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)

def predict_fn(input_data, model):
    print('Inferring sentiment of input data.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model.word_dict is None:
        raise Exception('Model has not been loaded properly, no word_dict.')
    
    # Process input_data so that it is ready to be sent to our model.
    data_X, data_len = convert_and_pad(model.word_dict, review_to_words(input_data))

    # Using data_X and data_len we construct an appropriate input tensor.
    data_pack = np.hstack((data_len, data_X))
    data_pack = data_pack.reshape(1, -1)
    
    data = torch.from_numpy(data_pack)
    data = data.to(device)

    # Make sure to put the model into evaluation mode
    model.eval()

    # Compute the result of applying the model to the input data.
    with torch.no_grad():
        output = model(data)
    
    result = np.round(output.cpu().numpy())

    return result
```

---

#### Deploying the model with custom inference

```python
from sagemaker.predictor import RealTimePredictor
from sagemaker.pytorch import PyTorchModel

class StringPredictor(RealTimePredictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super(StringPredictor, self).__init__(endpoint_name, sagemaker_session, content_type='text/plain')

model = PyTorchModel(model_data=estimator.model_data,
                     role=role,
                     framework_version='0.4.0',
                     entry_point='predict.py',
                     source_dir='serve',
                     predictor_cls=StringPredictor)
predictor = model.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
```

---

#### Testing the model

```python
import glob

def test_reviews(data_dir='../data/aclImdb', stop=250):
    
    results = []
    ground = []
    
    # We make sure to test both positive and negative reviews    
    for sentiment in ['pos', 'neg']:
        
        path = os.path.join(data_dir, 'test', sentiment, '*.txt')
        files = glob.glob(path)
        
        files_read = 0
        
        print('Starting ', sentiment, ' files')
        
        # Iterate through the files and send them to the predictor
        for f in files:
            with open(f) as review:
                # First, we store the ground truth (was the review positive or negative)
                if sentiment == 'pos':
                    ground.append(1)
                else:
                    ground.append(0)
                # Read in the review and convert to 'utf-8' for transmission via HTTP
                review_input = review.read().encode('utf-8')
                # Send the review to the predictor and store the results
                results.append(int(predictor.predict(review_input)))
                
            # Sending reviews to our endpoint one at a time takes a while so we
            # only send a small number of reviews
            files_read += 1
            if files_read == stop:
                break
            
    return ground, results

ground, results = test_reviews()
```

**Output:**
```
Starting  pos  files
Starting  neg  files
```

```python
from sklearn.metrics import accuracy_score
accuracy_score(ground, results)
```

**Output:**
```
0.858
```

---

### Step 7 (again): Use the model for the web app

So far we have been accessing our model endpoint by constructing a predictor object. Now we set up the web app using AWS services.

![Web App Architecture](https://raw.githubusercontent.com/shrikantnaidu/shrikantnaidu.github.io/main/_posts/assets/Web App Diagram.svg)

The diagram above gives an overview of how the various services work together:
- **SageMaker** - hosts the trained model
- **Lambda** - executes our inference code
- **API Gateway** - provides the public URL for the web app

#### Setting up a Lambda function

##### Part A: Create an IAM Role for the Lambda function

Using the AWS Console, navigate to the **IAM** page and click on **Roles**. Create a new role with **Lambda** as the service and attach the **AmazonSageMakerFullAccess** policy. Name it `LambdaSageMakerRole`.

##### Part B: Create a Lambda function

Navigate to AWS Lambda and create a function called `sentiment_analysis_func` with Python 3.6 runtime and the role you created.

```python
# We need to use the low-level library to interact with SageMaker since the SageMaker API
# is not available natively through Lambda.
import boto3

def lambda_handler(event, context):

    # The SageMaker runtime is what allows us to invoke the endpoint that we've created.
    runtime = boto3.Session().client('sagemaker-runtime')

    # Now we use the SageMaker runtime to invoke our endpoint, sending the review we were given
    response = runtime.invoke_endpoint(EndpointName='**ENDPOINT NAME HERE**',    # The name of the endpoint we created
                                       ContentType='text/plain',                 # The data format that is expected
                                       Body=event['body'])                       # The actual review

    # The response is an HTTP response whose body contains the result of our inference
    result = response['Body'].read().decode('utf-8')

    return {
        'statusCode': 200,
        'headers': { 'Content-Type': 'text/plain', 'Access-Control-Allow-Origin': '*' },
        'body': result
    }
```

Get the endpoint name:

```python
predictor.endpoint
```

**Output:**
```
'sagemaker-pytorch-2020-06-23-15-40-30-967'
```

---

#### Setting up API Gateway

1. Navigate to **Amazon API Gateway** and click **Get started**
2. Select **New API** and name it `sentiment_analysis_api`
3. Select **Actions** → **Create Method** → **POST**
4. Select **Lambda Function** and **Use Lambda Proxy integration**
5. Enter the Lambda function name and **Save**
6. Select **Actions** → **Deploy API** and create a new stage called `prod`

Copy the **Invoke URL** for your web app.

---

### Step 4: Deploying our web app

In the `website` folder, open `index.html` and replace **\*\*REPLACE WITH PUBLIC API URL\*\*** with the Invoke URL from API Gateway.

Now you can open `index.html` in your browser and use the provided site to interact with your SageMaker model.

**Important Note**: The endpoint must be running for the web app to work. Make sure to shut it down when not in use to avoid large AWS bills.

#### Delete the endpoint

```python
predictor.delete_endpoint()
```

---

### Conclusion

We've successfully developed a deep learning model that can classify the sentiment of movie reviews as either positive or negative, using AWS SageMaker. The model was trained on the IMDB dataset and achieved **85.8% accuracy**. We also deployed the model as a publicly accessible API using API Gateway and Lambda, and created a simple web app to interact with the model.

This project demonstrates the power of AWS SageMaker in building, training, and deploying machine learning models at scale.

The complete implementation can be found in the [GitHub repository](https://github.com/shrikantnaidu/Sentiment-Analysis-on-AWS-SageMaker).

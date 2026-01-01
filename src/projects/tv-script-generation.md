---
title: Generate TV Scripts with RNN
category: Deep Learning
date: 2020-06-08
client: Udacity (Project)
description: Building a recurrent neural network (RNN) to generate original TV scripts by training on Seinfeld dialogue data.
imageUrl: https://images.unsplash.com/photo-1522869635100-9f4c5e86aa37?auto=format&fit=crop&q=80&w=800
link: "https://github.com/shrikantnaidu/Generate-TV-Scripts"
tags:
  - RNN
  - LSTM
  - NLP
  - PyTorch
---

In this project, we'll generate our own [Seinfeld](https://en.wikipedia.org/wiki/Seinfeld) TV scripts using RNNs. We'll be using part of the [Seinfeld dataset](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv) of scripts from 9 seasons. The Neural Network we'll build will generate a new, "fake" TV script, based on patterns it recognizes in this training data.

### Get the Data

The data resides in `./data/Seinfeld_Scripts.txt`. As a first step, we'll load in this data and look at some samples. Then, we'll define and train an RNN to generate a new script!

```python
# load in data
import helper

data_dir = './data/Seinfeld_Scripts.txt'
text = helper.load_data(data_dir)
```

### Explore the Data

We play around with `view_line_range` to view different parts of the data. This gives us a sense of the data—that it is all lowercase text, and each new line of dialogue is separated by a newline character `\n`.

```python
view_line_range = (0, 10)

import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

lines = text.split('\n')
print('Number of lines: {}'.format(len(lines)))
word_count_line = [len(line.split()) for line in lines]
print('Average number of words in each line: {}'.format(np.average(word_count_line)))

print()
print('The lines {} to {}:'.format(*view_line_range))
print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))
```

**Output:**
```
Dataset Stats
Roughly the number of unique words: 46367
Number of lines: 109233
Average number of words in each line: 5.544240293684143

The lines 0 to 10:
jerry: do you know what this is all about? do you know, why were here? to be out, this is out...and out is one of the single most enjoyable experiences of life. people...did you ever hear people talking about we should go out? this is what theyre talking about...this whole thing, were all out now, no one is home. not one person here is home, were all out! there are people trying to find us, they dont know where we are. (on an imaginary phone) did you ring?, i cant find him. where did he go? he didnt tell me where he was going. he must have gone out. you wanna go out you get ready, you pick out the clothes, right? you take the shower, you get all ready, get the cash, get your friends, the car, the spot, the reservation...then youre standing around, what do you do? you go we gotta be getting back. once youre out, you wanna get back! you wanna go to sleep, you wanna get up, you wanna go out again tomorrow, right? where ever you are in life, its my feeling, youve gotta go. 

jerry: (pointing at georges shirt) see, to me, that button is in the worst possible spot. the second button literally makes or breaks the shirt, look at it. its too high! its in no-mans-land. you look like you live with your mother. 

george: are you through? 

jerry: you do of course try on, when you buy? 

george: yes, it was purple, i liked it, i dont actually recall considering the buttons. 
```

---

### Implement Pre-processing Functions

The first thing to do to any dataset is pre-processing. We implement the following pre-processing functions:
- Lookup Table
- Tokenize Punctuation

#### Lookup Table

To create a word embedding, we first need to transform the words to ids. We create two dictionaries:
- Dictionary to go from the words to an id: `vocab_to_int`
- Dictionary to go from the id to word: `int_to_vocab`

```python
import problem_unittests as tests
from collections import Counter

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """

    word_counts = Counter(text)
    
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    
    return (vocab_to_int, int_to_vocab)


tests.test_create_lookup_tables(create_lookup_tables)
```

**Output:**
```
Tests Passed
```

---

### Tokenize Punctuation

We'll be splitting the script into a word array using spaces as delimiters. However, punctuations like periods and exclamation marks can create multiple ids for the same word.

We create a dictionary for the following symbols where the symbol is the key and value is the token:

- Period ( **.** )
- Comma ( **,** )
- Quotation Mark ( **"** )
- Semicolon ( **;** )
- Exclamation mark ( **!** )
- Question mark ( **?** )
- Left Parentheses ( **(** )
- Right Parentheses ( **)** )
- Dash ( **-** )
- Return ( **\n** )

```python
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    tokens = {
                '.': '||period||',
                ',': '||comma||',
                '"': '||quotation_mark||',
                ';': '||semicolon||',
                '!': '||exclamation_mark||',
                '?': '||question_mark||',
                '(': '||left_parentheses||',
                ')': '||right_Parentheses||',
                '-': '||dash||',
                '\n': '||return||'
            }
    
    return tokens

tests.test_tokenize(token_lookup)
```

**Output:**
```
Tests Passed
```

---

### Pre-process all the data and save it

We pre-process all the data and save it to a file.

```python
# pre-process training data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
```

#### Check Point

This is the first checkpoint. If we have to restart the notebook, we can start from here. The preprocessed data has been saved to disk.

```python
import helper
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
```

---

### Build the Neural Network

In this section, we'll build the components necessary to build an RNN by implementing the RNN Module and forward and backpropagation functions.

#### Check Access to GPU

```python
import torch

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
```

### Input

We'll use [TensorDataset](http://pytorch.org/docs/master/data.html#torch.utils.data.TensorDataset) to provide a known format to our dataset; in combination with [DataLoader](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader), it will handle batching, shuffling, and other dataset iteration functions.

```python
data = TensorDataset(feature_tensors, target_tensors)
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
```

### Batching

Implement the `batch_data` function to batch `words` data into chunks of size `batch_size` using the `TensorDataset` and `DataLoader` classes.

For example, say we have these as input:
```
words = [1, 2, 3, 4, 5, 6, 7]
sequence_length = 4
```

Our first `feature_tensor` should contain the values:
```
[1, 2, 3, 4]
```

And the corresponding `target_tensor` should just be the next "word"/tokenized word value:
```
5
```

```python
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """

    # number of batches
    number_batches = len(words)//batch_size     
    
    # only take full batches
    words = words[:number_batches*batch_size]
        
    # x -> feature , y -> target
    x, y = [], []
    
    for ii in range(0, len(words)- sequence_length):
        x.append(words[ii:ii+sequence_length])
        y.append(words[ii + sequence_length])
    
       
    # convert numpy arrays to tensors
    x_tensors = torch.from_numpy(np.array(x))
    y_tensors = torch.from_numpy(np.array(y))
    
    
    # Dataset wrapping tensors
    data = TensorDataset(x_tensors, y_tensors)
    
    # multi-process iterators over the dataset (our data loader)
    data_loader = torch.utils.data.DataLoader(data, shuffle=True,
                                          batch_size=batch_size)
    
    # return a dataloader
    return data_loader
       

# Test the batch_data function
words = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
sequence_length = 3
data_loader = batch_data(words, sequence_length, 3)

for feature, target in data_loader:
    print("Feature: "+ str(feature))
    print("Target: "+str(target))
    print("Feature shape: "+str(feature.shape))
    print("Target shape: "+str(target.shape))
```

**Output:**
```
Feature: tensor([[  3,   4,   5],
                 [  2,   3,   4],
                 [  9,  10,  11]])
Target: tensor([  6,   5,  12])
Feature shape: torch.Size([3, 3])
Target shape: torch.Size([3])
Feature: tensor([[  5,   6,   7],
                 [  8,   9,  10],
                 [  1,   2,   3]])
Target: tensor([  8,  11,   4])
Feature shape: torch.Size([3, 3])
Target shape: torch.Size([3])
Feature: tensor([[ 4,  5,  6],
                 [ 7,  8,  9],
                 [ 6,  7,  8]])
Target: tensor([  7,  10,   9])
Feature shape: torch.Size([3, 3])
Target shape: torch.Size([3])
```

---

### Test your dataloader 

Below, we're generating some test text data and defining a dataloader using the function defined above. Then, we get some sample batch of inputs `sample_x` and targets `sample_y` from our dataloader.

Our sample_x should be of size `(batch_size, sequence_length)` or (10, 5) in this case and sample_y should just have one dimension: batch_size (10).

The targets, sample_y, are the *next* value in the ordered test_text data. So, for an input sequence `[ 28,  29,  30,  31,  32]` that ends with the value `32`, the corresponding output should be `33`.

```python
# test dataloader

test_text = range(50)
t_loader = batch_data(test_text, sequence_length=5, batch_size=10)

data_iter = iter(t_loader)
sample_x, sample_y = data_iter.next()

print(sample_x.shape)
print(sample_x)
print()
print(sample_y.shape)
print(sample_y)
```

**Output:**
```
torch.Size([10, 5])
tensor([[ 26,  27,  28,  29,  30],
        [  3,   4,   5,   6,   7],
        [ 29,  30,  31,  32,  33],
        [ 28,  29,  30,  31,  32],
        [ 22,  23,  24,  25,  26],
        [ 44,  45,  46,  47,  48],
        [ 10,  11,  12,  13,  14],
        [ 25,  26,  27,  28,  29],
        [  4,   5,   6,   7,   8],
        [ 36,  37,  38,  39,  40]])

torch.Size([10])
tensor([ 31,   8,  34,  33,  27,  49,  15,  30,   9,  41])
```

---

### Build the Neural Network

We implement an RNN using PyTorch's [Module class](http://pytorch.org/docs/master/nn.html#torch.nn.Module). To complete the RNN, we implement the following functions:
 - `__init__` - The initialize function
 - `init_hidden` - The initialization function for an LSTM/GRU hidden state
 - `forward` - Forward propagation function

**The output of this model should be the *last* batch of word scores** after a complete sequence has been processed. That is, for each input sequence of words, we only want to output the word scores for a single, most likely, next word.

#### Notes:
- Make sure to stack the outputs of the lstm to pass to your fully-connected layer: `lstm_output = lstm_output.contiguous().view(-1, self.hidden_dim)`
- Get the last batch of word scores by shaping the output of the final, fully-connected layer:

```python
# reshape into (batch_size, seq_length, output_size)
output = output.view(batch_size, -1, self.output_size)
# get last batch
out = output[:, -1]
```

```python
import torch.nn as nn

class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        
        # set class variables
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.dropout = nn.Dropout(0.20)
 
        # define model layers
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout = dropout, batch_first=True)
        
        
        # linear fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        
        
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        
        # first dimension is batch size
        batch_size = nn_input.size(0)
        
        # embeddings and lstm_out
        nn_input = nn_input.long()
        embeds = self.embedding(nn_input)
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # dropout and fully-connected layer
        output = self.dropout(lstm_out)
        output = self.fc(output)
        
        # reshape to be batch_size first
        output = output.view(batch_size, -1, self.output_size)
        out = output[:, -1] # get last batch of labels
        
        # return one batch of output word scores and the hidden state
        return out, hidden
        
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        
        # initialize hidden state with zero weights, and move to GPU if available
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

tests.test_rnn(RNN, train_on_gpu)
```

**Output:**
```
Tests Passed
```

---

### Define forward and backpropagation

Using the RNN class to apply forward and back propagation. This function will be called iteratively in the training loop:

```python
loss = forward_back_prop(decoder, decoder_optimizer, criterion, inp, target)
```

```python
def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    
    # move data to GPU, if available
    if (train_on_gpu):
        inp, target = inp.cuda(), target.cuda()
 
    # perform backpropagation and optimization
    hidden = tuple([each.data for each in hidden])

    # zero accumulated gradients
    rnn.zero_grad()

    # get the output from the model
    output, hidden = rnn(inp, hidden)

    # calculate the loss and perform backprop
    loss = criterion(output, target)
    loss.backward()
    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    clip = 5
    nn.utils.clip_grad_norm_(rnn.parameters(), clip)
    optimizer.step()    
    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hidden

tests.test_forward_back_prop(RNN, forward_back_prop, train_on_gpu)
```

**Output:**
```
Tests Passed
``` 

---

### Neural Network Training

With the structure of the network complete and data ready to be fed in the neural network, it's time to train it.

### Train Loop

The training loop is implemented in the `train_decoder` function. This function will train the network over all the batches for the number of epochs given.

```python
def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    batch_losses = []
    
    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)
        
        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            
            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
            
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []

    # returns a trained rnn
    return rnn
```

### Hyperparameters

Set and train the neural network with the following parameters:

#### Data parameters:
- **Sequence Length**: 10 (words in a sequence)
- **Batch Size**: 256

#### Training parameters:
- **Number of Epochs**: 15
- **Learning Rate**: 0.001

#### Model parameters:
- **Vocab Size**: len(vocab_to_int)
- **Output Size**: vocab_size
- **Embedding Dimension**: 200
- **Hidden Dimension**: 300
- **Number of RNN Layers**: 2
- **Show stats every n batches**: 500

---

### Train

We train the neural network on the pre-processed data. **We are aiming for a loss less than 3.5.**

```python
# create model and move to gpu if available
rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
if train_on_gpu:
    rnn.cuda()

# defining loss and optimization functions for training
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# training the model
trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)

# saving the trained model
helper.save_model('./save/trained_rnn', trained_rnn)
print('Model Trained and Saved')
```

**Output:**
```
Training for 15 epoch(s)...
Epoch:    1/15    Loss: 5.400850810050964
..
..
..
Epoch:   15/15    Loss: 3.294407793521881

Model Trained and Saved
```

---

#### Checkpoint

After running the above training cell, our model will be saved by name, `trained_rnn`, and if you save your notebook progress, **you can pause here and come back to this code at another time**.

```python
import torch
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
trained_rnn = helper.load_model('./save/trained_rnn')
```

---

## Generate TV Script

With the network trained and saved, we'll use it to generate a new, "fake" Seinfeld TV script.

#### Generate Text

To generate the text, the network needs to start with a single word and repeat its predictions until it reaches a set length. The `generate` function uses topk sampling to introduce some randomness in choosing the most likely next word.

```python
import torch.nn.functional as F

def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, predict_len=100):
    """
    Generate text using the neural network
    :param decoder: The PyTorch Module that holds the trained neural network
    :param prime_id: The word id to start the first prediction
    :param int_to_vocab: Dict of word id keys to word values
    :param token_dict: Dict of puncuation tokens keys to puncuation values
    :param pad_value: The value used to pad a sequence
    :param predict_len: The length of text to generate
    :return: The generated text
    """
    rnn.eval()
    
    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, sequence_length), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]
    
    for _ in range(predict_len):
        if train_on_gpu:
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)
        
        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0))
        
        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)
        
        # get the next word probabilities
        p = F.softmax(output, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
         
        # use top_k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()
        
        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p/p.sum())
        
        # retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)     
        
        # the generated word becomes the next "current sequence" and the cycle can continue
        current_seq = np.roll(current_seq, -1, 1)
        current_seq[-1][-1] = word_i
    
    gen_sentences = ' '.join(predicted)
    
    # Replace punctuation tokens
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')
    
    # return all the sentences
    return gen_sentences
```

---

### Generate a New Script

Set `gen_length` to the length of TV script you want to generate and set `prime_word` to one of the following to start the prediction:
- "jerry"
- "elaine"
- "george"
- "kramer"

```python
# run the cell multiple times to get different results!
gen_length = 400 # modify the length to your preference
prime_word = 'jerry' # name for starting the script

pad_word = helper.SPECIAL_WORDS['PADDING']
generated_script = generate(trained_rnn, vocab_to_int[prime_word + ':'], int_to_vocab, token_dict, vocab_to_int[pad_word], gen_length)
print(generated_script)
```
    
**Output:**
```
jerry:."

jerry:(to jerry) oh, yeah, yeah.

jerry:(to himself) you see?

elaine: oh, yeah...

jerry: hey.

kramer: well, i got to be a very interesting driver.(george nods)

kramer:(pointing at the table) you know, i just had to go to the bathroom. i was wondering if you want to go with him, i don't know what you want, i can't.

george:(laughs) oh, no... i just got a little more than the best way to be in the bathroom?

jerry: no, it's all the way.

george: oh, well. you don't have to go.

jerry:(confused) what?

elaine: well, you know, you know..

jerry:(pointing at the counter) hey.

jerry: hey!

elaine:(quietly) yeah. i got it. i don't want to have any money.

elaine:(pointing out) what is this?

george: i can't. i don't know...

george: oh, hi. hi jerry.

jerry: hi, hi.

elaine: hi, jerry.

elaine:(shouting) what are you talking about?

jerry: yeah, i don't know what i do.

george: i don't know. i don't know...

jerry: i don't have it. i mean, i don't want to be able to tell you this. you know what i want to say,"

george:" what happened?

kramer: well, i don't think you know, i just had a little.

kramer: yeah, i think i got the tape.

jerry: i don't know. but i don't know how to have a good time.

elaine: oh my god. you want to know?

jerry: well, i just don't want to have a
```

---

#### Save your favorite scripts

Once you have a script that you like (or find interesting), save it to a text file!

```python
# save script to a text file
f = open("generated_script_1.txt","w")
f.write(generated_script)
f.close()
```

---

### The TV Script is Not Perfect

It's ok if the TV script doesn't make perfect sense. It should look like alternating lines of dialogue. Here is one example of a few generated lines:

```
jerry: what about me?

jerry: i don't have to wait.

kramer:(to the sales table)

elaine:(to jerry) hey, look at this, i'm a good doctor.

newman:(to elaine) you think i have no idea of this...

elaine: oh, you better take the phone, and he was a little nervous.

kramer:(to the phone) hey, hey, jerry, i don't want to be a little bit.(to kramer and jerry) you can't.

jerry: oh, yeah. i don't even know, i know.

jerry:(to the phone) oh, i know.

kramer:(laughing) you know...(to jerry) you don't know.
```

We can see that there are multiple characters that say (somewhat) complete sentences, but it doesn't have to be perfect! It takes quite a while to get good results, and often, you'll have to use a smaller vocabulary (and discard uncommon words), or get more data.

---

### Conclusion

We've successfully developed an algorithm that generates original TV scripts by training on Seinfeld dialogue data. The model was initially built using RNNs, and we leveraged pre-processing functions to prepare the data for training. We then implemented the RNN Module and forward and backpropagation functions to build the neural network. Finally, we trained the network using a train loop and hyperparameters.

The complete implementation can be found in the [GitHub repository](https://github.com/shrikantnaidu/Generate-TV-Scripts).

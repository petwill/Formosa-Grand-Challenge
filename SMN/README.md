# Sequential Matching Network

## Prerequisite
- all `BOLD_WORDS` represent an argument that can be modified.

## Requirements
- python3 >= 3.6
- python3-package : **tensorflow-gpu>=1.2**, jieba, csv, tqdm, gensim, numpy
- zsh

## File Description
- cleanup.sh : clean up the messy training data
- pretrain.py : generate files for training
- pretest.py : generate a file for testing
- train.py : train the model
- test.py : test pretrained model
- config.py : configuration file for `train.py` and `test.py`
- go.sh : all-in-one file from generate TFRecord file to testing

## Usage
### Process
1. The project directory should look like the following one for convenience, or
   else many settings have to be changed:

```
   root
   +-- files listed in {File Description}
   +-- data
   |   +-- training_data
   |   |   +-- subtitle_no_TC
   |   |   +-- subtitle_with_TC
   |   +-- dic.txt.big
   +-- submission
```

`submission` is an empty directory for storing predictions for kaggle
submission

2. Run `cleanup.sh`, which will create a `cleaned_data` directory under `data`
   containing all text files for furthur usage. This action should only be done
   once.

3. Run `pretrain.py`, which will generate TFRecord file, word embedding and
   vocabulary list for
   training. The word embedding is trained on the original dataset with the aid of
   [gensim](https://radimrehurek.com/gensim/models/word2vec.html). Words that are
   not in the final vocabulary list are not replaced by UNK during training.

4. Run `train.py`, which will start the training process. Typically, at least 8G
   of GPU memory is required, though decreasing `BATCH_SIZE` will lower the
   consumption. Entropy per `INFO_EPOCH` epoch(s) will be printed on the terminal.

5. Run `pretest.py`, which will read the vocabulary list that is produced in step 3 in order to
   generate testing TFRecord file.

6. Run `test.py`, which will test the trained model on a testing dataset. The
   number of steps that the trained model have updated will be shown and make sure
   that this number is not zero. Generated predictions will be located in 
   the `submission` directory.

***Since some arguments between these files should be shared, it is 
recommended to just use "go.sh [PRETRAIN] [PRETEST] [TRAIN] [TEST]" for step 2~6 
instead of executing individually.***

### Help
To see the help mesage for `pretrain.py`, `pretest.py`, `train.py`, `test.py`,
type :
```
./*.py -h
```

## Note
1. `LOG_DIR` is the directory that store all the information for furthur usage, including graph definition, 
   tensorboard summaries and pretrained model. During training, the model will be saved for every
   `SAVE_MODEL_SECS` seconds, and will only kept the 5 most recent ones. When
   executing `train.py` or `test.py`, if there exists a log directory with the
   same name, it will automatically load in previous weights; otherwise it will create
   a fresh one.
2. To add new training data, first format the data according to the following
   rules:
   1. two complete dialogues should be seperated by exactly two newlines
   2. two sentences in a diagloue should be seperated by exactly one newline

   Then, move the file into `data/cleaned_data`.

## Todo
- [x] discrepancy of punctuation marks between testing and training
- [ ] add more data
- [x] add learning rate decay factor
- [ ] add dropout
- [x] add validation set
- [x] try bidirectional rnn (`USE_BIDIRECTIONAL` or `UB`)

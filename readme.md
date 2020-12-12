## Colab Notebook
Final version of our project code is available as Google Colab Notebook which is shared and available at   [SequenceClassification](https://colab.research.google.com/drive/1nhsCc1krBzPR6LKg3Qfwq_cxHv4sr_Ib?usp=sharing). Below are the several steps detailed to be execute to run the code end to end.
Since modelling methodology is Transformer based it would be recommended to use GPU for processing. Google Colab is the preferred environment to run the end to end process and generate the results.

## Testing the model and Reproducing answer.txt
User can run [RoBERTa_Model_Test](https://colab.research.google.com/drive/1S9g8dD7JmuT6JsJo1ysAa4e3nTCNakxk?usp=sharing) to use the checkpoint of our model and reproduce answer.txt. 

Tutorial: [Reproduce_Answers_with_Checkpoint](TOADDLINK)

Checkpoint available to Download from: [Checkpoint](https://drive.google.com/file/d/1z1IIeU1e7DgqtAyyPWE66QyAG7h1D-sT/view?usp=sharing)

See slide 11 for detailed step-by-step guidance [Presentation](https://drive.google.com/file/d/1z1IIeU1e7DgqtAyyPWE66QyAG7h1D-sT/view?usp=sharing)

## Contribution of Team Members
Each of us collaborated very closely in each step of reseach, experiment, and improvise. We have touch point scheduled on a weekly basis, where we shared the learning and resources, discuss our approach, and walk-through our codes. With that, each of us contributed 100% effort in each step of the project process.

## Below are the Steps in our Model Training and Output Generation 
(as in [SequenceClassification](https://colab.research.google.com/drive/1nhsCc1krBzPR6LKg3Qfwq_cxHv4sr_Ib?usp=sharing).)

## Environment Setup
First setup the environment, we will do the following steps here.
- Transformers Model Installation
- Hyper Parameter Tuning Library Installation
- Colab Setup

You will be required to authorize the code using your google account. Copy the authorization code generated and pass it in the notebook in the input box provided when you run  mount drive code. Below is the reference:
```py
# Colab setup
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```
Also please copy the train & test JSONL files provided in your google drive required for training and testing the models further.

Tutorial: [Environment Setup](https://drive.google.com/file/d/1p9c4u6m04NFf1mT4c-VoMP7q4Nm0p6U1/view?usp=sharing)
## Data Load
Next step is to load the Training & Test Datasets as Pandas dataframe. Please update correct data path where training and test dataset is copied in your google drive.
```py
datapath = r'/content/drive/My Drive/mcsds/cs-410-text-mining/project/ClassificationCompetition/data'
train_pddf = pd.read_json(datapath+'/train.jsonl', lines=True)
test_pddf = pd.read_json(datapath+'/test.jsonl', lines=True)
```
Above example suggests my train & test jsonl files are copied in my drive at '/mcsds/cs-410-text-mining/project/ClassificationCompetition/data' location.

Further run the data load section.
Reference: [Data Load & Preprocessing](https://drive.google.com/file/d/1RlV-zoHJgMvMbd3G2mHXWwuS2O52Njtc/view?usp=sharing)
## Data Preprocessing
Next step is to run the data preprocessing steps. Below are the different components of it:
### Feature Engineering
Create new features:
* Last Response: Extract the last response from the context since the current response was generated on Last this can be separately treated.
* Context Reversed: Reverse the context before feeding to transformers so that latest tweets are given more attention and incase if context is too big latest shall be considered.
* Combine all into a single
* Combine Current & Last Response into Single
### Sequence Structuring
Define how do we want to structure the different tweets, basically two approaches are followed:
* Combine into single: Last response only, Combine all tweets togeather or current and last.
* Two Sentence: (Current, Last Response) or (Current, Context Reversed).

### Transform to Datasets
Translate preprocessed dataframes to Transformer Datasets. This step is required to make our dataset translated into Transformer datasets construct.
Reference: [Data Load & Preprocessing](https://drive.google.com/file/d/1RlV-zoHJgMvMbd3G2mHXWwuS2O52Njtc/view?usp=sharing)

## Model Configurtion
Configure which model strategy to select, train test valid splits, performance metrics, training batch sizes etc. Below are the details:
1.   model_checkpoint: which model to use for text sequence classification. Roberta models are observed to give the maximum performance.
2.   task: specify how to structure the sequences as described in sequence structuring step. We have observed the maximum performance with 'response_context_rev_sep' structure. This format structures input as two sequence <response, context> where response is last tweet to be classified, and context tweets are the previous tweets in an reversed order of occurance.
3. metric_name: metric to be optimized while training. We have configured it to accuracy.
4. num_labels: 2, number of classes Sarcasm, Not Sarcasm
5. batch_size: 16 for roberta, 64 for bert otherwise we face out of memory issues.
6. train_test_split: to divide training data into train and test datasets.
7. test_valid_split: to divide test dataset into test and validation set. 
8. epoch: number of epochs to train model on.
9. weight_decay: determines how much an updating step influences the current value of the weights
10. learning_rate: weight update rule that causes the weights to exponentially decay to zero

Reference: [Model Config](https://drive.google.com/file/d/1IOWOvfrQgxSzDK7pQ4-U_X10sqRonLzs/view?usp=sharing)

## Tokenization
This step translates words to context tokens. Transformers Tokenizer tokenize the inputs (including converting the tokens to their corresponding IDs in the pretrained vocabulary) and put it in a format the model expects, as well as generate the other inputs that model requires.

Reference: [Tokenization & Single Model Fine Tuning](https://drive.google.com/file/d/1ZrYGUZZijx207dnLpsBSAiDdTbXFYDem/view?usp=sharing)

## Single Model Fine Tuning
Download the pretrained model and fine tune the selected model with arguments configured in the previous step.

Reference: [Tokenization & Single Model Fine Tuning](https://drive.google.com/file/d/1ZrYGUZZijx207dnLpsBSAiDdTbXFYDem/view?usp=sharing)
Reference: [Training Results](https://drive.google.com/file/d/11zO6pl_p0HuZu3ejrgEW06zRW2YR-oGp/view?usp=sharing)

## Test Validation
Validate the results on test data and compute the metrics.

Reference: [Validation Results](https://drive.google.com/file/d/1VaZCl6JHtoqAWgDCa6p7nz9QDsaCTfOx/view?usp=sharing)

## Hyper Parameter Tuning
**Could be only run with HIGH GPU environment**
Using Transformer Trainer utility which supports hyperparameter search using optuna or Ray Tune libraries which we have installed in our previous step. During hyperparameter tuning step, the Trainer will run several trainings, so it needs to have the model defined via a function (so it can be reinitialized at each new run) instead of just having it passed. The hyperparameter_search method returns a BestRun objects, which contains the value of the objective maximized and the hyperparameters it used for that run.

Reference: [Hyperparameter Tuning](https://drive.google.com/file/d/1J3pAIoJPyF7jeYBtJMQb63rz-fROB8th/view?usp=sharing)
# Best Run Selection & Training
**Could be only run with HIGH GPU environment**
To reproduce the best training run from our previous hyper parameter train setp we will set the best hyperparameters  TrainingArgument before training the model again.
Reference: [Hyperparameter Tuning](https://drive.google.com/file/d/1J3pAIoJPyF7jeYBtJMQb63rz-fROB8th/view?usp=sharing)


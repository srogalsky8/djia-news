""" 
Finetuning BERT - 

Best to use GPU on Google Colab. 
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing as pp
import random
import time
import torch

# %pip install transformers
from sklearn.metrics import f1_score
from transformers import (
    BertTokenizer,
    BertModel,
    BertForSequenceClassification,
    AdamW,
    BertConfig,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import (
    TensorDataset,
    random_split,
    DataLoader,
    SequentialSampler,
    RandomSampler,
)

# from google.colab import files
# files.upload()

# torch.cuda.is_available()
# cuda = torch.device('cuda')

""" 
Process train data. 
"""


def process_tweets(path):
    twt = pd.read_csv(path)
    add_row = twt.columns.to_numpy()
    twt.loc[len(twt.index)] = add_row
    twt.columns = ["Index", "Source", "Sentiment", "Tweet"]

    x = pd.DataFrame(twt["Tweet"])
    y = pd.DataFrame(twt["Sentiment"])

    y.drop(x[x.isnull().any(axis=1)].index.to_numpy(), inplace=True)
    y.reset_index(inplace=True)
    y.drop("index", axis=1, inplace=True)

    x.drop(x_tr[x.isnull().any(axis=1)].index.to_numpy(), inplace=True)
    x.reset_index(inplace=True)
    x.drop("index", axis=1, inplace=True)

    y = y.to_numpy()
    x = x.to_numpy()

    y[y == "Positive"] = 0
    y[y == "Negative"] = 1
    y[y == "Neutral"] = 2
    y[y == "Irrelevant"] = 3

    x = x.flatten()
    y = y.astype(int).flatten()

    return (x, y)


def sequential_tokenizer(dat, labs):
    input_ids = []
    attention_masks = []
    for sentence in dat:
        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=64,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
            trucation=True,
        )
        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])
    # Uncomment below to use cuda in colab
    # input_ids = torch.cat(input_ids, dim = 0).cuda()
    # attention_masks = torch.cat(attention_masks,dim=0).cuda()
    # labels = torch.tensor(labs).type(torch.Longtensor).cuda()
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labs).type(torch.Longtensor)

    return (input_ids, attention_masks, labels)


""" 
Process dat
"""

# tr_path_colab  = "./twitter_training.csv"
tr_path_local = "./data/twitter_training.csv"

(x_tr, y_tr) = process_tweets(tr_path_local)

# val_path_colab = "./twitter_validation.csv"
val_path_local = "./data/twitter_validation.csv"

(x_val, y_val) = process_tweets(val_path_local)

"""
Initialize tokenizer and tokenize dat
"""
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
(input_ids, attention_masks, labels) = sequential_tokenizer(x_tr, y_tr)
(input_ids_val, attention_masks_val, labels_val) = sequential_tokenizer(x_val, y_val)

"""
Generate dataloaders
"""
train_dat = TensorDataset(input_ids, attention_masks, labels)
val_dat = TensorDataset(input_ids_val, attention_masks_val, labels_val)
batch_size = 32
train_dataloader = DataLoader(
    train_dat, sampler=RandomSampler(train_dat), batch_size=batch_size
)
validation_dataloader = DataLoader(
    val_dat, sampler=SequentialSampler(val_dat), batch_size=batch_size
)
"""
Initialize model 
"""
# model = BertForSequenceClassification.from_pretrained(
#     "bert-base-uncased",
#     num_labels = 4,
#     output_attentions = False,
#     output_hidden_states = False).cuda()

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4,
    output_attentions=False,
    output_hidden_states=False,
)
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 5
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)


"""
Begin training loop
"""
seed_val = 256
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds.numpy(), axis=1).flatten()
    labels_flat = labels.numpy().flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds.numpy(), axis=1).flatten()
    labels_flat = labels.numpy().flatten()
    return f1_score(labels_flat, preds_flat, average="weighted")


training_stats = []
val_loss_tracker = 100
tolerance = 0.01
start = time.time()

for epoch_i in range(0, epochs):
    print("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
    total_train_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        # b_input_ids = batch[0].cuda()
        # b_input_mask = batch[1].cuda()
        # b_labels = batch[2].cuda()
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]
        model.zero_grad()
        outputs = model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels,
        )
        # loss = outputs.loss.cuda()
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_train_loss / len(train_dataloader)
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    # evaluation
    model.eval()
    total_eval_accuracy = 0
    total_eval_f1 = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    for batch in validation_dataloader:
        # b_input_ids = batch[0].cuda()
        # b_input_mask = batch[1].cuda()
        # b_labels = batch[2].cuda()

        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]

        with torch.no_grad():
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )
            # loss = outputs.loss.cuda()
            # logits = outputs.logits.cuda()
            loss = outputs.loss
            logits = outputs.logits

        total_eval_loss += loss.item()
        # total_eval_f1 += f1_score_func(logits.cpu(), b_labels.cpu())
        # total_eval_accuracy += flat_accuracy(logits.cpu(), b_labels.cpu())
        total_eval_f1 += f1_score_func(logits.cpu(), b_labels)
        total_eval_accuracy += flat_accuracy(logits.cpu(), b_labels)

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    avg_val_f1 = total_eval_f1 / len(validation_dataloader)
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    print("  F1: {0:.2f}".format(avg_val_f1))
    print("  Loss: {0:.2f}".format(avg_val_loss))
    if avg_val_loss < val_loss_tracker - tolerance:
        val_loss_tracker = avg_val_loss
        # torch.save(model.state_dict(), "best_model.pt")
        torch.save(model.state_dict(), "./saved_model/best_model.pt")
    elif avg_val_loss > val_loss_tracker + tolerance:
        end = time.time()
        print("Total Time:", start - end, ". Exiting")
        break

end = time.time()
total = start - end
print(total)

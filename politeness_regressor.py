from argparse import ArgumentParser
from typing import List, Dict

import numpy as np
import pytorch_lightning as pl
import sklearn.metrics
import sklearn.model_selection
import torch
import torch.optim
import torch.utils.data
import transformers
import pandas as pd
import random
import sklearn.metrics

try:
    from polyglot.text import Text
except:
    print("polyglot not installed. Cannot use --strategy_words")

class MyDataModule(pl.LightningDataModule):
    def __init__(self, train_file, test_file, binary, tokenizer, max_length, batch_size, strategy_words_replacement_negate=False, strategy_words=None, random_masking_ratio=None):
        super().__init__()
        self.train_file = train_file
        self.test_file = test_file
        self.binary = binary
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = tokenizer

        if strategy_words:
            self.strategy_words = pd.read_csv(strategy_words)
            self.strategy_words = set(list(self.strategy_words.values[:, 1:].reshape(-1)))
        else:
            self.strategy_words = None
        self.strategy_words_replacement_negate = strategy_words_replacement_negate
        self.random_masking_ratio = random_masking_ratio

    @staticmethod   
    def read_file(file_name, text_only=False):
        if file_name.split(".")[-1] == "csv":
            df = pd.read_csv(file_name)
            data = [(a, b) for a, b in zip(list(df['sentence']), df['score'])]
            if text_only:
                data = [t[0] for t in data]
        else:
            data = open(file_name).read().strip().split('\n')
        return data

    def setup(self, stage=None):
        if self.train_file:
            self.train_data = MyDataModule.read_file(self.train_file)
            self.train_data, self.val_data = sklearn.model_selection.train_test_split(self.train_data, shuffle=False, test_size=0.2)
        if self.test_file:
            self.test_data = MyDataModule.read_file(self.test_file)

    def prepare_dataloader(self, mode):
        if mode == "train":
            data = self.train_data
        elif mode == "val":
            data = self.val_data
        else:
            data = self.test_data

        # tokenized = self.tokenizer([t[0] for t in data], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        tokenized = MyDataModule.tokenize([t[0] for t in data], self.tokenizer, self.max_length, self.strategy_words_replacement_negate, self.strategy_words, self.random_masking_ratio)
        if self.binary:
            labels = torch.tensor([t[1] > 0 for t in data], dtype=int)
        else:
            labels = torch.tensor([t[1] for t in data])

        if mode == "train":
            weights = torch.zeros_like(labels)
            weights[labels == 0] = labels.shape[0] - labels.sum()
            weights[labels == 1] = labels.sum()
            return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tokenized['input_ids'], tokenized['attention_mask'], labels), batch_size=self.batch_size, sampler=torch.utils.data.WeightedRandomSampler(1 / weights, len(weights), replacement=True))
        else:
            return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tokenized['input_ids'], tokenized['attention_mask'], labels), batch_size=self.batch_size)

    @staticmethod
    def tokenize(data: List[str], tokenizer, max_length, strategy_words_replacement_negate, strategy_words, random_masking_ratio):
        if strategy_words is not None or random_masking_ratio is not None:
            tokenized_data = []
            for sentence in data:
                words = Text(sentence).words
                words = [t.lower() for t in words]
                if strategy_words:
                    words = [t if ((t in strategy_words) != strategy_words_replacement_negate) else tokenizer.mask_token for t in words]
                elif random_masking_ratio:
                    words = [t if random.random() <= random_masking_ratio else tokenizer.mask_token for t in words]
                tokenized_data.append(' '.join(words))
            out = tokenizer(tokenized_data, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            # out['attention_mask'] = torch.tensor(out['input_ids'] != tokenizer.pad_token_id, dtype=int)
            return out
        else:
            return tokenizer(data, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

    def train_dataloader(self):
        return self.prepare_dataloader("train")
        # return torch.utils.data.DataLoader(MyDataModule.CustomDataset1(self.tokenizer, self.train_data, self.max_length), batch_size=self.batch_size)

    def test_dataloader(self):
        return self.prepare_dataloader("test")
        # return torch.utils.data.DataLoader(MyDataModule.CustomDataset1(self.tokenizer, self.test_data, self.max_length), batch_size=self.batch_size)

    def val_dataloader(self):
        return self.prepare_dataloader("val")
        # return torch.utils.data.DataLoader(MyDataModule.CustomDataset1(self.tokenizer, self.val_data, self.max_length), batch_size=self.batch_size)


class RegressionModel(pl.LightningModule):
    def __init__(self, pretrained_model, binary, learning_rate, num_warmup_steps, tokenizer):
        super(RegressionModel, self).__init__()
        self.save_hyperparameters()
        self.pretrained_model = pretrained_model
        self.binary = binary
        self.learning_rate = learning_rate
        self.num_warmup_steps = num_warmup_steps
        self.tokenizer = tokenizer
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(self.pretrained_model, num_labels=2 if self.binary else 1)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
        loss = outputs['loss']
        ret = {"loss": loss}
        if self.binary:
            acc = torch.tensor(batch[2] == torch.argmax(outputs['logits']), dtype=float).mean().item()
            ret["acc"] = acc
        else:
            rmse = (torch.mean((batch[2] - outputs['logits'])**2)**0.5).item()
            ret["rmse"] = rmse

        return {"loss": loss, "log": ret}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, self.num_warmup_steps, len(self.trainer.datamodule.train_dataloader()) // self.trainer.accumulate_grad_batches)
        return [optimizer], [scheduler]

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, mode="test")

    def validation_step(self, batch, batch_idx, mode="val"):
        outputs = self.forward(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
        loss = outputs['loss']
        self.log("{}_loss".format(mode), loss, prog_bar=True)

        ret = {"loss": loss}
        if self.binary:
            preds = torch.argmax(outputs['logits'], axis=1).tolist()
            gold = batch[2].tolist()
            ret["preds"] = preds
            ret["gold"] = gold
            # f1 = sklearn.metrics.f1_score(gold, preds)
            # acc = sklearn.metrics.accuracy_score(gold, preds)
            # ret["acc"] = acc
            # ret["f1"] = f1
            # self.log("{}_acc".format(mode), acc, prog_bar=True)
            # self.log("{}_f1".format(mode), f1, prog_bar=True)
        else:
            preds = outputs['logits'].tolist()
            gold = batch[2].tolist()
            ret['preds'] = preds
            ret['gold'] = gold
            # rmse = (torch.mean((batch[2] - outputs['logits'])**2)**0.5).item()
            # self.log("{}_rmse".format(mode), rmse, prog_bar=True)
            # ret["rmse"] = rmse

        return {"loss": loss, "log": ret}

    def validation_epoch_end(self, outputs, mode="val"):
        gold = []
        preds = []
        for batch in outputs:
            gold.extend(batch['log']['gold'])
            preds.extend(batch['log']['preds'])
        if self.binary:
            f1 = sklearn.metrics.f1_score(gold, preds)
            acc = sklearn.metrics.accuracy_score(gold, preds)
            self.log("{}_acc".format(mode), acc, prog_bar=True)
            self.log("{}_f1".format(mode), f1, prog_bar=True)
        else:
            rmse = (torch.mean((torch.tensor(gold) - torch.tensor(preds))**2)**0.5).item()
            self.log("{}_rmse".format(mode), rmse, prog_bar=True)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, mode="test")

    def predict_step(self, batch, batch_idx):
        preds = self.forward(input_ids=batch[0], attention_mask=batch[1])
        if self.binary:
            ret = preds['logits'].tolist()
        else:
            ret = preds['logits'].view(-1).tolist()
        return ret

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("RegressionModel")
        parser.add_argument('--pretrained_model', type=str)
        parser.add_argument('--learning_rate', type=float, default="5e-6")
        parser.add_argument('--num_warmup_steps', type=float, default="0")
        return parent_parser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--load_model", type=str)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--model_save_location", type=str)
    parser.add_argument("--preds_save_location", type=str)
    parser.add_argument("--preds_save_logits", action="store_true")
    parser.add_argument("--strategy_words", type=str)
    parser.add_argument("--strategy_words_replacement_negate", action="store_true")
    parser.add_argument("--random_masking_ratio", type=float)
    parser = RegressionModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    print(args)

    pl.utilities.seed.seed_everything(seed=args.seed)
    if args.load_model:
        model = RegressionModel.load_from_checkpoint(args.load_model)
        tokenizer = model.tokenizer
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.pretrained_model)
        model = RegressionModel(pretrained_model=args.pretrained_model, binary=args.binary, learning_rate=args.learning_rate, num_warmup_steps=args.num_warmup_steps, tokenizer=tokenizer)
    trainer = pl.Trainer.from_argparse_args(args)

    dataset = MyDataModule(train_file=args.train_file, test_file=args.test_file, binary=model.binary, max_length=args.max_length, batch_size=args.batch_size, tokenizer=tokenizer, strategy_words_replacement_negate=args.strategy_words_replacement_negate, strategy_words=args.strategy_words, random_masking_ratio=args.random_masking_ratio)
    dataset.setup()

    if args.train:
        trainer.fit(model, dataset)

    if args.test:
        trainer.test(model, dataset.test_dataloader())

    if args.preds_save_location:
        data = MyDataModule.read_file(args.test_file, True)
        strategy_words = None
        if args.strategy_words:
            strategy_words = pd.read_csv(args.strategy_words)
            strategy_words = set(list(args.strategy_words.values[:, 1:].reshape(-1)))
        tokenized = MyDataModule.tokenize(data, tokenizer, args.max_length, args.strategy_words_replacement_negate, strategy_words, args.random_masking_ratio)
        input_data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tokenized['input_ids'], tokenized['attention_mask']), batch_size=args.batch_size)
        preds = trainer.predict(model, input_data, return_predictions=True)
        preds = [t for y in preds for t in y]
        preds = torch.tensor(preds)
        if model.binary:
            if args.preds_save_logits:
                preds = torch.softmax(preds, axis=1)[:, 1].tolist()
            else:
                preds = preds.argmax(axis=1).tolist()
        else:
            preds = preds.view(-1).tolist()
        preds = [str(t) for t in preds]

        with open(args.preds_save_location, 'w') as f:
            f.write('\n'.join(preds) + '\n')

    if args.model_save_location:
        trainer.save_checkpoint(args.model_save_location, weights_only=True)

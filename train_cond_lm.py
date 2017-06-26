
import os
import sys
import glob

seed = 1001
import random                   # nopep8
random.seed(seed)

import numpy as np

import torch                    # nopep8
try:
    torch.cuda.manual_seed(seed)
except:
    print('no NVIDIA driver found')
torch.manual_seed(seed)

import torch.nn as nn           # nopep8

from seqmod.modules.lm import ConditionalLM       # nopep8
from seqmod import utils as u  # nopep8

from seqmod.misc.trainer import ConditionalLMTrainer               # nopep8
from seqmod.misc.loggers import StdLogger, VisdomLogger  # nopep8
from seqmod.misc.optimizer import Optimizer              # nopep8
from seqmod.misc.dataset import Dict, ConditionalBlockDataset       # nopep8
from seqmod.misc.preprocess import text_processor        # nopep8
from seqmod.misc.early_stopping import EarlyStopping     # nopep8

import pandas as pd
from collections import Counter


class Metadata(object):

    def __init__(self, path='metainfo-master.csv'):

        self.path = path
        meta = pd.read_csv(path)
        meta = meta.set_index('filepath')
        selected = sorted('Aspe Cookson Coben'.split())
        self.idx = {}
        for s in selected:
            self.idx[s] = len(self.idx)
        self.meta = meta.loc[meta['author:lastname'].isin(selected)]
        self.default = np.zeros(len(self.idx), dtype='float32')

    def split_filenames(self, indir):
        conds, filenames = [], []
        for filepath in glob.glob(indir+'/*.txt'):
            fp = os.path.basename(filepath)
            fn, _ = os.path.splitext(fp)
            try:
                auth = self.meta.loc[fn]['author:lastname']
                filenames.append(filepath)
                conds.append(auth)
            except KeyError:
                pass

        from sklearn.model_selection import train_test_split
        train_conds, valid_conds, train_names, valid_names \
            = train_test_split(conds, filenames,
                               stratify=conds,
                               test_size=len(self.idx))
        train_conds, test_conds, train_names, test_names \
            = train_test_split(train_conds, train_names,
                               stratify=train_conds,
                               test_size=len(self.idx))
        
        return train_names, valid_names, test_names

    def conditions_from_path(self, filepath):
        fp = os.path.basename(filepath)
        fn, _ = os.path.splitext(fp)
        try:
            auth = self.meta.loc[fn]['author:lastname']
            v = self.default.copy()
            v[self.idx[auth]] = np.float32(1)
            return v
        except KeyError:
            return None


# Load data
def load_lines(names, processor=text_processor(),
               meta=None):
    lines, conditions = [], []
    for filename in names:
        conds = meta.conditions_from_path(filename)
        if isinstance(conds, np.ndarray):
            with open(filename) as f:
                for line in f:
                    line = line.strip()
                    if processor is not None:
                        line = processor(line)
                    if line:
                        lines.append(line)
                        conditions.append(conds)
    return lines, conditions


def load_from_file(path):
    if path.endswith('npy'):
        import numpy as np
        array = np.load(path).astype(np.int64)
        data = torch.LongTensor(array)
    elif path.endswith('.pt'):
        data = torch.load(path)
    else:
        raise ValueError('Unknown input format [%s]' % path)
    return data


# hook
def make_lm_check_hook(d, seed_text, max_seq_len=25, gpu=False,
                       method='sample', temperature=1, width=5,
                       early_stopping=None, validate=True,
                       seed_condition=None, nb_conditions=None):

    def hook(trainer, epoch, batch_num, checkpoint):
        trainer.log("info", "Checking training...")
        if validate:
            loss = trainer.validate_model()
            trainer.log("info", "Valid loss: %g" % loss)
            trainer.log("info", "Registering early stopping loss...")
            if early_stopping is not None:
                early_stopping.add_checkpoint(loss)
        trainer.log("info", "Generating text...")
        scores, hyps = trainer.model.generate(
            d, seed_text=seed_text, seed_condition=seed_condition,
            max_seq_len=max_seq_len, gpu=gpu, nb_conditions=nb_conditions,
            method=method, temperature=temperature, width=width)
        hyps = [u.format_hyp(score, hyp, hyp_num + 1, d)
                for hyp_num, (score, hyp) in enumerate(zip(scores, hyps))]
        trainer.log("info", '\n***' + ''.join(hyps) + "\n***")

    return hook


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--emb_dim', default=200, type=int)
    parser.add_argument('--hid_dim', default=200, type=int)
    parser.add_argument('--att_dim', default=0, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--word_dropout', default=0.0, type=float)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--deepout_layers', default=0, type=int)
    parser.add_argument('--deepout_act', default='MaxOut')
    # dataset
    parser.add_argument('--path', required=True)
    parser.add_argument('--processed', action='store_true')
    parser.add_argument('--dict_path', type=str)
    parser.add_argument('--max_size', default=1000000, type=int)
    parser.add_argument('--min_freq', default=1, type=int)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--num', action='store_true')
    parser.add_argument('--level', default='token')
    # training
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--bptt', default=20, type=int)
    parser.add_argument('--gpu', action='store_true')
    # - optimizer
    parser.add_argument('--optim', default='Adam', type=str)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--learning_rate_decay', default=0.5, type=float)
    parser.add_argument('--start_decay_at', default=5, type=int)
    parser.add_argument('--decay_every', default=1, type=int)
    parser.add_argument('--max_grad_norm', default=5., type=float)
    parser.add_argument('--early_stopping', default=-1, type=int)
    # - check
    parser.add_argument('--seed', default=None)
    parser.add_argument('--condition', default=None)
    parser.add_argument('--nb_conditions', type=int)
    parser.add_argument('--decoding_method', default='sample')
    parser.add_argument('--max_seq_len', default=25, type=int)
    parser.add_argument('--temperature', default=1, type=float)
    parser.add_argument('--checkpoint', default=200, type=int)
    parser.add_argument('--hooks_per_epoch', default=5, type=int)
    parser.add_argument('--log_checkpoints', action='store_true')
    parser.add_argument('--visdom_server', default='localhost')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--prefix', default='model', type=str)
    args = parser.parse_args()

    if args.processed:
        print("Loading preprocessed datasets...")
        assert args.dict_path, "Processed data requires DICT_PATH"
        data, d = load_from_file(args.path), u.load_model(args.dict_path)
        train, test, valid = BlockDataset(
            data, d, args.batch_size, bptt=args.bptt, gpu=args.gpu, fitted=True
        ).splits(test=0.1, dev=0.1)
        del data
    else:
        print("Processing datasets...")
        meta = Metadata(path='metainfo-master.csv')
        train_names, valid_names, test_names = meta.split_filenames(args.path)

        proc = text_processor(
            lower=args.lower, num=args.num, level=args.level)
        train_data, train_conditions = load_lines(names=train_names,
                                                  processor=proc,
                                                  meta=meta)
        valid_data, valid_conditions = load_lines(names=valid_names,
                                                  processor=proc,
                                                  meta=meta)
        test_data, test_conditions = load_lines(names=test_names,
                                                  processor=proc,
                                                  meta=meta)
        d = Dict(max_size=args.max_size, min_freq=args.min_freq,
                 eos_token=u.EOS, bos_token=u.BOS)
        d.fit(train_data, valid_data)
        train = ConditionalBlockDataset(
            examples=train_data, conditions=train_conditions, d=d,
            batch_size=args.batch_size, bptt=args.bptt, gpu=args.gpu)
        valid = ConditionalBlockDataset(
            examples=valid_data, conditions=valid_conditions, d=d,
            batch_size=args.batch_size, bptt=args.bptt, gpu=args.gpu,
            evaluation=True)
        test = ConditionalBlockDataset(
            examples=test_data, conditions=test_conditions, d=d,
            batch_size=args.batch_size, bptt=args.bptt, gpu=args.gpu,
            evaluation=True)
        del train_data, valid_data, test_data

    print(' * vocabulary size. %d' % len(d))
    print(' * number of train batches. %d' % len(train))

    print('Building model...')
    model = ConditionalLM(len(d), args.emb_dim, args.hid_dim,
               num_layers=args.layers, cell=args.cell, dropout=args.dropout,
               att_dim=args.att_dim, tie_weights=args.tie_weights,
               deepout_layers=args.deepout_layers,
               deepout_act=args.deepout_act, word_dropout=args.word_dropout,
               target_code=d.get_unk(),
               nb_conditions=args.nb_conditions)

    model.apply(u.make_initializer())
    if args.gpu:
        model.cuda()

    print(model)
    print('* number of parameters: %d' % model.n_params())

    optim = Optimizer(
        model.parameters(), args.optim, args.learning_rate, args.max_grad_norm,
        lr_decay=args.learning_rate_decay, start_decay_at=args.start_decay_at,
        decay_every=args.decay_every)
    criterion = nn.CrossEntropyLoss()

    # CrossEntropyLossate trainer
    trainer = ConditionalLMTrainer(model, {"train": train, "test": test, "valid": valid},
                        criterion, optim)

    STUB = [[1, 0, 0] for i in range(4)] + [[0, 1, 0] for i in range(3)] + [[0, 0, 1] for i in range(3)]
    # hooks
    early_stopping = None
    if args.early_stopping > 0:
        early_stopping = EarlyStopping(args.early_stopping)
    model_check_hook = make_lm_check_hook(
        d, method=args.decoding_method, temperature=args.temperature,
        max_seq_len=args.max_seq_len, seed_text=args.seed,
        #seed_condition=args.condition, gpu=args.gpu,
        seed_condition=STUB, gpu=args.gpu,
        #seed_condition=[[0, 0, 1, 0] for i in range(10)], gpu=args.gpu,
        early_stopping=early_stopping, nb_conditions=args.nb_conditions)
    num_checkpoints = len(train) // (args.checkpoint * args.hooks_per_epoch)
    trainer.add_hook(model_check_hook, num_checkpoints=num_checkpoints)

    # loggers
    visdom_logger = VisdomLogger(
        log_checkpoints=args.log_checkpoints, title=args.prefix, env='lm',
        server='http://' + args.visdom_server)
    trainer.add_loggers(StdLogger(), visdom_logger)

    trainer.train(args.epochs, args.checkpoint, gpu=args.gpu)

    if args.save:
        test_ppl = trainer.validate_model(test=True)
        print("Test perplexity: %g" % test_ppl)
        if args.save:
            f = '{prefix}.{cell}.{layers}l.{hid_dim}h.{emb_dim}e.{bptt}b.{ppl}'
            fname = f.format(ppl="%.2f" % test_ppl, **vars(args))
            if os.path.isfile(fname):
                answer = input("File [%s] exists. Overwrite? (y/n): " % fname)
                if answer.lower() not in ("y", "yes"):
                    print("Goodbye!")
                    sys.exit(0)
            print("Saving model to [%s]..." % fname)
            u.save_model(model, fname, d=d)

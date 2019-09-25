"""
Copyright 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#-*- coding: utf-8 -*-

import os
import sys
import time
import math
import argparse
import queue
import shutil
import random
import math
import time
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import Levenshtein as Lev 

import data.wavio as wavio
import data.label_loader as label_loader
#from data.loader import *
from data.specaug_loader import *
from models import EncoderRNN, DecoderRNN, Seq2seq
from models.EncoderTrans import EncoderTrans
from models.DecoderTrans import DecoderTrans
from models.Transformer import Transformer
from models.Transoptrimizer import TransformerOptimizer
from models.Solver import Solver
from models.Transloss import cal_performance
from utils.Visual import Visual

import nsml
from nsml import GPU_NUM, DATASET_PATH, DATASET_NAME, HAS_DATASET

char2index = dict()
index2char = dict()
SOS_token = 0
EOS_token = 0
PAD_token = 0
SAMPLE = True

# Configure dataset directory
if HAS_DATASET == False:
    if SAMPLE:
        DATASET_PATH = '/mnt/hdd0/datasets/Speech_sample'
    else:
        DATASET_PATH='/mnt/hdd0/datasets/Speech'
DATASET_PATH = os.path.join(DATASET_PATH, 'train')

def label_to_string(labels):
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == EOS_token:
                break
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sents.append(sent)

        return sents

# For CER
def char_distance(ref, hyp):
    ref = ref.replace(' ', '') 
    hyp = hyp.replace(' ', '') 

    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))

    return dist, length 

# For CER
def get_distance(ref_labels, hyp_labels, display=False):
    total_dist = 0
    total_length = 0
    for i in range(len(ref_labels)):
        ref = label_to_string(ref_labels[i])
        hyp = label_to_string(hyp_labels[i])
        dist, length = char_distance(ref, hyp)
        total_dist += dist
        total_length += length 
        if display:
            cer = total_dist / total_length
            logger.debug('%d (%0.4f)\n(%s)\n(%s)' % (i, cer, ref, hyp))
    return total_dist, total_length


def trainRNN(model, total_batch_size, queue, criterion, optimizer, device, train_begin, train_loader_count, print_batch=5, teacher_forcing_ratio=1, visual=None):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0
    batch = 0

    model.train()

    logger.info('train() start')

    begin = epoch_begin = time.time()

    while True:
        if queue.empty():
            logger.debug('queue is empty')

        feats, scripts, feat_lengths, script_lengths = queue.get()

        if feats.shape[0] == 0:
            # empty feats means closing one loader
            train_loader_count -= 1

            logger.debug('left train_loader: %d' % (train_loader_count))

            if train_loader_count == 0:
                break
            else:
                continue

        optimizer.zero_grad()

        feats = feats.to(device)
        scripts = scripts.to(device)

        src_len = scripts.size(1)
        target = scripts[:, 1:]

        model.module.flatten_parameters()
        logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=teacher_forcing_ratio)

        logit = torch.stack(logit, dim=1).to(device)

        y_hat = logit.max(-1)[1]

        loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
        total_loss += loss.item()
        total_num += sum(feat_lengths)

        display = random.randrange(0, 100) == 0
        dist, length = get_distance(target, y_hat, display=display)
        total_dist += dist
        total_length += length

        total_sent_num += target.size(0)

        loss.backward()
        optimizer.step()

        if visual:
            vis_log = {'Train Loss':total_loss / total_num, 'Train CER':total_dist / total_length}
            visual.log(vis_log)

        if batch % print_batch == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0

            logger.info('batch: {:4d}/{:4d}, loss: {:.4f}, cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'
                .format(batch,
                        #len(dataloader),
                        total_batch_size,
                        total_loss / total_num,
                        total_dist / total_length,
                        elapsed, epoch_elapsed, train_elapsed))
            begin = time.time()

            nsml.report(False,
                        step=trainRNN.cumulative_batch_count, train_step__loss=total_loss/total_num,
                        train_step__cer=total_dist/total_length)
        batch += 1
        trainRNN.cumulative_batch_count += 1

    logger.info('train() completed')
    return total_loss / total_num, total_dist / total_length

trainRNN.cumulative_batch_count = 0

def trainTrans(model, total_batch_size, queue, criterion, optimizer, device, train_begin, train_loader_count, print_batch=5, teacher_forcing_ratio=1, visual=None, label_smoothing=0.1):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0
    batch = 0

    model.train()

    logger.info('train() start')

    begin = epoch_begin = time.time()

    while True:
        if queue.empty():
            logger.debug('queue is empty')

        feats, scripts, feat_lengths, script_lengths = queue.get()

        if feats.shape[0] == 0:
            # empty feats means closing one loader
            train_loader_count -= 1
            logger.debug('left train_loader: %d' % (train_loader_count))
            if train_loader_count == 0:
                break
            else:
                continue

        optimizer.zero_grad()

        feats = feats.to(device)
        scripts = scripts.to(device)

        src_len = scripts.size(1)
        target = scripts[:, 1:]

        pred, gold = model(feats, feat_lengths, scripts)
        loss, n_correct = cal_performance(pred, gold, smoothing=label_smoothing)
        y_hat = pred.max(-1)[1]

        total_loss += loss.item()
        total_num += sum(feat_lengths)

        display = random.randrange(0, 100) == 0
        dist, length = get_distance(target, y_hat, display=display)
        total_dist += dist
        total_length += length

        total_sent_num += target.size(0)

        loss.backward()
        optimizer.step()

        if visual:
            vis_log = {'Train Loss':total_loss / total_num, 'Train CER':total_dist / total_length}
            visual.log(vis_log)

        if batch % print_batch == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0

            logger.info('batch: {:4d}/{:4d}, loss: {:.4f}, cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'
                .format(batch,
                        #len(dataloader),
                        total_batch_size,
                        total_loss / total_num,
                        total_dist / total_length,
                        elapsed, epoch_elapsed, train_elapsed))
            begin = time.time()

            nsml.report(False,
                        step=trainTrans.cumulative_batch_count, train_step__loss=total_loss/total_num,
                        train_step__cer=total_dist/total_length)
        batch += 1
        trainTrans.cumulative_batch_count += 1

    logger.info('train() completed')
    return total_loss / total_num, total_dist / total_length

trainTrans.cumulative_batch_count = 0

def evaluateRNN(model, dataloader, queue, criterion, device, visual=None):
    logger.info('evaluate() start')
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0

    model.eval()
 
    with torch.no_grad():
        while True:
            feats, scripts, feat_lengths, script_lengths = queue.get()
            if feats.shape[0] == 0:
                break

            feats = feats.to(device)
            scripts = scripts.to(device)

            src_len = scripts.size(1)
            target = scripts[:, 1:]

            model.module.flatten_parameters()
            logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=0.0)

            logit = torch.stack(logit, dim=1).to(device)
            y_hat = logit.max(-1)[1]

            loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
            total_loss += loss.item()
            total_num += sum(feat_lengths)

            display = random.randrange(0, 100) == 0
            dist, length = get_distance(target, y_hat, display=display)
            total_dist += dist
            total_length += length
            total_sent_num += target.size(0)
    if visual:                                                                      
        vis_log = {'Eval Loss':total_loss/total_num, 'Eval_CER':total_dist/total_length}
        visual.log(vis_log)
    logger.info('evaluate() completed')
    return total_loss / total_num, total_dist / total_length

def evaluateTrans(model, dataloader, queue, criterion, device, visual=None):
    logger.info('evaluate() start')
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0

    model.eval()
 
    with torch.no_grad():
        while True:
            feats, scripts, feat_lengths, script_lengths = queue.get()
            if feats.shape[0] == 0:
                break

            feats = feats.to(device)
            scripts = scripts.to(device)

            src_len = scripts.size(1)
            target = scripts[:, 1:]

            pred, gold = model(feats, feat_lengths, scripts)
            y_hat = pred.max(-1)[1]

            loss, n_correct = cal_performance(pred, gold, smoothing=label_smoothing)
            total_loss += loss.item()
            total_num += sum(feat_lengths)

            display = random.randrange(0, 100) == 0
            dist, length = get_distance(target, y_hat, display=display)
            total_dist += dist
            total_length += length
            total_sent_num += target.size(0)
    if visual:                                                                      
        vis_log = {'Eval Loss':total_loss/total_num, 'Eval_CER':total_dist/total_length}
        visual.log(vis_log)
    logger.info('evaluate() completed')
    return total_loss / total_num, total_dist / total_length


def bind_model(args, model, optimizer=None):
    def load(filename, **kwargs):    # model load
        state = torch.load(os.path.join(filename, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Model loaded')

    def save(filename, **kwargs):    # model save
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(filename, 'model.pt'))

    def infer(wav_path):             # model inference
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input = get_spectrogram_feature(wav_path, args.use_stft, args.mels, args.mode).unsqueeze(0)
        input = input.to(device)

        logit = model(input_variable=input, input_lengths=None, teacher_forcing_ratio=0)
        logit = torch.stack(logit, dim=1).to(device)

        y_hat = logit.max(-1)[1]
        hyp = label_to_string(y_hat)

        return hyp[0]

    nsml.bind(save=save, load=load, infer=infer) # 'nsml.bind' function must be called at the end.

def split_dataset(config, wav_paths, script_paths, valid_ratio=0.05):
    train_loader_count = config.workers
    records_num = len(wav_paths)
    batch_num = math.ceil(records_num / config.batch_size)

    valid_batch_num = math.ceil(batch_num * valid_ratio)
    train_batch_num = batch_num - valid_batch_num

    batch_num_per_train_loader = math.ceil(train_batch_num / config.workers)

    train_begin = 0
    train_end_raw_id = 0
    train_dataset_list = list()

    for i in range(config.workers):

        train_end = min(train_begin + batch_num_per_train_loader, train_batch_num)

        train_begin_raw_id = train_begin * config.batch_size
        train_end_raw_id = train_end * config.batch_size

        train_dataset_list.append(BaseDataset(
                                        wav_paths[train_begin_raw_id:train_end_raw_id],
                                        script_paths[train_begin_raw_id:train_end_raw_id],
                                        SOS_token, EOS_token, config.use_stft, config.mels, config.mode))
        train_begin = train_end 

    valid_dataset = BaseDataset(wav_paths[train_end_raw_id:], script_paths[train_end_raw_id:], SOS_token, EOS_token, config.use_stft, config.mels)

    return train_batch_num, train_dataset_list, valid_dataset

def main():

    global char2index
    global index2char
    global SOS_token
    global EOS_token
    global PAD_token

    parser = argparse.ArgumentParser(description='Speech hackathon Baseline')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden size of model (default: 256)')
    parser.add_argument('--layer_size', type=int, default=3, help='number of layers of model (default: 3)')
    parser.add_argument('--bidirectional', action='store_true', help='use bidirectional RNN for encoder (default: False)')
    parser.add_argument('--use_attention', action='store_true', help='use attention between encoder-decoder (default: False)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training (default: 32)')
    parser.add_argument('--workers', type=int, default=4, help='number of workers in dataset loader (default: 4)')
    parser.add_argument('--max_epochs', type=int, default=10, help='number of max epochs in training (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-04, help='learning rate (default: 0.0001)')
    parser.add_argument('--teacher_forcing', type=float, default=0.5, help='teacher forcing ratio in decoder (default: 0.5)')
    parser.add_argument('--max_len', type=int, default=80, help='maximum characters of sentence (default: 80)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--save_name', type=str, default='model', help='the name of model in nsml or local')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument("--visdom", type=bool, default=False)
    parser.add_argument("--use_stft", type=bool, default=False, help="use stft or log mel + specaugmentation")
    parser.add_argument("--mels", type=int, default=256)
    parser.add_argument("--use_rnn", type=bool, default=False)

    # Low Frame Rate (stacking and skipping frames)
    parser.add_argument('--LFR_m', default=4, type=int, help='Low Frame Rate: number of frames to stack')
    parser.add_argument('--LFR_n', default=3, type=int, help='Low Frame Rate: number of frames to skip')
    # EncoderTrans
    parser.add_argument('--d_input', default=80, type=int, help='Dim of encoder input (before LFR)')
    parser.add_argument('--n_layers_enc', default=2, type=int, help='Number of encoder stacks')
    parser.add_argument('--n_head', default=8, type=int, help='Number of Multi Head Attention (MHA)')
    parser.add_argument('--d_k', default=64, type=int, help='Dimension of key')
    parser.add_argument('--d_v', default=64, type=int, help='Dimension of value')
    parser.add_argument('--d_model', default=512, type=int, help='Dimension of model')
    parser.add_argument('--d_inner', default=2048, type=int, help='Dimension of inner')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate')
    parser.add_argument('--pe_maxlen', default=5000, type=int, help='Positional Encoding max len')
    # Decoder Trans
    parser.add_argument('--d_word_vec', default=512, type=int, help='Dim of decoder embedding')
    parser.add_argument('--n_layers_dec', default=2, type=int, help='Number of decoder stacks')
    parser.add_argument('--tgt_emb_prj_weight_sharing', default=1, type=int, help='share decoder embedding with decoder projection')
    # TransLoss
    parser.add_argument('--label_smoothing', default=0.1, type=float, help='label smoothing')
    # Optimizer
    parser.add_argument('--k', default=1.0, type=float, help='tunable scalar multiply to learning rate')
    parser.add_argument('--warmup_steps', default=4000, type=int, help='warmup steps')
    args = parser.parse_args()

    char2index, index2char = label_loader.load_label('./data/hackathon.labels')
    SOS_token = char2index['<s>']
    EOS_token = char2index['</s>']
    PAD_token = char2index['_']

    # Setting seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Setting device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    # Feature extractor
    if args.use_stft:
        feature_size = N_FFT / 2 + 1
    else:
        feature_size = args.mels

    # Actual model
    if args.use_rnn:    # RNN structure
        
        # Define model
        enc = EncoderRNN(feature_size, args.hidden_size,
                        input_dropout_p=args.dropout, dropout_p=args.dropout,
                        n_layers=args.layer_size, bidirectional=args.bidirectional, 
                        rnn_cell='gru', variable_lengths=False)
        dec = DecoderRNN(len(char2index), args.max_len, args.hidden_size * (2 if args.bidirectional else 1),
                        SOS_token, EOS_token,
                        n_layers=args.layer_size, rnn_cell='gru', bidirectional=args.bidirectional,
                        input_dropout_p=args.dropout, dropout_p=args.dropout, use_attention=args.use_attention)
        model = Seq2seq(enc, dec)
        model.flatten_parameters()
        
        # Parameters initialization
        for param in model.parameters():
            param.data.uniform_(-0.08, 0.08)

        model = nn.DataParallel(model).to(device)

        optimizer = optim.Adam(model.module.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)

        bind_model(args, model, optimizer)

        if args.pause == 1:
            nsml.paused(scope=locals())

        if args.mode != "train":
            return

        data_list = os.path.join(DATASET_PATH, 'train_data', 'data_list.csv')
        wav_paths = list()
        script_paths = list()

        with open(data_list, 'r') as f:
            for line in f:
                # line: "aaa.wav,aaa.label"
                wav_path, script_path = line.strip().split(',')
                wav_paths.append(os.path.join(DATASET_PATH, 'train_data', wav_path))
                script_paths.append(os.path.join(DATASET_PATH, 'train_data', script_path))

        best_loss = 1e10
        best_cer = 1e10
        begin_epoch = 0

        # load all target scripts for reducing disk i/o
        target_path = os.path.join(DATASET_PATH, 'train_label')
        load_targets(target_path)

        train_batch_num, train_dataset_list, valid_dataset = split_dataset(args, wav_paths, script_paths, valid_ratio=0.05)

        logger.info('start')
        
        if args.visdom:
            train_visual = Visual(train_batch_num)
            eval_visual = Visual(1)

        train_begin = time.time()

        for epoch in range(begin_epoch, args.max_epochs):

            train_queue = queue.Queue(args.workers * 2)

            train_loader = MultiLoader(train_dataset_list, train_queue, args.batch_size, args.workers)
            train_loader.start()

            if args.visdom:
                train_loss, train_cer = trainRNN(model, train_batch_num, train_queue, criterion, optimizer, device, train_begin, args.workers, 10, args.teacher_forcing, train_visual)
            else:
                train_loss, train_cer = trainRNN(model, train_batch_num, train_queue, criterion, optimizer, device, train_begin, args.workers, 10, args.teacher_forcing)

            logger.info('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))

            train_loader.join()

            valid_queue = queue.Queue(args.workers * 2)
            valid_loader = BaseDataLoader(valid_dataset, valid_queue, args.batch_size, 0)
            valid_loader.start()

            if args.visdom:
                eval_loss, eval_cer = evaluateRNN(model, valid_loader, valid_queue, criterion, device, eval_visual)
            else:
                eval_loss, eval_cer = evaluateRNN(model, valid_loader, valid_queue, criterion, device)

            logger.info('Epoch %d (Evaluate) Loss %0.4f CER %0.4f' % (epoch, eval_loss, eval_cer))

            valid_loader.join()

            nsml.report(False,
                step=epoch, train_epoch__loss=train_loss, train_epoch__cer=train_cer,
                eval__loss=eval_loss, eval__cer=eval_cer)

            best_loss_model = (eval_loss < best_loss)
            best_cer_model = (eval_cer < best_cer)
            nsml.save(args.save_name)

            if best_loss_model:
                nsml.save('best_loss')
                best_loss = eval_loss
            if best_cer_model:
                nsml.save('best_cer')
                best_cer = eval_cer

    else:           # Transformer structure
        # Define model
        enc = EncoderTrans(feature_size, args.n_layers_enc, args.n_head,
                      args.d_k, args.d_v, args.d_model, args.d_inner,
                      dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        dec = DecoderTrans(SOS_token, EOS_token, len(char2index),
                      args.d_word_vec, args.n_layers_dec, args.n_head,
                      args.d_k, args.d_v, args.d_model, args.d_inner,
                      dropout=args.dropout,
                      tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                      pe_maxlen=args.pe_maxlen)
        model = Transformer(enc, dec)

        # Parameter initialization
        for param in model.parameters():
            param.data.uniform_(-0.08, 0.08)
        model = nn.DataParallel(model).to(device)

        optimizer = TransformerOptimizer(  torch.optim.Adam(model.parameters(), 
                                            betas=(0.9, 0.98), eps=1e-09),
                                            args.k, args.d_model, args.warmup_steps)
        criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)
        bind_model(args, model, optimizer)

        if args.pause == 1:
            nsml.paused(scope=locals())

        if args.mode != "train":
            return

        data_list = os.path.join(DATASET_PATH, 'train_data', 'data_list.csv')
        wav_paths = list()
        script_paths = list()

        with open(data_list, 'r') as f:
            for line in f:
                # line: "aaa.wav,aaa.label"
                wav_path, script_path = line.strip().split(',')
                wav_paths.append(os.path.join(DATASET_PATH, 'train_data', wav_path))
                script_paths.append(os.path.join(DATASET_PATH, 'train_data', script_path))

        best_loss = 1e10
        best_cer = 1e10
        begin_epoch = 0

        # load all target scripts for reducing disk i/o
        target_path = os.path.join(DATASET_PATH, 'train_label')
        load_targets(target_path)

        train_batch_num, train_dataset_list, valid_dataset = split_dataset(args, wav_paths, script_paths, valid_ratio=0.05)

        logger.info('start')
        
        train_begin = time.time()

        for epoch in range(begin_epoch, args.max_epochs):

            train_queue = queue.Queue(args.workers * 2)

            train_loader = MultiLoader(train_dataset_list, train_queue, args.batch_size, args.workers)
            train_loader.start()

            if args.visdom:
                train_loss, train_cer = trainTrans(model, train_batch_num, train_queue, criterion, optimizer, device, train_begin, args.workers, 10, args.teacher_forcing, train_visual)
            else:
                train_loss, train_cer = trainTrans(model, train_batch_num, train_queue, criterion, optimizer, device, train_begin, args.workers, 10, args.teacher_forcing, label_smoothing=args.label_smoothing)

            logger.info('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))

            train_loader.join()

            valid_queue = queue.Queue(args.workers * 2)
            valid_loader = BaseDataLoader(valid_dataset, valid_queue, args.batch_size, 0)
            valid_loader.start()

            if args.visdom:
                eval_loss, eval_cer = evaluateTrans(model, valid_loader, valid_queue, criterion, device, eval_visual)
            else:
                eval_loss, eval_cer = evaluateTrans(model, valid_loader, valid_queue, criterion, device)

            logger.info('Epoch %d (Evaluate) Loss %0.4f CER %0.4f' % (epoch, eval_loss, eval_cer))

            valid_loader.join()

            nsml.report(False,
                step=epoch, train_epoch__loss=train_loss, train_epoch__cer=train_cer,
                eval__loss=eval_loss, eval__cer=eval_cer)

            best_loss_model = (eval_loss < best_loss)
            best_cer_model = (eval_cer < best_cer)
            nsml.save(args.save_name)

            if best_loss_model:
                nsml.save('best_loss')
                best_loss = eval_loss
            if best_cer_model:
                nsml.save('best_cer')
                best_cer = eval_cer

if __name__ == "__main__":
    main()

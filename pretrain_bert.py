# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain BERT"""

from apex import amp
from comet_ml import Experiment
import os
import random
import numpy as np
import torch

from sentence_encoders.arguments import get_args
from sentence_encoders.configure_data import configure_data
from sentence_encoders.learning_rates import AnnealingLR
from sentence_encoders.model import BertModel
from sentence_encoders.model import get_params_for_weight_decay_optimization
from sentence_encoders.model import DistributedDataParallel as DDP
from sentence_encoders.optim import Adam
from sentence_encoders.utils import Timers
from sentence_encoders.utils import save_checkpoint
from sentence_encoders.utils import load_checkpoint


def get_model(tokenizer, args):
    """Build the model."""

    print('building BERT model ...')
    model = BertModel(tokenizer, args)
    print(' > number of parameters: {}'.format(
        sum([p.nelement() for p in model.parameters()])), flush=True)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Wrap model for distributed training.
    if args.world_size > 1:
        model = DDP(model)

    return model


def get_optimizer(model, args):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    param_groups = model.get_params()

    # Use Adam.
    optimizer = Adam(param_groups,
                     lr=args.lr, weight_decay=args.weight_decay)

    return optimizer


def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_tokens * args.epochs
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step)

    return lr_scheduler


def setup_model_and_optimizer(args, tokenizer):
    """Setup model and optimizer."""

    model = get_model(tokenizer, args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)
    criterion_cls = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1)
    criterion_reg = torch.nn.MSELoss(reduce=False)
    criterion = (criterion_cls, criterion_reg)

    if args.load is not None:
        epoch, i, total_iters = load_checkpoint(model, optimizer,
                                                lr_scheduler, args)
        if args.resume_dataloader:
            args.epoch = epoch
            args.mid_epoch_iters = i
            args.total_iters = total_iters

    return model, optimizer, lr_scheduler, criterion


def get_batch(data):
    ''' get_batch subdivides the source data into chunks of
    length args.seq_length. If source is equal to the example
    output of the data loading example, with a seq_length limit
    of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the data loader. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM. A Variable representing an appropriate
    shard reset mask of the same dimensions is also returned.
    '''
    aux_labels = {}
    for mode, label in data['aux_labels'].items():
        aux_labels[mode] = torch.autograd.Variable(label.long()).cuda()
    num_sentences = data['n']
    num_tokens = sum(data['num_tokens']).item()
    tokens = []
    types = []
    tasks = []
    loss_mask = []
    lm_labels = []
    att_mask = []
    for i in range(min(num_sentences)):
        suffix = "_" + str(i)
        tokens.append(torch.autograd.Variable(data['text' + suffix].long()).cuda())
        types.append(torch.autograd.Variable(data['types' + suffix].long()).cuda())
        tasks.append(torch.autograd.Variable(data['task' + suffix].long()).cuda())
        att_mask.append(1 - torch.autograd.Variable(data['pad_mask' + suffix].byte()).cuda())
        lm_labels.append((data['mask_labels' + suffix]).long())
        loss_mask.append((data['mask' + suffix]).float())

    lm_labels = torch.autograd.Variable(torch.cat(lm_labels, dim=0).long()).cuda()
    loss_mask = torch.autograd.Variable(torch.cat(loss_mask, dim=0).float()).cuda()

    return (tokens, types, tasks, aux_labels, loss_mask, lm_labels, att_mask, num_tokens)


def forward_step(data, model, criterion, modes, args):
    """Forward step."""
    criterion_cls, criterion_reg = criterion
    # Get the batch.
    batch = get_batch(data)

    tokens, types, tasks, aux_labels, loss_mask, lm_labels, att_mask, num_tokens = batch
    if "rg" in modes:
        aux_labels['rg'] = torch.autograd.Variable(torch.arange(tokens[0].shape[0]).long()).cuda()
    if "fs" in modes:
        aux_labels['fs'] = torch.autograd.Variable(torch.ones(args.batch_size * 2 * args.seq_length).long()).cuda()
    # Forward model.
    scores = model(modes, tokens, types, tasks, att_mask, checkpoint_activations=args.checkpoint_activations)
    assert list(scores.keys()) == modes

    losses = {}
    for mode, score in scores.items():
        if mode == "mlm":
            mlm_loss = criterion_cls(score.view(-1, args.data_size).contiguous().float(),
                                 lm_labels.contiguous().view(-1).contiguous())
            loss_mask = loss_mask.contiguous()
            loss_mask = loss_mask.view(-1)
            losses["mlm"] = torch.sum(mlm_loss * loss_mask.view(-1).float()) / loss_mask.sum()
        elif mode in ["wlen", "tf", "tf_idf"]: # use regression
            losses[mode] = criterion_reg(score.view(-1).contiguous().float(),
                                         aux_labels[mode].view(-1).contiguous().float()).mean()
        else:
            # TODO do I need separate criterion sentences?
            score = score.view(-1, 2) if mode in ["corrupt_tok", "cap"] else score
            losses[mode] = criterion_cls(score.contiguous().float(),
                                         aux_labels[mode].view(-1).contiguous()).mean()
    return losses, num_tokens


def backward_step(optimizer, model, losses, args):
    """Backward step."""
    # Backward pass.
    optimizer.zero_grad()
    total_loss = sum(losses.values())
    with amp.scale_loss(total_loss, optimizer) as scaled_loss:
         scaled_loss.backward()
    #total_loss.backward()

    # Reduce across processes.
    losses_reduced = losses
    if args.world_size > 1:
        losses_reduced = [[k,v] for k,v in losses_reduced.items()]
        reduced_losses = torch.cat([x[1].view(1) for x in losses_reduced])
        torch.distributed.all_reduce(reduced_losses.data)
        reduced_losses.data = reduced_losses.data / args.world_size
        model.allreduce_params(reduce_after=False,
                               fp32_allreduce=False)#args.fp32_allreduce)
        losses_reduced = {losses_reduced[i][0]: reduced_losses[i] for i in range(len(losses_reduced))}
    # Clipping gradients helps prevent the exploding gradient.
    if args.clip_grad > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

    return losses_reduced


def train_step(input_data, model, criterion, optimizer, lr_scheduler, modes, args):
    """Single training step."""
    # Forward model for one step.
    losses, num_tokens = forward_step(input_data, model, criterion, modes, args)
    # Calculate gradients, reduce across processes, and clip.
    losses_reduced = backward_step(optimizer, model, losses, args)
    # Update parameters.
    optimizer.step()
    # Update learning rate.
    skipped_iter = 0
    return losses_reduced, skipped_iter, num_tokens


def train_epoch(epoch, model, optimizer, train_data, lr_scheduler, criterion, timers, experiment, metrics, past_iters, args):
    """Train one full epoch."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_losses = {}

    # Iterations.
    max_tokens = args.train_tokens
    log_tokens = 0
    tot_tokens = 0
    iteration = 0
    tot_iteration = 0
    skipped_iters = 0
    if args.resume_dataloader:
        iteration = args.mid_epoch_iters
        args.resume_dataloader = False

    # Data iterator.
    modes = args.modes.split(',')
    if args.incremental:
        modes = modes[:epoch]
    train_data.dataset.set_args(modes, past_iters)
    data_iters = iter(train_data)

    timers('interval time').start()
    while tot_tokens < max_tokens:
        while True:
            try:
                losses, skipped_iter, num_tokens = train_step(next(data_iters),
                              model,
                              criterion,
                              optimizer,
                              lr_scheduler,
                              modes,
                              args)
                break
            except (TypeError, RuntimeError) as e:
                print("Ooops, caught: '{}', continuing...".format(e))

        log_tokens += num_tokens
        tot_tokens += num_tokens
        lr_scheduler.step(step_num=(epoch-1) * max_tokens + tot_tokens)
        skipped_iters += skipped_iter
        iteration += 1
        # Update losses.
        for mode, loss in losses.items():
            total_losses[mode] = total_losses.get(mode, 0.0) + loss.data.detach().float()

        # Logging.
        if log_tokens > args.log_interval:
            log_tokens = 0
            learning_rate = optimizer.param_groups[0]['lr']
            avg_loss = {}
            for mode, v in total_losses.items():
                avg_loss[mode] = v.item() / iteration

            elapsed_time = timers('interval time').elapsed()
            log_string = ' epoch{:2d} |'.format(epoch)
            log_string += ' tokens {:8d}/{:8d} |'.format(tot_tokens, max_tokens)
            log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
                elapsed_time * 1000.0 / iteration)
            log_string += ' learning rate {:.3E} |'.format(learning_rate)
            for mode, v in avg_loss.items():
                log_string += ' {} loss {:.3E} |'.format(mode, v)
            print(log_string, flush=True)
            #print(iteration)
            total_losses = {}

            experiment.set_step(tot_iteration + past_iters)
            metrics['learning_rate'] = learning_rate
            for mode, v in avg_loss.items():
                metrics[mode] = v

            experiment.log_metrics(metrics)
            tot_iteration += iteration
            iteration = 0

        # Checkpointing
        if args.save and args.save_iters and iteration % args.save_iters == 0:
            total_iters = args.train_iters * (epoch-1) + iteration
            model_suffix = 'model/%d.pt' % (total_iters)
            save_checkpoint(model_suffix, epoch, iteration, model, optimizer,
                            lr_scheduler, args)

    print("Learnt using {} tokens over {} iterations this epoch".format(tot_tokens, tot_iteration))
    return tot_iteration, skipped_iters

def evaluate(epoch, data_source, model, criterion, elapsed_time, args, test=False):
    """Evaluation."""
    import time
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_losses = {}
    max_tokens = args.eval_tokens
    tokens = 0
    modes = args.modes.split(',')
    data_source.dataset.set_args(modes, 0)
    data_iters = iter(data_source)
    start_time = time.time()
    with torch.no_grad():
        iteration = 0
        while tokens < max_tokens:
            # Forward evaluation.
            
            while True:
                try:
                    losses, num_tokens = forward_step(next(data_iters), model, criterion, modes, args)
                    break
                except (TypeError, RuntimeError) as e:
                    print("Ooops, caught: '{}', continuing".format(e))

            tokens += num_tokens
            # Reduce across processes.
            if isinstance(model, DDP):
                # reduced_losses = torch.cat((lm_loss.view(1), nsp_loss.view(1)))
                losses_reduced = [[k, v] for k, v in losses.items()]
                reduced_losses = torch.cat([x[1].view(1) for x in losses_reduced])
                torch.distributed.all_reduce(reduced_losses.data)
                reduced_losses.data = reduced_losses.data / args.world_size
                model.allreduce_params(reduce_after=False,
                                       fp32_allreduce=False)  # args.fp32_allreduce)
                losses = {losses_reduced[i][0]: reduced_losses[i] for i in range(len(losses_reduced))}
            
            assert list(losses.keys()) == modes
            for mode, loss in losses.items():
                total_losses[mode] = total_losses.get(mode, 0.0) + loss.data.detach().float().item()
            iteration += 1
            #print("Done iteration", iteration, time.time() - start_time)

    print("Evaluated using {} tokens over {} iterations.".format(tokens, iteration))

    # Move model back to the train mode.
    model.train()

    avg_loss = {}
    for mode, v in total_losses.items():
        avg_loss[mode] = v / args.eval_iters

    tot_loss = sum(avg_loss.values())
    sep_char = '=' if test else '-'
    print(sep_char * 100)
    log_string = '| End of training | '.format(epoch) if test else '| End of epoch {:3d} | '
    log_string += 'time: {:5.2f}s | valid loss {:.4E} | '.format(epoch, elapsed_time, tot_loss)
    for mode, v in avg_loss.items():
        log_string += ' {} loss {:.3E} |'.format(mode, v)
    print(log_string, flush=True)
    print(sep_char * 100)

    return tot_loss


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    if args.world_size > 1:
        init_method = 'tcp://'
        master_ip = os.getenv('MASTER_ADDR', 'localhost')
        master_port = os.getenv('MASTER_PORT', '6000')
        init_method += master_ip + ':' + master_port
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.world_size, rank=args.rank,
            init_method=init_method)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


def main():
    """Main training program."""

    print('Pretrain BERT model')
    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()

    experiment = Experiment(api_key='1jl4lQOnJsVdZR6oekS6WO5FI',
                           project_name=args.model_type,
                           auto_param_logging=False, auto_metric_logging=False,
                           disabled=(not args.track_results))
    experiment.log_parameters(vars(args))
    metrics = {}

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # Data stuff.
    data_config = configure_data()
    data_config.set_defaults(data_set_type='BERT', transpose=False)
    (train_data, val_data, test_data), tokenizer = data_config.apply(args)
    args.data_size = tokenizer.num_tokens

    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler, criterion = setup_model_and_optimizer(
        args, tokenizer)

    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    timers("total time").start()
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        total_iters = 0
        skipped_iters = 0
        start_epoch = 1
        best_val_loss = float('inf')
        # Resume data loader if necessary.
        if args.resume_dataloader:
            start_epoch = args.epoch
            total_iters = args.total_iters
            train_data.batch_sampler.start_iter = total_iters % len(train_data)
        # For all epochs. 
        for epoch in range(start_epoch, args.epochs+1):
            if args.shuffle:
                train_data.batch_sampler.sampler.set_epoch(epoch+args.seed)
            timers('epoch time').start()
            iteration, skipped = train_epoch(epoch, model, optimizer,
                                             train_data, lr_scheduler,
                                             criterion, timers, experiment,
                                             metrics, total_iters, args)

            elapsed_time = timers('epoch time').elapsed()
            total_iters += iteration
            skipped_iters += skipped
            val_loss = evaluate(epoch, val_data, model, criterion, elapsed_time, args)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.save:
                    best_path = 'best/model.pt'
                    print('saving best model to:',
                           os.path.join(args.save, best_path))
                    save_checkpoint(best_path, epoch+1, total_iters, model,
                                    optimizer, lr_scheduler, args)


    except KeyboardInterrupt:
        print('-' * 100)
        print('Exiting from training early')
        if args.save and False: # WARNING I disabled this to save memory, but may be necessary in the future
            cur_path = 'current/model.pt'
            print('saving current model to:',
                   os.path.join(args.save, cur_path))
            save_checkpoint(cur_path, epoch, total_iters, model, optimizer,
                            lr_scheduler, args)
        exit()

    if args.save and False: # WARNING I disabled this to save memory, but may be necessary in the future
        final_path = 'final/model.pt'
        print('saving final model to:', os.path.join(args.save, final_path))
        save_checkpoint(final_path, args.epochs, total_iters, model, optimizer,
                        lr_scheduler, args)

    if test_data is not None:
        # Run on test data.
        print('entering test')
        elapsed_time = timers("total time").elapsed()
        evaluate(epoch, test_data, model, criterion, elapsed_time, args, test=True)


if __name__ == "__main__":
    main()

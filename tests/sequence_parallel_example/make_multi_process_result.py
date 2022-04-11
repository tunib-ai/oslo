import os

import colossalai
import torch
from colossalai.amp import AMP_TYPE
from colossalai.context import ParallelMode
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.engine.schedule import PipelineSchedule
from colossalai.kernel import LayerNorm
from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import FusedAdam
from colossalai.utils import MultiTimer, is_using_pp, get_current_device

from data import build_train_valid_test_data_iterators
from data import build_train_valid_test_datasets
from data.bert_helper import get_batch_for_sequence_parallel, SequenceParallelDataIterator
from data.datasets.dataset_utils import get_indexed_dataset_, get_train_valid_test_split_
from data.tokenizer import initialize_tokenizer, get_padded_vocab_size
from loss_func.bert_loss import BertLoss
from lr_scheduler import AnnealingLR
from make_check_dataset import build_train_valid_test_data_iterators_for_check
from model.bert import BertForPretrain, build_pipeline_bert

from initialize import initialize

# precision matters
torch.set_default_dtype(torch.float64)


def get_tensor_shape():
    if not gpc.is_initialized(ParallelMode.PIPELINE):
        return None

    dp_size = gpc.get_world_size(ParallelMode.DATA)
    if gpc.is_initialized(ParallelMode.SEQUENCE):
        seq_size = gpc.get_world_size(ParallelMode.SEQUENCE)
    else:
        seq_size = 1
    tensor_shape = (gpc.config.SEQ_LENGTH // seq_size,
                    gpc.config.GLOBAL_BATCH_SIZE // dp_size // gpc.config.NUM_MICRO_BATCHES,
                    gpc.config.HIDDEN_SIZE)
    return tensor_shape


def process_batch_data(batch_data):
    tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = batch_data
    if gpc.is_first_rank(ParallelMode.PIPELINE):
        data = dict(input_ids=tokens,
                    attention_masks=padding_mask,
                    tokentype_ids=types,
                    lm_labels=lm_labels)
    else:
        data = dict(attention_masks=padding_mask,
                    tokentype_ids=types,
                    lm_labels=lm_labels)
    label = dict(loss_mask=loss_mask,
                 sentence_order=sentence_order)
    return data, label


def split_example_for_sequence_parallel(example):
    tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = example

    print('get_batch_for_sequence_parallel', f'{torch.distributed.get_rank()}' * 30)

    # get tensor parallel local rank
    global_rank = torch.distributed.get_rank()
    local_world_size = 1 if not gpc.is_initialized(ParallelMode.TENSOR) else gpc.get_world_size(ParallelMode.TENSOR)
    local_rank = global_rank % local_world_size
    seq_length = tokens.size(1)
    sub_seq_length = seq_length // local_world_size
    sub_seq_start = local_rank * sub_seq_length
    sub_seq_end = (local_rank+1) * sub_seq_length

    # # Unpack.
    tokens = tokens[:, sub_seq_start:sub_seq_end].long()
    types = types[:, sub_seq_start:sub_seq_end].long()
    sentence_order = sentence_order.long()
    loss_mask = loss_mask[:, sub_seq_start:sub_seq_end].float()
    lm_labels = lm_labels[:, sub_seq_start:sub_seq_end].long()
    padding_mask = padding_mask.long()

    return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask


def main():
    # initialize
    colossalai.launch_from_torch(
        config='./config_multi.py',
        seed=1234,
        backend='nccl')

    logger = get_dist_logger()

    single_process_result_path = 'results/single_process'
    result_path = 'results/multi_process'
    os.makedirs(result_path, exist_ok=True)

    # build dataloader
    initialize_tokenizer(gpc.config.VOCAB_FILE_PATH, tokenizer_type='BertWordPieceLowerCase')
    VOCAB_SIZE = get_padded_vocab_size()
    trainloader, validloader, testloader = build_train_valid_test_data_iterators_for_check(
        train_iters=gpc.config.TRAIN_ITERS,
        global_batch_size=gpc.config.GLOBAL_BATCH_SIZE,
        eval_interval=gpc.config.EVAL_INTERVAL,
        eval_iters=gpc.config.EVAL_ITERS,
    )

    logger.info("Dataloaders are built", ranks=[0])

    # build model
    if hasattr(gpc.config, 'fp16') and gpc.config.fp16.get('mode') == AMP_TYPE.NAIVE:
        is_naive_fp16 = True
    else:
        is_naive_fp16 = False

    kwargs = dict(
        vocab_size=30592,   # dimension in checkpoint
        hidden_size=gpc.config.HIDDEN_SIZE,
        max_sequence_length=gpc.config.SEQ_LENGTH,
        num_attettion_heads=gpc.config.NUM_ATTENTION_HEADS,
        convert_fp16_to_fp32_in_softmax=True,
        is_naive_fp16=is_naive_fp16,
        add_binary_head=gpc.config.ADD_BINARY_HEAD
    )

    # no PP
    model = BertForPretrain(num_layers=gpc.config.DEPTH, **kwargs)

    # model = model.half()

    # weight initialization and save
    checkpoint_name = 'weight_sp.pth'
    checkpoint_path = os.path.join(single_process_result_path, checkpoint_name)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        logger.info(f"loaded checkpoint {checkpoint_path}", ranks=[0])
    else:
        raise Exception

    logger.info(f"Model is built with softmax in fp32 = {is_naive_fp16}", ranks=[0])

    total_numel = 0
    for p in model.parameters():
        total_numel += p.numel()
    logger.info(f"This model has {total_numel} parameters")

    # build criterion
    criterion = BertLoss()
    logger.info("Criterion is built", ranks=[0])

    # layernorm and bias has no weight decay
    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in model.modules():
        if isinstance(module_, LayerNorm):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])

    logger.info(
        f"without weight decay param: {len(no_weight_decay_params['params'])}, with weight decay param: {len(weight_decay_params['params'])}")

    # optimizer
    optimizer = FusedAdam((weight_decay_params, no_weight_decay_params),
                          lr=gpc.config.LR,
                          weight_decay=gpc.config.WEIGHT_DECAY)

    # optimizer = torch.optim.Adam(model.parameters(), lr=gpc.config.LR,
    #                              weight_decay=gpc.config.WEIGHT_DECAY)
    # logger.info("Optimizer is built", ranks=[0])

    # lr scheduler
    # follow Megatron-LM setting
    # warmup_steps = int(gpc.config.DECAY_ITERS * gpc.config.WARMUP_FRACTION)
    # lr_scheduler = AnnealingLR(optimizer=optimizer,
    #                            max_lr=gpc.config.LR,
    #                            min_lr=gpc.config.MIN_LR,
    #                            warmup_steps=warmup_steps,
    #                            decay_steps=gpc.config.DECAY_ITERS,
    #                            decay_style='linear'
    #                            )
    # logger.info(f"LR Scheduler is built with {warmup_steps} warmup steps and {gpc.config.DECAY_ITERS} decay steps")

    # init
    engine, *dummy = initialize(
        model,
        optimizer,
        criterion,
    )

    checkpoint_name = f'weight_{torch.distributed.get_rank()}.pth'
    checkpoint_path = os.path.join(result_path, checkpoint_name)
    torch.save(engine.model.state_dict(), checkpoint_path)

    # schedule
    schedule = None
    tensor_shape = get_tensor_shape()
    # if use_pipeline:
    #     logger.info('Build PipelineSchedule', ranks=[0])
    #     schedule = PipelineSchedule(gpc.config.NUM_MICRO_BATCHES,
    #                                 tensor_shape=tensor_shape, scatter_gather_tensors=False,
    #                                 batch_data_process_func=process_batch_data)
    #     schedule.pre_processing(engine)

    # build timer
    timer = MultiTimer()
    skip_iters = 0

    # build loss tracker
    accumulated_train_loss = torch.zeros(1, dtype=torch.float32).cuda()
    accumulated_eval_loss = torch.zeros(1, dtype=torch.float32).cuda()

    # build data iters for pipeline parallel
    # if use_pipeline:
    #     train_data_iter = SequenceParallelDataIterator(trainloader)
    #     valid_data_iter = SequenceParallelDataIterator(validloader)

    # for step in range(1, gpc.config.TRAIN_ITERS + 1):
    #     timer.start('train-iterations')
    #     engine.train()
    #     # no PP
    #     tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = get_batch_for_sequence_parallel(trainloader)
    #     engine.zero_grad()
    #     lm_loss, sop_output = engine(tokens, padding_mask, types, lm_labels)
    #     train_loss = engine.criterion(lm_loss, sop_output, loss_mask, sentence_order)
    #     engine.backward(train_loss)
    #     _ = engine.step()
    #     timer.stop('train-iterations', keep_in_history=True)
    #
    #     if not gpc.is_initialized(ParallelMode.PIPELINE) or gpc.is_last_rank(ParallelMode.PIPELINE):
    #         accumulated_train_loss += train_loss

    example_filename = 'example.pth'
    example_filepath = os.path.join(single_process_result_path, example_filename)
    grad_filename = f'grad_{torch.distributed.get_rank()}.pth'
    grad_filepath = os.path.join(result_path, grad_filename)
    output_filename = f'output_{torch.distributed.get_rank()}.pth'
    output_filepath = os.path.join(result_path, output_filename)
    ks_filename = f'ks_{torch.distributed.get_rank()}.pth'
    ks_filepath = os.path.abspath(os.path.join(result_path, ks_filename))

    engine.train()
    if os.path.exists(example_filepath):
        example = torch.load(example_filepath, map_location='cpu')
        example = split_example_for_sequence_parallel(example)
        example = map(lambda x: x.to(get_current_device()), example)
        tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = example
    else:
        raise Exception

    example_filename = f'example_{torch.distributed.get_rank()}.pth'
    example_filepath = os.path.join(result_path, example_filename)
    torch.save((tokens, types, sentence_order, loss_mask, lm_labels, padding_mask), example_filepath)

    engine.zero_grad()
    (lm_loss, sop_output), output = engine(tokens, padding_mask, types, lm_labels, ks_filepath)
    output.append(lm_loss.detach())

    # train_loss = engine.criterion(lm_loss, sop_output, loss_mask, sentence_order)
    train_loss = engine.criterion(lm_loss, None, loss_mask, sentence_order)

    output.append(train_loss.detach())

    engine.backward(train_loss)

    torch.distributed.barrier()

    torch.save(output, output_filepath)
    torch.distributed.barrier()

    # example is wrong!; example reduce_mean the loss
    # However, loss is need to be done reduce_sum in SP group, reduce_mean in DP group
    engine._all_reduce_gradients()
    torch.distributed.barrier()

    grads = {}
    for name, m in engine.model.named_parameters():
        grads[name] = m.grad

    torch.save(grads, grad_filepath)


if __name__ == "__main__":
    main()

import os
import math

import torch


def check_example():
    single_example_path = 'results/single_process/example.pth'
    single_ddp_example_paths = [
        'results/single_process_ddp/example_0.pth',
        'results/single_process_ddp/example_1.pth',
    ]

    single_example = torch.load(single_example_path, map_location='cpu')
    single_ddp_examples = [torch.load(p, map_location='cpu') for p in single_ddp_example_paths]

    # tokens, types, sentence_order, loss_mask, lm_labels, padding_mask
    do_concat = [True, True, True, True, True, True]

    for i in range(len(do_concat)):
        if do_concat[i]:    # sub-sequences
            subsequences = [ex[i] for ex in single_ddp_examples]
            assert torch.allclose(single_example[i], torch.cat(subsequences, 0))
        else:
            for j in range(len(single_ddp_examples)):
                assert torch.allclose(single_example[i], single_ddp_examples[j][i])


def check_outputs():
    single_output_path = 'results/single_process/output.pth'
    single_ddp_output_paths = [
        'results/single_process_ddp/output_0.pth',
        'results/single_process_ddp/output_1.pth',
    ]

    single_output = torch.load(single_output_path, map_location='cpu')
    single_ddp_outputs = [torch.load(p, map_location='cpu') for p in single_ddp_output_paths]

    # need torch.float64 to compare
    # 0: pos_ids
    # 1: attention_masks
    # 2: embedding
    # --- BertLayer ---
    # 3: layer_norm output
    # 4: query
    # 5: key
    # 6: value
    # 7: attention scores
    # 8: attention_probs
    # 9: context_layer
    # 10: attention output
    # 11: attention bias
    # 12: residual
    # 13: layer_norm input <- == attention output???
    # 14: layer_norm output
    # 15: mlp_output
    # 16: mlp_bias
    # 17: residual <- == layer_norm input (7)
    # --- end ---
    # 18: BertLayer output
    # 19: final layer_norm
    # 20: lm_loss
    # 21: train_loss
    # layer_norm makes small difference to big difference

    print(f'train_loss: {single_output[21]}, {single_ddp_outputs[0][21]}, {single_ddp_outputs[1][21]}')

    assert torch.allclose(single_output[21], (single_ddp_outputs[0][21] + single_ddp_outputs[1][21])/2.)

    for i in range(len(single_output)-1):   # except 21; final loss
        subbatch = [mo[i] for mo in single_ddp_outputs]

        single = single_output[i]

        single_shape = single.shape
        subseq_shape = subbatch[0].shape
        axis = None
        for j in range(len(single_shape)):
            if single_shape[j] != subseq_shape[j]:
                axis = j
                break

        if axis is not None:
            single_ddp = torch.cat(subbatch, axis)
        else:
            # attention bias; 0
            single_ddp = subbatch[0]

        if single.dtype == torch.bool:
            single = single.long()
            single_ddp = single_ddp.long()

        # print((i, torch.max(torch.abs(single-multi)), single.shape, subseq_shape))

        assert torch.allclose(single, single_ddp), (i, torch.max(torch.abs(single-single_ddp)), single, single_ddp)


def check_grads():
    single_grad_path = 'results/single_process/grad_sp.pth'
    multi_grad_paths = [
        'results/single_process_ddp/grad_sp_0.pth',
        'results/single_process_ddp/grad_sp_1.pth',
    ]

    single_grad = torch.load(single_grad_path, map_location='cpu')
    multi_grads = [torch.load(p, map_location='cpu') for p in multi_grad_paths]

    # for k, v in multi_grads[0].items():
    #     if v is not None:
    #         print(k, torch.allclose(v, multi_grads[1][k]))

    for k, v in single_grad.items():
        g = torch.zeros_like(v)
        for i in range(len(multi_grads)):
            g = multi_grads[i][k]

        # print(v)
        # print(g)

        if torch.allclose(v, g):
            print(f'{k} allclose')
        else:
            gap = torch.max(torch.abs(v-g))
            print(f'{k} FAILED!!!!!!!!!!!!!', gap)
            print(v)
            print(g)


if __name__ == "__main__":
    check_example()
    check_outputs()
    check_grads()

import os
import math

import torch


def check_example():
    single_example_path = 'results/single_process/example.pth'
    multi_example_paths = [
        'results/multi_process/example_0.pth',
        'results/multi_process/example_1.pth',
    ]

    single_example = torch.load(single_example_path, map_location='cpu')
    multi_examples = [torch.load(p, map_location='cpu') for p in multi_example_paths]

    # tokens, types, sentence_order, loss_mask, lm_labels, padding_mask
    do_concat = [True, True, False, True, True, False]

    for i in range(len(do_concat)):
        if do_concat[i]:    # sub-sequences
            subsequences = [ex[i] for ex in multi_examples]
            assert torch.allclose(single_example[i], torch.cat(subsequences, -1))
        else:
            for j in range(len(multi_examples)):
                assert torch.allclose(single_example[i], multi_examples[j][i])


def check_weights():
    single_weight_path = 'results/single_process/weight_sp.pth'
    multi_weight_paths = [
        'results/multi_process/weight_0.pth',
        'results/multi_process/weight_1.pth',
    ]

    single_weight = torch.load(single_weight_path, map_location='cpu')
    multi_weights = [torch.load(p, map_location='cpu') for p in multi_weight_paths]

    for k, v in single_weight.items():
        for i in range(len(multi_weights)):
            assert torch.allclose(v, multi_weights[i][k])


def check_outputs():
    single_output_path = 'results/single_process/output.pth'
    multi_output_paths = [
        'results/multi_process/output_0.pth',
        'results/multi_process/output_1.pth',
    ]

    single_output = torch.load(single_output_path, map_location='cpu')
    multi_outputs = [torch.load(p, map_location='cpu') for p in multi_output_paths]

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

    # print(f'train_loss: {single_output[21]}, {multi_outputs[0][21]}, {multi_outputs[1][21]}')

    for i in range(len(single_output)):
        subsequences = [mo[i] for mo in multi_outputs]

        single = single_output[i]

        single_shape = single.shape
        subseq_shape = subsequences[0].shape
        axis = None
        for j in range(len(single_shape)):
            if single_shape[j] != subseq_shape[j]:
                axis = j
                break

        if axis is not None:
            multi = torch.cat(subsequences, axis)
        else:
            # attention bias; 0
            multi = subsequences[0]

        if single.dtype == torch.bool:
            single = single.long()
            multi = multi.long()

        # print((i, torch.max(torch.abs(single-multi)), single.shape, subseq_shape))

        assert torch.allclose(single, multi), (i, torch.max(torch.abs(single-multi)), single, multi)


def check_grads_colossalai():
    single_grad_path = 'results/single_process/grad_sp.pth'
    multi_grad_paths = [
        'results/multi_process/grad_0.pth',
        'results/multi_process/grad_1.pth',
    ]

    single_grad = torch.load(single_grad_path, map_location='cpu')
    multi_grads = [torch.load(p, map_location='cpu') for p in multi_grad_paths]

    for k, v in single_grad.items():
        for i in range(len(multi_grads)):
            if torch.allclose(v, multi_grads[i][k]):
                print(f'{k} allclose')
            else:
                print(f'{k} FAILED!!!!!!!!!!!!!')


def check_grads():
    single_grad_path = 'results/single_process/grad_sp.pth'
    multi_grad_paths = [
        'results/multi_process/grad_0.pth',
        'results/multi_process/grad_1.pth',
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


def check_attention():
    single_output_path = 'results/single_process/output.pth'
    multi_output_paths = [
        'results/multi_process/output_0.pth',
        'results/multi_process/output_1.pth',
    ]

    single_output = torch.load(single_output_path, map_location='cpu')
    multi_outputs = [torch.load(p, map_location='cpu') for p in multi_output_paths]
    output_size = (4, 2, 32, 32)

    # need to calculate in GPU!
    # if matmul is done in CPU, the result will be different!!
    single_query = single_output[4].transpose(0, 1).cuda().contiguous().type(torch.float64)
    single_key = single_output[5].transpose(0, 1).cuda().contiguous().type(torch.float64)
    single_attention = torch.matmul(single_query, single_key.transpose(2, 1))

    norm_factor = math.sqrt(single_attention.size(-1))
    single_attention /= norm_factor

    # make placeholder before reshape
    multi_attention = torch.zeros_like(single_attention)

    single_attention = single_attention.view(*output_size)
    single_attention_saved = single_output[7].cuda().type(torch.float64)

    half = multi_attention.size(1) // 2
    sub_q_1 = multi_outputs[0][4].transpose(0, 1).cuda().contiguous().type(torch.float64)
    sub_q_2 = multi_outputs[1][4].transpose(0, 1).cuda().contiguous().type(torch.float64)
    sub_k_1 = multi_outputs[0][5].transpose(0, 1).cuda().contiguous().type(torch.float64)
    sub_k_2 = multi_outputs[1][5].transpose(0, 1).cuda().contiguous().type(torch.float64)

    multi_attention[:, :half, :half] = torch.matmul(sub_q_1, sub_k_1.transpose(2, 1)) / norm_factor
    multi_attention[:, :half, half:] = torch.matmul(sub_q_1, sub_k_2.transpose(2, 1)) / norm_factor
    multi_attention[:, half:, :half] = torch.matmul(sub_q_2, sub_k_1.transpose(2, 1)) / norm_factor
    multi_attention[:, half:, half:] = torch.matmul(sub_q_2, sub_k_2.transpose(2, 1)) / norm_factor
    multi_attention = multi_attention.view(*output_size)

    # for i in range(32):
    #     print(single_attention[0][0][i])
    #     print(multi_attention[0][0][i])
    #     print('-' * 30)

    multi_attention_saved = [mo[7].cuda() for mo in multi_outputs]
    multi_attention_saved = torch.cat(multi_attention_saved, -2).cuda().type(torch.float64)

    print(torch.allclose(single_attention, single_attention_saved))

    print(torch.allclose(single_attention_saved, multi_attention))

    print(torch.allclose(multi_attention, multi_attention_saved))

    print(torch.allclose(single_attention, multi_attention_saved))


def check_ks():
    single_ks_path = 'results/single_process/ks.pth'
    multi_ks_paths = [
        'results/multi_process/ks_0.pth',
        'results/multi_process/ks_1.pth',
    ]

    single_ks = torch.load(single_ks_path, map_location='cpu')
    multi_ks = [torch.load(p, map_location='cpu') for p in multi_ks_paths]

    print(torch.allclose(multi_ks[0][0], multi_ks[1][1]))
    print(torch.allclose(multi_ks[0][1], multi_ks[1][0]))
    print(torch.allclose(single_ks[0], torch.cat([multi_ks[0][0], multi_ks[1][0]], 1)))


if __name__ == "__main__":
    check_example()
    check_weights()
    check_outputs()
    check_grads()
    check_attention()
    check_ks()

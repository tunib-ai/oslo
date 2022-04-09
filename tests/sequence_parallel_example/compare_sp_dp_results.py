import os
import math

import torch


def check_example():
    single_example_path = 'results/single_process/example.pth'
    multi_example_paths = [
        ['results/multi_process_ddp/example_0_0.pth', 'results/multi_process_ddp/example_0_1.pth'],
        ['results/multi_process_ddp/example_1_0.pth', 'results/multi_process_ddp/example_1_1.pth']
    ]

    single_example = torch.load(single_example_path, map_location='cpu')
    multi_examples = [[torch.load(p, map_location='cpu') for p in ps] for ps in multi_example_paths]

    # sequence dimension concat
    # tokens, types, sentence_order, loss_mask, lm_labels, padding_mask
    do_concat = [True, True, False, True, True, False]

    for i in range(len(do_concat)):
        if do_concat[i]:    # sub-sequences
            subsequences = [[ex[i] for ex in examples] for examples in multi_examples]
            fullsequences = [torch.cat(ss, -1) for ss in subsequences]   # sequence dimension concat
            multi_result = torch.cat(fullsequences, 0)   # batch dimension concat
            assert torch.allclose(single_example[i], multi_result)
        else:
            for j in range(len(multi_examples)):
                # batch dimension concat only
                multi_result = [multi_examples[j][k][i] for k in range(len(multi_examples[j]))]
                multi_result = torch.cat(multi_result, 0)
                assert torch.allclose(single_example[i], multi_result)


def check_grads():
    single_grad_path = 'results/single_process/grad_sp.pth'
    multi_grad_paths = [
        'results/multi_process_ddp/grad_sp_0_0.pth',
        'results/multi_process_ddp/grad_sp_0_1.pth',
        'results/multi_process_ddp/grad_sp_1_0.pth',
        'results/multi_process_ddp/grad_sp_1_1.pth',
    ]

    single_grad = torch.load(single_grad_path, map_location='cpu')
    multi_grads = [torch.load(p, map_location='cpu') for p in multi_grad_paths]

    for k, v in single_grad.items():
        for i in range(len(multi_grads)):
            g = multi_grads[i][k]

            if torch.allclose(v, g):
                print(f'{k} allclose')
            else:
                gap = torch.max(torch.abs(v-g))
                print(f'{k} FAILED!!!!!!!!!!!!!', gap)
                print(v)
                print(g)


if __name__ == "__main__":
    check_example()
    check_grads()

    # since intermediate outputs' dimensions are complex, check_output is skipped

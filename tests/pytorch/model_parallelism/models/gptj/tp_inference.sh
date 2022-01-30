# USAGE:   ``sh ./tp_inference.sh $NUM_GPUS``
# EXAMPLE: ``sh ./tp_inference.sh 4``

NUM_GPUS=$1

python -m torch.distributed.launch \
       --nproc_per_node="$NUM_GPUS" \
       ../../testcases/inference.py \
       --task=causal-lm \
       --model=anton-l/gpt-j-tiny-random \
       --tokenizer=gpt2 \
       --tensor_parallel_size="$NUM_GPUS"

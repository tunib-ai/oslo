# USAGE:   ``sh ./tp_merging.sh $NUM_GPUS``
# EXAMPLE: ``sh ./tp_merging.sh 4``

NUM_GPUS=$1

python -m torch.distributed.launch \
       --nproc_per_node="$NUM_GPUS" \
       ../../testcases/merging.py \
       --task=causal-lm \
       --model=EleutherAI/gpt-neo-125M \
       --tokenizer=gpt2 \
       --tensor_parallel_size="$NUM_GPUS"

# USAGE:   ``sh ./merging.sh $NUM_GPUS $CONFIG``
# EXAMPLE: ``sh ./merging.sh 4 ../../config/model_parallelism.json``

NUM_GPUS=$1
CONFIG=$2

python -m torch.distributed.launch \
       --nproc_per_node="$NUM_GPUS" \
       ../../testcases/merging.py \
       --task=causal-lm \
       --model=EleutherAI/gpt-neo-125M \
       --tokenizer=gpt2 \
       --tensor_parallel_size="$NUM_GPUS" \
       --config="$CONFIG"

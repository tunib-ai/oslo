# USAGE:   ``sh ./merging.sh $NUM_GPUS $CONFIG``
# EXAMPLE: ``sh ./merging.sh 4 ../../config/model_parallelism.json``

NUM_GPUS=$1
CONFIG=$2

python -m torch.distributed.launch \
       --nproc_per_node="$NUM_GPUS" \
       ../../testcases/merging.py \
       --task=seq2seq-lm \
       --model=facebook/blenderbot-400M-distill \
       --tensor_parallel_size="$NUM_GPUS" \
       --config="$CONFIG"

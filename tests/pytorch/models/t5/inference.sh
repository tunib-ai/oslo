# USAGE:   ``sh ./inference.sh $NUM_GPUS $CONFIG``
# EXAMPLE: ``sh ./inference.sh 4 ../../config/model_parallelism.json``

NUM_GPUS=$1
CONFIG=$2

python -m torch.distributed.launch \
       --nproc_per_node="$NUM_GPUS" \
       ../../testcases/inference.py \
       --task=seq2seq-lm \
       --model=t5-base \
       --input="translate English to German: My name is Wolfgang and I live in Berlin" \
       --tensor_parallel_size="$NUM_GPUS" \
       --config="$CONFIG"

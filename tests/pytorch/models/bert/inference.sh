# USAGE:   ``sh ./inference.sh $NUM_GPUS $CONFIG``
# EXAMPLE: ``sh ./inference.sh 4 ../../config/model_parallelism.json``

NUM_GPUS=$1
CONFIG=$2

python -m torch.distributed.launch \
       --nproc_per_node="$NUM_GPUS" \
       ../../testcases/inference.py \
       --task=masked-lm \
       --model=bert-base-cased \
       --tensor_parallel_size="$NUM_GPUS" \
       --config="$CONFIG"

python -m torch.distributed.launch \
       --nproc_per_node="$NUM_GPUS" \
       ../../testcases/inference.py \
       --task=sequence-classification \
       --model=textattack/bert-base-uncased-SST-2 \
       --tensor_parallel_size="$NUM_GPUS" \
       --config="$CONFIG"

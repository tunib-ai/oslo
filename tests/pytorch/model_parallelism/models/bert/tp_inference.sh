# USAGE:   ``sh ./tp_inference.sh $NUM_GPUS``
# EXAMPLE: ``sh ./tp_inference.sh 4``

NUM_GPUS=$1

python -m torch.distributed.launch \
       --nproc_per_node="$NUM_GPUS" \
       ../../testcases/inference.py \
       --task=masked-lm \
       --model=bert-base-cased \
       --tensor_parallel_size="$NUM_GPUS"

python -m torch.distributed.launch \
       --nproc_per_node="$NUM_GPUS" \
       ../../testcases/inference.py \
       --task=sequence-classification \
       --model=textattack/bert-base-uncased-SST-2 \
       --tensor_parallel_size="$NUM_GPUS"

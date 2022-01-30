# USAGE:   ``sh ./tp_inference.sh $NUM_GPUS``
# EXAMPLE: ``sh ./tp_inference.sh 4``

NUM_GPUS=$1

python -m torch.distributed.launch \
       --nproc_per_node="$NUM_GPUS" \
       ../../testcases/inference.py \
       --task=masked-lm \
       --model=google/electra-base-generator \
       --tensor_parallel_size="$NUM_GPUS"

python -m torch.distributed.launch \
       --nproc_per_node="$NUM_GPUS" \
       ../../testcases/inference.py \
       --task=sequence-classification \
       --model=EMBEDDIA/bertic-tweetsentiment \
       --tensor_parallel_size="$NUM_GPUS"

# USAGE:   ``sh ./training.sh $NUM_GPUS $BATCH_SIZE $SEQ_LENGTH $TRAIN_STEP $SAVE_INTERVAL $CONFIG``
# EXAMPLE: ``sh ./training.sh 4 4 512 50 25 ../../config/model_parallelism.json``

NUM_GPUS=$1
BATCH_SIZE=$2
SEQ_LENGTH=$3
TRAIN_STEP=$4
SAVE_INTERVAL=$5

python -m torch.distributed.launch \
       --nproc_per_node="$NUM_GPUS" \
       ../../testcases/training.py \
       --task=sequence-classification \
       --model=tals/albert-base-mnli \
       --batch_size="$BATCH_SIZE" \
       --sequence_length="$SEQ_LENGTH" \
       --train_step="$TRAIN_STEP" \
       --save_interval="$SAVE_INTERVAL" \
       --tensor_parallel_size="$NUM_GPUS" \
       --config="$CONFIG"

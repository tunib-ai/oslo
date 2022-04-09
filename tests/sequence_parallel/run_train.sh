# USAGE "sh ./run_train.sh 4 4 256 50 25"

NUM_GPUS=$1
BATCH_SIZE=$2
SEQ_LENGTH=$3
TRAIN_STEP=$4
SAVE_INTERVAL=$5

torchrun \
       --nproc_per_node="$NUM_GPUS" \
       training.py \
       --seed=42 \
       --model_name_or_path=sshleifer/tiny-gpt2 \
       --batch_size="$BATCH_SIZE" \
       --sequence_length="$SEQ_LENGTH" \
       --train_step="$TRAIN_STEP" \
       --save_interval="$SAVE_INTERVAL"
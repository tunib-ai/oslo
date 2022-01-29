# USAGE:   ``sh ./tp_inference.sh $NUM_GPUS``
# EXAMPLE: ``sh ./tp_inference.sh 4``

NUM_GPUS=$1

deepspeed --num_gpus="$NUM_GPUS" ../../testcases/inference.py \
          --task=causal-lm \
          --model=gpt2 \
          --tensor_parallel_size="$NUM_GPUS" \

deepspeed --num_gpus="$NUM_GPUS" ../../testcases/inference.py \
          --task=sequence-classification \
          --model=MiBo/SAGPT2 \
          --tokenizer=gpt2 \
          --tensor_parallel_size="$NUM_GPUS" \

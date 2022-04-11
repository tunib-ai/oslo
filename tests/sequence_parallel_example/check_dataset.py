import colossalai
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.context import ParallelMode

from data import build_train_valid_test_datasets
from data.tokenizer import initialize_tokenizer, get_padded_vocab_size
from data.datasets.dataset_utils import _build_train_valid_test_datasets, get_indexed_dataset_, get_train_valid_test_split_


DSET_TYPE_STD = 'standard_bert'
DSET_TYPE_ICT = 'ict'

DSET_TYPES = [DSET_TYPE_ICT, DSET_TYPE_STD]


def _build_train_valid_test_datasets_to_check(data_prefix, data_impl, splits_string,
                                     train_valid_test_num_samples,
                                     max_seq_length, masked_lm_prob,
                                     short_seq_prob, seed, skip_warmup,
                                     binary_head,
                                     dataset_type='standard_bert'):
    logger = get_dist_logger()

    if dataset_type not in DSET_TYPES:
        raise ValueError("Invalid dataset_type: ", dataset_type)

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    if dataset_type == DSET_TYPE_ICT:
        args = get_args()
        title_dataset = get_indexed_dataset_(args.titles_data_path,
                                             data_impl,
                                             skip_warmup)

    # Get start and end indices of train/valid/train into doc-idx
    # Note that doc-idx is desinged to be num-docs + 1 so we can
    # easily iterate over it.
    total_num_of_documents = indexed_dataset.doc_idx.shape[0] - 1
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    logger.info('\n > dataset split:')

    def print_split_stats(name, index):
        start_index = indexed_dataset.doc_idx[splits[index]]
        end_index = indexed_dataset.doc_idx[splits[index + 1]]
        logger.info('\n    {}:'.format(name) +
                    '\n     document indices in [{}, {}) total of {} documents'.format(
                        splits[index],
                        splits[index + 1],
                        splits[index + 1] - splits[index]) +
                    '\n     sentence indices in [{}, {}) total of {} sentences'.format(
                        start_index,
                        end_index,
                        end_index - start_index),
                    ranks=[0])
    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        from data.datasets.bert_dataset import BertDataset
        dataset = None
        if splits[index + 1] > splits[index]:
            # Get the pointer to the original doc-idx so we can set it later.
            doc_idx_ptr = indexed_dataset.get_doc_idx()
            # Slice the doc-idx
            start_index = splits[index]
            # Add +1 so we can index into the dataset to get the upper bound.
            end_index = splits[index + 1] + 1
            # New doc_idx view.
            indexed_dataset.set_doc_idx(doc_idx_ptr[start_index:end_index])
            # Build the dataset accordingly.
            kwargs = dict(
                name=name,
                data_prefix=data_prefix,
                num_epochs=None,
                max_num_samples=train_valid_test_num_samples[index],
                max_seq_length=max_seq_length,
                seed=seed,
                binary_head=binary_head
            )

            dataset = indexed_dataset
            # if dataset_type == DSET_TYPE_ICT:
            #     args = get_args()
            #     dataset = ICTDataset(
            #         block_dataset=indexed_dataset,
            #         title_dataset=title_dataset,
            #         query_in_block_prob=args.query_in_block_prob,
            #         use_one_sent_docs=args.use_one_sent_docs,
            #         **kwargs
            #     )
            # else:
            #     dataset = BertDataset(
            #         indexed_dataset=indexed_dataset,
            #         masked_lm_prob=masked_lm_prob,
            #         short_seq_prob=short_seq_prob,
            #         **kwargs
            #     )

            # Set the original pointer so dataset remains the main dataset.
            indexed_dataset.set_doc_idx(doc_idx_ptr)
            # Checks.
            assert indexed_dataset.doc_idx[0] == 0
            assert indexed_dataset.doc_idx.shape[0] == \
                   (total_num_of_documents + 1)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return train_dataset, valid_dataset, test_dataset


def main():
    # initialize
    colossalai.launch_from_torch(
        config='./config.py',
        seed=1234,
        backend='nccl')

    logger = get_dist_logger()

    # build dataloader
    # Data loader only on rank 0 of each model parallel group.
    if not gpc.is_initialized(ParallelMode.TENSOR) or gpc.get_local_rank(ParallelMode.TENSOR) == 0:
        initialize_tokenizer(gpc.config.VOCAB_FILE_PATH, tokenizer_type='BertWordPieceLowerCase')
        VOCAB_SIZE = get_padded_vocab_size()
        train_dataset, valid_dataset, test_dataset = _build_train_valid_test_datasets(
            train_valid_test_num_samples=[8, 8, 8],
            data_prefix=gpc.config.DATA_PATH,
            data_impl='mmap',
            splits_string='949,50,1',
            max_seq_length=gpc.config.SEQ_LENGTH,
            masked_lm_prob=0.15,
            short_seq_prob=0.1,
            seed=1234,
            skip_warmup=True,
            binary_head=False,
        )

        print(train_dataset[0]['text'])


if __name__ == "__main__":
    main()

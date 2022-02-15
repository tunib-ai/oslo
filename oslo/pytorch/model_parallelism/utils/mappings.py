import copy
import importlib


def update_module_arguments(module, **kwargs):
    for k, v in kwargs.items():
        setattr(module, k, v)


class TensorParallelismInfo(object):
    """
    A class to describe tensor parallelization information.

    Args:
        name (Tuple[str]): the name of parameter
        combined_qkv (bool): combined qkv or not
        parallel (bool): parallelizable param or not
        reverse (bool): reversed param or not
    """

    def __init__(self, *name, combined_qkv: bool = False, reverse: bool = False):
        self.name = name
        self.combined_qkv = combined_qkv
        self.reverse = reverse

    def __str__(self):
        return f"{self.__class__.__qualname__}({self.name})"

    def __repr__(self):
        return self.__str__()


Column = type("Column", (TensorParallelismInfo,), {})
Row = type("Row", (TensorParallelismInfo,), {})
Update = type("Update", (TensorParallelismInfo,), {})


class TensorParallelismMapping(object):
    __MAPPING__ = dict(
        Albert=[
            Column("query", "key", "value", "ffn"),
            Row("attention.dense", "ffn_output"),
            Update("num_attention_heads", "all_head_size"),
        ],
        Bart=[
            Column("q_proj", "k_proj", "v_proj", "fc1"),
            Row("out_proj", "fc2"),
            Update("embed_dim", "num_heads"),
        ],
        Bert=[
            Column("query", "key", "value", "intermediate.dense"),
            Row("output.dense"),
            Update("num_attention_heads", "all_head_size"),
        ],
        T5=[
            Column("q", "k", "v", "DenseReluDense.wi"),
            Row("o", "DenseReluDense.wo", "relative_attention_bias"),
            Update("d_model", "n_heads", "inner_dim"),
        ],
        GPT2=[
            Column("c_attn", reverse=True, combined_qkv=True),
            Column("c_fc", "q_attn", reverse=True),
            Row("c_proj", reverse=True),
            Update("embed_dim", "split_size", "num_heads"),
        ],
        GPTNeo=[
            Column("q_proj", "k_proj", "v_proj", "c_fc"),
            Row("out_proj", "c_proj"),
            Update("embed_dim", "num_heads"),
        ],
        GPTJ=[
            Column("q_proj", "k_proj", "v_proj", "fc_in"),
            Row("out_proj", "fc_out"),
            Update("embed_dim", "num_attention_heads"),
        ],
        Electra=[
            Column("query", "key", "value", "intermediate.dense"),
            Row("output.dense"),
            Update("num_attention_heads", "all_head_size"),
        ],
        Roberta=[
            Column("query", "key", "value", "intermediate.dense"),
            Row("output.dense"),
            Update("num_attention_heads", "all_head_size"),
        ],
    )

    def __init__(self):
        cache_mapping = {}

        for cls_name, mapping in self.__MAPPING__.items():
            cls = self._load_class_by_model_name(cls_name)
            cache_mapping[cls] = []

            for elem in mapping:
                for name in elem.name:
                    copy_elem = copy.deepcopy(elem)
                    copy_elem.name = name
                    cache_mapping[cls].append(copy_elem)

        self.__MAPPING__ = {cls: {} for cls in cache_mapping}
        # clear exist mapping rather than making new mapping dict

        for cls, mapping in cache_mapping.items():
            for elem in mapping:
                if elem.__class__.__qualname__ in self.__MAPPING__[cls]:
                    self.__MAPPING__[cls][elem.__class__.__qualname__].append(elem)
                else:
                    self.__MAPPING__[cls][elem.__class__.__qualname__] = [elem]

    @staticmethod
    def _load_class_by_model_name(model_name):
        """
        Load base class obj by class name

        Args:
            model_name (str): model name (e.g. Bert, GPT2, T5, ...)

        Returns:
            class: XXXPreTrainedModel
        """
        transformers = importlib.import_module("transformers")
        cls = getattr(transformers, f"{model_name}PreTrainedModel", None)
        if cls is None:
            cls = getattr(transformers, f"{model_name}PretrainedModel", None)
        if cls is None:
            raise ValueError(f"Can not import the model named {cls}.")
        return cls

    def get_mapping(self, model):
        """
        Get mapping by model obj

        Args:
            model (PreTrainedModel): model object (e.g. BertForSequenceClassification)

        Returns:
            dict: mapping by model
        """
        mapping_by_model = None
        for cls, mapping in self.__MAPPING__.items():
            if isinstance(model, cls):
                mapping_by_model = mapping

        assert mapping_by_model is not None, (
            f"Currently, {model.__class__.__qualname__} is not supported. "
            f"The current supported models are {list(self.__MAPPING__.keys())}"
        )
        return mapping_by_model

    def column_parallel_params(self, model):
        """
        Get list of column parallel param elements

        Args:
            model (PreTrainedModel): model obj

        Returns:
            List[Column]: list of column parallel param elements
        """
        mapping = self.get_mapping(model)
        if mapping is not None:
            return mapping["Column"]

    def row_parallel_params(self, model):
        """
        Get list of row parallel param elements

        Args:
            model (PreTrainedModel): model obj

        Returns:
            List[Row]: list of row parallel param elements
        """
        mapping = self.get_mapping(model)
        if mapping is not None:
            return mapping["Row"]

    def update_attrs(self, model):
        """
        Get list of update attribute elements

        Args:
            model (PreTrainedModel): model obj

        Returns:
            List[Update]: list of update attribute elements
        """
        mapping = self.get_mapping(model)
        if mapping is not None:
            return mapping["Update"]

    def search(self, model, param_name):
        """
        Get element by parameter name

        Args:
            model (PreTrainedModel): model obj

        Returns:
            TensorParallelismInfo: element by parameter name
        """
        mapping = self.get_mapping(model)
        count_contain_elem_in_param = 0
        param_split = param_name.split(".")
        first_check = []

        for elems in mapping.values():
            for elem in elems:
                if elem.name in param_name:
                    first_check.append(elem)

        for elem in first_check:
            elem_split = elem.name.split(".")
            for split in elem_split:
                if split in param_split:
                    count_contain_elem_in_param += 1
            if count_contain_elem_in_param == len(elem_split):
                return elem

        return None

    def is_combined_qkv_param(self, model, param_name):
        """
        Check whether the param is combined qkv or not

        Args:
            model (PreTrainedModel): model obj
            param_name (str): name of parameter

        Returns:
            bool: whether the param is combined qkv or not
        """
        elem = self.search(model, param_name)
        if elem is not None:
            return elem.combined_qkv

    def get_combined_qkv_degree(self, model, param_name, module):
        """
        Get combined qkv degree

        Args:
            model (PreTrainedModel): model obj
            param_name (str): name of parameter
            module (nn.Module): module that has `weight` parameter

        Returns:
            int: combined qkv degree
        """
        if self.is_combined_qkv_param(model, param_name) and hasattr(module, "weight"):
            bigger = max(module.weight.size(0), module.weight.size(1))
            smaller = min(module.weight.size(0), module.weight.size(1))
            return bigger // smaller
        return 1

    def is_reversed_param(self, model, param_name):
        """
        Check whether the parameter is reversed or not

        Args:
            model (PreTrainedModel): model obj
            param_name (str): name of parameter

        Returns:
            bool: whether the param is reversed or not
        """
        elem = self.search(model, param_name)
        if elem is not None:
            return elem.reverse

    def is_column_parallel(self, model, param_name):
        """
        Check whether the parameter is column parallelizable or not

        Args:
            model (PreTrainedModel): model obj
            param_name (str): name of parameter

        Returns:
            bool: whether the param is column parallelizable or not
        """
        elem = self.search(model, param_name)
        if elem is not None:
            return isinstance(elem, Column)

    def is_row_parallel(self, model, param_name):
        """
        Check whether the parameter is row parallelizable or not

        Args:
            model (PreTrainedModel): model obj
            param_name (str): name of parameter

        Returns:
            bool: whether the param is row parallelizable or not
        """
        elem = self.search(model, param_name)
        if elem is not None:
            return isinstance(elem, Row)

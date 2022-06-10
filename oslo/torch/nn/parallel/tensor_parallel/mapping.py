import copy


class TensorParallelInfo(object):
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


Column = type("Column", (TensorParallelInfo,), {})
Row = type("Row", (TensorParallelInfo,), {})
Update = type("Update", (TensorParallelInfo,), {})
Head = type("Head", (TensorParallelInfo,), {})


class TensorParallelMapping(object):
    __MAPPING__ = {}

    def __init__(self, tp_mapping=None):
        if isinstance(tp_mapping, dict):
            self.__MAPPING__.update(tp_mapping)
        elif tp_mapping is not None:
            raise ValueError("The argument `tp_mapping` must be None or dict.")

        cache_mapping = {}
        for cls, mapping in self.__MAPPING__.items():
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
            TensorParallelInfo: element by parameter name
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

    def is_head(self, model, param_name):
        """
        Check whether the parameter is lm head or not

        Args:
            model (PreTrainedModel): model obj
            param_name (str): name of parameter

        Returns:
            bool: whether the param is lm head or not
        """
        elem = self.search(model, param_name)
        if elem is not None:
            return isinstance(elem, Head)

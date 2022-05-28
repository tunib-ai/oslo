import copy


class ExpertParallelInfo(object):
    """
    A class to describe expert parallelization information.

    Args:
        name (Tuple[str]): the name of parameter
        reverse (bool): reversed param or not
    """

    def __init__(self, *name, reverse: bool = False):
        self.name = name
        self.reverse = reverse

    def __str__(self):
        return f"{self.__class__.__qualname__}({self.name})"

    def __repr__(self):
        return self.__str__()


Front = type("Front", (ExpertParallelInfo,), {})
Behind = type("Behind", (ExpertParallelInfo,), {})


class ExpertParallelMapping(object):
    __MAPPING__ = {}

    def __init__(self, ep_mapping=None):
        if isinstance(ep_mapping, dict):
            self.__MAPPING__.update(ep_mapping)
        elif ep_mapping is not None:
            raise ValueError("The argument `ep_mapping` must be None or dict")

        cache_mapping = {}
        for cls, mapping in self.__MAPPING__.items():
            cache_mapping[cls] = []

            for elem in mapping:
                for name in elem.name:
                    copy_elem = copy.deepcopy(elem)
                    copy_elem.name = name
                    cache_mapping[cls].append(copy_elem)

        self.__MAPPING__ = {cls: {} for cls in cache_mapping}

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

    def search(self, model, param_name):
        """
        Get element by parameter name

        Args:
            model (PreTrainedModel): model obj

        Returns:
            ExpertParallelInfo: element by parameter name
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

    def is_front_parallel(self, model, param_name):
        """
        Check whether the parameter is front parallelizable or not

        Args:
            model (PreTrainedModel): model obj
            param_name (str): name of parameter

        Returns:
            bool: whether the param is front parallelizable or not
        """

        elem = self.search(model, param_name)
        if elem is not None:
            return isinstance(elem, Front)

    def is_behind_parallel(self, model, param_name):
        """
        Check whether the parameter is behind parallelizable or not

        Args:
            model (PreTrainedModel): model obj
            param_name (str): name of parameter

        Returns:
            bool: whether the param is behind parallelizable or not
        """
        elem = self.search(model, param_name)
        if elem is not None:
            return isinstance(elem, Behind)

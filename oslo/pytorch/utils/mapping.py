import importlib


class nMapping(object):
    __MAPPING__ = {}

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
        if cls is not None:
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

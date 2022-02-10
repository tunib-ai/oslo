from oslo.pytorch.kernel_fusion.params.bert import BertParams
from oslo.pytorch.utils import Mapping


class KernelFusionMapping(Mapping):
    __MAPPING__ = {
        "Bert": BertParams,
    }

    def __init__(self):
        cache_mapping = {}

        for cls_name, mapping in self.__MAPPING__.items():
            cls = self._load_class_by_model_name(cls_name)
            cache_mapping[cls] = mapping

        self.__MAPPING__ = cache_mapping

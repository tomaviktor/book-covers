from dataclasses import dataclass, asdict, is_dataclass
import copy
import inspect


class TypeDict(dict):
    def __init__(self, t, *args, **kwargs):
        super(TypeDict, self).__init__(*args, **kwargs)

        if not isinstance(t, type):
            raise TypeError("t must be a type")

        self._type = t

    @property
    def type(self):
        return self._type


# copy of the internal function _is_dataclass_instance
def is_dataclass_instance(obj):
    return is_dataclass(obj) and not is_dataclass(obj.type)


def _from_dict_inner(obj):
    # reconstruct the dataclass using the type tag
    if is_dataclass_dict(obj):
        result = {}
        type_args = [
            x[0] for x in inspect.getmembers(obj.type, lambda a: not(inspect.isroutine(a))) if not x[0].startswith("__")
        ]
        for name, data in obj.items():
            if name in type_args:
                result[name] = _from_dict_inner(data)
        return obj.type(**result)

    # exactly the same as before (without the tuple clause)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_from_dict_inner(v) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((_from_dict_inner(k), _from_dict_inner(v))
                         for k, v in obj.items())
    else:
        return copy.deepcopy(obj)


def is_dataclass_dict(obj):
    return isinstance(obj, TypeDict)


def from_dict(obj):
    if not is_dataclass_dict(obj):
        raise TypeError("from_dict() should be called on TypeDict instances")
    return _from_dict_inner(obj)


@dataclass
class Config:
    token: str = None
    seed: int = 0
    learning_rate: float = 3e-4
    betas: tuple = (0.9, 0.95)
    num_labels: int = 207572
    model_name: str = 'microsoft/resnet-50'
    batch_size: int = 16
    image_size: int = 256
    grayscale: bool = False
    augmentation: bool = True
    debug: bool = False
    cache_dir: str = None
    num_epochs: int = 3
    perc_warmup_steps: float = 0.3
    steps_per_epoch: int = None

    def dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(obj: dict):
        t_dict = TypeDict(Config, **obj)
        cls = from_dict(t_dict)
        return cls


r"""
Modified from Habitat codebase

Import the global registry object using

.. code:: py

    from visimu.core.registry import registry

Various decorators for registry different kind of classes with unique keys

-   Register a model: ``@registry.register_model``
-   Register a metric: ``@registry.register_metric``
-   Register a dataset: ``@registry.register_dataset``

"""
import abc
import collections
from typing import Any, Callable, DefaultDict, Optional, Type

from torch.utils.data import Dataset
from pytorch_lightning import LightningModule
from torchmetrics import Metric

from typing import Any, Dict, List, Optional


class Singleton(type):
    _instances: Dict["Singleton", "Singleton"] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]

class Registry(metaclass=Singleton):
    mapping: DefaultDict[str, Any] = collections.defaultdict(dict)

    @classmethod
    def _register_impl(
        cls,
        _type: str,
        to_register: Optional[Any],
        name: Optional[str],
        assert_type: Optional[Type] = None,
    ) -> Callable:
        def wrap(to_register):
            if assert_type is not None:
                assert issubclass(
                    to_register, assert_type
                ), "{} must be a subclass of {}".format(
                    to_register, assert_type
                )
            register_name = to_register.__name__ if name is None else name

            cls.mapping[_type][register_name] = to_register
            return to_register

        if to_register is None:
            return wrap
        else:
            return wrap(to_register)

    @classmethod
    def register_curriculm(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl(
            "curriculm", to_register, name
        )

    @classmethod
    def _get_impl(cls, _type: str, name: str) -> Type:
        return cls.mapping[_type].get(name, None)

    @classmethod
    def get_curriculm(cls, name: str):
        return cls._get_impl("curriculm", name)

registry = Registry()

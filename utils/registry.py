from typing import Dict, Type, TypeVar
from torch.utils.data import Dataset
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

T = TypeVar("T")


class Registry:
    """Registry for managing dataset, model and learning rate scheduler classes."""

    def __init__(self):
        self._dataset_mapping: Dict[str, Type[Dataset]] = {}
        self._model_mapping: Dict[str, Type[Module]] = {}
        self._optimizer_mapping: Dict[str, Type[Optimizer]] = {}
        self._lr_scheduler_mapping: Dict[str, Type[LRScheduler]] = {}

    def _register_class(
        self,
        name: str,
        cls: Type[T],
        mapping: Dict[str, Type[T]],
        base_cls: Type[T],
        type_name: str,
    ) -> Type[T]:
        """Helper method to register a class in the specified mapping."""
        if not issubclass(cls, base_cls):
            raise TypeError(f"Class must inherit from {base_cls.__name__}")

        if name in mapping:
            raise KeyError(
                f"Name '{name}' already registered for {type_name}: {mapping[name]}"
            )

        mapping[name] = cls
        return cls

    def register_dataset(self, name: str):
        """Register a dataset class."""

        def wrapper(cls: Type[Dataset]) -> Type[Dataset]:
            return self._register_class(
                name, cls, self._dataset_mapping, Dataset, "dataset"
            )

        return wrapper

    def register_model(self, name: str):
        """Register a model class."""

        def wrapper(cls: Type[Module]) -> Type[Module]:
            return self._register_class(name, cls, self._model_mapping, Module, "model")

        return wrapper

    def register_optimizer(self, name: str):
        """Register an optimizer class."""

        def wrapper(cls: Type[Optimizer]) -> Type[Optimizer]:
            return self._register_class(
                name, cls, self._optimizer_mapping, Optimizer, "optimizer"
            )

        return wrapper

    def register_lr_scheduler(self, name: str):
        """Register a learning rate scheduler class."""

        def wrapper(cls: Type[LRScheduler]) -> Type[LRScheduler]:
            return self._register_class(
                name, cls, self._lr_scheduler_mapping, LRScheduler, "lr scheduler"
            )

        return wrapper

    def get_dataset_class(self, name: str) -> Type[Dataset]:
        """Get registered dataset class by name."""
        return self._dataset_mapping.get(name)

    def get_model_class(self, name: str) -> Type[Module]:
        """Get registered model class by name."""
        return self._model_mapping.get(name)

    def get_optimizer_class(self, name: str) -> Type[Optimizer]:
        """Get registered optimizer class by name."""
        return self._optimizer_mapping.get(name)

    def get_lr_scheduler_class(self, name: str) -> Type[LRScheduler]:
        """Get registered learning rate scheduler class by name."""
        return self._lr_scheduler_mapping.get(name)

    def list_datasets(self) -> list[str]:
        """List all registered dataset names."""
        return sorted(self._dataset_mapping.keys())

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return sorted(self._model_mapping.keys())

    def list_optimizers(self) -> list[str]:
        """List all registered optimizer names."""
        return sorted(self._optimizer_mapping.keys())

    def list_lr_schedulers(self) -> list[str]:
        """List all registered learning rate scheduler names."""
        return sorted(self._lr_scheduler_mapping.keys())


registry = Registry()


def test_registry():
    """Test the Registry class functionality."""
    registry = Registry()

    # Test model registration
    @registry.register_model("test_model")
    class TestModel(Module):
        pass

    assert registry.get_model_class("test_model") == TestModel
    assert "test_model" in registry.list_models()

    # Test dataset registration
    @registry.register_dataset("test_dataset")
    class TestDataset(Dataset):
        pass

    assert registry.get_dataset_class("test_dataset") == TestDataset
    assert "test_dataset" in registry.list_datasets()

    # Test optimizer registration
    @registry.register_optimizer("test_optimizer")
    class TestOptimizer(Optimizer):
        def __init__(self):
            super().__init__([], {})

        def step(self):
            pass

    assert registry.get_optimizer_class("test_optimizer") == TestOptimizer
    assert "test_optimizer" in registry.list_optimizers()

    # Test lr scheduler registration
    @registry.register_lr_scheduler("test_scheduler")
    class TestScheduler(LRScheduler):
        pass

    assert registry.get_lr_scheduler_class("test_scheduler") == TestScheduler
    assert "test_scheduler" in registry.list_lr_schedulers()

    # Test invalid registration
    try:

        @registry.register_model("invalid")
        class InvalidModel:  # Not a Module subclass
            pass

        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Test getting non-existent class
    assert registry.get_model_class("nonexistent") is None

    print("All registry tests passed!")


if __name__ == "__main__":
    test_registry()

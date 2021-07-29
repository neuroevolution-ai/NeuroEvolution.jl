import abc

registered_optimizer_classes = {}


def get_optimizer_class(optimizer_class_name: str):
    if optimizer_class_name in registered_optimizer_classes:
        return registered_optimizer_classes[optimizer_class_name]
    else:
        raise RuntimeError("No valid optimizer")


class IOptimizer(abc.ABC):

    @abc.abstractmethod
    def ask(self):
        pass

    @abc.abstractmethod
    def tell(self, rewards):
        pass

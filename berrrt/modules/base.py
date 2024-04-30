from berrrt.modules.berrrt import BERRRTModel
from berrrt.modules.berrrt_gate import BERRRTGateModel
from berrrt.modules.bert import BERTModel
from berrrt.modules.berrrt_early_exit import BERRRTEarlyExitModel


class ModulesFactory:
    modules = {
        "berrrt": BERRRTModel,
        "bert": BERTModel,
        "berrrt_gate": BERRRTGateModel,
        "berrrt_early_exit" : BERRRTEarlyExitModel,
    }

    def __init__(
        self,
        model_type: str,
    ) -> None:
        self.model_type = model_type

        if model_type not in self.modules:
            raise ValueError(f"Unknown model type: {model_type}")

    def create_model(self, *args, **kwargs):
        return self.modules[self.model_type](*args, **kwargs)

from flwr.server.strategy import FedAvg
from flwr.common import (
    parameters_to_ndarrays,
    ndarrays_to_parameters,                     
)


class CustomFedAvg(FedAvg):
    def __init__(self, net, lossf, testloader, fraction_fit=0.1, fraction_eval=0.1, min_fit_clients=2, min_eval_clients=2, min_available_clients=2, eval_fn=None, on_fit_config_fn=None, on_evaluate_config_fn=None, accept_failures=True, initial_parameters=None):
        super().__init__(fraction_fit=fraction_fit,
                         fraction_eval=fraction_eval,
                         min_fit_clients=min_fit_clients,
                         min_eval_clients=min_eval_clients,
                         min_available_clients=min_available_clients,
                         eval_fn=eval_fn,
                         on_fit_config_fn=on_fit_config_fn,
                         on_evaluate_config_fn=on_evaluate_config_fn,
                         accept_failures=accept_failures,
                         initial_parameters=initial_parameters)
        self.net = net
        self.lossf = lossf
        self.testloader = testloader
        
    def evaluate(self, parameters):
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        self.set_parameters(self.net, parameters_ndarrays)
        loss, accuracy = self.lossf(self.net, self.testloader)
        return loss, {"accuracy": accuracy}
    
    def set_parameters(self, net, parameters):    
        net.set_parameters(parameters, strict=True)
    
    def get_parameters(self, net):
        return net.get_parameters()
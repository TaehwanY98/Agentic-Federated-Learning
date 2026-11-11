from flwr.client.numpy_client import NumPyClient

class client(NumPyClient):
    def __init__(self, net, lossf, trainloader, optimizer):
        super().__init__()
    def fit(self, parameters, config):
        return super().fit(parameters, config)
    
    def evaluate(self, parameters, config):
        return super().evaluate(parameters, config)

    def get_parameters(self, config):
        return super().get_parameters(config)
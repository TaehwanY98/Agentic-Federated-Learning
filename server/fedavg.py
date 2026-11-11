from flwr.server.strategy import FedAvg
from logging import WARNING, ERROR, DEBUG
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    
)
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from model.LLM_decision_agent import AFL_Decision_Agent_Call

class AgenticFedAvg(FedAvg):
    def __init__(self, agent, net, lossf, testloader, fraction_fit=0.1, fraction_eval=0.1, min_fit_clients=2, min_eval_clients=2, min_available_clients=2, eval_fn=None, on_fit_config_fn=None, on_evaluate_config_fn=None, accept_failures=True, initial_parameters=None):
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
        self.agent = agent
        self.net = net
        self.lossf = lossf
        self.testloader = testloader

    def aggregate_fit(self,
            server_round: int,
            results: list[tuple[ClientProxy, FitRes]],
            failures: list[tuple[ClientProxy, FitRes] | BaseException],
        ):
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        loss, metric_scores = self.task_evaluation()
        num_samples = [fit_res.num_examples for _, fit_res in results]
        AFL_decision = AFL_Decision_Agent_Call(self.agent, num_samples=str(num_samples), num_class=str(self.testloader.dataset.targets), eval_metrics_scores=f"loss: {loss}, metric_scores: {metric_scores}")

        heterogeneity, evaluation_score, description = AFL_decision.get('heterogeneity'), AFL_decision.get('evaluation_score'), AFL_decision.get('description')

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
        
    def task_evaluation(self, parameters):
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        self.set_parameters(self.net, parameters_ndarrays)
        loss, accuracy = self.lossf(self.net, self.testloader)
        return loss, {"accuracy": accuracy}
    
    def set_parameters(self, net, parameters):    
        net.set_parameters(parameters, strict=True)
    
    def get_parameters(self, net):
        return net.get_parameters()
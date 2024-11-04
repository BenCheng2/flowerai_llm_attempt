"""fedobd: A Flower / HuggingFace app."""
from typing import Optional, Callable, List, Tuple
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, NDArrays, Parameters, \
    MetricsAggregationFn
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from transformers import AutoModelForSequenceClassification

from fedobd.task import get_weights
from functools import reduce
from flwr.common import Scalar
import numpy as np


class CustomFedAvg(FedAvg):
    def __init__(
            self,
            *,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, dict[str, Scalar]],
                    Optional[tuple[float, dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            inplace: bool = True,
            shapes: List[Tuple[int, ...]] = None,
            blocks: List[np.ndarray] = None
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            inplace=inplace
        )
        self.shapes = shapes
        self.blocks = blocks
        self.num_runs = 0

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, fit_res.metrics)
            for _, fit_res in results
        ]
        aggregated_ndarrays = self._aggregate_helper(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        self.num_runs += 1

        return parameters_aggregated, {}

    def _aggregate_helper(self, results: list[tuple[NDArrays, int, dict[str, Scalar]]]) -> NDArrays:
        num_examples_total = 0

        weighted_weights = []

        for weights, num_examples, metrics in results:
            layer_weights = []

            pointer = 0
            for index in range(len(self.shapes)):
                if str(index) in metrics:
                    layer = weights[pointer]
                    pointer += 1
                    layer_weights.append(layer * num_examples)
                    num_examples_total += num_examples
                else:
                    # read the i-th shape from shapes and create empty list matching that size
                    # shape = self.shapes[index]
                    # layer_weights.append(np.zeros(shape))
                    layer_weights.append(self.blocks[index] * num_examples)
            weighted_weights.append(layer_weights)

        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]

        return weights_prime

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)

        if self.num_runs > 3:
            return num_available_clients, self.min_available_clients
        else:
            return max(num_clients, self.min_fit_clients), self.min_available_clients



def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize global model
    model_name = context.run_config["model-name"]
    num_labels = context.run_config["num-labels"]
    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    weights = get_weights(net)
    initial_parameters = ndarrays_to_parameters(weights)

    shapes = [w.shape for w in weights]
    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        initial_parameters=initial_parameters,
        shapes=shapes,
        blocks=weights
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)

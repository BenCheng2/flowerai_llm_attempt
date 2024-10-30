"""fedobd: A Flower / HuggingFace app."""
import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, Scalar, NDArrays
from transformers import AutoModelForSequenceClassification

from fedobd.task import get_weights, load_data, set_weights, test, train


# Flower client
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train(self.net, self.trainloader, epochs=self.local_epochs, device=self.device)
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        return float(loss), len(self.testloader), {"accuracy": accuracy}

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        if config["round"] < 3:
            return self.get_parameters_for_sending(self.net, 0.1)
        else:
            return self.get_parameters_for_phase_2(self.net)

    def get_parameters_for_sending(self, net, dropout_rate: float) -> NDArrays:
        old_blocks = self.old_weights
        current_blocks = get_weights(net)
        diff_dictionary = {}
        for i in range(len(old_blocks)):
            diff_dictionary[i] = np.linalg.norm(old_blocks[i] - current_blocks[i])
        retain_rate = 1 - dropout_rate
        retain_count = int(retain_rate * len(old_blocks))

        sorted_indices = np.argsort(list(diff_dictionary.values()))

        retained_indices = sorted_indices[-retain_count:]

        new_blocks = []
        for i in range(len(old_blocks)):
            if i in retained_indices:
                # new_blocks.append(current_blocks[i])
                new_blocks.append(current_blocks[i].astype(np.uint8))
            else:
                new_blocks.append(None)

        return new_blocks

    def get_parameters_for_phase_2(self, net) -> NDArrays:
        old_blocks = self.old_weights
        current_blocks = get_weights(net)

        new_blocks = []
        for i in range(len(old_blocks)):
            fine_tuned_block = np.round(current_blocks[i], decimals=4).astype(np.uint8)
            new_blocks.append(fine_tuned_block)

        return new_blocks

def client_fn(context: Context):

    # Get this client's dataset partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    model_name = context.run_config["model-name"]
    trainloader, valloader = load_data(partition_id, num_partitions, model_name)

    # Load model
    num_labels = context.run_config["num-labels"]
    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)

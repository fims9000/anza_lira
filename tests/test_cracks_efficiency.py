import torch

from cracks_experiment.efficiency import parameter_count


def test_parameter_count_counts_only_trainable_parameters() -> None:
    model = torch.nn.Linear(3, 2)
    assert parameter_count(model) == 8
    model.bias.requires_grad_(False)
    assert parameter_count(model) == 6

import torch

from vasae.metrics.base import Aggregator, MetricComposer
from vasae.models.sae import SAEOutput


@torch.no_grad()
def evaluate(
    model, data_loader, metrics: MetricComposer, device, logger, max_batchsize=0
):

    aggregator = Aggregator()
    model.eval()
    for batch_i, data in enumerate(data_loader):
        data, display_text = data["activations"], data["display_text"]
        data = data.to(device)
        output: SAEOutput = model(data)
        decoded = output.hidden_states_recon

        eval_outcomes = metrics.compute(
            {"hidden_states": data, "hidden_states_recon": decoded, "display_text": display_text}
        )

        logger.info(f"{batch_i}/{len(data_loader)}")
        batchsize = data.size(0)
        aggregator.add(
            {
                "loss": output.loss.detach().cpu().item(),
                "loss_reconst": output.recon_loss,
                "loss_l1": output.l1_loss,
                "logitlens_acc": eval_outcomes["logitlens_acc"],
                "loss_lowrank": output.loss_lowrank,
            },
            batchsize,
        )

        if max_batchsize > 0 and batch_i > max_batchsize:
            break

    return aggregator.compute()

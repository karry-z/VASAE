import numpy as np
import torch

from vasae.metrics.interface import Aggregator, MetricComposer
from vasae.metrics.logitlens import LogitLensAccuracy
from vasae.models.sae_hf import SAEOutput


@torch.no_grad()
def evaluate(model, data_loader, metrics: MetricComposer, device, logger):
    data_ids = []
    reconst_ids = []
    loss_reconst_per_emb = []
    loss_l1 = []
    aggregator = Aggregator()
    model.eval()
    for batch_i, data in enumerate(data_loader):
        data, display_text = data["activations"], data["display_text"]
        data = data.to(device)
        output: SAEOutput = model(data)

        decoded = output.hidden_states_recon
        loss_reconst_per_emb.extend(output.loss_per_sample.flatten().detach().tolist())
        if output.l1_loss is not None:
            loss_l1.append(output.l1_loss.detach().item())

        eval_outcomes = metrics.compute(
            {"data": data, "decoded": decoded, "display_text": display_text}
        )
        data_ids.extend(eval_outcomes["data_ids"])
        reconst_ids.extend(eval_outcomes["recons_ids"])
        logger.info(f"{batch_i}/{len(data_loader)}")
        batchsize = data.size(0)
        aggregator.add(
            {
                "loss": output.loss.detach().cpu().item(),
                "loss_reconst": output.recon_loss,
                "loss_l1": output.l1_loss,
                "logitlens_acc": eval_outcomes["logitlens_acc"],
            },
            batchsize,
        )

    total_acc = LogitLensAccuracy().compute(reconst_ids, data_ids)

    res = {
        "loss_reconst": np.array(loss_reconst_per_emb).mean().item(),
        "loss_reconst_std": np.array(loss_reconst_per_emb).std().item(),
        "acc": total_acc,
    }
    if loss_l1:
        res["loss_l1"] = np.array(loss_l1).mean().item()
    return res

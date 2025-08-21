import torch
import torch.nn.functional as F

def mc_predict_mean_probs(model, X, T=100):
    model.train()
    probs_list = []
    with torch.no_grad():
        for _ in range(T):
            logits_t = model(X)
            probs_t = F.softmax(logits_t, dim=1)
            probs_list.append(probs_t.unsqueeze(0))
    probs_mc = torch.cat(probs_list, dim=0)
    model.eval()
    return probs_mc.mean(0)

def generate_mc_outputs(model, X, T=1000, metric="mc_estimate", labels=None):
    model.train()
    outputs = []
    with torch.no_grad():
        for _ in range(T):
            out = model(X)
            outputs.append(out.unsqueeze(0))
    outputs = torch.cat(outputs, dim=0)
    
    all_probs = torch.softmax(outputs, dim=2)
    mean_probs = all_probs.mean(dim=0)
    var_pred = outputs.var(dim=0)

    if metric == "mc_estimate":
        metric_value = mean_probs
    elif metric == "variance":
        metric_value = var_pred
    elif metric == "predictive_entropy":
        metric_value = -(mean_probs * (mean_probs + 1e-12).log()).sum(dim=1)
    elif metric == "relative_norm":
        if labels is None:
            raise ValueError("labels must be provided for relative_norm metric")
        labels_onehot = F.one_hot(labels, num_classes=mean_probs.size(1)).float()
        diff_norm = torch.norm(mean_probs - labels_onehot, dim=1)
        denom = torch.max(torch.norm(mean_probs, dim=1), torch.norm(labels_onehot, dim=1))
        metric_value = diff_norm / (denom + 1e-12)
    else:
        raise ValueError("Metric must be 'mc_estimate', 'variance', 'predictive_entropy' or 'relative_norm'")
    
    model.eval()
    return outputs, mean_probs, metric_value
import torch
import torch.nn.functional as F

def mc_predict_mean_probs(model, X, T):
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
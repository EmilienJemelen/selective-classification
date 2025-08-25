import torch
import torch.nn.functional as F

def generate_mc_outputs(model, X, T, metrics="mc_estimate", labels=None):
    model.train()  # dropout actif en inférence
    outputs = []
    with torch.no_grad():
        for _ in range(T):
            out = model(X)                      
            outputs.append(out.unsqueeze(0))    
    
    outputs = torch.cat(outputs, dim=0)         
    all_probs   = torch.softmax(outputs, dim=2)  
    mean_probs  = all_probs.mean(dim=0)          # moyenne des softmax
    var_pred  = outputs.var(dim=0)                # variance des logits
   
    results = {}
    for metric in metrics:
        if metric == "mc_estimate":
            results[metric] = mean_probs
        elif metric == "variance":
            results[metric] = var_pred
        elif metric == "predictive_entropy":
            results[metric] = -(mean_probs * (mean_probs + 1e-12).log()).sum(dim=1) # +1e-12 pour éviter log(0)
        elif metric == "relative_norm":
            if labels is None:
                raise ValueError("labels must be provided for relative_norm metric")
            labels_onehot = F.one_hot(labels, num_classes=mean_probs.size(1)).float()
            diff_norm = torch.norm(mean_probs - labels_onehot, dim=1)   # ||ŷ̄ - Y||
            denom = torch.max(torch.norm(mean_probs, dim=1), torch.norm(labels_onehot, dim=1))
            results[metric] = diff_norm / (denom + 1e-12)                  
        else:
            raise ValueError(f"Metric {metric} non reconnue")
    model.eval()  # remettre le modèle en mode eval à la fin
    return outputs, mean_probs, results
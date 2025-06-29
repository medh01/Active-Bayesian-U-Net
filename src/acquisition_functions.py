import torch, torch.nn.functional as F

def random_score(model, imgs, **kwargs):
    """Generates random scores for acquisition.

    This function assigns a random score to each image in the batch.

    Args:
        model (torch.nn.Module): The neural network model (not used in this function,
                                   but included for consistency).
        imgs (torch.Tensor): A batch of input images with shape (B, C, H, W).
        **kwargs: Additional keyword arguments (not used).

    Returns:
        torch.Tensor: A tensor of random scores with shape (B,),
                      where B is the batch size.
    """
    return torch.rand(imgs.size(0), device=imgs.device) # Shape: (B,)

def entropy(model, imgs, T=8, num_classes=4):
    """Computes the entropy score for active learning.

    This function calculates the predictive entropy for each image by performing
    `T` stochastic forward passes through the model (with dropout enabled).

    Args:
        model (torch.nn.Module): The neural network model.
        imgs (torch.Tensor): A batch of input images with shape (B, C, H, W).
        T (int, optional): Number of Monte Carlo samples (stochastic forward passes).
                           Defaults to 8.
        num_classes (int, optional): The number of output classes.
                                     Defaults to 4.

    Returns:
        torch.Tensor: A tensor of entropy scores with shape (B,),
                      where B is the batch size. The score is the spatial sum
                      of the entropy map for each image.
    """
    model.train() # Keep dropout ON for stochasticity
    probs_sum = torch.zeros(
        imgs.size(0), num_classes, *imgs.shape[2:], device=imgs.device
    ) # Shape: (B, C, H, W)

    for _ in range(T):
        with torch.amp.autocast('cuda'):
            logits = model(imgs) # Shape: (B, C, H, W)
            probs  = F.softmax(logits, 1) # Shape: (B, C, H, W)
        probs_sum += probs

    probs_mean = probs_sum / T # Shape: (B, C, H, W)
    ent = -(probs_mean * probs_mean.log()).sum(dim=1) # Shape: (B, H, W)
    return ent.sum(dim=(1, 2)) # Shape: (B,)


def BALD(model, imgs, T=8, num_classes=4):
    """Computes the Bayesian Active Learning by Disagreement (BALD) score.

    BALD quantifies the mutual information between the model predictions and the
    model parameters, given the input. It is calculated as the difference between
    the predictive entropy and the expected entropy.

    Args:
        model (torch.nn.Module): The neural network model.
        imgs (torch.Tensor): A batch of input images with shape (B, C, H, W).
        T (int, optional): Number of Monte Carlo samples (stochastic forward passes).
                           Defaults to 8.
        num_classes (int, optional): The number of output classes.
                                     Defaults to 4.

    Returns:
        torch.Tensor: A tensor of BALD scores with shape (B,),
                      representing the mean BALD score over the spatial dimensions
                      for each image.
    """
    model.train()
    probs_sum = torch.zeros(
        imgs.size(0), num_classes, *imgs.shape[2:], device=imgs.device
    ) # Shape: (B, C, H, W)
    entropies_sum = torch.zeros(
        imgs.size(0), *imgs.shape[2:], device=imgs.device
    ) # Shape: (B, H, W)

    for _ in range(T):
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                logits = model(imgs) # Shape: (B, C, H, W)
                probs = F.softmax(logits, dim=1) # Shape: (B, C, H, W)

        # Accumulate probabilities for calculating predictive entropy
        probs_sum += probs

        # Calculate entropy of the current prediction and accumulate it
        # H[y|x, Θ_t] = -Σ_c (p_c * log(p_c))
        entropy_t = -(probs * torch.log(probs + 1e-12)).sum(dim=1) # Shape: (B, H, W)
        entropies_sum += entropy_t

    # 1. Calculate Predictive Entropy: H[y|x]
    probs_mean = probs_sum / T # Shape: (B, C, H, W)
    predictive_entropy = -(probs_mean * torch.log(probs_mean + 1e-12)).sum(dim=1) # Shape: (B, H, W)

    # 2. Calculate Expected Entropy: E[H[y|x, Θ]]
    expected_entropy = entropies_sum / T # Shape: (B, H, W)

    # 3. Compute BALD score for each pixel
    # I(y; Θ|x) = H[y|x] - E[H[y|x, Θ]]
    bald_map = predictive_entropy - expected_entropy # Shape: (B, H, W)

    return bald_map.mean(dim=(1, 2)) # Shape: (B,)



def committee_kl_divergence(model, imgs, T=8, num_classes=4):
    """Computes the Committee KL-Divergence score for active learning.

    This function calculates the Kullback-Leibler (KL) divergence between a
    deterministic standard prediction (model.eval()) and a Monte Carlo posterior
    average prediction (model.train() with dropout) for each image.

    Args:
        model (torch.nn.Module): The neural network model.
        imgs (torch.Tensor): A batch of input images with shape (B, C, H, W).
        T (int, optional): Number of Monte Carlo samples (stochastic forward passes).
                           Defaults to 8.
        num_classes (int, optional): The number of output classes.
                                     Defaults to 4.

    Returns:
        torch.Tensor: A tensor of KL-Divergence scores with shape (B,),
                      representing the mean KL divergence over the spatial dimensions
                      for each image.
    """
    B, _, H, W = imgs.shape
    device     = imgs.device

    # 1) Monte Carlo posterior under dropout
    model.train()
    all_probs = torch.zeros(T, B, num_classes, H, W, device=device) # Shape: (T, B, C, H, W)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i in range(T):
            logits      = model(imgs) # Shape: (B, C, H, W)
            all_probs[i] = F.softmax(logits, dim=1) # Shape: (B, C, H, W)
    posterior = all_probs.mean(dim=0) # Shape: (B, C, H, W)

    # 2) Deterministic “standard” prediction
    model.eval()   # Turn OFF dropout here
    with torch.no_grad(), torch.amp.autocast('cuda'):
        logits        = model(imgs) # Shape: (B, C, H, W)
        standard_probs = F.softmax(logits, dim=1) # Shape: (B, C, H, W)

    # 3) Compute per-pixel KL(standard || posterior)
    eps      = 1e-12
    P        = torch.clamp(standard_probs,   min=eps)
    Q        = torch.clamp(posterior,        min=eps)
    kl_map   = P * (P.log() - Q.log()) # Shape: (B, C, H, W)
    kl_pixel = kl_map.sum(dim=1) # Shape: (B, H, W)
    kl_score = kl_pixel.mean(dim=(1, 2)) # Shape: (B,)

    return kl_score

def committee_js_divergence(model, imgs, T=8, num_classes=4):
    """Computes a per-image Jensen-Shannon (JS) divergence score.

    The JS divergence measures the similarity between two probability distributions.
    Here, it's computed between:
      - `p`: The standard (deterministic) softmax prediction from the model (dropout off).
      - `Q`: The Monte Carlo average softmax prediction over `T` stochastic forward passes (dropout on).

    The JS divergence is defined as:
    JS(p || Q) = 0.5 * KL(p || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (p + Q).

    Args:
        model (torch.nn.Module): The neural network model.
        imgs (torch.Tensor): A batch of input images with shape (B, C, H, W).
        T (int, optional): Number of Monte Carlo samples (stochastic forward passes).
                           Defaults to 8.
        num_classes (int, optional): The number of output classes.
                                     Defaults to 4.

    Returns:
        torch.Tensor: A tensor of JS-Divergence scores with shape (B,),
                      representing the mean JS divergence over all pixels for each image.
    """
    B, _, H, W = imgs.shape
    device = imgs.device

    # 1) Monte Carlo posterior Q
    model.train()  # Keep dropout on
    all_probs = torch.zeros(T, B, num_classes, H, W, device=device) # Shape: (T, B, C, H, W)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i in range(T):
            logits = model(imgs) # Shape: (B, C, H, W)
            all_probs[i] = F.softmax(logits, dim=1) # Shape: (B, C, H, W)
    Q = all_probs.mean(dim=0) # Shape: (B, C, H, W)

    # 2) Deterministic standard prediction p
    model.eval()  # Turn dropout off
    with torch.no_grad(), torch.amp.autocast('cuda'):
        logits = model(imgs) # Shape: (B, C, H, W)
        p = F.softmax(logits, dim=1) # Shape: (B, C, H, W)

    # 3) Build mixture M and clamp for numerical stability
    p = torch.clamp(p, min=eps)
    Q = torch.clamp(Q, min=eps)
    M = torch.clamp(0.5 * (p + Q), min=eps)

    # 4) Compute ½ KL(p‖M) + ½ KL(Q‖M) per pixel
    kl_p_m = p * (p.log() - M.log()) # Shape: (B, C, H, W)
    kl_q_m = Q * (Q.log() - M.log()) # Shape: (B, C, H, W)
    js_map   = 0.5 * (kl_p_m + kl_q_m).sum(dim=1) # Shape: (B, H, W)
    js_score = js_map.mean(dim=(1, 2)) # Shape: (B,)

    return js_score
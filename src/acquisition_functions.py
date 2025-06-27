import torch, torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

def random_score(model, imgs,**kwargs):
    return torch.rand(imgs.size(0), device=imgs.device)

def entropy(model, imgs, T=8, num_classes=4):
    model.train()                      # keep dropout ON for stochasticity
    probs_sum = torch.zeros(
        imgs.size(0), num_classes, *imgs.shape[2:], device=imgs.device)

    for _ in range(T):
        with torch.amp.autocast('cuda'):
            logits = model(imgs)               # B,C,H,W
            probs  = F.softmax(logits, 1)
        probs_sum += probs

    probs_mean = probs_sum / T                 # p̄
    ent = -(probs_mean * probs_mean.log()).sum(dim=1)  # B,H,W
    return ent.sum(dim=(1, 2))                # spatial sum → (B,)


def BALD(model, imgs, T=8, num_classes=4): #bald_map = predictive_entropy - expected_entropy
    model.train()
    probs_sum = torch.zeros(
        imgs.size(0), num_classes, *imgs.shape[2:], device=imgs.device
    )
    entropies_sum = torch.zeros(
        imgs.size(0), *imgs.shape[2:], device=imgs.device
    )

    for _ in range(T):
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                logits = model(imgs)  # Shape: (B, C, H, W)
                probs = F.softmax(logits, dim=1)  # Shape: (B, C, H, W)

        # Accumulate probabilities for calculating predictive entropy
        probs_sum += probs

        # Calculate entropy of the current prediction and accumulate it
        # H[y|x, Θ_t] = -Σ_c (p_c * log(p_c))
        entropy_t = -(probs * torch.log(probs + 1e-12)).sum(dim=1)  # Shape: (B, H, W)
        entropies_sum += entropy_t

        # 1. Calculate Predictive Entropy: H[y|x]
        # First, get the mean probability distribution over T passes
    probs_mean = probs_sum / T
    # Then, calculate the entropy of this mean distribution
    predictive_entropy = -(probs_mean * torch.log(probs_mean + 1e-12)).sum(dim=1)  # (B, H, W)

    # 2. Calculate Expected Entropy: E[H[y|x, Θ]]
    # This is the average of the entropies from each pass
    expected_entropy = entropies_sum / T  # (B, H, W)

    # 3. Compute BALD score for each pixel
    # I(y; Θ|x) = H[y|x] - E[H[y|x, Θ]]
    bald_map = predictive_entropy - expected_entropy  # (B, H, W)

    # Return the mean BALD score over the spatial dimensions for each image
    return bald_map.mean(dim=(1, 2))  # Shape: (B,)

def committee_kl_divergence(model, imgs, T=8, num_classes=4):
    B, _, H, W = imgs.shape
    device     = imgs.device

    # 1) Monte Carlo posterior under dropout
    model.train()
    all_probs = torch.zeros(T, B, num_classes, H, W, device=device)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i in range(T):
            logits      = model(imgs)
            all_probs[i] = F.softmax(logits, dim=1)
    posterior = all_probs.mean(dim=0)  # (B, C, H, W)

    # 2) Deterministic “standard” prediction
    model.eval()   # <-- turn OFF dropout here
    with torch.no_grad(), torch.amp.autocast('cuda'):
        logits        = model(imgs)
        standard_probs = F.softmax(logits, dim=1)

    # 3) Compute per-pixel KL(standard || posterior)
    eps      = 1e-12
    P        = torch.clamp(standard_probs,   min=eps)
    Q        = torch.clamp(posterior,        min=eps)
    kl_map   = P * (P.log() - Q.log())        # shape (B, C, H, W)
    kl_pixel = kl_map.sum(dim=1)              # (B, H, W)
    kl_score = kl_pixel.mean(dim=(1, 2))      # (B,)

    return kl_score

# def score_unlabeled_pool(unl_loader, model, device="cuda", num_classes=4):
#     model.to(device)
#     scores, fnames = [], []
#
#     with torch.no_grad():
#         for imgs, names in tqdm(unl_loader, desc="Scoring", leave=False):
#             imgs = imgs.to(device)
#             s    = entropy(model, imgs, T=8, num_classes=num_classes)
#             scores.extend(s.cpu().tolist())
#             fnames.extend(names)
#
#     return dict(zip(fnames, scores))   # { filename : entropy }

def committee_js_divergence(model, imgs, T=8, num_classes=4, eps=1e-12):
    """
    Computes a per-image JS divergence score between:
      p = standard (deterministic) softmax
      Q = MCMC average softmax over T stochastic forward passes
    JS = 0.5*KL(p||M) + 0.5*KL(Q||M),  M = 0.5*(p+Q)
    Returns: Tensor of shape (B,) with the mean JS over all pixels.
    """
    B, _, H, W = imgs.shape
    device = imgs.device

    # --- 1) Monte Carlo posterior Q ---
    model.train()  # keep dropout on
    all_probs = torch.zeros(T, B, num_classes, H, W, device=device)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i in range(T):
            logits = model(imgs)                  # (B,C,H,W)
            all_probs[i] = F.softmax(logits, dim=1)
    Q = all_probs.mean(dim=0)                   # (B,C,H,W)

    # --- 2) Deterministic standard prediction p ---
    model.eval()  # turn dropout off
    with torch.no_grad(), torch.amp.autocast('cuda'):
        logits = model(imgs)
        p = F.softmax(logits, dim=1)            # (B,C,H,W)

    # --- 3) Build mixture M and clamp for numerical stability --- to not get log(0)
    p = torch.clamp(p, min=eps)
    Q = torch.clamp(Q, min=eps)
    M = torch.clamp(0.5 * (p + Q), min=eps)

    # --- 4) Compute ½ KL(p‖M) + ½ KL(Q‖M) per pixel ---
    # KL(p‖M) per class: p * (log p – log M)
    kl_p_m = p * (p.log() - M.log())
    kl_q_m = Q * (Q.log() - M.log())
    # sum over classes → (B,H,W), then average spatially → (B,)
    js_map   = 0.5 * (kl_p_m + kl_q_m).sum(dim=1)
    js_score = js_map.mean(dim=(1, 2))

    return js_score



# def score_unlabeled_pool(unl_loader, model, acq_type="entropy", T=8, num_classes=4, device="cuda"):
#     """
#     Scores unlabeled pool using specified acquisition function
#
#     Args:
#         unl_loader: DataLoader for unlabeled images
#         model: Model to use for scoring
#         acq_type: Type of acquisition function ('entropy', 'bald', 'kl_divergence')
#         T: Number of Monte Carlo samples
#         num_classes: Number of output classes
#         device: Device to use for computation
#
#     Returns:
#         Dictionary of {filename: score}
#     """
#     model.to(device)
#     model.train()  # Ensure dropout is enabled for all acquisition functions
#
#     # Map acquisition type to scoring function
#     acquisition_functions = {
#         "entropy": entropy,
#         "bald": BALD,
#         "kl_divergence": committee_kl_divergence
#     }
#
#     # Validate acquisition type
#     if acq_type not in acquisition_functions:
#         raise ValueError(f"Invalid acquisition type: {acq_type}. "
#                          f"Valid options are {list(acquisition_functions.keys())}")
#
#     # Select the scoring function
#     score_func = acquisition_functions[acq_type]
#
#     scores, fnames = [], []
#
#     with torch.no_grad():
#         for imgs, names in tqdm(unl_loader, desc=f"Scoring ({acq_type})", leave=False):
#             imgs = imgs.to(device)
#
#             # Handle different parameter requirements
#             if acq_type == "kl_divergence":
#                 s = score_func(model, imgs, T=T, num_classes=num_classes)
#             else:
#                 s = score_func(model, imgs, T=T, num_classes=num_classes)
#
#             scores.extend(s.cpu().tolist())
#             fnames.extend(names)
#
#     return dict(zip(fnames, scores))

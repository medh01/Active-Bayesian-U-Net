# Step 2: Test if acquisition functions give different scores
import torch
import numpy as np
import random

import sys, os
sys.path.append(os.path.abspath("../src"))   # path relative to current file

def test_acquisition_functions():
    # Set seeds consistently
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Import your functions
    from bayesian_unet import BayesianUNet
    from acquisition_functions import entropy, BALD, committee_js_divergence, committee_kl_divergence

    # Create model
    model = BayesianUNet(1, 4, [64, 128, 256, 512], 0.1)

    # Create dummy images (batch of 5 images)
    dummy_imgs = torch.randn(5, 1, 128, 128)  # Adjust size as needed

    print("Image batch shape:", dummy_imgs.shape)

    print("Testing acquisition functions with dummy data...")
    print("=" * 50)

    # Test each acquisition function
    functions_to_test = {
        "entropy": entropy,
        "bald": BALD,
        "js_divergence": committee_js_divergence,
        "kl_divergence": committee_kl_divergence
    }

    results = {}

    for func_name, func in functions_to_test.items():
        print(f"\nTesting {func_name}:")

        # IMPORTANT: Reset seed before each function call
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        try:
            scores = func(model, dummy_imgs, T=5, num_classes=4)
            results[func_name] = scores.cpu().numpy()
            print(f"  Scores: {scores.cpu().numpy()}")
            print(f"  Mean: {scores.mean().item():.6f}")
            print(f"  Std: {scores.std().item():.6f}")
            print(f"  Min: {scores.min().item():.6f}")
            print(f"  Max: {scores.max().item():.6f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 50)
    print("COMPARISON:")

    # Compare results
    func_names = list(results.keys())
    if len(func_names) >= 2:
        for i in range(len(func_names)):
            for j in range(i + 1, len(func_names)):
                func1, func2 = func_names[i], func_names[j]
                scores1, scores2 = results[func1], results[func2]

                # Check if scores are identical
                are_identical = np.allclose(scores1, scores2, atol=1e-6)
                correlation = np.corrcoef(scores1, scores2)[0, 1]

                print(f"{func1} vs {func2}:")
                print(f"  Identical: {are_identical}")
                print(f"  Correlation: {correlation:.4f}")
                print(f"  Max difference: {np.max(np.abs(scores1 - scores2)):.8f}")


if __name__ == "__main__":
    test_acquisition_functions()
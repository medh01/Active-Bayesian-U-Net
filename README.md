# Active Bayesian U-Net
![Computer Vision](https://img.shields.io/badge/topic-Computer%20Vision-teal)
![Semantic Segmentation](https://img.shields.io/badge/task-Semantic%20Segmentation-blueviolet)
![Medical Imaging](https://img.shields.io/badge/domain-Medical%20Imaging-red)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-%23EE4C2C)
![Python](https://img.shields.io/badge/python-3.11.11-blue)

This project is an implementation of **an Active Learning Framework based on Bayesian U-Net** to improve the segmentation of
**human embryo images**. The objective is to optimize **annotation** efforts while maintaining high segmentation
performance.
> *Note: 1*
This implementation is inspired from this research paper [Active Learning with Bayesian U-Net for Efficient Semantic Image Segmentation by Isah Charles Saidu and Lehel Csató](https://doi.org/10.3390/jimaging7020037).

> *Note: 2*
I highly recommend reading the paper if you are looking to dive deep into this project



## Table of Contents

- [Repository Structure](#repository-structure)
- [Dependency Graph](#dependency-graph)
- [Data Preparation](#data-preparation)
- [Prerequisites](#prerequisites)
- [Running the Project on kaggle](#Running-the-Project-on-kaggle)
- [Implementation Details](#implementation-details)
- [Contact](#contact)

## Repository Structure

```
Active-Bayesian-U-Net/
└── docs/                         # contains images for the README.md file
└── examples/                     # contains examples of images and their masks
└── experiments/                  # contains the results of the experiments
└── src/ 
    ├── mask_converter.py        # To Convert the 3 binary masks (ICM, TE, ZP) into an RGB mask                     
    ├── active_learning_loop.py  # Active Learning Pipeline implementation 
    ├── acquisition_functions.py # random, Entropy, BALD, KL-Divergence, JS-Divergence
    ├── active_learning_utils.py # resetting and creating active learning pools, scoring and moving unlabeled data
    ├── bayesian_unet_parts.py   # DoubleConv, Encoder, Decoder
    ├── bayesian_unet.py         # Bayesian U-Net implementation
    ├── train_eval.py            # train and evaluate one epoch
    ├── metrics.py               # custom loss functions (Tversky Loss) and Evaluation metrics (Dice, accuracy)
    ├── data_loading.py          # Preprocessing and Loading the Blastocyst Data
    
└── .gitignore
└── requirements.txt
└── README.md

```

## Dependency Graph

<p align="center">
  <img src="./docs/dependancy%20graph.png" alt="Dependancy Diagram" width="700"/>
</p>

## Data Preparation

> **Note:** The full dataset is private, but three example images and corresponding RGB masks are included in the `examples/` folder.

The original dataset consisted of four folders:

- `images/`: original grayscale blastocyst images  
- `GT_ICM/`: binary mask for Inner Cell Mass  
- `GT_TE/`: binary mask for Trophectoderm  
- `GT_ZP/`: binary mask for Zona Pellucida

We merged these three binary masks into a single RGB mask with 5 classes:

- **Blue** channel: Inner Cell Mass (ICM)  
- **Red** channel: Trophectoderm (TE)  
- **Green** channel: Zona Pellucida (ZP)
- **Yellow** channel: blastocoel (BL)
- **BLack** channel: background

> **Note:** We don't have an explicit blastocoel mask, so to isolate this region I:
> 1. Filled holes in the ZP mask  
> 2. Subtracted the ring itself to get the interior  
> 3. Removed any overlap with the ICM and TE masks

Below is a table showing an example blastocyst image, its three binary masks (ICM, TE, ZP), and the combined RGB mask:

|                       Original Image                       |                             ICM Mask                             |                            TE Mask                            |                            ZP Mask                            |                       RGB Mask                       |
|:----------------------------------------------------------:|:----------------------------------------------------------------:|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:----------------------------------------------------:|
| ![Original](./examples/images/Blast_PCRM_1201754%20D5.BMP) | ![ICM](./examples/GT_ICM/Blast_PCRM_1201754%20D5%20ICM_MASK.BMP) | ![TE](./examples/GT_TE/Blast_PCRM_1201754%20D5%20TE_MASK.BMP) | ![ZP](./examples/GT_ZP/Blast_PCRM_1201754%20D5%20ZP_MASK.BMP) | ![RGB](./examples/masks/Blast_PCRM_1201754%20D5.png) |


## Prerequisites

- Python 3.11.11 or higher
- Core Packages: torch,matplotlib, tqdm, numpy, Pillow, pandas
> *Note: Check requirements.txt*

## Running the Project on kaggle
Follow these steps in a Kaggle notebook to reproduce the experiments:

1. **Import the Dataset**  
   - In your Kaggle notebook, click **Add data** and select the `Blastocyst dataset`.

2. **Clone the GitHub Repository**  
    ```bash
    !git clone https://github.com/medh01/Active-Bayesian-U-Net.git

3. **Install Dependencies**
    ```bash
   %pip install -r Active-Bayesian-U-Net/requirements.txt

4. Prepare the Data Directory
   - Create a local data folder and copy images & masks from the Kaggle dataset
    
   ```bash
    %%bash
    # ← edit this to match your dataset’s folder under /kaggle/input
    DATASET_PATH="<PATH_TO_EMBRYO_DATASET>"   # e.g. /kaggle/input/embryo-images-and-masks

    # where you want to store a local copy
    WORKING_DATA="/kaggle/working/data"

    # create the directory
    mkdir -p "$WORKING_DATA"

    # copy images & masks
    cp -r "$DATASET_PATH/images" "$WORKING_DATA/"
    cp -r "$DATASET_PATH/masks"  "$WORKING_DATA/"

5. Add the Source to Your Python Path
    ```bash
   import sys
    sys.path.append("/kaggle/working/Active-Bayesian-U-Net/src")

6. Launch an active learning loop
    - Feel free to adjust any of the parameters below to suit your needs
    ```Python
   from active_learning_loop import active_learning_loop

    df = active_learning_loop(
    BASE_DIR          = "/kaggle/working/data",
    LABEL_SPLIT_RATIO = 0.1,
    TEST_SPLIT_RATIO  = 0.2,
    augment = True,
    sample_size       = 10,
    acquisition_type  = acq,
    mc_runs           = 5,
    batch_size        = 4,
    lr                = 1e-3,
    loop_iterations   = None,
    seed              = 1,
    device            = "cuda"
    )

7. Run experiments to compare different acquisition functions
- Feel free tro adjust any of the parameters below to suit your needs
   
## Implementation Details

### Data Encoding & Preprocessing

#### Mask Encoding
I used **label encoding** to convert colour‐coded masks into integer class labels for training. Each pixel’s RGB value is mapped to a class index as follows:

* Inner Cell Mass (ICM, blue channel) → 0

* Trophectoderm (TE, red channel) → 1

* Zona Pellucida (ZP, green channel) → 2

* blastocoel (BL, yellow channel) → 3 

* Background (black) → 4

#### Size Standardization
To ensure all inputs share the same spatial dimensions, every grayscale image and its corresponding mask are padded (using ImageOps.pad) to a fixed size of 256x256 pixels. 
This preserves the original aspect ratio, fills any extra borders with background values, and guarantees consistent tensor shapes for efficient batched training.

## Bayesian U-Net

### Bayesian U-Net Architecture
![](docs/Bayesian%20U-Net%20architecture.png)

### Loss and Evaluation Metrics
> implemented in `src/metrics.py`
* Loss function \
The loss function plays a crucial role in updating model weights and enhancing overall accuracy.  
However, in human embryo segmentation, we face significant class imbalance like shown in the table:
<p align="center">
  <img src="./docs/class%20imbalance.png" alt="Class imbalance" width="200"/>
</p>

That being said, I evaluated several loss functions, including cross-entropy, weighted cross-entropy, Dice loss, Tversky loss, and their combinations to solve the problem of class imbalance. 
I found that **a weighted cross-entropy + Tversky loss (excluding the background)** yielded the best results

1. Wighted Cross Entropy \
Weighted cross-entropy extends the standard cross-entropy loss by multiplying each class’s loss term by a pre-computed weight, typically the inverse of its frequency in the training set. \
This loss was used to emphasise minority classes. 
This way, rare classes (e.g. ICM or TE) contribute more to the total loss, so the network receives stronger gradient signals when it misclassifies them.

| Basic Cross Entropy                                                     | Weighted Cross Entropy                                                        |
|:------------------------------------------------------------------------:|:----------------------------------------------------------------------------:|
| $$\mathcal{L}_{CE} = -\sum_{c=1}^{C} y_c \,\log p_c$$                     | $$\mathcal{L}_{WCE} = -\sum_{c=1}^{C} w_c\,y_c\,\log p_c$$                     |



2. Tversky Loss

$$\mathcal{L}_{T} = 1 \;-\; \frac{\mathrm{TP}}{\mathrm{TP} + \alpha\,\mathrm{FP} + \beta\,\mathrm{FN}}$$

I chose **α = 0.3** and **β = 0.7** to make the Tversky loss penalise false negatives more than false positives.

In practice, this weighting:

- **Boosts recall** for small, rare structures (like the ICM) by punishing missed pixels more heavily  
- **Reduces under-segmentation**, ensuring the model doesn’t “play it safe” and omit tiny regions  

> **In a nutshell:**  
> α < β focuses the loss on catching every positive pixel, even at the cost of a few extra false alarms.

3. Combining the Two Losses
Combining **Weighted Cross-Entropy** (WCE) with **Tversky Loss** gives two complementary strengths:

**Pixel-wise balancing (WCE)**  
- Scales each pixel’s loss by its class weight.  
- Ensures rare classes (ICM, TE, ZP, BL) contribute strongly to the gradient, preventing the model from defaulting to the background.

**Overlap-based focus (Tversky)**  
- Directly optimises region overlap, with tunable penalties for false positives vs. false negatives.  
- Encourages accurate boundaries and robust detection of small structures.

* Metrics

$$
\text{Dice coefficient: }\quad 
\mathrm{Dice} = \frac{2\,\lvert P \cap G\rvert}{\lvert P\rvert + \lvert G\rvert}
$$

<p align="center">
  <img src="./docs/dice%20coefficient.png" alt="Dice Coefficient" width="400"/>
</p>

$$
\text{Pixel accuracy: }\quad 
\mathrm{Accuracy} = \frac{\text{correct pixels}}{\text{total pixels}}
$$

> I trained a Bayesian U-Net and visualized the segmentation outputs, you can find the full Jupyter notebook in ./experiments 

### Active Learning Pipeline
![](docs/active%20learning%20pipeline.png)

#### Splitting the Data Pool 
![](docs/splitting%20data.png)

>*Note: You can tweak the splitting ratios | check active_learning_pool.py*

#### Active learning algorithm 
Active learning loop was implemented following strictly this algorithm 
Refer to paper for more insights 
![](./docs/active%20learning%20algorithm.png)

#### Acquisition Functions

##### Notation

- (i = 1, ..... , B\) indexes the **image** in a batch of size \(B\).

- \(c = 1, ..... , C\) indexes the **class** channel (C = 4) in your case).

- \(t = 1, ..... , T\) indexes the **Monte Carlo** forward pass (dropout sample).

- (x,y) denotes spatial pixel coordinates.

- $$z^{(t)}_{i,c}(x,y)$$ is the logit for image \(i\), class \(c\), pixel \((x,y)\) on pass \(t\).

- The softmax probability on pass \(t\) is then:

$$p^{(t)}(i,c)(x,y) = \frac{\exp\left(z^{(t)}(i,c)(x,y)\right)}{\sum_{k=1}^C \exp\left(z^{(t)}(i,k)(x,y)\right)}$$
---

##### 1. Random sampling

$$
s_i = U(0,1)
$$

---

##### 2. Entropy sampling

1. **Mean predictive**

$$
\bar p_{i,c}(x,y) = \frac{1}{T} \sum_{t=1}^T p^{(t)}_{i,c}(x,y).
$$

2. **Entropy map**

$$
H_i(x,y) = -\sum_{c=1}^C \bar p_{i,c}(x,y)\,\ln \bar p_{i,c}(x,y).
$$

3. **Image score**

$$
s_i = \frac{1}{H\,W} \sum_{x=1}^H \sum_{y=1}^W H_i(x,y).
$$

---

##### 3. BALD

##### 1. Predictive entropy**

$$
H\bigl[\bar p_i(x,y)\bigr] = -\sum_{c=1}^C \bar p_{i,c}(x,y)\,\ln \bar p_{i,c}(x,y).
$$

##### 2. Expected Entropy

The expected entropy over the T MC passes is:

$$\mathbb{E}\left[H\left[p^{(t)}(i)(x,y)\right]\right] = \frac{1}{T} \sum_{t=1}^T \left[ -\sum_{c=1}^C p^{(t)}(i,c)(x,y) \ln p^{(t)}(i,c)(x,y) \right]$$

##### 3. BALD Map & Score

The pixel-wise BALD (Bayesian Active Learning by Disagreement) map is:

$$\mathrm{BALD}_i(x,y) = H\bigl[\bar p_i(x,y)\bigr] - \mathbb{E}_t\bigl[H\bigl(p^{(t)}_i(x,y)\bigr)\bigr]$$

The image-level acquisition score is:

$$s_i = \frac{1}{H\,W} \sum_{x=1}^H \sum_{y=1}^W \mathrm{BALD}_i(x,y)$$

##### 5. Committee KL-Divergence

1. **Posterior mean**:
   $$\bar p_{i,c}(x,y) = \frac{1}{T}\sum_{t=1}^T p^{(t)}_{i,c}(x,y)$$

2. **Deterministic prediction** (no dropout):
  $\mathbb{E}\big[H\big[p^{(t)}(i)(x,y)\big]\big] = \frac{1}{T} \sum_{t=1}^T \Big[ -\sum_{c=1}^C p^{(t)}(i,c)(x,y) \ln p^{(t)}(i,c)(x,y) \Big]$
3. **KL map & score**:
  $KL_i = \sum_{c=1}^C p_{ic} \ln(p_{ic} / q_{ic})$
   
   $$s_i = \frac{1}{H\,W} \sum_{x=1}^H \sum_{y=1}^W \mathrm{KL}_i(x,y)$$

##### 6. Committee JS-Divergence

Let:
$$Q_{i,c}(x,y) = \bar p_{i,c}(x,y), \quad p^*_{i,c}(x,y)$$

Define the mixture:
$$M_{ic}(x,y) = \frac{1}{2}(p_{ic}^*(x,y) + Q_{ic}(x,y))$$

The per-pixel JS divergence is:
$$\mathrm{JS} = \frac{1}{2} \sum_{c=1}^C p_{c} + \frac{1}{2} \sum_{c=1}^C Q_{c}$$

Finally, the image-level JS score is:
$$s_i = \frac{1}{H\,W} \sum_{x=1}^H \sum_{y=1}^W \mathrm{JS}_i(x,y)$$

## Results

![](./docs/results plot.png)

![](./docs/results table.png)

![](./docs/results summary.png)

## Contact

If you have any questions or suggestions, please open an issue or contact the maintainer at `<https://www.linkedin.com/in/mohamed-heni-178a9819b/>`.


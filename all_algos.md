# PyLDL Algorithm Reference

A categorized description of every LDL, LE, classifier, loss, and auxiliary algorithm listed in the [README](README.md) and implemented under [pyldl/algorithms/](pyldl/algorithms/). Non-neural-network methods are summarized briefly; neural-network methods get fuller treatment covering architecture, loss, and optimization.

---

## 1. Problem Transformation (PT)

Classical decomposition that converts LDL into weighted single-label classification/regression.

### `PT_Bayes` *(Geng 2016)*
Replicates each sample `c` times (one per class), samples the replicas with probability proportional to their description degrees `D[i, j]`, then fits a Gaussian naive Bayes classifier with `predict_proba` giving the label distribution. ([_problem_transformation.py:46-57](pyldl/algorithms/_problem_transformation.py#L46-L57))

### `PT_SVM` *(Geng 2016)*
Same replica/resample preprocessing as `PT_Bayes`, but the base classifier is a `LinearSVC` wrapped in `CalibratedClassifierCV` so `predict_proba` is available. ([_problem_transformation.py:60-67](pyldl/algorithms/_problem_transformation.py#L60-L67))

### `LDSVR` *(Geng & Hou 2015)*
"LDL via Support Vector Regression." Maps each `D[:, j]` through a logit transform, then fits a `MultiOutputRegressor(SVR)` with `gamma` set from the median pairwise distance of `X`. Predictions are pushed back through a sigmoid and re-normalized. ([_problem_transformation.py:82-91](pyldl/algorithms/_problem_transformation.py#L82-L91))

---

## 2. Specialized Algorithms (SA)

Direct optimization of a MaxEnt (linear-softmax) model against KL divergence.

### `SA_BFGS` *(Geng 2016)*
Model: `D̂ = softmax(XW)`. Fits `W` by minimizing `KL(D ‖ D̂)` with `scipy.optimize.minimize(..., method='L-BFGS-B')`, supplying both loss and the analytic gradient `Xᵀ(D̂ − D)`. Convex, so it finds the global optimum. ([_specialized_algorithms.py:44-65](pyldl/algorithms/_specialized_algorithms.py#L44-L65))

### `SA_IIS` *(Geng 2016)*
Same model and loss as `SA_BFGS`, but solved via **Improved Iterative Scaling** — a coordinate-wise procedure that derives a per-parameter 1-D nonlinear equation and solves each with `scipy.optimize.fsolve`. Slower than BFGS in practice but monotonically decreases KL by construction. ([_specialized_algorithms.py:68-112](pyldl/algorithms/_specialized_algorithms.py#L68-L112))

### `LALOT` *(Zhao & Zhou 2018)*
"Label distribution via Optimal Transport." Same `_SA` base — a MaxEnt model — but with an OT-based loss instead of KL (see `_specialized_algorithms.py`). Brief variant of the SA family for ordered/structured label spaces.

---

## 3. Algorithm Adaptation — Non-Neural

### `AA_KNN` *(Geng 2016)*
`NearestNeighbors` with `k=5` by default; prediction is the average label distribution of a test point's `k` training neighbors. Zero parameters to learn beyond the neighbor index. ([_algorithm_adaptation.py:14-29](pyldl/algorithms/_algorithm_adaptation.py#L14-L29))

---

## 4. Algorithm Adaptation — Neural Networks

### `AA_BP` *(Geng 2016)*
The reference MLP-for-LDL baseline. A 3-layer network `Dense(n_hidden, sigmoid) → Dense(n_outputs, softmax)` trained by SGD on per-sample MSE against the target label distribution. No custom hooks — the class inherits all defaults from `BaseGD + BaseDeepLDL` ([base.py:402-416](pyldl/algorithms/base.py#L402-L416), [base.py:485-558](pyldl/algorithms/base.py#L485-L558)) and overrides nothing. Training loop uses a manual `GradientTape` + `apply_gradients` step so that downstream subclasses can expose batch-index-aware losses; for `AA_BP` itself this is equivalent to `keras.Model.fit`. ([_algorithm_adaptation.py:38-44](pyldl/algorithms/_algorithm_adaptation.py#L38-L44))

### `CPNN` *(Geng, Yin, & Zhou 2013)*
"Conditional Probability Neural Network." A 3-layer network that concatenates each feature vector with a one-hot-like label index and outputs a scalar conditional probability; softmax is applied across the `n_outputs` outputs of each sample to recover a distribution. Loss is KL divergence; the optimizer is `RProp` (resilient backprop, imported from `pyldl.algorithms.utils`). The `mode` argument toggles three input encodings: `'none'` (scalar label index), `'binary'` (one-hot label), and `'augment'` (one-hot plus top-class re-weighted replication of training samples). ([_algorithm_adaptation.py:48-95](pyldl/algorithms/_algorithm_adaptation.py#L48-L95))

### `BCPNN` *(Yang, Sun, & Sun 2017)*
Shortcut for `CPNN(mode='binary')` — uses the one-hot label encoding rather than the scalar index of the original CPNN. ([_algorithm_adaptation.py:99-115](pyldl/algorithms/_algorithm_adaptation.py#L99-L115))

### `ACPNN` *(Yang, Sun, & Sun 2017)*
Shortcut for `CPNN(mode='augment')` — replicates each training sample `v` times with weighted boosted top-class description degrees, which acts as a targeted oversampling scheme before backprop. ([_algorithm_adaptation.py:119-135](pyldl/algorithms/_algorithm_adaptation.py#L119-L135))

### `LDLF` *(Shen et al. 2017)*
"LDL Forests" — deep neural decision forests adapted to LDL. A 3-layer MLP produces a sigmoid-activated "decision" vector of size `n_latent`; for each of `n_estimators` trees, a random subset `phi_i` of those decisions routes samples probabilistically down a complete binary tree of depth `n_depth`, yielding per-leaf soft assignments `mu`. Each tree has learnable leaf distributions `pi_i` over outputs; the predicted distribution is `pi_i * mu`. Loss: KL divergence against the target, averaged over trees. Inside the loss function, `pi_i` is updated in closed form (the E-step of the variational decision forest) while the underlying network is updated by Adam. Prediction is the simplex projection (`proj`) of the per-tree averaged output. ([_algorithm_adaptation.py:139-236](pyldl/algorithms/_algorithm_adaptation.py#L139-L236))

### `Duo_LDL` *(Żychowski & Mańdziuk 2021)*
"Duo-LDL." Instead of predicting the label distribution directly, the 3-layer MLP (`tanh` output) predicts all pairwise *differences* `D[:, i] − D[:, j]` — an output of size `n_outputs · (n_outputs − 1)`. At prediction time the pairwise differences are averaged into per-label scores which are then re-normalized into a distribution. Training targets `C = concat[D − roll(D, i)]` for `i = 1..n_outputs-1` with MSE loss, trained by Adam with `batch_size=50`. ([_algorithm_adaptation.py:240-267](pyldl/algorithms/_algorithm_adaptation.py#L240-L267))

### `BD_LDL` *(Liu et al. 2021)*
"Bi-Directional LDL." Closed-form linear model `W ∈ ℝ^(n_features × n_outputs)` obtained by solving a Sylvester equation `AW + WB = C` with `A = XᵀX + β I`, `B = α DᵀD`, `C = (1+α) XᵀD`. Not a neural network despite its placement in `_algorithm_adaptation.py`; prediction is the simplex projection of `XW`. ([_algorithm_adaptation.py:270-289](pyldl/algorithms/_algorithm_adaptation.py#L270-L289))

---

## 5. Regularization / Constraint-Based LDL

All are softmax-linear or numpy-linear models with auxiliary regularizers — most use L-BFGS (via `BaseBFGS`) or ADMM (via `BaseADMM`).

### `LDLLC` *(Jia et al. 2018)*
"LDL with Label Correlation." Softmax-linear `D̂ = softmax(XW)` plus a correlation-aware regularizer that penalizes pairwise Euclidean distance between label-specific weight vectors, signed by the Pearson correlation of their columns. Optimized by L-BFGS on flattened weights. ([_ldllc.py](pyldl/algorithms/_ldllc.py))

### `LDLSF` *(Ren et al. 2019a)*
"LDL with Specific Features." ADMM solve of a Frobenius-norm LDL loss with sparsity (`L1` soft-thresholding on `W1`) and group-sparsity (`L2,1` via `solvel21` on `W2`) regularizers plus a correlation-based Laplacian term. Inner gradient/objective accelerated with `numba.jit`. ([_ldlsf.py](pyldl/algorithms/_ldlsf.py))

### `LDL_LCLR` *(Ren et al. 2019b)*
"LDL with Label Correlations + Low-Rank." ADMM over four variables `{W, S, E, Z}` — the softmax-linear weights `W`, a label-correlation matrix `S`, an `L2,1` outlier matrix `E`, and a low-rank surrogate `Z` enforced via singular value thresholding. KMeans on `D` supplies cluster-wise pairwise Euclidean penalties. Inner W/S updates use `numba` + L-BFGS. ([_ldl_lclr.py](pyldl/algorithms/_ldl_lclr.py))

### `LDL_SCL` *(Jia et al. 2021)*
"LDL via Sample Correlations Locally." Deep method despite the constraint flavor. A 2-layer softmax model trained by Adam; an additional learnable `C ∈ ℝ^(n_samples × n_clusters)` acts as a per-sample soft assignment to `n_clusters` KMeans centroids of the training label distributions, and `W ∈ ℝ^(n_clusters × n_outputs)` is added inside the softmax logits. Loss: KL divergence + a correlation term that pulls assigned centroids close to the prediction + a barrier term `1/C`. At inference, an SVR is fit on training `(X, C)` to extrapolate cluster memberships to test points. ([_ldl_scl.py](pyldl/algorithms/_ldl_scl.py))

### `LDL_LRR` *(Jia et al. 2023a)*
"LDL with Label Ranking Relation." Softmax-linear model trained by L-BFGS; on top of the KL loss, a pairwise ranking cross-entropy term is added — `P[i,j,k]` indicates whether `D[i,j] − D[i,k] > 0.5`, and the model's sigmoided difference `sigmoid(D̂[i,j] − D̂[i,k])` is trained to match it, weighted by the squared target difference. Adds a ranking signal to preserve label ordering. ([_ldl_lrr.py](pyldl/algorithms/_ldl_lrr.py))

### `LDL_DPA` *(Jia et al. 2023b)*
"LDL with Description-degree Percentile Average." Softmax-linear L-BFGS optimizer with two added terms: (1) a ranking loss that rewards the model for concentrating mass on labels with higher description-degree rank; (2) a variance-matching term that penalizes the gap between per-sample variance of true and predicted distributions. ([_ldl_dpa.py](pyldl/algorithms/_ldl_dpa.py))

### `TLRLDL` / `TKLRLDL` *(Kou et al. 2024)*
"(Top-)k Low-Rank LDL." ADMM-solved linear model where the label matrix is first binarized — either by a threshold (`TLRLDL`) or by selecting top-`k` labels per sample (`TKLRLDL`) — and a low-rank regularizer (SVT) is applied to an auxiliary projection `O` of the weight matrix. ([_lrldl.py](pyldl/algorithms/_lrldl.py))

### `LDL_HVLC` *(Lin et al. 2024)*
"LDL with Horizontal & Vertical Label Correlation." A 2-layer linear model augmented with a learnable `n_outputs × n_outputs` label-correlation matrix `M`. For each sample, a kNN-weighted neighbor distribution `C` (rows of `D` averaged over neighbors) is added inside the softmax as `C @ M`. Loss: KL divergence + horizontal term (Pearson-weighted similarity of each sample to its neighborhood) + vertical term (Pearson-weighted squared pairwise differences across labels) + L2 regularization. Trained by SGD with momentum; the model weights and `M` get alternating gradient steps. ([_ldl_hvlc.py](pyldl/algorithms/_ldl_hvlc.py))

### `RKNN_LDL` *(Wang et al. 2025)*
"Residual kNN-LDL." A 2-layer softmax network (optionally kernelized via RBF on `X`) blended with an `AA_KNN` baseline through a per-sample learnable gate `rho ∈ [0, 1]^n_samples`: the prediction is `rho · D_aaknn + (1 − rho) · model(X)`. A second learnable matrix `Z` captures inter-sample label-correlation residuals over the kNN adjacency graph. Loss: MSE against the target + graph-smoothed correlation regularizer (sparse or dense) + L2. Optimizer: SGD; `rho` and the network weights are updated in alternating steps, and `Z` is updated in closed form after each batch. At prediction, kNN-averaged `rho` gates the weighted blend. ([_rknn_ldl.py](pyldl/algorithms/_rknn_ldl.py))

---

## 6. Ensemble Methods

### `RG4LDL` *(Tan et al. 2025)*
Single-model "ensemble": an RBM trained with simulated annealing learns a sigmoid embedding `H = σ(XW + c)`, and one `SA_BFGS` is fit on `(H, D)`. "RG" = renormalization group, referencing the annealing temperature schedule. ([_ensemble.py:18-45](pyldl/algorithms/_ensemble.py#L18-L45))

### `DF_LDL` *(González et al. 2021b)*
"Decomposition & Fusion." One-vs-one pairwise decomposition: for every label pair `(i, j)`, two `SA_BFGS` models are trained on the subsets where `D[:, i] ≥ D[:, j]` and the complement. Fusion uses an `AA_KNN` as a routing oracle to pick which of the two pairwise experts to query per sample, then averages. ([_ensemble.py:48-89](pyldl/algorithms/_ensemble.py#L48-L89))

### `StructRF` / `StructTree` *(Chen et al. 2018)*
Structured random forest over label distributions. Each `StructTree` recursively KMeans-clusters the label distributions at a node into two groups, then finds the best `(feature, threshold)` split (Cython `best_split`) to separate them; leaves predict the mean training label distribution. `StructRF` bags `n_estimators=20` such trees on `sampling_ratio=0.8` subsets (without replacement) and averages. ([_ensemble.py:92-161](pyldl/algorithms/_ensemble.py#L92-L161))

### `LDLogitBoost` *(Xing et al. 2016)*
Multiclass LogitBoost adapted to LDL. Per round and per class, fits a `DecisionTreeRegressor` to the Newton working response `(D − softmax(F)) / (P(1−P))` weighted by `P(1−P)`, then updates per-class logits `F[:, j]` with a centered shrinkage factor `lr · ((K−1)/K)`. Prediction is softmax of the accumulated logits. ([_ensemble.py:164-201](pyldl/algorithms/_ensemble.py#L164-L201))

### `AdaBoostLDL`
Boosting variant where base `SA_BFGS` models are trained on weighted resamples; sample weights are updated by the per-sample LDL loss (default `sort_loss`). Final prediction averages estimators weighted by their cumulative training loss. ([_ensemble.py:204-237](pyldl/algorithms/_ensemble.py#L204-L237))

---

## 7. LDL Classifiers

Use `predict_proba` for distributions and `predict` for argmax labels.

### `LDL4C` *(Wang & Geng 2019)*
Linear-softmax model fit by L-BFGS on a composite loss: per-sample entropy-weighted MAE + hinge margin between the top-two predicted description degrees (`max(0, 1 − (p_top1 − p_top2) / rho)`) + L2. Forces a clear separation between the most and second-most likely labels. ([_classifier.py:12-37](pyldl/algorithms/_classifier.py#L12-L37))

### `LDL_HR` *(Wang & Geng 2021a)*
"LDL Highest-vs-Rest." L-BFGS on a linear-softmax model with: MAE against a one-hot of the argmax label; hinge margin between the top label and each non-top label; MAE over the rest of the distribution; L2. Emphasizes classification-style correctness while preserving distributional fidelity on the rest. ([_classifier.py:40-75](pyldl/algorithms/_classifier.py#L40-L75))

### `LDLM` *(Wang & Geng 2021b)*
"LDL via Margins." 2-layer softmax model trained by SGD. Three margin terms: (1) prediction margin `|L − D̂|₁ − rho` (forces top-class confidence), (2) label margin `1 − (p_top − p_rest)/rho` (pairwise separation), and (3) a "second" margin penalty tied to the gap between the top-two non-argmax predicted values. L2 on the model. ([_classifier.py:78-135](pyldl/algorithms/_classifier.py#L78-L135))

---

## 8. Incomplete LDL

Fit with a mask matrix indicating observed entries; generate via `pyldl.utils.random_missing`.

### `IncomLDL` *(Xu & Zhou 2017)*
ADMM optimization of the standard softmax-linear LDL objective with: (1) a masked observation term using `qpsolvers.solve_qp` to project onto the simplex per sample, and (2) a singular-value-thresholding step (`svt`) that enforces low-rank structure on the completed prediction. ([_incomplete.py:9-37](pyldl/algorithms/_incomplete.py#L9-L37))

### `WInLDL` *(Li & Chen 2024)*
"Weighted Incomplete LDL." ADMM with weighted imputation — a per-entry weight matrix `Q = Q1 + Q2` grows with iteration count on missing entries and down-weights observed entries by `2^(1−D)`. Linear-system updates rather than QP. ([_incomplete.py:40-67](pyldl/algorithms/_incomplete.py#L40-L67))

---

## 9. Oversampling for LDL

### `SSG_LDL` *(González et al. 2021a)*
"Synthetic Sample Generation for LDL." SMOTE-style oversampler that picks seed samples proportional to a mixed feature+label distance, finds `k` neighbors in concatenated `(X, D)` space, and generates new samples as linear interpolations between seed and neighbor with averaged neighbor label distributions. Call via `fit_transform`. ([_ssg_ldl.py](pyldl/algorithms/_ssg_ldl.py))

---

## 10. Domain Adaptation for LDL

### `LDL_DA` *(Wu, Li, & Jia 2025)*
Neural encoder-decoder for cross-domain LDL. Encoder + softmax decoder trained jointly on source `(sX, sD)` and target `(tX, tD)` with three losses: (1) KL on both domains' predictions; (2) contrastive alignment in the latent space — pairs with similar labels (measured via Jensen-Shannon divergence on `D`) are pulled together, dissimilar pairs pushed apart via cosine similarity or a margin-euclidean variant; (3) prototype alignment — entropy-weighted class prototypes computed via `unsorted_segment_sum` are matched between source, target, and joint sets with MSE. Optional decoder-only fine-tune on the target after pretraining. Provides static helpers `augment` (zero-pad feature concatenation), `reorder_D` (label semantic alignment), and `pairwise_jsd`. ([_ldl_da.py](pyldl/algorithms/_ldl_da.py))

---

## 11. Subtask-Based LDL (`S-LDL` family, Wu, Li, & Jia 2025)

All share a *subtask construction* step: an L-BFGS-optimized `W ∈ [0, 1]^(t × n_outputs)` selects `t` label subsets that (a) emphasize low-average-degree labels and (b) are cosine-diverse; subsets with binarized `W > 0.9` become the tuple of index lists `combi`.

### `Shallow_S_LDL`
Stacks shallow predictors (default `LDSVR`): one per subtask produces a vector over its subset labels; these predictions concatenate with `X` and feed a final `LDSVR` that outputs the full distribution. ([_s_ldl.py:68-97](pyldl/algorithms/_s_ldl.py#L68-L97))

### `S_KLD`, `S_LRR`, `S_SCL`, `S_QFD2`, `S_CJS`
Deep variants via `_DeepSLDL`: an encoder produces a representation `rep`; a subtask decoder produces a latent of size `sum(len(subset))` partitioned by subtask, where each partition is softmaxed and MAE-trained toward the corresponding normalized subset of `D` (weighted by per-sample mass in that subset); a main decoder concatenates `rep + latent` and produces the final distribution. The specialized loss `loss_function` varies per subclass:
- `S_KLD` — plain KL divergence.
- `S_LRR` — KL + `LDL_LRR.ranking_loss` (ranking auxiliary).
- `S_SCL` — KL + `LDL_SCL.scl_loss` (sample-correlation auxiliary with its own learnable `C`, `W`).
- `S_QFD2` — KL replaced by `qfd2` (quadratic-form distance squared).
- `S_CJS` — KL replaced by `cjs` (cumulative Jensen-Shannon). 

Trained by Adam. ([_s_ldl.py:100-221](pyldl/algorithms/_s_ldl.py#L100-L221))

---

## 12. Specialty Deep LDL

### `Delta_LDL` *(Li et al. 2025)*
Deep LDL trained to minimize an integral of `sigmoid(KL(D ‖ D̂) − δ)` over `δ ∈ [0, δ_max]`, computed via adaptive Simpson's rule. `δ_max` is set to the mean KL between each training `D[i]` and the uniform distribution, so the objective smoothly upper-bounds the probability that KL exceeds thresholds across a range — a calibration-aware surrogate loss. Uses the default 3-layer softmax MLP, trained by Adam. ([_delta_ldl.py](pyldl/algorithms/_delta_ldl.py))

### `SNEFY_LDL` *(Zhang et al. 2025)*
"Squared Neural Family LDL." Encoder `3-layer MLP(X) → features ∈ ℝ^n_hidden`; learnable `W ∈ ℝ^(n_outputs × n_hidden)` and `V ∈ ℝ^(n_latent × n_hidden)`. The label distribution is modeled as a *squared neural density* on the simplex via a Dirichlet-like kernel: `D(x | X) ∝ ‖Vᵀ exp(log(D) · W + features + b)‖² / Z(X)`, where `Z(X)` is computed in closed form from products of `Gamma(W + 1)` entries. The loss is the negative log-likelihood of `D` under this density (treating each sample `D[i]` as a point on the simplex). Trained by Adam with `batch_size=64`; `W` is clipped to maintain validity. `predict` computes posterior mean and optionally variance (`return_uncertainty=True`) via closed-form moments of the induced Dirichlet — making this the only algorithm in the library that natively exposes epistemic uncertainty. ([_snefy_ldl.py](pyldl/algorithms/_snefy_ldl.py))

---

## 13. Label Enhancement (LE)

Recovers a label distribution `D` from logical labels `L`.

### `FCM` *(Xu, Liu, & Geng 2019)*
Fuzzy c-means clustering on `X` yields membership matrix `u`; the label-membership association `Lᵀu` is softmaxed via max-product composition to recover `D`. ([_label_enhancement.py:22-35](pyldl/algorithms/_label_enhancement.py#L22-L35))

### `KM` *(Xu, Liu, & Geng 2019)*
Kernel method: per-label RBF-kernel distance from each sample to the positive-label cluster centroid is converted to a score `1 − √(s²/max s²)`, masked by `L`, softmaxed into `D`. ([_label_enhancement.py:38-58](pyldl/algorithms/_label_enhancement.py#L38-L58))

### `LP` *(Xu, Liu, & Geng 2019)*
Label propagation on a normalized Gaussian-similarity graph of `X`: `D ← α P D + (1 − α) L` iterated 500 times, then softmaxed. ([_label_enhancement.py:61-77](pyldl/algorithms/_label_enhancement.py#L61-L77))

### `ML` *(Xu, Liu, & Geng 2019)*
Manifold learning: constructs a locally-linear neighbor reconstruction matrix `W` via `barycenter_kneighbors_graph`; for each label, solves a QP that balances manifold preservation `(I−W)ᵀ(I−W)` against the logical-label constraint. Softmax recovers `D`. ([_label_enhancement.py:80-111](pyldl/algorithms/_label_enhancement.py#L80-L111))

### `GLLE` *(Xu, Liu, & Geng 2019)*
"Graph Laplacian LE." A shallow model trained by L-BFGS on an RBF-kernel representation `P` of `X`. Loss: MSE(L, D) + Laplacian smoothness `tr(Dᵀ G D)` (with `G` from a k-NN graph) + cluster-wise decorrelation penalty enforced by per-cluster learnable matrices `E_i` updated by inner SGD alongside the L-BFGS outer loop. Normalizes `E_i` rows after each inner step. ([_label_enhancement.py:114-191](pyldl/algorithms/_label_enhancement.py#L114-L191))

### `LEVI` *(Xu et al. 2020)*
"LE via Variational Inference." VAE-style model: the encoder `(X, L) → (μ, σ²)` samples a latent `z ~ N(μ, σ²)` whose softmax is the recovered `D`; two decoders reconstruct `X` (MSE) and `L` (BCE). Loss = `‖L − z‖² + α (KL(q ‖ N(0,I)) + rec_X + rec_L)`. Trained by Adam. Uses `tensorflow_probability` for the distributions. ([_label_enhancement.py:194-251](pyldl/algorithms/_label_enhancement.py#L194-L251))

### `LIBLE` *(Zheng, Zhu, & Tang 2023)*
"Latent-Information-Based LE." Four-network encoder-decoder: encoder `X → (μ, σ²)` → latent `h`; decoders reconstruct `L` (MSE), `D` (weighted by a learnable heteroscedastic scale `g`), and the scale itself. Loss = `‖L − L̂‖² + α · KL(q ‖ N(0,I)) + β · (g⁻² · ‖L − D̂‖² + log g²)`, a heteroscedastic negative log-likelihood with KL regularization. Trained by Adam. ([_label_enhancement.py:254-303](pyldl/algorithms/_label_enhancement.py#L254-L303))

### `ConLE` *(Wang et al. 2023)*
"Contrastive LE." Two encoders (`encoder_Z` on `X`, `encoder_Q` on `L`) produce latent codes concatenated and fed to a softmax decoder. Loss combines: (1) a symmetric InfoNCE contrastive term between `Z` and `Q` with temperature `tau`; (2) MSE reconstruction `‖L − D̂‖²`; (3) a threshold margin enforcing that non-L positions in `D̂` stay below L positions by at least `threshold`. Trained by SGD. ([_label_enhancement.py:306-360](pyldl/algorithms/_label_enhancement.py#L306-L360))

---

## 14. Loss Functions (Engineering)

Standalone loss functions from [loss_function_engineering.py](pyldl/algorithms/loss_function_engineering.py) — mix into any `BaseDeep*` subclass by overriding `loss_function`.

- `unimodal_loss` *(Li et al. 2022)* — hinge penalty encouraging the predicted distribution to be unimodal around the scalar target label `y` (for ordered labels).
- `concentrated_loss` *(Li et al. 2022)* — Gaussian NLL of the predicted distribution's mean/variance against the scalar `y`.
- `cad` *(Wen et al. 2023)* — Cumulative Absolute Difference; L1 between cumulative distributions, summed over all prefix lengths.
- `qfd2` *(Wen et al. 2023)* — Quadratic Form Distance squared with a triangular similarity matrix `A_jk = 1 − |j−k|/(n−1)`.
- `cjs` *(Wen et al. 2023)* — Cumulative Jensen-Shannon divergence over all prefix lengths of the distribution.

---

## Notes on Base Classes

All algorithms inherit from one of the base classes in [base.py](pyldl/algorithms/base.py):

- `BaseLDL` — sklearn-style LDL interface (`fit`, `predict`, `score`).
- `BaseDeepLDL` = `BaseLDL` + keras `Model` — adds tf-tensor casting, `.keras` serialization.
- `BaseGD` — manual SGD loop with `GradientTape`; enables index-aware loss signatures (`_loss(X, Y, start, end)`).
- `BaseAdam` — `BaseGD` with Adam default.
- `BaseBFGS` — `tfp.optimizer.lbfgs_minimize` wrapper; requires flattened params and a `_loss(params_1d)` signature.
- `BaseADMM` — primal/dual/error stopping criteria, `_update_W/_update_Z/_update_V` hooks for alternating solves.
- `BaseEnsemble` — holds `_estimator` and `_estimators`; supports iteration/indexing.
- `BaseLDLClassifier` / `BaseDeepLDLClassifier` — `predict_proba` for the distribution, `predict` for argmax labels.
- `BaseIncomLDL` — incomplete-LDL variant with `repair(D, mask)` to handle missing-entry cleanup.

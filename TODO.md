# TODO: Tools for Long-Memory, Non-Stationary, and Self-Similar Processes

A roadmap for extending the toolkit based on theoretical foundations connecting distributed relaxation, heavy tails, and cascade dynamics.

---

## 1. Distributed Relaxation Analysis

The core mechanism: superpositions of exponential decays with heterogeneous rate constants generate power-law tails (via Laplace transform / Tauberian theorems).

### 1.1 Inverse Laplace Transform (Spectral Distribution Recovery)
- [ ] Implement regularized inverse Laplace transform (CONTIN-style algorithm)
- [ ] Tikhonov (L2) regularization with automatic parameter selection
- [ ] L1 regularization option for sparse/sharp spectral features
- [ ] Non-negativity constraints (physical requirement from Bernstein's theorem)
- [ ] Output: estimated g(λ) distribution of relaxation rates

### 1.2 Spectral Distribution Diagnostics
- [ ] Test whether g(λ) ~ λ^(α-1) near λ → 0
- [ ] If yes, predict power-law tail exponent α
- [ ] Confidence intervals on the exponent
- [ ] Visualization: log-log plot of recovered spectrum

### 1.3 Prony Series Decomposition
- [ ] Fit discrete sum of exponentials: f(t) = Σ aᵢ exp(-λᵢ t)
- [ ] Automatic selection of number of terms
- [ ] Extract discrete approximation to spectral distribution

**Key references**: Provencher (1982) CONTIN; Istratov & Vyvenko (1999) Rev. Sci. Instrum.

---

## 2. Power-Law Fitting (Rigorous)

Log-log regression is insufficient. Need proper statistical methodology.

### 2.1 Clauset-Shalizi-Newman Method
- [ ] Maximum likelihood estimation of exponent α (not least-squares)
- [ ] x_min estimation via Kolmogorov-Smirnov statistic
- [ ] Monte Carlo goodness-of-fit p-values
- [ ] Wrapper around `powerlaw` package with consistent API

### 2.2 Alternative Distribution Comparison
- [ ] Likelihood ratio tests against:
  - Exponential
  - Log-normal
  - Stretched exponential
  - Truncated power-law
- [ ] Report which distribution is statistically preferred
- [ ] Visualization: empirical CDF vs fitted alternatives

### 2.3 Discrete Power-Law Support
- [ ] Handle integer-valued data (cascade sizes, event counts)
- [ ] Zeta distribution fitting

**Key reference**: Clauset, Shalizi & Newman (2009) SIAM Review

---

## 3. Stretched Exponential (KWW) Analysis

The Kohlrausch-Williams-Watts function exp(-(t/τ)^β) appears ubiquitously in disordered systems.

### 3.1 KWW Fitting
- [ ] Nonlinear least squares fit for τ and β
- [ ] Confidence intervals
- [ ] Goodness-of-fit diagnostics

### 3.2 Spectral Representation
- [ ] Given β, compute the implied spectral distribution ρ(s; β)
- [ ] Analytical expressions for special cases (β = 1/2, 1/3, 2/3)
- [ ] Numerical computation for general β
- [ ] Verify ρ(s; β) ~ s^(β-1) for small s

### 3.3 Mittag-Leffler Functions
- [ ] Implement E_α(z) and E_α,β(z)
- [ ] Fit Mittag-Leffler relaxation to data
- [ ] Connection to fractional differential equations

**Key references**: Johnston (2006) Phys. Rev. B; Gorenflo et al. (2020) Mittag-Leffler Functions

---

## 4. Detrended Fluctuation Analysis (DFA)

More robust than R/S for non-stationary data with trends.

### 4.1 Standard DFA
- [ ] DFA-1 through DFA-4 (polynomial detrending orders)
- [ ] Scaling exponent estimation
- [ ] Crossover detection (different scaling regimes)
- [ ] Comparison/validation against R/S Hurst estimator

### 4.2 Multifractal DFA (MF-DFA)
- [ ] q-order fluctuation functions F_q(s)
- [ ] Generalized Hurst exponent h(q)
- [ ] Multifractal spectrum f(α) via Legendre transform
- [ ] Singularity spectrum width as heterogeneity measure

### 4.3 DFA Diagnostics
- [ ] Confidence intervals via surrogate data
- [ ] Distinguish true long-memory from trend artifacts

**Key references**: Peng et al. (1994); Kantelhardt et al. (2002) MF-DFA

---

## 5. Waiting Time / Inter-Event Analysis

For event-based data: when do things happen?

### 5.1 Waiting Time Distribution
- [ ] Extract inter-event times from timestamps
- [ ] Fit candidate distributions (exponential, power-law, Weibull, etc.)
- [ ] Statistical comparison using AIC/BIC or likelihood ratios

### 5.2 Burstiness Metrics
- [ ] Burstiness parameter B = (σ - μ)/(σ + μ)
- [ ] Memory coefficient M (correlation of consecutive waiting times)
- [ ] Goh-Barabási burstiness characterization

### 5.3 Temporal Clustering
- [ ] Detect clustering vs regularity vs randomness
- [ ] Allan factor analysis
- [ ] Fano factor across timescales

**Key reference**: Goh & Barabási (2008) EPL

---

## 6. Fractional Differencing

Extend derivative analysis to non-integer orders.

### 6.1 Fractional Difference Operator
- [ ] Implement (1 - L)^d for arbitrary d ∈ ℝ
- [ ] Truncated binomial expansion for practical computation
- [ ] Fast implementation via FFT

### 6.2 ARFIMA Parameter Estimation
- [ ] Estimate d parameter from data
- [ ] Whittle estimator (frequency domain)
- [ ] GPH estimator (log-periodogram regression)
- [ ] Connection to Hurst: H = d + 0.5

### 6.3 Fractional Integration for Simulation
- [ ] Generate fractionally integrated noise
- [ ] Complement existing HK simulation methods

**Key references**: Hosking (1981); Geweke & Porter-Hudak (1983)

---

## 7. Complete Monotonicity Testing

Bernstein's theorem: any completely monotone function is a Laplace transform.

### 7.1 Derivative Sign Testing
- [ ] Estimate derivatives numerically from data
- [ ] Test sign alternation: f ≥ 0, f' ≤ 0, f'' ≥ 0, f''' ≤ 0, ...
- [ ] Statistical test for complete monotonicity

### 7.2 Implications
- [ ] If completely monotone → guaranteed representation as superposition of exponentials
- [ ] If not → something more complex than distributed relaxation

### 7.3 Diagnostic Visualization
- [ ] Plot successive derivatives
- [ ] Highlight sign violations

**Key reference**: Schilling, Song & Vondraček (2012) Bernstein Functions

---

## 8. Cascade Size Distribution Analysis

For systems that produce discrete events/cascades.

### 8.1 Distribution Characterization
- [ ] Histogram with log-binning
- [ ] Empirical CDF / CCDF
- [ ] Power-law vs exponential vs bimodal classification

### 8.2 Bimodality Detection
- [ ] Hartigan's dip test
- [ ] Gaussian mixture model fitting
- [ ] Identify "small cascades" vs "catastrophic cascades" modes

### 8.3 Tail Index Estimation
- [ ] Hill estimator
- [ ] Pickands estimator
- [ ] Moment estimator
- [ ] Comparison and diagnostics

### 8.4 Connection to Theory
- [ ] Given cascade size distribution, infer properties of underlying process
- [ ] Test "nat 20" model: geometric decay with path-dependent amplification

---

## 9. Timescale Separation Diagnostics

For coupled multi-component systems.

### 9.1 Characteristic Timescale Estimation
- [ ] Fit exponential/KWW to individual component relaxation
- [ ] Extract τ for each component
- [ ] Build distribution of timescales across components

### 9.2 Separation Ratio
- [ ] Compute τ_slow / τ_fast for system
- [ ] Classify: coupled (ratio ~ 1) vs separated (ratio >> 1)
- [ ] Identify which components are "frozen" on a given observation timescale

### 9.3 Stratification Analysis
- [ ] Given threshold timescale T, partition components into:
  - Dynamic (τ < T): can respond during observation
  - Frozen (τ > T): effectively static during observation
- [ ] Analyze how partition changes with T

### 9.4 Application to Networks
- [ ] Given network with node-specific timescales, compute effective response
- [ ] Predict aggregate relaxation behavior from component distribution
- [ ] Test Laplace transform prediction: g(λ) shape → f(t) tail

---

## 10. Future Extensions

Lower priority but potentially valuable.

### 10.1 Wavelet-Based Methods
- [ ] Wavelet transform modulus maxima (WTMM) for multifractal analysis
- [ ] Wavelet coherence for coupled processes
- [ ] Scale-dependent correlation analysis

### 10.2 Recurrence Analysis
- [ ] Recurrence plots
- [ ] Recurrence quantification analysis (RQA)
- [ ] Detect determinism vs stochasticity

### 10.3 Information-Theoretic Measures
- [ ] Transfer entropy between coupled processes
- [ ] Mutual information across timescales
- [ ] Complexity measures (permutation entropy, sample entropy)

### 10.4 Extreme Value Theory
- [ ] Block maxima method
- [ ] Peaks over threshold
- [ ] GEV / GPD fitting
- [ ] Return level estimation

---

## Implementation Notes

### Language Strategy
- **Python**: Numerical heavy-lifting, integration with scientific stack
- **R**: Statistical testing, time series econometrics, visualization

### Testing Philosophy
- Synthetic data with known properties for validation
- Compare multiple estimators on same data
- Publish diagnostic plots, not just point estimates

### Documentation
- Each tool should explain the theory briefly
- Link to key references
- Provide interpretation guidance (what does this number mean?)

---

## References

### Foundational
- Feller (1971) *An Introduction to Probability Theory* Vol. II
- Bingham, Goldie & Teugels (1987) *Regular Variation*
- Schilling, Song & Vondraček (2012) *Bernstein Functions*

### Methods
- Clauset, Shalizi & Newman (2009) Power-law distributions in empirical data. *SIAM Review*
- Provencher (1982) CONTIN. *Comput. Phys. Commun.*
- Kantelhardt et al. (2002) Multifractal DFA. *Physica A*

### Applications
- Metzler & Klafter (2000) The random walk's guide to anomalous diffusion. *Phys. Rep.*
- Dutta & Horn (1981) Low-frequency fluctuations in solids: 1/f noise. *Rev. Mod. Phys.*

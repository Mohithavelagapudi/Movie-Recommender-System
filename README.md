# 🎬 **Movie Recommendation System using Matrix Factorization (Truncated SVD)**  

> _A mathematically grounded exploration of latent factor modeling for personalized movie recommendations._

---

## 🧩 **1. Abstract**

Recommender systems play a crucial role in **mitigating information overload** by inferring user preferences from historical interactions.  
This project develops a **Movie Recommendation Engine** using **Matrix Factorization**, specifically **Truncated Singular Value Decomposition (SVD)**, to uncover hidden structures in the user–item rating matrix.

By decomposing the rating matrix into **orthogonal user and item feature spaces**, we extract **latent dimensions** representing taste patterns such as genre affinity and stylistic preferences.  
The system balances expressiveness and generalization by retaining only the **top-k singular values**, yielding a **low-rank approximation** that preserves the majority of spectral energy.

We minimize **Root Mean Squared Error (RMSE)** across observed ratings and compare:
- 📉 **Closed-form SVD**
- ⚙️ **Iterative Optimization** (Gradient Descent, Alternating Least Squares)

<p align="center">
  <img src="image (25).png" alt="Matrix factorization" width="1000"/>
</p>

> ✨ This work emphasizes both **mathematical rigor** (orthogonality, rank reduction, spectral energy) and **practical considerations** in evaluation and deployment.

---

## 🎯 **2. Problem Statement**

Given a **sparse user–movie rating matrix**, the goal is to **predict unobserved ratings** to enable personalized recommendations.

### ⚠️ Key Challenges
| Challenge | Description |
|------------|-------------|
| **Sparsity** | Most users rate only a few movies. |
| **Scalability** | The dataset can involve thousands of users and items. |
| **Overfitting** | High-rank reconstructions memorize noise instead of learning meaningful patterns. |

### 🧠 **Objective**
Learn **latent representations** capturing intrinsic taste dimensions through **linear algebraic factorization**.

---

## 🧮 **3. Mathematical Foundations**

Let the **user–item rating matrix** be:

$$
R \in \mathbb{R}^{m \times n}
$$

where missing entries are unobserved.

---
### 🧱 **Full Singular Value Decomposition (SVD)**

The full SVD factorizes the rating matrix as:

$$
R = U \Sigma V^{\top}
$$

where  


- **U** ∈ ℝ<sup>m×m</sup>: Orthonormal user singular vectors  
- **Σ** ∈ ℝ<sup>m×n</sup>: Diagonal matrix of singular values σ₁ ≥ σ₂ ≥ … ≥ σᵣ  
- **V** ∈ ℝ<sup>n×n</sup>: Orthonormal item singular vectors
  
---

### 🔹 **Truncated (Rank-k) Approximation**

To reduce dimensionality, we keep only the top **k singular values**:

$$
R_k = U_k \Sigma_k V_k^{T}
$$

The rank-k approximation minimizes the Frobenius norm error:

$$
|| R - R_k ||_F = \min_{rank(X)=k} || R - X ||_F
$$

---

### 🧾 **Energy Retention**

Energy retained after truncation is given by:

$$
\text{Energy}(k) = 
\frac{\sum_{i=1}^{k} \sigma_i^2}
{\sum_{i=1}^{r} \sigma_i^2}
$$

> This quantifies how much of the total variance (spectral energy) is preserved in the top-k components.

---

### 💡 **Latent Feature Representation**

Latent user and item features are constructed as:

$$
\begin{aligned}
P &= U_k \Sigma_k^{1/2} \quad \text{(User feature matrix)} \\
Q &= \Sigma_k^{1/2} V_k^{\top} \quad \text{(Item feature matrix)}
\end{aligned}
$$

The predicted rating for user \( u \) and item \( i \) is:

$$
\hat{R}_{u,i} = P_{u,:} \cdot Q_{:,i}
$$

<p align="center">
  <img src="image (24).png" alt="Matrix factorization" width="1000"/>
</p>

---

## 🔁 **4. Alternative Optimization (Implicit Factorization)**

Instead of direct SVD, latent vectors can be learned by minimizing the reconstruction loss:

$$
L = 
\sum_{(u,i)\in\Omega} (R_{u,i} - p_u \cdot q_i)^2 
+ \lambda (||p_u||^2 + ||q_i||^2)
$$

where:
- Ω = set of observed user–item pairs  
- λ = regularization parameter  

---

### ⚙️ **Gradient Descent Updates**

$$
\begin{aligned}
p_u &\leftarrow p_u + \eta (e_{u,i} q_i - \lambda p_u) \\
q_i &\leftarrow q_i + \eta (e_{u,i} p_u - \lambda q_i)
\end{aligned}
$$

with error term:

$$
e_{u,i} = R_{u,i} - p_u \cdot q_i
$$

---

### 🔄 **Alternating Least Squares (ALS)**

For fixed item matrix \( Q \), optimize user factors \( P \) via **ridge regression**, and vice versa:

1. Fix \( Q \), solve for \( P \)  
2. Fix \( P \), solve for \( Q \)  

Repeat until convergence.

---

## 🧪 **5. Evaluation Metric**

Model performance is measured using **Root Mean Squared Error (RMSE):**

$$
\text{RMSE} = 
\sqrt{
\frac{1}{|\Omega|}
\sum_{(u,i)\in\Omega}
\left( R_{u,i} - \hat{R}_{u,i} \right)^2
}
$$

---
## 🧭 **6. Key Insights**

| 💡 **Concept** | 🧮 **Description** |
|----------------|--------------------|
| **Orthogonality** | Ensures that latent features (columns of **U** and **V**) are **orthogonal**, meaning **UᵀU = I** and **VᵀV = I**. This guarantees that each latent dimension captures unique, uncorrelated information. |
| **Rank Reduction** | By using a **truncated rank-k approximation** **Rₖ = UₖΣₖVₖᵀ**, the model retains only the most significant singular values—reducing noise and preventing overfitting. |
| **Spectral Energy Concentration** | Measures how much of the total variance (energy) is captured by the top-k components:<br><br> $$ Energy(k) = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2} $$ <br>Higher energy retention indicates stronger representation of dominant behavioral patterns. |


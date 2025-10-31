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

Let  

\[
R \in \mathbb{R}^{m \times n}
\]  

represent the **user–item rating matrix**, where missing entries are unobserved.

---

### 🧱 **Full Singular Value Decomposition (SVD)**

\[
R = U \, \Sigma \, V^{\top}
\]

Where:

- \( U \in \mathbb{R}^{m \times m} \): Orthonormal **user singular vectors**  
- \( \Sigma \in \mathbb{R}^{m \times n} \): Diagonal matrix of **singular values** \( \sigma_1 \geq \sigma_2 \geq ... \geq \sigma_r \)  
- \( V \in \mathbb{R}^{n \times n} \): Orthonormal **item singular vectors**

---

### 🔹 **Truncated (Rank-k) Approximation**

\[
R_k = U_k \, \Sigma_k \, V_k^{\top} = \underset{\text{rank}(X)=k}{\operatorname{argmin}} \| R - X \|_F
\]

Only the **top-k singular values** are retained to reduce dimensionality.

#### 🧾 **Energy Retention**

\[
\text{Energy}(k) = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2}
\]

> This ensures that we capture the most informative components of user–item interactions.

---

### 💡 **Latent Feature Representation**

\[
P = U_k \, \Sigma_k^{1/2} \quad \text{(User feature matrix)}
\]  
\[
Q = \Sigma_k^{1/2} \, V_k^{\top} \quad \text{(Item feature matrix)}
\]

Predicted rating for user \( u \) and item \( i \):

\[
\hat{R}_{u,i} = P_{u,:} \cdot Q_{:,i}
\]

---

## 🔁 **4. Alternative Optimization (Implicit Factorization)**

Instead of computing SVD directly, we can **learn the latent factors** by minimizing reconstruction loss:

\[
L = \sum_{(u,i)\in\Omega} (R_{u,i} - p_u \cdot q_i)^2 + \lambda (\|p_u\|^2 + \|q_i\|^2)
\]

where:
- \( \Omega \): Set of observed ratings  
- \( \lambda \): Regularization parameter  

---

### ⚙️ **Gradient Descent Updates**

\[
p_u \leftarrow p_u + \eta (e_{u,i} q_i - \lambda p_u)
\]  
\[
q_i \leftarrow q_i + \eta (e_{u,i} p_u - \lambda q_i)
\]  

where:

\[
e_{u,i} = R_{u,i} - p_u \cdot q_i
\]

---

### 🔄 **Alternating Least Squares (ALS)**

For fixed \( Q \), optimize each \( p_u \) via **ridge regression**, then alternate:

1. Fix \( Q \), solve for \( P \)  
2. Fix \( P \), solve for \( Q \)  

Repeat until convergence.

---

## 🧪 **5. Evaluation Metric**

Model performance is assessed using **Root Mean Squared Error (RMSE):**

\[
\text{RMSE} = \sqrt{ \frac{1}{|\Omega|} \sum_{(u,i)\in\Omega} (R_{u,i} - \hat{R}_{u,i})^2 }
\]

---

## 🧰 **6. How to Run**

### 📂 **Setup**

1. **Acquire Dataset** (e.g., [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/))  
2. Place the file `ratings.csv` in the project root directory.

### 🛠️ **Install Dependencies**

```bash
pip install numpy pandas scipy scikit-learn
```
## 🧭 **7. Key Insights**

| 💡 **Concept** | 🧮 **Description** |
|----------------|--------------------|
| **Orthogonality** | Ensures that latent features (columns of \( U \) and \( V \)) are **orthogonal**, meaning \( U^{\top}U = I \) and \( V^{\top}V = I \). This guarantees that each latent dimension captures unique, uncorrelated information. |
| **Rank Reduction** | By using a **truncated rank-k approximation** \( R_k = U_k \Sigma_k V_k^{\top} \), the model retains only the most significant singular values—reducing noise and preventing overfitting. |
| **Spectral Energy Concentration** | Measures how much of the total variance (energy) is captured by the top-k components:  \[ \text{Energy}(k) = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2} \]  Higher energy retention indicates stronger representation of dominant behavioral patterns. |

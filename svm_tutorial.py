"""
=============================================================
 SVM TUTORIAL — Linear & Non-Linear + Conditional Probability
 Dataset: Student Exam Performance (Custom / Unique)
=============================================================
TOPICS COVERED:
  1. Dataset Creation (unique)
  2. Preprocessing  (scaling, train/test split)
  3. Linear SVM     (kernel='linear')
  4. Non-Linear SVM (kernel='rbf', 'poly')
  5. Conditional Probability via Platt Scaling
  6. Decision Boundary Visualisation
  7. Evaluation Metrics
"""

# pip install numpy pandas scikit-learn matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1. CREATE UNIQUE DATASET — "Student Exam Performance"
#    Features:
#      study_hours  – avg hours studied per day
#      sleep_hours  – avg sleep per night
#      prev_score   – score in previous exam (%)
#      attendance   – attendance percentage
#      assignments  – assignments submitted (out of 20)
#    Target:
#      grade → 1=Pass / 0=Fail
# ─────────────────────────────────────────────────────────────
np.random.seed(42)
N = 300

study_hours = np.random.uniform(1, 10, N)
sleep_hours = np.random.uniform(4,  9, N)
prev_score  = np.random.uniform(30, 100, N)
attendance  = np.random.uniform(40, 100, N)
assignments = np.random.randint(5, 21, N)

pass_prob = (
    0.25 * (study_hours / 10) +
    0.15 * (sleep_hours /  9) +
    0.30 * (prev_score  / 100) +
    0.20 * (attendance  / 100) +
    0.10 * (assignments / 20)
)
pass_prob += np.random.normal(0, 0.08, N)
pass_prob  = np.clip(pass_prob, 0, 1)
grade      = (pass_prob >= 0.50).astype(int)

df = pd.DataFrame({
    "study_hours": np.round(study_hours, 2),
    "sleep_hours": np.round(sleep_hours, 2),
    "prev_score" : np.round(prev_score,  2),
    "attendance" : np.round(attendance,  2),
    "assignments": assignments,
    "grade"      : grade
})
df["grade_label"] = df["grade"].map({1: "Pass", 0: "Fail"})

print("=" * 60)
print("  STUDENT EXAM PERFORMANCE DATASET")
print("=" * 60)
print(df.head(10).to_string(index=False))
print(f"\nShape  : {df.shape}")
print(f"\nClass Distribution:\n{df['grade_label'].value_counts()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nBasic Statistics:")
print(df.describe().round(2))

# ─────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PREPROCESSING")
print("=" * 60)

feature_cols = ["study_hours", "sleep_hours", "prev_score",
                "attendance", "assignments"]
X = df[feature_cols].values
y = df["grade"].values

# 2a. Train-Test Split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain size : {X_train.shape[0]}")
print(f"Test  size : {X_test.shape[0]}")

# 2b. StandardScaler → mean=0, std=1
#     z = (x - μ) / σ
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("\nBefore Scaling (row 0):", X_train[0])
print("After  Scaling (row 0):", np.round(X_train_sc[0], 4))
print("Mean  :", np.round(scaler.mean_,  2))
print("Std   :", np.round(scaler.scale_, 2))

# ─────────────────────────────────────────────────────────────
# 3. LINEAR SVM
#    Decision boundary : w·x + b = 0
#    Margin            : M = 2 / ‖w‖
#    Objective         : min ½‖w‖² + C·Σξᵢ
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  LINEAR SVM  (kernel='linear', C=1)")
print("=" * 60)

svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train_sc, y_train)
y_pred_linear = svm_linear.predict(X_test_sc)

print(f"\nAccuracy : {accuracy_score(y_test, y_pred_linear):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_linear, target_names=["Fail","Pass"]))
print(f"Support Vectors : {svm_linear.n_support_.sum()}  "
      f"(Fail={svm_linear.n_support_[0]}, Pass={svm_linear.n_support_[1]})")

# ─────────────────────────────────────────────────────────────
# 4. NON-LINEAR SVM
#    RBF  kernel : K(xᵢ,xⱼ) = exp(−γ‖xᵢ−xⱼ‖²)
#    Poly kernel : K(xᵢ,xⱼ) = (γ xᵢ·xⱼ + r)^d
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  NON-LINEAR SVM — RBF Kernel  (C=10, γ=scale)")
print("=" * 60)

svm_rbf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train_sc, y_train)
y_pred_rbf = svm_rbf.predict(X_test_sc)

print(f"\nAccuracy : {accuracy_score(y_test, y_pred_rbf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rbf, target_names=["Fail","Pass"]))
print(f"Support Vectors : {svm_rbf.n_support_.sum()}")

print("\n" + "─" * 40)
print("  NON-LINEAR SVM — Polynomial Kernel  (degree=3, C=5)")
print("─" * 40)

svm_poly = SVC(kernel='poly', degree=3, C=5.0, random_state=42)
svm_poly.fit(X_train_sc, y_train)
y_pred_poly = svm_poly.predict(X_test_sc)

print(f"\nAccuracy : {accuracy_score(y_test, y_pred_poly):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_poly, target_names=["Fail","Pass"]))

# ─────────────────────────────────────────────────────────────
# 5. CONDITIONAL PROBABILITY — Platt Scaling
#
#    P(y=+1 | x) =         1
#                    ─────────────────────
#                    1 + exp(A·f(x) + B)
#
#    f(x) = SVM decision function
#    A, B = parameters fitted via maximum likelihood
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CONDITIONAL PROBABILITY — Platt Scaling")
print("=" * 60)
print("""
  Formula:
       P(y = Pass | x) =          1
                           ─────────────────────
                           1 + exp( A·f(x) + B )

  f(x) = SVM decision score  (raw, before thresholding)
  A, B = calibration parameters (learned by MLE on held-out data)
""")

svm_prob = SVC(kernel='rbf', C=10.0, gamma='scale',
               probability=True, random_state=42)
svm_prob.fit(X_train_sc, y_train)
proba = svm_prob.predict_proba(X_test_sc)

print("  Sample | P(Fail) | P(Pass) | Predicted | Actual")
print("  " + "─" * 52)
for i in range(10):
    pred   = "Pass" if svm_prob.predict(X_test_sc[[i]])[0] == 1 else "Fail"
    actual = "Pass" if y_test[i] == 1 else "Fail"
    print(f"  {i+1:^6} | {proba[i,0]:.4f}  | {proba[i,1]:.4f}  |"
          f"  {pred:^9}| {actual}")

# ─────────────────────────────────────────────────────────────
# 6. HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  GRID SEARCH — Best Hyperparameters (RBF)")
print("=" * 60)

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1]}
grid = GridSearchCV(SVC(kernel='rbf', random_state=42),
                    param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_sc, y_train)

print(f"\nBest Parameters : {grid.best_params_}")
print(f"Best CV Accuracy: {grid.best_score_:.4f}")
y_pred_best = grid.best_estimator_.predict(X_test_sc)
print(f"Test Accuracy   : {accuracy_score(y_test, y_pred_best):.4f}")

# ─────────────────────────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────────────────────────

def plot_boundary(model, X2, y, title, ax):
    h = 0.05
    x0_min, x0_max = X2[:,0].min()-1, X2[:,0].max()+1
    x1_min, x1_max = X2[:,1].min()-1, X2[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x0_min, x0_max, h),
                         np.arange(x1_min, x1_max, h))
    grid_pts = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_pts).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap='coolwarm')
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5)
    ax.scatter(X2[:,0], X2[:,1], c=y, cmap='coolwarm',
               edgecolors='k', s=40, linewidths=0.5)
    ax.set_xlabel("study_hours (scaled)")
    ax.set_ylabel("sleep_hours (scaled)")
    ax.set_title(title)

X2_train = X_train_sc[:, :2]
X2_test  = X_test_sc[:, :2]

m_lin  = SVC(kernel='linear', C=1.0,  random_state=42).fit(X2_train, y_train)
m_rbf  = SVC(kernel='rbf',    C=10.0, gamma='scale',  random_state=42).fit(X2_train, y_train)
m_poly = SVC(kernel='poly',   C=5.0,  degree=3,       random_state=42).fit(X2_train, y_train)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("SVM Decision Boundaries (study_hours vs sleep_hours)",
             fontsize=14, fontweight='bold')
plot_boundary(m_lin,  X2_test, y_test, "Linear SVM (C=1)",          axes[0])
plot_boundary(m_rbf,  X2_test, y_test, "RBF SVM (C=10, γ=scale)",   axes[1])
plot_boundary(m_poly, X2_test, y_test, "Poly SVM (C=5, degree=3)",  axes[2])

p0 = mpatches.Patch(color='blue', alpha=0.5, label='Fail (0)')
p1 = mpatches.Patch(color='red',  alpha=0.5, label='Pass (1)')
fig.legend(handles=[p0, p1], loc='lower center', ncol=2, fontsize=11)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("svm_decision_boundaries.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: svm_decision_boundaries.png")

# Confusion matrices
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
fig2.suptitle("Confusion Matrices", fontsize=14, fontweight='bold')
for ax, y_pred, title in zip(
        axes2,
        [y_pred_linear, y_pred_rbf, y_pred_poly],
        ["Linear SVM", "RBF SVM", "Poly SVM"]):
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                           display_labels=["Fail","Pass"]).plot(
        ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title)
plt.tight_layout()
plt.savefig("svm_confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: svm_confusion_matrices.png")

# Probability distribution
fig3, ax3 = plt.subplots(figsize=(8, 4))
proba_pass = svm_prob.predict_proba(X_test_sc)[:, 1]
ax3.hist([proba_pass[y_test==0], proba_pass[y_test==1]],
         bins=20, label=["Fail (actual)", "Pass (actual)"],
         color=["tomato", "steelblue"], alpha=0.75, edgecolor='k')
ax3.axvline(0.5, color='k', linestyle='--', label='Threshold (0.5)')
ax3.set_xlabel("P(Pass | x)  — Conditional Probability")
ax3.set_ylabel("Count")
ax3.set_title("Platt-Scaled Conditional Probability Distribution")
ax3.legend()
plt.tight_layout()
plt.savefig("svm_probability_dist.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: svm_probability_dist.png")

# Feature importance (linear SVM weights)
fig4, ax4 = plt.subplots(figsize=(8, 4))
coef = np.abs(svm_linear.coef_[0])
colors_b = ['#4361ee','#3a0ca3','#7209b7','#f72585','#4cc9f0']
bars = ax4.barh(feature_cols, coef, color=colors_b, edgecolor='k')
ax4.set_xlabel("|Weight|  (Linear SVM)")
ax4.set_title("Feature Importance via Linear SVM Coefficients")
for bar, val in zip(bars, coef):
    ax4.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig("svm_feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: svm_feature_importance.png")

# ─────────────────────────────────────────────────────────────
# 8. SUMMARY TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MODEL COMPARISON SUMMARY")
print("=" * 60)
summary = pd.DataFrame({
    "Model"   : ["Linear SVM", "RBF SVM", "Poly SVM (d=3)", "Best Grid SVM"],
    "Kernel"  : ["linear",     "rbf",     "poly",            "rbf (tuned)"],
    "Accuracy": [accuracy_score(y_test, y_pred_linear),
                 accuracy_score(y_test, y_pred_rbf),
                 accuracy_score(y_test, y_pred_poly),
                 accuracy_score(y_test, y_pred_best)]
})
summary["Accuracy"] = summary["Accuracy"].map("{:.4f}".format)
print(summary.to_string(index=False))

print("""
═══════════════════════════════════════════════════════════
  KEY FORMULAS RECAP
═══════════════════════════════════════════════════════════
  Linear SVM:
      Decision  : f(x) = w·x + b
      Classify  : +1 if f(x)≥0, else −1
      Margin    : M = 2 / ‖w‖
      Objective : min ½‖w‖² + C·Σξᵢ
                  s.t. yᵢ(w·xᵢ+b) ≥ 1 − ξᵢ,  ξᵢ ≥ 0

  RBF Kernel (Non-Linear):
      K(xᵢ,xⱼ) = exp(−γ · ‖xᵢ−xⱼ‖²)
      γ = 1/(n_features · Var(X))  when gamma='scale'

  Polynomial Kernel (Non-Linear):
      K(xᵢ,xⱼ) = (γ · xᵢᵀxⱼ + r)^d

  Conditional Probability (Platt Scaling):
      P(y=+1|x) = 1 / (1 + exp(A·f(x) + B))
      → A, B fitted by MLE on validation fold
═══════════════════════════════════════════════════════════
""")

print("ALL DONE!")

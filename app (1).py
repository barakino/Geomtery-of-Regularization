import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -- Application Setup --
st.set_page_config(page_title="Regularization Geometry", layout="wide")
st.title("Ridge vs. Lasso Regularization Geometry")
st.markdown("""
This application demonstrates the geometric equivalence of penalized regression. 
Adjust the **Budget (t)** and the unconstrained **OLS Estimates** to see how the shape of the constraint boundary dictates the optimal solution.
""")

# -- Sidebar Controls --
st.sidebar.header("Parameters")
reg_type = st.sidebar.radio("Regularization Type", ("Lasso (L1 Norm)", "Ridge (L2 Norm)"))
t = st.sidebar.slider("Constraint Budget (t)", min_value=0.5, max_value=6.0, value=2.5, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Unconstrained OLS Center**")
beta1_hat = st.sidebar.slider("β1 Estimate", min_value=0.0, max_value=6.0, value=3.5, step=0.1)
beta2_hat = st.sidebar.slider("β2 Estimate", min_value=0.0, max_value=6.0, value=2.5, step=0.1)

# -- Mathematical Helper Functions --
# Assuming spherical loss (orthogonal features X^T X = I) for clear textbook visualization
def get_ridge_opt(b1_hat, b2_hat, t):
    """Calculates the projection onto the L2 ball (Circle)"""
    norm = np.sqrt(b1_hat**2 + b2_hat**2)
    if norm <= t:
        return b1_hat, b2_hat
    return b1_hat * t / norm, b2_hat * t / norm

def get_lasso_opt(b1_hat, b2_hat, t):
    """Calculates the projection onto the L1 ball (Diamond) via soft thresholding"""
    # Work with absolute values for symmetry, restore signs later
    s1, s2 = np.sign(b1_hat), np.sign(b2_hat)
    b1_abs, b2_abs = abs(b1_hat), abs(b2_hat)

    if b1_abs + b2_abs <= t:
        return b1_hat, b2_hat

    # Project onto the line x + y = t
    delta = (b1_abs + b2_abs - t) / 2
    opt1_abs = b1_abs - delta
    opt2_abs = b2_abs - delta

    # If projection falls outside the quadrant, snap exactly to the corner (Sparsity)
    if opt1_abs < 0:
        opt1_abs = 0
        opt2_abs = t
    elif opt2_abs < 0:
        opt1_abs = t
        opt2_abs = 0

    return s1 * opt1_abs, s2 * opt2_abs

# -- Calculate Optimal Point --
if reg_type == "Ridge (L2 Norm)":
    opt_b1, opt_b2 = get_ridge_opt(beta1_hat, beta2_hat, t)
else:
    opt_b1, opt_b2 = get_lasso_opt(beta1_hat, beta2_hat, t)

optimal_loss = (opt_b1 - beta1_hat)**2 + (opt_b2 - beta2_hat)**2

# -- Plotting --
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-7, 7)
ax.set_ylim(-7, 7)
ax.axhline(0, color='black', linewidth=1.5)
ax.axvline(0, color='black', linewidth=1.5)
ax.set_xlabel(r"$\beta_1$", fontsize=16)
ax.set_ylabel(r"$\beta_2$", fontsize=16)

# 1. Draw Constraint Region
if reg_type == "Ridge (L2 Norm)":
    constraint = plt.Circle((0, 0), t, color='#4A90E2', alpha=0.2, label='Ridge Constraint ($L_2$)')
    ax.add_patch(constraint)
    ax.plot(0, 0, color='#4A90E2', label='_nolegend_') # Dummy for legend styling
else:
    diamond = plt.Polygon([[t, 0], [0, t], [-t, 0], [0, -t]], color='#4A90E2', alpha=0.2, label='Lasso Constraint ($L_1$)')
    ax.add_patch(diamond)

# 2. Draw Loss Contours
b1, b2 = np.meshgrid(np.linspace(-7, 7, 400), np.linspace(-7, 7, 400))
Z = (b1 - beta1_hat)**2 + (b2 - beta2_hat)**2

if optimal_loss > 0:
    # Draw the specific contour that perfectly touches the constraint
    ax.contour(b1, b2, Z, levels=[optimal_loss], colors='#D0021B', linestyles='dashed', linewidths=2)
    # Draw outer contours for depth
    ax.contour(b1, b2, Z, levels=[optimal_loss + 3, optimal_loss + 9, optimal_loss + 18], colors='gray', alpha=0.4)
else:
    ax.contour(b1, b2, Z, levels=[2, 6, 12, 20], colors='gray', alpha=0.4)

# 3. Plot Points
ax.plot(beta1_hat, beta2_hat, 'ko', markersize=8, label='Unconstrained OLS ($\hat{\beta}_{LS}$)')
ax.plot(opt_b1, opt_b2, 'ro', markersize=8, label='Optimal Penalized $\hat{\beta}$')

# Draw dashed line connecting origin to OLS to show trajectory
ax.plot([0, beta1_hat], [0, beta2_hat], 'k--', alpha=0.3)

ax.legend(loc='upper left', fontsize=12)
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.7)

# -- Layout Rendering --
col1, col2 = st.columns([2, 1])

with col1:
    st.pyplot(fig)

with col2:
    st.subheader("Results")
    st.markdown(f"**Optimal $\beta_1$:** `{opt_b1:.3f}`")
    st.markdown(f"**Optimal $\beta_2$:** `{opt_b2:.3f}`")
    
    st.markdown("---")
    
    # Check for sparsity
    is_sparse = (np.isclose(opt_b1, 0) or np.isclose(opt_b2, 0)) and optimal_loss > 0
    
    if reg_type == "Lasso (L1 Norm)" and is_sparse:
        st.success("🎯 **Sparsity Achieved!**\n\nThe expanding loss contour has intercepted the sharp corner of the Lasso diamond. One of the coefficients has been driven exactly to zero, performing automatic feature selection.")
    elif reg_type == "Ridge (L2 Norm)":
        st.info("🔄 **Proportional Shrinkage**\n\nThe smooth circular boundary of Ridge forces the coefficients to shrink proportionally toward zero, but they rarely hit exactly zero.")
    else:
        st.warning("Increase the OLS center or decrease the budget to see the constraints take effect!")

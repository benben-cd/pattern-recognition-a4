"""
Pattern Recognition and Image Processing - Assignment A4
Student ID: 2025213362
Name: Min Yizhou

This application implements the following functions:
1. Least Squares Linear Regression
2. KNN Classifier (with real CIFAR-10 image data support)
3. CIFAR Linear Classifier Template Visualization
4. KNN Visualization with Different K Values
5. SGD vs Momentum Gradient Descent
6. Loss Function Calculation Demo

Agent Used: Kimi Code CLI (Moonshot AI)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# Force English fonts only - no Chinese font dependencies
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

from matplotlib.patches import FancyArrowPatch, Patch
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from PIL import Image
import warnings
import pickle
import urllib.request
import tarfile
from collections import Counter
from matplotlib.colors import ListedColormap
warnings.filterwarnings('ignore')

# ==================== CIFAR-10 Data Management ====================

CIFAR10_CLASSES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
                   'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

@st.cache_data
def download_and_load_cifar10():
    """Download and load CIFAR-10 dataset"""
    data_dir = './cifar10_data'
    cifar_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    
    if not os.path.exists(cifar_dir):
        return None, None, None
    
    try:
        # Load training data
        X_train = []
        y_train = []
        
        for i in range(1, 6):
            filepath = os.path.join(cifar_dir, f'data_batch_{i}')
            with open(filepath, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                X_train.append(batch[b'data'])
                y_train.extend(batch[b'labels'])
        
        X_train = np.vstack(X_train)
        y_train = np.array(y_train)
        
        # Load test data
        test_filepath = os.path.join(cifar_dir, 'test_batch')
        with open(test_filepath, 'rb') as f:
            test_batch = pickle.load(f, encoding='bytes')
            X_test = test_batch[b'data']
            y_test = np.array(test_batch[b'labels'])
        
        # Reshape to (N, 32, 32, 3)
        X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        # Subsample
        np.random.seed(42)
        train_idx = np.random.choice(len(X_train), 2000, replace=False)
        test_idx = np.random.choice(len(X_test), 500, replace=False)
        
        X_train = X_train[train_idx]
        y_train = y_train[train_idx]
        X_test = X_test[test_idx]
        y_test = y_test[test_idx]
        
        return (X_train, y_train), (X_test, y_test), CIFAR10_CLASSES
    except Exception as e:
        return None, None, None

def download_cifar10_data():
    """Download CIFAR-10 dataset"""
    data_dir = './cifar10_data'
    os.makedirs(data_dir, exist_ok=True)
    
    base_url = "https://www.cs.toronto.edu/~kriz/"
    filename = "cifar-10-python.tar.gz"
    filepath = os.path.join(data_dir, filename)
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            progress_bar.progress(int(percent))
            status_text.text(f"Download Progress: {percent:.1f}%")
        
        status_text.text("Downloading CIFAR-10 dataset (~170MB)...")
        urllib.request.urlretrieve(base_url + filename, filepath, reporthook=download_progress)
        
        status_text.text("Extracting...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(data_dir)
        
        os.remove(filepath)
        progress_bar.empty()
        status_text.empty()
        return True
    except Exception as e:
        st.error(f"Download failed: {e}")
        return False

def check_cifar10_status():
    """Check CIFAR-10 data status"""
    data_dir = './cifar10_data/cifar-10-batches-py'
    if os.path.exists(data_dir):
        return download_and_load_cifar10()
    return None, None, None

# ==================== Page Configuration ====================

st.set_page_config(
    page_title="Pattern Recognition A4 - 2025213362",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Helper Functions ====================

def generate_linear_classifier_weights(n_classes=10):
    """Generate simulated linear classifier weight templates"""
    np.random.seed(42)
    weights = np.zeros((n_classes, 32, 32, 3))
    
    for c in range(n_classes):
        if c == 0:  # Airplane
            weights[c, :, :, 2] = 0.6
            weights[c, 10:22, 8:24, :] = [0.9, 0.9, 0.95]
            weights[c, 14:18, 4:28, 2] = 0.8
        elif c == 1:  # Automobile
            weights[c, 20:32, :, 1] = 0.3
            weights[c, 12:22, 8:24, 0] = 0.8
        elif c == 2:  # Bird
            weights[c, :, :, 2] = 0.5
            weights[c, 12:20, 12:20, 0] = 0.9
        elif c == 3:  # Cat
            weights[c, :, :, 0] = 0.4
            weights[c, 10:26, 10:22, 0] = 0.9
        elif c == 4:  # Deer
            weights[c, :, :, 1] = 0.4
            weights[c, 12:26, 10:22, 0] = 0.6
        elif c == 5:  # Dog
            weights[c, 22:32, :, 1] = 0.4
            weights[c, 12:24, 10:24, 0] = 0.7
        elif c == 6:  # Frog
            weights[c, :, :, 1] = 0.4
            weights[c, 14:24, 10:22, 1] = 0.9
        elif c == 7:  # Horse
            weights[c, 22:32, :, 1] = 0.35
            weights[c, 10:26, 10:22, 0] = 0.6
        elif c == 8:  # Ship
            weights[c, 18:32, :, 2] = 0.7
            weights[c, 12:20, 10:22, :] = [0.9, 0.9, 0.9]
        elif c == 9:  # Truck
            weights[c, 22:32, :, :] = 0.3
            weights[c, 10:24, 6:26, :] = 0.6
    
    weights += np.random.randn(*weights.shape) * 0.05
    weights = np.clip(weights, 0, 1)
    
    return weights, CIFAR10_CLASSES

# ==================== Module 1: Least Squares ====================

def least_squares_module():
    st.markdown('<h2 style="color: #ff7f0e;">📈 Least Squares Linear Regression</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">
        <b>Least Squares Principle:</b><br>
        Find the best-fit line that minimizes the sum of squared residuals.
        </div>
        """, unsafe_allow_html=True)
        
        n_samples = st.slider("Sample Count", 10, 200, 50, key="ls_samples")
        noise_level = st.slider("Noise Level", 0.0, 5.0, 1.0, 0.1, key="ls_noise")
        regularization = st.slider("L2 Regularization (lambda)", 0.0, 1.0, 0.0, 0.01, key="ls_reg")
        
        st.markdown(r"""
        **Loss Function:**
        $$L(w) = \\sum_{i=1}^{N}(y_i - w \\cdot x_i)^2 + \\lambda\\|w\\|^2$$
        """)
    
    with col2:
        np.random.seed(42)
        X = np.linspace(0, 10, n_samples)
        true_w, true_b = 2.5, 1.0
        y = true_w * X + true_b + np.random.randn(n_samples) * noise_level
        
        X_design = np.vstack([X, np.ones_like(X)]).T
        
        if regularization > 0:
            I = np.eye(2)
            I[1, 1] = 0
            w = np.linalg.inv(X_design.T @ X_design + regularization * I) @ X_design.T @ y
        else:
            w = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
        
        predicted_w, predicted_b = w[0], w[1]
        y_pred = predicted_w * X + predicted_b
        mse = mean_squared_error(y, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X, y, c='blue', alpha=0.5, s=50, label='Training Data')
        ax.plot(X, true_w * X + true_b, 'g--', linewidth=2, label=f'True: y = {true_w}x + {true_b}')
        ax.plot(X, y_pred, 'r-', linewidth=2, label=f'Fitted: y = {predicted_w:.3f}x + {predicted_b:.3f}')
        
        for i in range(0, n_samples, max(1, n_samples//10)):
            ax.plot([X[i], X[i]], [y[i], y_pred[i]], 'k:', alpha=0.3)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'Least Squares Linear Regression (MSE: {mse:.4f})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.success(f"Slope: {predicted_w:.4f}, Intercept: {predicted_b:.4f}, MSE: {mse:.4f}")

# ==================== Module 2: KNN Classifier ====================

def knn_module():
    st.markdown('<h2 style="color: #ff7f0e;">🎯 K-Nearest Neighbors (KNN) Image Classifier</h2>', unsafe_allow_html=True)
    
    # Check data status
    cifar10_status = check_cifar10_status()
    has_real_data = cifar10_status[0] is not None
    
    if has_real_data:
        (X_train_real, y_train_real), (X_test_real, y_test_real), _ = cifar10_status
    else:
        X_train_real = y_train_real = X_test_real = y_test_real = None
    
    col_dl1, col_dl2 = st.columns([1, 1])
    with col_dl1:
        if not has_real_data:
            st.warning("Real CIFAR-10 data not downloaded (~170MB)")
            if st.button("Download CIFAR-10 Dataset", type="primary"):
                with st.spinner("Downloading..."):
                    if download_cifar10_data():
                        st.success("Download complete! Please refresh")
                        st.rerun()
        else:
            st.success(f"Real CIFAR-10 data loaded: {len(X_train_real)} train, {len(X_test_real)} test")
    
    data_mode = st.radio("Select Data Mode", 
                        ["Simulated Data (Fast Demo)", "Real CIFAR-10 Data"],
                        index=0 if not has_real_data else 1,
                        disabled=not has_real_data)
    
    use_real_data = (data_mode == "Real CIFAR-10 Data") and has_real_data
    
    # Image upload feature
    st.subheader("Upload Image for Classification")
    uploaded_file = st.file_uploader("Upload test image (JPG/PNG)", type=['jpg', 'jpeg', 'png'], key="knn_upload")
    
    if uploaded_file is not None:
        col_up1, col_up2 = st.columns([1, 2])
        
        with col_up1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Test Image", use_container_width=True)
            
            k_value_upload = st.slider("K Value", 1, 10, 3, key="knn_k_upload")
            metric_upload = st.selectbox("Distance Metric", ["euclidean", "manhattan"], 
                                        format_func=lambda x: "L2 (Euclidean)" if x == "euclidean" else "L1 (Manhattan)",
                                        key="knn_metric_upload")
        
        with col_up2:
            if has_real_data and use_real_data:
                # Use real CIFAR-10 data
                img_resized = image.resize((32, 32))
                img_np = np.array(img_resized)
                
                img_flat = img_np.flatten().reshape(1, -1)
                X_train_flat = X_train_real.reshape(X_train_real.shape[0], -1)
                
                knn = KNeighborsClassifier(n_neighbors=k_value_upload, metric=metric_upload)
                knn.fit(X_train_flat, y_train_real)
                
                prediction = knn.predict(img_flat)[0]
                distances, indices = knn.kneighbors(img_flat, n_neighbors=k_value_upload)
                
                st.success(f"Predicted Class: {CIFAR10_CLASSES[prediction]}")
                
                st.subheader(f"{k_value_upload} Nearest Neighbors:")
                cols = st.columns(min(k_value_upload, 5))
                for i, idx in enumerate(indices[0][:5]):
                    with cols[i % 5]:
                        neighbor_img = X_train_real[idx]
                        true_label = CIFAR10_CLASSES[y_train_real[idx]]
                        st.image(neighbor_img, caption=f"{true_label}\nDist:{distances[0][i]:.1f}", width=80)
            else:
                st.info("Using simulated data (based on color features)")
                
                img_array = np.array(image.resize((64, 64)))
                r_mean, g_mean, b_mean = np.mean(img_array[:,:,0]), np.mean(img_array[:,:,1]), np.mean(img_array[:,:,2])
                
                class_color_hints = {
                    0: (135, 206, 235), 1: (255, 0, 0), 2: (255, 215, 0), 3: (255, 140, 0),
                    4: (139, 90, 43), 5: (160, 82, 45), 6: (0, 255, 0), 7: (139, 69, 19),
                    8: (0, 0, 139), 9: (128, 128, 128)
                }
                
                distances_to_classes = []
                for class_id in range(10):
                    base_color = np.array(class_color_hints[class_id])
                    test_color = np.array([r_mean, g_mean, b_mean])
                    dist = np.linalg.norm(base_color - test_color)
                    distances_to_classes.append((class_id, dist))
                
                distances_to_classes.sort(key=lambda x: x[1])
                predicted_class = distances_to_classes[0][0]
                
                st.success(f"Predicted Class: {CIFAR10_CLASSES[predicted_class]} (based on color)")
                st.write(f"Color Features: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}")
    
    st.divider()
    st.subheader("KNN Classification Demo")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        k_value = st.slider("K Value", 1, 20, 5, key="knn_k")
        metric = st.selectbox("Distance Metric", ["euclidean", "manhattan"], 
                             format_func=lambda x: "L2 (Euclidean)" if x == "euclidean" else "L1 (Manhattan)")
        
        selected_classes = st.multiselect("Select Classes", list(range(10)), [0, 1, 2, 3],
                                         format_func=lambda x: f"{x}: {CIFAR10_CLASSES[x]}")
        
        if len(selected_classes) < 2:
            st.warning("Please select at least 2 classes")
            return
    
    with col2:
        if use_real_data:
            # Use real data
            train_mask = np.isin(y_train_real, selected_classes)
            test_mask = np.isin(y_test_real, selected_classes)
            
            X_train_filtered = X_train_real[train_mask]
            y_train_filtered = y_train_real[train_mask]
            X_test_filtered = X_test_real[test_mask]
            y_test_filtered = y_test_real[test_mask]
            
            label_map = {old: new for new, old in enumerate(selected_classes)}
            y_train_mapped = np.array([label_map[y] for y in y_train_filtered])
            y_test_mapped = np.array([label_map[y] for y in y_test_filtered])
            
            X_train_flat = X_train_filtered.reshape(len(X_train_filtered), -1)
            X_test_flat = X_test_filtered.reshape(len(X_test_filtered), -1)
            
            knn = KNeighborsClassifier(n_neighbors=min(k_value, len(X_train_flat)), metric=metric)
            knn.fit(X_train_flat, y_train_mapped)
            
            y_pred = knn.predict(X_test_flat)
            accuracy = accuracy_score(y_test_mapped, y_pred)
            
            st.success(f"Accuracy: {accuracy:.2%}")
            
            st.subheader("Class Examples")
            cols = st.columns(min(5, len(selected_classes)))
            for idx, class_id in enumerate(selected_classes[:5]):
                with cols[idx]:
                    class_images = X_train_real[y_train_real == class_id]
                    if len(class_images) > 0:
                        st.image(class_images[0], caption=CIFAR10_CLASSES[class_id], width=100)
        else:
            # Use simulated data
            class_color_hints = {0: (135, 206, 235), 1: (255, 0, 0), 2: (255, 215, 0), 3: (255, 140, 0),
                               4: (139, 90, 43), 5: (160, 82, 45), 6: (0, 255, 0), 7: (139, 69, 19),
                               8: (0, 0, 139), 9: (128, 128, 128)}
            
            np.random.seed(42)
            n_samples = 30
            X, y = [], []
            
            for idx, class_id in enumerate(selected_classes):
                base_color = class_color_hints[class_id]
                features_r = np.random.randn(n_samples) * 15 + base_color[0]
                features_g = np.random.randn(n_samples) * 15 + base_color[1]
                X.extend(np.column_stack([features_r, features_g]))
                y.extend([idx] * n_samples)
            
            X, y = np.array(X), np.array(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            knn = KNeighborsClassifier(n_neighbors=min(k_value, len(X_train)), metric=metric)
            knn.fit(X_train, y_train)
            
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            x_min, x_max = X[:, 0].min() - 30, X[:, 0].max() + 30
            y_min, y_max = X[:, 1].min() - 30, X[:, 1].max() + 30
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 2), np.arange(y_min, y_max, 2))
            Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            
            colors_list = ['#FFCCCC', '#CCFFCC', '#CCCCFF', '#FFFFCC', '#FFCCFF']
            cmap = ListedColormap(colors_list[:len(selected_classes)])
            ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
            
            for idx, class_id in enumerate(selected_classes):
                color = np.array(class_color_hints[class_id]) / 255.0
                mask_train, mask_test = y_train == idx, y_test == idx
                ax.scatter(X_train[mask_train, 0], X_train[mask_train, 1], c=[color], edgecolors='k', s=50)
                ax.scatter(X_test[mask_test, 0], X_test[mask_test, 1], c=[color], edgecolors='k', s=100, marker='*')
            
            ax.set_xlabel('Mean Red Channel', fontsize=12)
            ax.set_ylabel('Mean Green Channel', fontsize=12)
            ax.set_title(f'KNN Classification (K={k_value}, Accuracy: {accuracy:.2%})')
            
            legend_elements = [Patch(facecolor=np.array(class_color_hints[c])/255, edgecolor='k', 
                                   label=CIFAR10_CLASSES[c]) for c in selected_classes]
            ax.legend(handles=legend_elements)
            st.pyplot(fig)

# ==================== Module 3: KNN Comparison ====================

def knn_comparison_module():
    st.markdown('<h2 style="color: #ff7f0e;">🔍 KNN Visualization with Different K Values</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">
    <b>Effect of K Value:</b><br>
    - <b>K=1</b>: Complex decision boundary, prone to overfitting<br>
    - <b>Large K</b>: Smooth decision boundary, may underfit<br>
    - <b>Moderate K</b>: Balance between fitting and generalization
    </div>
    """, unsafe_allow_html=True)
    
    class_color_hints = {0: (135, 206, 235), 1: (255, 0, 0), 2: (255, 215, 0), 3: (255, 140, 0),
                       4: (139, 90, 43), 5: (160, 82, 45), 6: (0, 255, 0), 7: (139, 69, 19),
                       8: (0, 0, 139), 9: (128, 128, 128)}
    
    selected_classes = st.multiselect("Select Classes", list(range(10)), [0, 1, 2],
                                     format_func=lambda x: f"{x}: {CIFAR10_CLASSES[x]}")
    
    if len(selected_classes) < 2:
        st.warning("Please select at least 2 classes")
        return
    
    k_values = st.multiselect("Select K Values", [1, 3, 5, 10, 15], default=[1, 5, 10])
    
    if len(k_values) == 0:
        return
    
    # Generate simulated data
    np.random.seed(42)
    n_samples = 50
    X, y = [], []
    
    for idx, class_id in enumerate(selected_classes):
        base_color = class_color_hints[class_id]
        features_r = np.random.randn(n_samples) * 20 + base_color[0]
        features_g = np.random.randn(n_samples) * 20 + base_color[1]
        X.extend(np.column_stack([features_r, features_g]))
        y.extend([idx] * n_samples)
    
    X, y = np.array(X), np.array(y)
    
    # Create subplots
    n_cols = min(3, len(k_values))
    n_rows = (len(k_values) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)
        
        x_min, x_max = X[:, 0].min() - 20, X[:, 0].max() + 20
        y_min, y_max = X[:, 1].min() - 20, X[:, 1].max() + 20
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        colors_list = ['#FFCCCC', '#CCFFCC', '#CCCCFF', '#FFFFCC', '#FFCCFF', '#CCFFFF', '#FFD700']
        cmap = ListedColormap(colors_list[:len(selected_classes)])
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
        
        for idx_c, class_id in enumerate(selected_classes):
            mask = y == idx_c
            color_norm = np.array(class_color_hints[class_id]) / 255.0
            ax.scatter(X[mask, 0], X[mask, 1], c=[color_norm], edgecolors='k', s=30)
        
        ax.set_title(f'K = {k}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Mean Red Channel')
        ax.set_ylabel('Mean Green Channel')
    
    for idx in range(len(k_values), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Legend
    legend_elements = [Patch(facecolor=np.array(class_color_hints[c])/255, edgecolor='k', 
                           label=CIFAR10_CLASSES[c]) for c in selected_classes]
    fig_legend, ax_legend = plt.subplots(figsize=(8, 1))
    ax_legend.legend(handles=legend_elements, loc='center', ncol=len(selected_classes))
    ax_legend.axis('off')
    st.pyplot(fig_legend)

# ==================== Module 4: Linear Classifier Templates ====================

def linear_classifier_templates_module():
    st.markdown('<h2 style="color: #ff7f0e;">🖼️ CIFAR-10 Linear Classifier Template Visualization</h2>', unsafe_allow_html=True)
    
    st.info("""
    Note: The following templates are hand-designed simulated templates for demonstrating 
    the visualization principle of linear classifiers. Real CIFAR-10 linear classifier 
    templates are obtained by training on 50,000 images.
    """)
    
    weights, class_names = generate_linear_classifier_weights(n_classes=10)
    
    # Display templates
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(10):
        template = weights[i]
        template = (template - template.min()) / (template.max() - template.min() + 1e-8)
        axes[i].imshow(template)
        axes[i].set_title(class_names[i], fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle('Linear Classifier Simulated Templates (CIFAR-10 Style)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    **Template Interpretation:**
    - Each template represents an "ideal" image of a class
    - Decision rule is w^T x: if w_i is large, then large x_i values indicate the class
    - During classification, compute dot product (similarity) between input and each template
    - The class with highest score is the predicted result
    """)

# ==================== Module 5: SGD and Momentum ====================

def sgd_momentum_module():
    st.markdown('<h2 style="color: #ff7f0e;">🚀 SGD vs Momentum Gradient Descent Visualization</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">
    <b>Gradient Descent Optimization Comparison:</b><br>
    <b>SGD:</b> x_{t+1} = x_t - alpha * grad f(x_t)<br>
    <b>SGD + Momentum:</b> v_{t+1} = rho * v_t + grad f(x_t),  x_{t+1} = x_t - alpha * v_{t+1}<br>
    Momentum accumulates historical gradients to accelerate convergence and escape local minima.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        terrain_type = st.selectbox("Loss Function Terrain", 
                                   ["Bowl (Convex)", "Wave (Multi-modal)", "Canyon (Ill-conditioned)"],
                                   index=2)
        
        learning_rate = st.slider("Learning Rate", 0.001, 0.5, 0.05, 0.001)
        momentum = st.slider("Momentum (rho)", 0.0, 0.99, 0.9, 0.01)
        n_iterations = st.slider("Iterations", 10, 200, 50)
        compare_both = st.checkbox("Show both SGD and Momentum", value=True)
    
    with col2:
        def loss_function(x, y, terrain):
            if terrain == "Bowl (Convex)":
                return x**2 + y**2
            elif terrain == "Wave (Multi-modal)":
                return np.sin(3*x) * np.cos(3*y) + 0.1 * (x**2 + y**2)
            else:  # Canyon
                return 10*x**2 + y**2
        
        def gradient(x, y, terrain):
            eps = 1e-5
            dx = (loss_function(x+eps, y, terrain) - loss_function(x-eps, y, terrain)) / (2*eps)
            dy = (loss_function(x, y+eps, terrain) - loss_function(x, y-eps, terrain)) / (2*eps)
            return np.array([dx, dy])
        
        x_range, y_range = (-2, 2), (-2, 2)
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = loss_function(X, Y, terrain_type)
        
        def optimize(use_momentum=False):
            pos = np.array([1.5, 1.5])
            velocity = np.zeros(2)
            path = [pos.copy()]
            
            for _ in range(n_iterations):
                grad = gradient(pos[0], pos[1], terrain_type)
                
                if use_momentum:
                    velocity = momentum * velocity + grad
                    pos = pos - learning_rate * velocity
                else:
                    pos = pos - learning_rate * grad
                
                path.append(pos.copy())
            
            return np.array(path)
        
        path_sgd = optimize(use_momentum=False)
        path_momentum = optimize(use_momentum=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        contour = ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8)
        
        if compare_both:
            ax.plot(path_sgd[:, 0], path_sgd[:, 1], 'r-o', markersize=4, linewidth=1.5, label='SGD', alpha=0.7)
            ax.plot(path_momentum[:, 0], path_momentum[:, 1], 'b-s', markersize=4, linewidth=1.5, label='SGD + Momentum', alpha=0.7)
        
        ax.scatter(path_sgd[0, 0], path_sgd[0, 1], c='green', s=200, marker='*', label='Start', zorder=5)
        
        ax.set_xlabel('Parameter w1', fontsize=12)
        ax.set_ylabel('Parameter w2', fontsize=12)
        ax.set_title(f'Gradient Descent Optimization Path - {terrain_type}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

# ==================== Module 6: Loss Function Demo ====================

def loss_demo_module():
    st.markdown('<h2 style="color: #ff7f0e;">📊 Loss Function Calculation Demo</h2>', unsafe_allow_html=True)
    
    loss_type = st.selectbox("Select Loss Function Type", 
                            ["Mean Squared Error (MSE)", "Cross-Entropy Loss", "Hinge Loss (SVM)"])
    
    if loss_type == "Mean Squared Error (MSE)":
        st.subheader("Mean Squared Error (MSE)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; font-family: monospace; text-align: center;'>L = (1/N) * sum((y_i - y_pred_i)^2)</div>", unsafe_allow_html=True)
            
            y_true = st.slider("True Value y", -5.0, 5.0, 2.0, 0.1)
            y_pred = st.slider("Predicted Value y_pred", -5.0, 5.0, 1.0, 0.1)
            
            error = y_true - y_pred
            loss = error ** 2
            
            st.info(f"Error: {error:.2f}, Squared Error: {loss:.4f}")
        
        with col2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            x_vals = np.linspace(-5, 5, 100)
            ax1.plot(x_vals, x_vals, 'g--', label='y = y_pred')
            ax1.scatter([y_pred], [y_true], c='red', s=200, zorder=5)
            ax1.plot([y_pred, y_pred], [y_true, y_pred], 'r-', linewidth=2)
            ax1.set_xlabel('Predicted Value y_pred')
            ax1.set_ylabel('True Value y')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            errors = np.linspace(-5, 5, 100)
            losses = errors ** 2
            ax2.plot(errors, losses, 'b-', linewidth=2)
            ax2.scatter([error], [loss], c='red', s=200, zorder=5)
            ax2.set_xlabel('Error')
            ax2.set_ylabel('Loss L')
            ax2.set_title('MSE Loss Function')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    elif loss_type == "Cross-Entropy Loss":
        st.subheader("Cross-Entropy Loss")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("<div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; font-family: monospace; text-align: center;'>L = -log(p_y)</div>", unsafe_allow_html=True)
            
            s_cat = st.slider("Cat Score", -10.0, 10.0, 3.2, 0.1)
            s_car = st.slider("Car Score", -10.0, 10.0, 5.1, 0.1)
            s_frog = st.slider("Frog Score", -10.0, 10.0, -1.7, 0.1)
            
            true_class = st.selectbox("True Class", ["Cat", "Car", "Frog"])
            
            scores = np.array([s_cat, s_car, s_frog])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / np.sum(exp_scores)
            
            class_map = {"Cat": 0, "Car": 1, "Frog": 2}
            true_idx = class_map[true_class]
            loss = -np.log(probs[true_idx] + 1e-10)
            
            st.success(f"Loss for '{true_class}': {loss:.4f}")
        
        with col2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            classes = ['Cat', 'Car', 'Frog']
            ax1.bar(classes, scores, color=['orange', 'blue', 'green'])
            ax1.set_ylabel('Score (Logits)')
            ax1.set_title('Class Scores')
            
            colors = ['red' if i == true_idx else 'skyblue' for i in range(3)]
            bars = ax2.bar(classes, probs, color=colors)
            ax2.set_ylabel('Probability')
            ax2.set_title('Softmax Probability Distribution')
            
            for i, (s, p) in enumerate(zip(scores, probs)):
                ax1.text(i, s + 0.3, f'{s:.2f}', ha='center')
                ax2.text(i, p + 0.02, f'{p:.4f}', ha='center')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    else:  # Hinge Loss
        st.subheader("Hinge Loss (SVM Loss)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("<div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; font-family: monospace; text-align: center;'>L = sum_{j!=y} max(0, s_j - s_y + delta)</div>", unsafe_allow_html=True)
            
            s_correct = st.slider("Correct Class Score", -5.0, 10.0, 3.0, 0.1)
            s_other1 = st.slider("Other Class 1 Score", -5.0, 10.0, 1.0, 0.1)
            s_other2 = st.slider("Other Class 2 Score", -5.0, 10.0, 2.0, 0.1)
            delta = st.slider("Margin delta", 0.0, 3.0, 1.0, 0.1)
            
            margins = [max(0, s - s_correct + delta) for s in [s_other1, s_other2]]
            loss = sum(margins)
            
            st.info(f"Loss: {margins[0]:.2f} + {margins[1]:.2f} = {loss:.2f}")
            if loss == 0:
                st.success("Loss is 0! Correct class score is high enough")
            else:
                st.warning(f"Loss is {loss:.2f}, needs more optimization")
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            scores = [s_correct, s_other1, s_other2]
            labels = ['Correct', 'Other 1', 'Other 2']
            colors = ['green', 'red', 'red']
            
            bars = ax.bar(labels, scores, color=colors, alpha=0.7)
            ax.axhline(y=s_correct - delta, color='blue', linestyle='--', label=f'Safety Margin')
            ax.axhline(y=s_correct, color='green', linestyle='-', label=f'Correct Class Score')
            
            for i, (s, m) in enumerate(zip([s_other1, s_other2], margins)):
                if m > 0:
                    ax.annotate('', xy=(i+1, s), xytext=(i+1, s_correct - delta),
                               arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
                    ax.text(i+1, (s + s_correct - delta)/2, f'Loss={m:.2f}', 
                           ha='center', fontsize=10, color='orange', fontweight='bold')
            
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('SVM Hinge Loss Visualization', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)

# ==================== Main Application ====================

def main():
    st.markdown('<h1 style="text-align: center; color: #1f77b4;">Pattern Recognition and Image Processing - Assignment A4</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Student ID: 2025213362 | Name: Min Yizhou</p>', unsafe_allow_html=True)
    
    st.sidebar.markdown("## Navigation Menu")
    
    module = st.sidebar.radio(
        "Select Module",
        [
            "Home",
            "Least Squares Linear Regression",
            "KNN Image Classifier",
            "KNN Comparison",
            "CIFAR Linear Classifier Templates",
            "SGD vs Momentum",
            "Loss Function Demo"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### Student Info
    - **ID**: 2025213362
    - **Name**: Min Yizhou
    
    ### Tools Used
    - **AI Agent**: Kimi Code CLI (Moonshot AI)
    - **Language**: Python 3.11
    - **Framework**: Streamlit
    - **Dataset**: CIFAR-10 (Real + Simulated)
    """)
    
    if module == "Home":
        st.markdown("""
        ## Welcome to Pattern Recognition Assignment A4
        
        This application includes 6 functional modules:
        
        ### Module List
        
        1. **Least Squares Linear Regression**
           - Least squares linear regression example
           - Supports L2 regularization
        
        2. **KNN Image Classifier**
           - Supports real CIFAR-10 data (requires download ~170MB)
           - Supports simulated data for quick demo
           - Upload your own images for classification
        
        3. **KNN Comparison**
           - Decision boundary comparison with different K values
           - Observe overfitting vs underfitting
        
        4. **CIFAR Linear Classifier Templates**
           - Linear classifier weight visualization
        
        5. **SGD vs Momentum**
           - Gradient descent optimization comparison
        
        6. **Loss Function Demo**
           - MSE, Cross-Entropy, Hinge Loss
        
        ### Data Download
        To use real CIFAR-10 data, click the download button in the KNN module.
        """)
    elif "Least Squares" in module:
        least_squares_module()
    elif "KNN Image Classifier" in module:
        knn_module()
    elif "KNN Comparison" in module:
        knn_comparison_module()
    elif "CIFAR Linear Classifier" in module:
        linear_classifier_templates_module()
    elif "SGD vs Momentum" in module:
        sgd_momentum_module()
    elif "Loss Function" in module:
        loss_demo_module()

if __name__ == "__main__":
    main()

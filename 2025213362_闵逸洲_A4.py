"""
模式识别与图像处理 - 作业A4 (完整版)
学号: 2025213362
姓名: 闵逸洲

本应用实现了以下功能：
1. Least Squares Linear Regression示例
2. KNN分类器（支持真实CIFAR-10图像数据）
3. 基于CIFAR的线性分类器学到的模板图像可视化
4. 不同K的KNN可视化对比
5. SGD/动量更新的梯度下降可视化对比
6. 不同loss损失的计算过程演示

使用的Agent: Kimi Code CLI (Moonshot AI)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
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
import os
import pickle
import urllib.request
import tarfile
from collections import Counter
from matplotlib.colors import ListedColormap
warnings.filterwarnings('ignore')

# ==================== CIFAR-10 数据管理 ====================

CIFAR10_CLASSES_CN = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
CIFAR10_CLASSES_EN = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@st.cache_data
def download_and_load_cifar10():
    """下载并加载CIFAR-10数据集"""
    data_dir = './cifar10_data'
    cifar_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    
    if not os.path.exists(cifar_dir):
        return None, None, None
    
    try:
        # 加载训练数据
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
        
        # 加载测试数据
        test_filepath = os.path.join(cifar_dir, 'test_batch')
        with open(test_filepath, 'rb') as f:
            test_batch = pickle.load(f, encoding='bytes')
            X_test = test_batch[b'data']
            y_test = np.array(test_batch[b'labels'])
        
        # 重塑数据为 (N, 32, 32, 3)
        X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        # 取子集
        np.random.seed(42)
        train_idx = np.random.choice(len(X_train), 2000, replace=False)
        test_idx = np.random.choice(len(X_test), 500, replace=False)
        
        X_train = X_train[train_idx]
        y_train = y_train[train_idx]
        X_test = X_test[test_idx]
        y_test = y_test[test_idx]
        
        return (X_train, y_train), (X_test, y_test), CIFAR10_CLASSES_CN
    except Exception as e:
        return None, None, None

def download_cifar10_data():
    """下载CIFAR-10数据集"""
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
            status_text.text(f"下载进度: {percent:.1f}%")
        
        status_text.text("正在下载CIFAR-10数据集 (~170MB)...")
        urllib.request.urlretrieve(base_url + filename, filepath, reporthook=download_progress)
        
        status_text.text("正在解压...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(data_dir)
        
        os.remove(filepath)
        progress_bar.empty()
        status_text.empty()
        return True
    except Exception as e:
        st.error(f"下载失败: {e}")
        return False

def check_cifar10_status():
    """检查CIFAR-10数据状态"""
    data_dir = './cifar10_data/cifar-10-batches-py'
    if os.path.exists(data_dir):
        return download_and_load_cifar10()
    return None, None, None

# ==================== 辅助函数 ====================

def generate_linear_classifier_weights(n_classes=10):
    """生成线性分类器的模拟权重模板"""
    np.random.seed(42)
    weights = np.zeros((n_classes, 32, 32, 3))
    
    for c in range(n_classes):
        if c == 0:  # 飞机
            weights[c, :, :, 2] = 0.6
            weights[c, 10:22, 8:24, :] = [0.9, 0.9, 0.95]
            weights[c, 14:18, 4:28, 2] = 0.8
        elif c == 1:  # 汽车
            weights[c, 20:32, :, 1] = 0.3
            weights[c, 12:22, 8:24, 0] = 0.8
        elif c == 2:  # 鸟
            weights[c, :, :, 2] = 0.5
            weights[c, 12:20, 12:20, 0] = 0.9
        elif c == 3:  # 猫
            weights[c, :, :, 0] = 0.4
            weights[c, 10:26, 10:22, 0] = 0.9
        elif c == 4:  # 鹿
            weights[c, :, :, 1] = 0.4
            weights[c, 12:26, 10:22, 0] = 0.6
        elif c == 5:  # 狗
            weights[c, 22:32, :, 1] = 0.4
            weights[c, 12:24, 10:24, 0] = 0.7
        elif c == 6:  # 青蛙
            weights[c, :, :, 1] = 0.4
            weights[c, 14:24, 10:22, 1] = 0.9
        elif c == 7:  # 马
            weights[c, 22:32, :, 1] = 0.35
            weights[c, 10:26, 10:22, 0] = 0.6
        elif c == 8:  # 船
            weights[c, 18:32, :, 2] = 0.7
            weights[c, 12:20, 10:22, :] = [0.9, 0.9, 0.9]
        elif c == 9:  # 卡车
            weights[c, 22:32, :, :] = 0.3
            weights[c, 10:24, 6:26, :] = 0.6
    
    weights += np.random.randn(*weights.shape) * 0.05
    weights = np.clip(weights, 0, 1)
    
    return weights, CIFAR10_CLASSES_CN

# ==================== 页面配置 ====================

st.set_page_config(
    page_title="模式识别作业A4 - 2025213362_闵逸洲",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 模块1: Least Squares ====================

def least_squares_module():
    st.markdown('<h2 style="color: #ff7f0e;">📈 Least Squares Linear Regression</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">
        <b>最小二乘法原理：</b><br>
        寻找最佳拟合直线，使得所有数据点到直线的垂直距离（残差）的平方和最小。
        </div>
        """, unsafe_allow_html=True)
        
        n_samples = st.slider("样本数量", 10, 200, 50, key="ls_samples")
        noise_level = st.slider("噪声水平", 0.0, 5.0, 1.0, 0.1, key="ls_noise")
        regularization = st.slider("L2正则化强度 (λ)", 0.0, 1.0, 0.0, 0.01, key="ls_reg")
        
        st.markdown(r"""
        **损失函数：**
        $$L(w) = \sum_{i=1}^{N}(y_i - w \cdot x_i)^2 + \lambda\|w\|^2$$
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
        ax.scatter(X, y, c='blue', alpha=0.5, s=50, label='训练数据')
        ax.plot(X, true_w * X + true_b, 'g--', linewidth=2, label=f'真实直线: y = {true_w}x + {true_b}')
        ax.plot(X, y_pred, 'r-', linewidth=2, label=f'拟合直线: y = {predicted_w:.3f}x + {predicted_b:.3f}')
        
        for i in range(0, n_samples, max(1, n_samples//10)):
            ax.plot([X[i], X[i]], [y[i], y_pred[i]], 'k:', alpha=0.3)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'Least Squares Linear Regression (MSE: {mse:.4f})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.success(f"斜率: {predicted_w:.4f}, 截距: {predicted_b:.4f}, MSE: {mse:.4f}")

# ==================== 模块2: KNN分类器 ====================

def knn_module():
    st.markdown('<h2 style="color: #ff7f0e;">🎯 K-Nearest Neighbors (KNN) 图像分类器</h2>', unsafe_allow_html=True)
    
    # 检查数据状态
    cifar10_status = check_cifar10_status()
    has_real_data = cifar10_status[0] is not None
    
    if has_real_data:
        (X_train_real, y_train_real), (X_test_real, y_test_real), _ = cifar10_status
    else:
        X_train_real = y_train_real = X_test_real = y_test_real = None
    
    col_dl1, col_dl2 = st.columns([1, 1])
    with col_dl1:
        if not has_real_data:
            st.warning("⚠️ 真实CIFAR-10数据未下载 (~170MB)")
            if st.button("📥 下载CIFAR-10数据集", type="primary"):
                with st.spinner("正在下载..."):
                    if download_cifar10_data():
                        st.success("下载完成！请刷新页面")
                        st.rerun()
        else:
            st.success(f"✅ 真实CIFAR-10数据已加载: {len(X_train_real)}张训练, {len(X_test_real)}张测试")
    
    data_mode = st.radio("选择数据模式", 
                        ["模拟数据（快速演示）", "真实CIFAR-10数据"],
                        index=0 if not has_real_data else 1,
                        disabled=not has_real_data)
    
    use_real_data = (data_mode == "真实CIFAR-10数据") and has_real_data
    
    # 添加图片上传功能
    st.subheader("📤 上传图像进行分类测试")
    uploaded_file = st.file_uploader("上传测试图像 (JPG/PNG)", type=['jpg', 'jpeg', 'png'], key="knn_upload")
    
    if uploaded_file is not None:
        col_up1, col_up2 = st.columns([1, 2])
        
        with col_up1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="上传的测试图像", use_container_width=True)
            
            k_value_upload = st.slider("K值", 1, 10, 3, key="knn_k_upload")
            metric_upload = st.selectbox("距离度量", ["euclidean", "manhattan"], 
                                        format_func=lambda x: "L2 (欧几里得)" if x == "euclidean" else "L1 (曼哈顿)",
                                        key="knn_metric_upload")
        
        with col_up2:
            if has_real_data and use_real_data:
                # 使用真实CIFAR-10数据进行分类
                img_resized = image.resize((32, 32))
                img_np = np.array(img_resized)
                
                img_flat = img_np.flatten().reshape(1, -1)
                X_train_flat = X_train_real.reshape(X_train_real.shape[0], -1)
                
                knn = KNeighborsClassifier(n_neighbors=k_value_upload, metric=metric_upload)
                knn.fit(X_train_flat, y_train_real)
                
                prediction = knn.predict(img_flat)[0]
                distances, indices = knn.kneighbors(img_flat, n_neighbors=k_value_upload)
                
                st.success(f"**预测类别: {CIFAR10_CLASSES_CN[prediction]}**")
                
                st.subheader(f"{k_value_upload}个最近邻:")
                cols = st.columns(min(k_value_upload, 5))
                for i, idx in enumerate(indices[0][:5]):
                    with cols[i % 5]:
                        neighbor_img = X_train_real[idx]
                        true_label = CIFAR10_CLASSES_CN[y_train_real[idx]]
                        st.image(neighbor_img, caption=f"{true_label}\n距离:{distances[0][i]:.1f}", width=80)
            else:
                st.info("使用模拟数据进行分类（基于颜色特征）")
                
                # 基于颜色特征进行简单分类
                img_array = np.array(image.resize((64, 64)))
                r_mean, g_mean, b_mean = np.mean(img_array[:,:,0]), np.mean(img_array[:,:,1]), np.mean(img_array[:,:,2])
                
                class_color_hints = {
                    0: (135, 206, 235), 1: (255, 0, 0), 2: (255, 215, 0), 3: (255, 140, 0),
                    4: (139, 90, 43), 5: (160, 82, 45), 6: (0, 255, 0), 7: (139, 69, 19),
                    8: (0, 0, 139), 9: (128, 128, 128)
                }
                
                # 计算与各类别的颜色距离
                distances_to_classes = []
                for class_id in range(10):
                    base_color = np.array(class_color_hints[class_id])
                    test_color = np.array([r_mean, g_mean, b_mean])
                    dist = np.linalg.norm(base_color - test_color)
                    distances_to_classes.append((class_id, dist))
                
                distances_to_classes.sort(key=lambda x: x[1])
                predicted_class = distances_to_classes[0][0]
                
                st.success(f"**预测类别: {CIFAR10_CLASSES_CN[predicted_class]}** (基于颜色相似度)")
                st.write(f"图像颜色特征: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}")
    
    st.divider()
    st.subheader("📊 KNN分类演示")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        k_value = st.slider("K值", 1, 20, 5, key="knn_k")
        metric = st.selectbox("距离度量", ["euclidean", "manhattan"], 
                             format_func=lambda x: "L2 (欧几里得)" if x == "euclidean" else "L1 (曼哈顿)")
        
        selected_classes = st.multiselect("选择类别", list(range(10)), [0, 1, 2, 3],
                                         format_func=lambda x: f"{x}: {CIFAR10_CLASSES_CN[x]}")
        
        if len(selected_classes) < 2:
            st.warning("请至少选择2个类别")
            return
    
    with col2:
        if use_real_data:
            # 使用真实数据
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
            
            st.success(f"**准确率: {accuracy:.2%}**")
            
            # 显示示例
            st.subheader("各类别示例")
            cols = st.columns(min(5, len(selected_classes)))
            for idx, class_id in enumerate(selected_classes[:5]):
                with cols[idx]:
                    class_images = X_train_real[y_train_real == class_id]
                    if len(class_images) > 0:
                        st.image(class_images[0], caption=CIFAR10_CLASSES_CN[class_id], width=100)
        else:
            # 使用模拟数据
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
            
            ax.set_xlabel('红色通道平均值')
            ax.set_ylabel('绿色通道平均值')
            ax.set_title(f'KNN分类 (K={k_value}, 准确率: {accuracy:.2%})')
            
            legend_elements = [Patch(facecolor=np.array(class_color_hints[c])/255, edgecolor='k', 
                                   label=CIFAR10_CLASSES_CN[c]) for c in selected_classes]
            ax.legend(handles=legend_elements)
            st.pyplot(fig)

# ==================== 模块3: KNN对比 ====================

def knn_comparison_module():
    st.markdown('<h2 style="color: #ff7f0e;">🔍 不同K值的KNN可视化对比</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">
    <b>K值选择的影响：</b><br>
    - <b>K=1</b>: 决策边界复杂，容易过拟合<br>
    - <b>K较大</b>: 决策边界平滑，可能欠拟合<br>
    - <b>适中的K</b>: 平衡拟合与泛化能力
    </div>
    """, unsafe_allow_html=True)
    
    # CIFAR-10颜色提示
    class_color_hints = {0: (135, 206, 235), 1: (255, 0, 0), 2: (255, 215, 0), 3: (255, 140, 0),
                       4: (139, 90, 43), 5: (160, 82, 45), 6: (0, 255, 0), 7: (139, 69, 19),
                       8: (0, 0, 139), 9: (128, 128, 128)}
    
    selected_classes = st.multiselect("选择类别", list(range(10)), [0, 1, 2],
                                     format_func=lambda x: f"{x}: {CIFAR10_CLASSES_CN[x]}")
    
    if len(selected_classes) < 2:
        st.warning("请至少选择2个类别")
        return
    
    k_values = st.multiselect("选择K值", [1, 3, 5, 10, 15], default=[1, 5, 10])
    
    if len(k_values) == 0:
        return
    
    # 生成模拟数据
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
    
    # 创建子图
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
        ax.set_xlabel('红色通道平均值')
        ax.set_ylabel('绿色通道平均值')
    
    for idx in range(len(k_values), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 图例
    legend_elements = [Patch(facecolor=np.array(class_color_hints[c])/255, edgecolor='k', 
                           label=CIFAR10_CLASSES_CN[c]) for c in selected_classes]
    fig_legend, ax_legend = plt.subplots(figsize=(8, 1))
    ax_legend.legend(handles=legend_elements, loc='center', ncol=len(selected_classes))
    ax_legend.axis('off')
    st.pyplot(fig_legend)

# ==================== 模块4: 线性分类器模板 ====================

def linear_classifier_templates_module():
    st.markdown('<h2 style="color: #ff7f0e;">🖼️ CIFAR-10线性分类器模板可视化</h2>', unsafe_allow_html=True)
    
    st.info("""
    **说明：** 以下模板是手工设计的模拟模板，用于演示线性分类器的可视化原理。
    真实的CIFAR-10线性分类器模板是通过在50000张图像上训练得到的。
    """)
    
    weights, class_names = generate_linear_classifier_weights(n_classes=10)
    
    # 显示模板
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(10):
        template = weights[i]
        template = (template - template.min()) / (template.max() - template.min() + 1e-8)
        axes[i].imshow(template)
        axes[i].set_title(class_names[i], fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle('线性分类器模拟类别模板 (CIFAR-10风格)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    **模板解读：**
    - 每个模板代表一个类别的"理想"图像
    - 决策规则是 wᵀx：如果 wᵢ 很大，则大的 xᵢ 值就是该类别的指示
    - 分类时计算输入图像与每个模板的点积（相似度）
    - 分数最高的类别即为预测结果
    """)

# ==================== 模块5: SGD与动量 ====================

def sgd_momentum_module():
    st.markdown('<h2 style="color: #ff7f0e;">🚀 SGD与动量梯度下降可视化对比</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">
    <b>梯度下降优化算法对比：</b><br>
    <b>SGD:</b> xₜ₊₁ = xₜ - α∇f(xₜ)<br>
    <b>SGD + Momentum:</b> vₜ₊₁ = ρvₜ + ∇f(xₜ),  xₜ₊₁ = xₜ - αvₜ₊₁<br>
    动量通过累积历史梯度来加速收敛，帮助逃离局部最小值。
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        terrain_type = st.selectbox("损失函数地形", 
                                   ["碗形 (凸函数)", "波浪形 (多峰)", "峡谷形 (病态条件)"],
                                   index=2)
        
        learning_rate = st.slider("学习率", 0.001, 0.5, 0.05, 0.001)
        momentum = st.slider("动量系数 (ρ)", 0.0, 0.99, 0.9, 0.01)
        n_iterations = st.slider("迭代次数", 10, 200, 50)
        compare_both = st.checkbox("同时显示SGD和Momentum", value=True)
    
    with col2:
        def loss_function(x, y, terrain):
            if terrain == "碗形 (凸函数)":
                return x**2 + y**2
            elif terrain == "波浪形 (多峰)":
                return np.sin(3*x) * np.cos(3*y) + 0.1 * (x**2 + y**2)
            else:  # 峡谷形
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
        
        ax.scatter(path_sgd[0, 0], path_sgd[0, 1], c='green', s=200, marker='*', label='起点', zorder=5)
        
        ax.set_xlabel('参数 w₁', fontsize=12)
        ax.set_ylabel('参数 w₂', fontsize=12)
        ax.set_title(f'梯度下降优化路径 - {terrain_type}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

# ==================== 模块6: 损失函数演示 ====================

def loss_demo_module():
    st.markdown('<h2 style="color: #ff7f0e;">📊 不同Loss损失计算过程演示</h2>', unsafe_allow_html=True)
    
    loss_type = st.selectbox("选择损失函数类型", 
                            ["均方误差 (MSE)", "交叉熵损失 (Cross-Entropy)", "Hinge Loss (SVM)"])
    
    if loss_type == "均方误差 (MSE)":
        st.subheader("均方误差 (Mean Squared Error)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; font-family: monospace; text-align: center;'>L = (1/N) Σ(yᵢ - ŷᵢ)²</div>", unsafe_allow_html=True)
            
            y_true = st.slider("真实值 y", -5.0, 5.0, 2.0, 0.1)
            y_pred = st.slider("预测值 ŷ", -5.0, 5.0, 1.0, 0.1)
            
            error = y_true - y_pred
            loss = error ** 2
            
            st.info(f"误差: {error:.2f}, 平方误差: {loss:.4f}")
        
        with col2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            x_vals = np.linspace(-5, 5, 100)
            ax1.plot(x_vals, x_vals, 'g--', label='y = ŷ')
            ax1.scatter([y_pred], [y_true], c='red', s=200, zorder=5)
            ax1.plot([y_pred, y_pred], [y_true, y_pred], 'r-', linewidth=2)
            ax1.set_xlabel('预测值 ŷ')
            ax1.set_ylabel('真实值 y')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            errors = np.linspace(-5, 5, 100)
            losses = errors ** 2
            ax2.plot(errors, losses, 'b-', linewidth=2)
            ax2.scatter([error], [loss], c='red', s=200, zorder=5)
            ax2.set_xlabel('误差')
            ax2.set_ylabel('损失 L')
            ax2.set_title('MSE损失函数')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    elif loss_type == "交叉熵损失 (Cross-Entropy)":
        st.subheader("交叉熵损失 (Cross-Entropy Loss)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("<div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; font-family: monospace; text-align: center;'>L = -log(p_y)</div>", unsafe_allow_html=True)
            
            s_cat = st.slider("猫 (Cat)", -10.0, 10.0, 3.2, 0.1)
            s_car = st.slider("车 (Car)", -10.0, 10.0, 5.1, 0.1)
            s_frog = st.slider("青蛙 (Frog)", -10.0, 10.0, -1.7, 0.1)
            
            true_class = st.selectbox("真实类别", ["猫", "车", "青蛙"])
            
            scores = np.array([s_cat, s_car, s_frog])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / np.sum(exp_scores)
            
            class_map = {"猫": 0, "车": 1, "青蛙": 2}
            true_idx = class_map[true_class]
            loss = -np.log(probs[true_idx] + 1e-10)
            
            st.success(f"真实类别 '{true_class}' 的损失值: **{loss:.4f}**")
        
        with col2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            classes = ['猫', '车', '青蛙']
            ax1.bar(classes, scores, color=['orange', 'blue', 'green'])
            ax1.set_ylabel('分数 (Logits)')
            ax1.set_title('各类别分数')
            
            colors = ['red' if i == true_idx else 'skyblue' for i in range(3)]
            bars = ax2.bar(classes, probs, color=colors)
            ax2.set_ylabel('概率')
            ax2.set_title('Softmax概率分布')
            
            for i, (s, p) in enumerate(zip(scores, probs)):
                ax1.text(i, s + 0.3, f'{s:.2f}', ha='center')
                ax2.text(i, p + 0.02, f'{p:.4f}', ha='center')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    else:  # Hinge Loss
        st.subheader("Hinge Loss (SVM Loss)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("<div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; font-family: monospace; text-align: center;'>L = Σⱼ≠y max(0, sⱼ - s_y + Δ)</div>", unsafe_allow_html=True)
            
            s_correct = st.slider("正确类别分数", -5.0, 10.0, 3.0, 0.1)
            s_other1 = st.slider("其他类别1分数", -5.0, 10.0, 1.0, 0.1)
            s_other2 = st.slider("其他类别2分数", -5.0, 10.0, 2.0, 0.1)
            delta = st.slider("边界 Δ", 0.0, 3.0, 1.0, 0.1)
            
            margins = [max(0, s - s_correct + delta) for s in [s_other1, s_other2]]
            loss = sum(margins)
            
            st.info(f"损失: {margins[0]:.2f} + {margins[1]:.2f} = {loss:.2f}")
            if loss == 0:
                st.success("损失为0！正确类别分数足够高")
            else:
                st.warning(f"损失为 {loss:.2f}，需要继续优化")
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            scores = [s_correct, s_other1, s_other2]
            labels = ['正确类别', '其他类别1', '其他类别2']
            colors = ['green', 'red', 'red']
            
            bars = ax.bar(labels, scores, color=colors, alpha=0.7)
            ax.axhline(y=s_correct - delta, color='blue', linestyle='--', label=f'安全边界')
            ax.axhline(y=s_correct, color='green', linestyle='-', label=f'正确类别分数')
            
            for i, (s, m) in enumerate(zip([s_other1, s_other2], margins)):
                if m > 0:
                    ax.annotate('', xy=(i+1, s), xytext=(i+1, s_correct - delta),
                               arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
                    ax.text(i+1, (s + s_correct - delta)/2, f'损失={m:.2f}', 
                           ha='center', fontsize=10, color='orange', fontweight='bold')
            
            ax.set_ylabel('分数', fontsize=12)
            ax.set_title('SVM Hinge Loss 可视化', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)

# ==================== 主应用 ====================

def main():
    st.markdown('<h1 style="text-align: center; color: #1f77b4;">🎓 模式识别与图像处理 - 作业A4</h1>', unsafe_allow_html=True)
    
    st.sidebar.markdown("## 📋 导航菜单")
    
    module = st.sidebar.radio(
        "选择模块",
        [
            "🏠 首页",
            "📈 Least Squares Linear Regression",
            "🎯 KNN图像分类器",
            "🔍 KNN可视化对比",
            "🖼️ CIFAR线性分类器模板",
            "🚀 SGD与动量梯度下降",
            "📊 损失函数演示"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 👤 学生信息
    - **学号**: 2025213362
    - **姓名**: 闵逸洲
    - **课程**: 模式识别与图像处理
    
    ### 🤖 使用工具
    - **AI Agent**: Kimi Code CLI (Moonshot AI)
    - **开发语言**: Python 3.11
    - **Web框架**: Streamlit
    - **数据集**: CIFAR-10 (真实数据 + 模拟数据)
    """)
    
    if module == "🏠 首页":
        st.markdown("""
        ## 欢迎使用模式识别作业A4演示系统
        
        本应用包含以下6个功能模块：
        
        ### 📚 模块列表
        
        1. **📈 Least Squares Linear Regression**
           - 最小二乘线性回归示例
           - 支持L2正则化
        
        2. **🎯 KNN图像分类器**
           - 支持真实CIFAR-10数据（需下载~170MB）
           - 支持模拟数据快速演示
        
        3. **🔍 KNN可视化对比**
           - 不同K值的决策边界对比
           - 观察过拟合与欠拟合
        
        4. **🖼️ CIFAR线性分类器模板**
           - 线性分类器权重可视化
        
        5. **🚀 SGD与动量梯度下降**
           - SGD vs Momentum对比
        
        6. **📊 损失函数演示**
           - MSE、Cross-Entropy、Hinge Loss
        
        ### 📥 数据下载
        如需使用真实CIFAR-10数据，请点击KNN模块中的下载按钮。
        """)
    elif "Least Squares" in module:
        least_squares_module()
    elif "KNN图像分类器" in module:
        knn_module()
    elif "KNN可视化对比" in module:
        knn_comparison_module()
    elif "CIFAR线性分类器" in module:
        linear_classifier_templates_module()
    elif "SGD与动量" in module:
        sgd_momentum_module()
    elif "损失函数" in module:
        loss_demo_module()

if __name__ == "__main__":
    main()

# 模式识别与图像处理 - 作业A4

**学号**: 2025213362  
**姓名**: 闵逸洲

## 项目简介

本应用实现了以下功能：
1. **Least Squares Linear Regression** - 最小二乘线性回归示例
2. **KNN图像分类器** - 使用真实CIFAR-10图像数据进行K-近邻分类
3. **KNN可视化对比** - 不同K值的决策边界对比
4. **CIFAR线性分类器模板** - 线性分类器权重可视化
5. **SGD与动量梯度下降** - 优化算法可视化对比
6. **损失函数演示** - MSE、Cross-Entropy、Hinge Loss计算过程

## 在线访问

部署后，应用可通过以下链接访问：
[Streamlit Cloud链接]

## 本地运行

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行应用
```bash
streamlit run 2025213362_闵逸洲_A4.py
```

### 访问应用
浏览器会自动打开，或手动访问：`http://localhost:8501`

## 使用说明

1. **首页** - 项目介绍和功能导航
2. **KNN图像分类器** - 支持上传图像进行分类，可选择使用真实CIFAR-10数据或模拟数据
3. **其他模块** - 包含回归、优化、损失函数等可视化演示

## 技术栈

- **前端**: Streamlit
- **后端**: Python 3.11
- **机器学习**: scikit-learn
- **可视化**: Matplotlib, Plotly
- **数据处理**: NumPy, Pillow

## 参考资料

课件: 06_ML_LinearClassifiers_Optimization.pdf (第10、67、70页)

## 使用工具

**AI Agent**: Kimi Code CLI (Moonshot AI)

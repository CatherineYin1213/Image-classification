import streamlit as st
import numpy as np
from PIL import Image
import joblib
from SVM_HOG import getHOGfeat, LinearSVM, extract_hog_features
import base64
from io import BytesIO

# 加载预训练的SVM模型
model_path = 'svm_model_hog.pkl'
svm_model = joblib.load(model_path)

# 加载标准化模型
scaler_path = 'scaler.pkl'
scaler = joblib.load(scaler_path)

# 加载均值图像
mean_image_path = 'mean_image.npy'
mean_image = np.load(mean_image_path)

# HOG参数配置，确保与训练时相同
hog_params = {
    'orientations': 8,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2)
}

# 图像预处理和特征提取函数
def preprocess_and_extract_features(image):
    # 将PIL图像转换为numpy数组，并调整类型为uint8
    image = np.array(image.convert('RGB')).astype(np.float32)

    # 展平图像并减去均值图像
    flat_image = np.reshape(image, -1).astype(np.float32)
    flat_image -= mean_image
    # 添加偏置项
    flat_image = np.hstack([flat_image, 1])

    # 调用extract_hog_features处理输入的图片
    hog_features = extract_hog_features([flat_image])

    # 将HOG特征与处理后的图像特征合并
    combined_features = np.hstack([hog_features.flatten(), flat_image])

    # 标准化特征
    standardized_features = scaler.transform([combined_features])

    return standardized_features

# 类别标签
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Streamlit UI
st.title('Image Classification')
uploaded_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_image is not None:
    display_image = Image.open(uploaded_image)
    # 将图像转换为Base64编码
    buffered = BytesIO()
    display_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # 使用HTML和CSS将图片居中显示
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{img_str}" width="300">
        </div>
        """,
        unsafe_allow_html=True
    )

    # 预处理图像并预测
    features = preprocess_and_extract_features(display_image)
    prediction = svm_model.predict(features)
    predicted_label = classes[prediction[0]]
    st.write(f"Predicted Label: {predicted_label}")
else:
    st.warning('Please upload an image.')

import numpy as np
import cv2
from tqdm import tqdm
from math import sqrt, atan2
from sklearn.preprocessing import StandardScaler

class LinearSVM:
    def __init__(self):
        self.W = None  # 权重矩阵

    def loss(self, X, y, reg):
        num_train = X.shape[0]
        scores = X.dot(self.W)
        correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)
        margins = np.maximum(0, scores - correct_class_scores + 1)
        margins[np.arange(num_train), y] = 0
        loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(self.W ** 2)

        inter_mat = np.zeros_like(scores)
        inter_mat[margins > 0] = 1
        inter_mat[np.arange(num_train), y] = -np.sum(inter_mat, axis=1)
        dW = X.T.dot(inter_mat) / num_train + reg * self.W

        return loss, dW

    def train(self, X, y, learning_rate, reg, num_iters, batch_size, verbose=False):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        loss_history = []
        for it in range(num_iters):
            idx_batch = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[idx_batch]
            y_batch = y[idx_batch]

            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print(f'Iteration {it} / {num_iters}: Loss {loss}')
        return loss_history

    def predict(self, X):
        return np.argmax(X.dot(self.W), axis=1)

def getHOGfeat(image, stride=8, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    # 初始化参数
    cx, cy = pixels_per_cell
    bx, by = cells_per_block
    sx, sy = image.shape
    gx = np.zeros(image.shape, dtype=np.float32)
    gy = np.zeros(image.shape, dtype=np.float32)
    eps = 1e-5

    # 计算梯度 gx 和 gy
    gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
    gy[1:-1, :] = image[2:, :] - image[:-2, :]
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.rad2deg(np.arctan2(gy, gx + eps)) % 360

    # 初始化方向直方图
    orientation_histogram = np.zeros((int(sx / cx), int(sy / cy), orientations))
    for i in range(orientations):
        # 处理每个方向
        temp_orientation = np.where((orientation >= (i * 360 / orientations)) &
                                    (orientation < ((i + 1) * 360 / orientations)),
                                    magnitude, 0)
        for r in range(int(sx / cx)):
            for c in range(int(sy / cy)):
                orientation_histogram[r, c, i] = temp_orientation[r*cx:(r+1)*cx, c*cy:(c+1)*cy].sum()

    # 归一化特征块
    n_cellsx, n_cellsy = int(sx / cx), int(sy / cy)
    n_blocksx, n_blocksy = (n_cellsx - bx + 1), (n_cellsy - by + 1)
    normalised_blocks = np.zeros((n_blocksy, n_blocksx, by * bx * orientations))
    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = orientation_histogram[y:y + by, x:x + bx, :].flatten()
            normalised_blocks[y, x, :] = block / np.sqrt(np.sum(block**2) + eps)

    return normalised_blocks.ravel()

def extract_hog_features(images):
    hog_features = []
    for image in tqdm(images, desc="Extracting HOG features"):
        if len(image) == 3073:  # Assuming the last element is extra and needs to be removed
            image = image[:-1]  # Remove the last element
        image = image.reshape(32, 32, 3)  # Reshape flat array into 32x32x3 RGB image
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)  # Ensure the image type is uint8
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        image = cv2.resize(image, (64, 64))  # Resize image to 64x64 for HOG
        hog_feature = getHOGfeat(image)
        hog_features.append(hog_feature)
    return np.array(hog_features)
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.cluster import KMeans
from tqdm import tqdm

# モデルのロード
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features

# 画像特徴抽出
image_directory = 'D:\麦判別AI作成\wheat_images'  # 画像ディレクトリを正しく設定
image_paths = [os.path.join(image_directory, x) for x in os.listdir(image_directory)]
features_list = [extract_features(img_path, base_model) for img_path in tqdm(image_paths)]

# K-means クラスタリング
kmeans = KMeans(n_clusters=2, random_state=22)
kmeans.fit(features_list)

# クラスタリング結果の表示関数
def display_cluster_images(cluster_labels, image_paths, cluster_number, num_images=10):
    fig, axs = plt.subplots(1, num_images, figsize=(20, 2))
    fig.suptitle(f'Images from Cluster {cluster_number}', fontsize=16)
    selected_images = [img for img, label in zip(image_paths, cluster_labels) if label == cluster_number][:num_images]
    for i, img_path in enumerate(selected_images):
        img = image.load_img(img_path, target_size=(224, 224))
        axs[i].imshow(img)
        axs[i].axis('off')
    plt.show()

# クラスタ 0 の画像を表示
display_cluster_images(kmeans.labels_, image_paths, cluster_number=0)

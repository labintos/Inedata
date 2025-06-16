import os
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.cluster import KMeans
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# ResNet50を使用して画像から特徴を抽出する関数
def extract_features(image_dir):
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    image_paths = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.lower().endswith(('png', 'jpg', 'jpeg'))]
    features_list = []

    for img_path in tqdm(image_paths, desc="Extracting features"):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        features = model.predict(preprocessed_img)
        flattened_features = features.flatten()
        normalized_features = flattened_features / np.linalg.norm(flattened_features)
        features_list.append(normalized_features)
    return features_list, image_paths

# K-meansクラスタリングを実行し、結果を表示する関数
def cluster_images(features_list, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=22)
    kmeans.fit(features_list)
    return kmeans.labels_

# クラスタに基づく画像の整理
def organize_images_by_cluster(labels, image_paths, output_dir='clusters'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for cluster_id in set(labels):
        cluster_dir = os.path.join(output_dir, f'cluster_{cluster_id}')
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
        
        for label, img_path in zip(labels, image_paths):
            if label == cluster_id:
                img_name = os.path.basename(img_path)
                os.rename(img_path, os.path.join(cluster_dir, img_name))

# 画像のクラスタリングと整理を実行するメイン関数
def main():
    image_dir = 'D:\麦判別AI作成\downloaded_images'  # 画像が保存されているディレクトリのパス
    n_clusters = 10  # 生成するクラスタの数

    # 画像から特徴を抽出
    features_list, image_paths = extract_features(image_dir)

    # 画像をクラスタリング
    labels = cluster_images(features_list, n_clusters)

    # クラスタごとに画像を整理
    organize_images_by_cluster(labels, image_paths)

if __name__ == "__main__":
    main()

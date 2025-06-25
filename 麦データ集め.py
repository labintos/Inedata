import os
import requests
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.cluster import KMeans
from tqdm import tqdm
from io import BytesIO

# 画像をダウンロードして保存する関数
def fetch_and_save_images(api_key, query, max_images=100, download_dir='downloaded_images'):
    URL = "https://pixabay.com/api/"
    params = {
        'key': api_key,
        'q': query,
        'image_type': 'photo',
        'per_page': 100
    }
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    images_downloaded = 0
    page = 1
    while images_downloaded < max_images:
        params['page'] = page
        response = requests.get(URL, params=params)
        data = response.json()
        for img in data['hits']:
            if images_downloaded >= max_images:
                break
            img_response = requests.get(img['webformatURL'])
            img_path = os.path.join(download_dir, f'img_{images_downloaded}.jpg')
            with open(img_path, 'wb') as f:
                f.write(img_response.content)
            images_downloaded += 1
        page += 1

# 特徴抽出とクラスタリング
def cluster_images(image_dir, n_clusters=10):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    image_paths = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
    features_list = []

    for img_path in tqdm(image_paths):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        features = model.predict(preprocessed_img)
        features_list.append(features.flatten())

    kmeans = KMeans(n_clusters=n_clusters, random_state=22)
    kmeans.fit(features_list)
    return kmeans.labels_, image_paths

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

# メイン処理部分
API_KEY = 
query = 'wheat field'
max_images = 500

# 画像のダウンロードと保存
fetch_and_save_images(API_KEY, query, max_images)

# 画像のクラスタリング
labels, paths = cluster_images('downloaded_images', n_clusters=10)

# クラスタごとに画像を整理
organize_images_by_cluster(labels, paths)

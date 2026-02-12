import cv2
import csv
import os
import mediapipe as mp
import copy
import itertools
from hand_detector import HandDetector, calc_landmark_list

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization (MENGGUNAKAN LOGIC YANG SAMA DENGAN JAVASCRIPT)
    # Cari nilai absolut maksimum untuk normalisasi
    max_value = 0
    for v in temp_landmark_list:
        if abs(v) > max_value:
            max_value = abs(v)
    
    if max_value == 0:
        return temp_landmark_list

    return [v / max_value for v in temp_landmark_list]

def process_dataset():
    # Folder dataset lokal
    dataset_path = 'dataset'
    
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"Folder '{dataset_path}' dibuat. Silakan masukkan folder label (A, B, C, dll) di dalamnya.")
        return

    # Output CSV
    csv_path = 'model/keypoint_classifier/keypoint.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Label mapping (berdasarkan nama folder di dalam dataset/)
    labels = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    if not labels:
        print(f"Tidak ada folder label di dalam '{dataset_path}'.")
        return

    label_csv_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'
    with open(label_csv_path, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(f"{label}\n")

    print(f"Memproses dataset dari folder: {dataset_path}")
    print(f"Label yang ditemukan: {labels}")

    detector = HandDetector(num_hands=1)

    with open(csv_path, 'w', newline="") as f:
        writer = csv.writer(f)
        
        for i, label in enumerate(labels):
            label_dir = os.path.join(dataset_path, label)
            print(f"Memproses label {label} ({i+1}/{len(labels)})...")
            
            processed_images = 0
            for img_name in os.listdir(label_dir):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(label_dir, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Data Augmentation:
                # 1. Original
                # 2. Horizontal Flip
                # 3. Rotations (-15, -10, -5, 5, 10, 15 degrees)
                # 4. Brightness & Contrast variants
                
                for flip_mode in [None, 1]:
                    base_img = image_rgb
                    if flip_mode is not None:
                        base_img = cv2.flip(image_rgb, flip_mode)
                    
                    variants = [base_img]
                    
                    # Rotation variants (Lebih banyak sudut)
                    h, w = base_img.shape[:2]
                    center = (w // 2, h // 2)
                    for angle in [-15, -10, -5, 5, 10, 15]:
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        rotated = cv2.warpAffine(base_img, M, (w, h))
                        variants.append(rotated)
                    
                    # Brightness variants
                    variants.append(cv2.convertScaleAbs(base_img, alpha=1.3, beta=20)) # Lebih terang
                    variants.append(cv2.convertScaleAbs(base_img, alpha=0.7, beta=-20)) # Lebih gelap
                    
                    # Contrast variants
                    variants.append(cv2.convertScaleAbs(base_img, alpha=1.5, beta=0)) 

                    for img_variant in variants:
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_variant)
                        detector.results = detector.detector.detect(mp_image)
                        
                        if detector.results and detector.results.hand_landmarks:
                            for hand_landmarks in detector.results.hand_landmarks:
                                landmark_list = calc_landmark_list(img_variant, hand_landmarks)
                                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                                
                                writer.writerow([i, *pre_processed_landmark_list])
                                processed_images += 1
            
            print(f"  Berhasil memproses {processed_images} data (termasuk augmentation) untuk label {label}")

    print("\nPemrosesan dataset selesai.")
    print("Sekarang Anda bisa menjalankan 'python train_model.py' untuk melatih model.")

if __name__ == "__main__":
    process_dataset()

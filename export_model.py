import pickle
import json
import numpy as np
import os

def export_model():
    model_path = 'model/keypoint_classifier/keypoint_classifier.pkl'
    label_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'
    output_path = 'static/model_data.json'

    if not os.path.exists('static'):
        os.makedirs('static')

    if not os.path.exists(model_path):
        print("Model pkl tidak ditemukan!")
        return

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Export Random Forest trees
    forest_data = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        tree_data = {
            'children_left': tree.children_left.tolist(),
            'children_right': tree.children_right.tolist(),
            'threshold': tree.threshold.tolist(),
            'feature': tree.feature.tolist(),
            'value': tree.value.tolist()
        }
        forest_data.append(tree_data)

    # Load labels
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    final_data = {
        'forest': forest_data,
        'labels': labels
    }

    with open(output_path, 'w') as f:
        json.dump(final_data, f)
    
    print(f"Model berhasil di-export ke {output_path}")

if __name__ == '__main__':
    export_model()

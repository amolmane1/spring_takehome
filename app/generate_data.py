#!/usr/bin/env python3
import os
import json
import numpy as np
from sklearn.datasets import fetch_lfw_people
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

def generate_data():
    # Reproducibility
    np.random.seed(42)
    
    # Ensure output directory exists
    os.makedirs('data', exist_ok=True)

    # 1) Load all LFW faces, then pick 100 at random
    lfw = fetch_lfw_people(
        min_faces_per_person=1,
        resize=0.5,
        color=False,
        funneled=True,
        download_if_missing=True
    )
    all_images  = lfw.images    # (n_samples, h, w)
    all_targets = lfw.target   # (n_samples,)
    names_map   = lfw.target_names

    # sample exactly 100 images upfront ---
    n_samples = 100
    chosen_idx = np.random.choice(len(all_images), size=n_samples, replace=False)
    images  = all_images[chosen_idx]
    targets = all_targets[chosen_idx]
    # ---------------------------------------------

    # 2) Set up face detector + embedder
    mtcnn  = MTCNN(image_size=160, margin=0)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    embeddings = []
    names      = []

    # 3) Process exactly those 100 images
    for img_array, tgt in zip(images, targets):
        # grayscale → uint8 RGB PIL
        arr = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(arr).convert('RGB')

        # detect & crop face
        face_t = mtcnn(img)
        if face_t is None:
            # skip if no face found
            continue

        # embed & normalize
        with torch.no_grad():
            emb = resnet(face_t.unsqueeze(0))            # [1,512]
            emb = emb / emb.norm(dim=1, keepdim=True)

        embeddings.append(emb.squeeze().cpu().numpy().tolist())
        names.append(names_map[tgt])

    # 4) Sample net worths for however many embeddings you got
    net_worths = np.random.normal(
        loc=500_000,    # mean $500k
        scale=50_000,  # sd $50k
        size=len(embeddings)
    ).tolist()

    # 5) Write out JSON
    ref_data = {
        "embedding": embeddings,   # list of [512]-length lists
        "name":      names,        # list of strings
        "net worth": net_worths    # list of floats
    }
    with open('data/reference_data.json', 'w') as f:
        json.dump(ref_data, f)

    print(f"Saved {len(embeddings)} entries to data/reference_data.json")

if __name__ == "__main__":
    generate_data()
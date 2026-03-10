import argparse
import shutil
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from annoy import AnnoyIndex
from scipy.cluster.hierarchy import fcluster, linkage
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor

# Main function to process webdataset shards and cluster them based on aggregated video-clip embeddings.
def process_images(image_directory, model_name, threshold, batch_size, num_frames):
    image_directory = Path(image_directory)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings_file = image_directory / "embeddings.npy"
    regenerate_embeddings = check_and_load_embeddings(embeddings_file)

    model = AutoModel.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    allowed_extensions = {".tar"}

    shards_to_paths, all_shard_ids = get_images_to_paths(image_directory, allowed_extensions)
    valid_shard_ids, damaged_shard_ids, all_embeddings = generate_embeddings(
        all_shard_ids,
        shards_to_paths,
        model,
        processor,
        device,
        batch_size,
        num_frames,
        regenerate_embeddings,
        embeddings_file,
    )

    if regenerate_embeddings:
        np.save(embeddings_file, all_embeddings)

    if len(all_embeddings) < 2:
        print("Not enough valid shards to cluster.")
        organize_images(shards_to_paths, image_directory, {}, damaged_shard_ids)
        return

    print("Building Annoy index...")
    annoy_index = build_annoy_index(all_embeddings)

    print("Computing distance matrix...")
    distances = compute_distance_matrix(all_embeddings, annoy_index)

    print("Applying hierarchical clustering...")
    labels = apply_clustering(distances, threshold)

    shard_id_clusters = build_image_clusters(valid_shard_ids, labels)
    organize_images(shards_to_paths, image_directory, shard_id_clusters, damaged_shard_ids)

# Check for existing embeddings file and load it if found, otherwise generate new embeddings
def check_and_load_embeddings(embeddings_file):
    if embeddings_file.exists():
        use_existing_embeddings = input("Embeddings file found. Do you want to use existing embeddings? (Y/N) ").strip().lower()
        if use_existing_embeddings in ("", "y", "yes"):
            print("Loading embeddings from file...")
            return False
    return True

# Get the paths of all tar shards in the given directory and return the shard ids and their paths.
def get_images_to_paths(image_directory, allowed_extensions):
    images_to_paths = {
        image_path.name: image_path
        for image_path in image_directory.iterdir()
        if image_path.suffix.lower() in allowed_extensions
    }
    return images_to_paths, list(images_to_paths.keys())

# Generate one embedding per tar shard by averaging normalized clip embeddings across its video samples.
def generate_embeddings(all_image_ids, images_to_paths, model, processor, device, batch_size, num_frames, regenerate_embeddings, embeddings_file):
    if not regenerate_embeddings:
        return all_image_ids, set(), np.load(embeddings_file)

    valid_image_ids, damaged_image_ids, all_embeddings = [], set(), []
    progress_bar = tqdm(total=len(all_image_ids), desc="Generating shard embeddings")

    for shard_id in all_image_ids:
        shard_embedding = build_shard_embedding(
            images_to_paths[shard_id],
            model,
            processor,
            device,
            batch_size,
            num_frames,
        )
        if shard_embedding is None:
            print(f"\nError processing shard {images_to_paths[shard_id]}, marking as corrupted or empty.")
            damaged_image_ids.add(shard_id)
        else:
            valid_image_ids.append(shard_id)
            all_embeddings.append(shard_embedding)
        progress_bar.update(1)

    progress_bar.close()
    return valid_image_ids, damaged_image_ids, all_embeddings

def build_shard_embedding(shard_path, model, processor, device, batch_size, num_frames):
    clip_embeddings = []

    for clip_frames in iter_shard_videos(shard_path, num_frames):
        clip_embedding = build_clip_embedding(clip_frames, model, processor, device, batch_size)
        if clip_embedding is not None:
            clip_embeddings.append(clip_embedding)

    if not clip_embeddings:
        return None

    shard_tensor = torch.stack(clip_embeddings, dim=0).mean(dim=0)
    shard_tensor = torch.nn.functional.normalize(shard_tensor, p=2, dim=-1)
    return shard_tensor.numpy()


def build_clip_embedding(clip_frames, model, processor, device, batch_size):
    frame_embeddings = []
    for start_idx in range(0, len(clip_frames), batch_size):
        batch_frames = clip_frames[start_idx : start_idx + batch_size]
        inputs = processor(images=batch_frames, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = extract_image_features(model, inputs)

        outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
        frame_embeddings.append(outputs.cpu())

    if not frame_embeddings:
        return None

    clip_tensor = torch.cat(frame_embeddings, dim=0)
    if clip_tensor.shape[0] == 0:
        return None

    return clip_tensor.reshape(-1)


def iter_shard_videos(shard_path, num_frames):
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}

    try:
        with tarfile.open(shard_path, "r") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                if Path(member.name).suffix.lower() not in video_extensions:
                    continue

                extracted = tar.extractfile(member)
                if extracted is None:
                    continue

                try:
                    video_bytes = extracted.read()
                    clip_frames = sample_video_frames(video_bytes, num_frames)
                    if clip_frames:
                        yield clip_frames
                except Exception:
                    continue
    except tarfile.TarError:
        return


def sample_video_frames(video_bytes, num_frames):
    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video_file:
        temp_video_file.write(video_bytes)
        temp_video_file.flush()

        video = cv2.VideoCapture(temp_video_file.name)
        try:
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                return []

            sample_indices = np.linspace(0, total_frames - 1, num=min(num_frames, total_frames), dtype=int)
            frames = []

            for frame_idx in sample_indices:
                video.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                success, frame = video.read()
                if not success:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

            if not frames:
                return []

            while len(frames) < num_frames:
                frames.append(frames[-1].copy())

            return frames
        finally:
            video.release()


def extract_image_features(model, model_inputs):
    if hasattr(model, "get_image_features"):
        return model.get_image_features(**model_inputs)

    outputs = model(**model_inputs)

    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        return outputs.image_embeds

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output

    raise AttributeError("Model does not expose image embeddings via get_image_features, image_embeds, or pooler_output.")

# Build an Annoy index using the generated shard embeddings.
def build_annoy_index(all_embeddings):
    embeddings = np.array(all_embeddings)
    n_dimensions = embeddings.shape[1]

    annoy_index = AnnoyIndex(n_dimensions, "angular")
    for i, embedding in enumerate(embeddings):
        annoy_index.add_item(i, embedding)

    annoy_index.build(100)
    return annoy_index

# Compute the distance matrix of the embeddings using the Annoy index.
def compute_distance_matrix(all_embeddings, annoy_index):
    n = len(all_embeddings)
    distances = []

    for i in range(n):
        for j in range(i + 1, n):
            distance = annoy_index.get_distance(i, j)
            distances.append(distance)

    return distances

# Apply hierarchical clustering on the computed distance matrix with the given threshold.
def apply_clustering(distances, threshold):
    condensed_distances = np.array(distances)
    Z = linkage(condensed_distances, method="average", optimal_ordering=True)
    return fcluster(Z, t=threshold, criterion="distance")

# Build clusters of shard ids based on the clustering labels.
def build_image_clusters(all_image_ids, labels):
    image_id_clusters = defaultdict(set)

    for image_id, cluster_label in zip(all_image_ids, labels):
        image_id_clusters[cluster_label].add(image_id)

    return image_id_clusters

# Organize shards into separate folders for clusters, unique shards, and corrupted shards.
def organize_images(images_to_paths, image_directory, image_id_clusters, damaged_image_ids):
    clustered_image_ids = set()

    for idx, image_id_cluster in enumerate(image_id_clusters.values()):
        if len(image_id_cluster) < 2:
            continue

        clustered_image_ids.update(image_id_cluster)
        move_images_to_directory(image_directory, f"cluster_{idx}", image_id_cluster, images_to_paths)

    unique_image_ids = set(images_to_paths.keys()) - set(damaged_image_ids) - clustered_image_ids
    move_images_to_directory(image_directory, "unique", unique_image_ids, images_to_paths)

    if damaged_image_ids:
        move_images_to_directory(image_directory, "corrupted", damaged_image_ids, images_to_paths)

# Move shards to the specified folder within the image_directory.
def move_images_to_directory(image_directory, folder_name, image_ids, images_to_paths):
    output_directory = image_directory / folder_name
    output_directory.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        source = images_to_paths[image_id]
        destination = output_directory / source.name
        shutil.move(source, destination)

def main():
    parser = argparse.ArgumentParser(description="Finding conceptually similar webdataset tar shards using SigLIP/CLIP-style video-clip embeddings and hierarchical clustering.")
    parser.add_argument("--image_directory", type=str, required=True, help="Path to the directory containing the webdataset .tar shards to cluster.")
    parser.add_argument(
        "--model_name",
        "--clip_model",
        dest="model_name",
        type=str,
        default="google/siglip2-so400m-patch14-384",
        help="Hugging Face vision-language model used to generate per-frame embeddings. SigLIP 2 models such as google/siglip2-so400m-patch14-384 are supported.",
    )
    parser.add_argument("--threshold", type=float, default=0.22, help="Threshold for hierarchical clustering on shard embeddings. Lower values will reduce false positives but may miss more. (default: 0.22)")
    parser.add_argument("--batch_size", type=int, default=192, help="Batch size for encoding sampled video frames. Higher values will require more VRAM. (default: 192)")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of uniformly sampled frames per clip. Each clip embedding is the concatenation of these frame embeddings. (default: 8)")
    args = parser.parse_args()

    process_images(args.image_directory, args.model_name, args.threshold, args.batch_size, args.num_frames)

if __name__ == "__main__":
    main()

import argparse
import json
import os
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from sklearn.cluster import KMeans
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor

# Main function to process webdataset shards and cluster video clips based on clip embeddings.
def process_images(image_directory, model_name, num_clusters, batch_size, num_frames):
    image_directory = Path(image_directory)
    distributed, rank, world_size, local_rank, device = init_distributed()

    embeddings_file = image_directory / "clip_embeddings.npy"
    clip_ids_file = image_directory / "clip_ids.npy"
    corrupted_file = image_directory / "corrupted_shards.json"
    allowed_extensions = {".tar"}
    shards_to_paths, all_shard_ids = get_images_to_paths(image_directory, allowed_extensions)

    regenerate_embeddings = check_and_load_embeddings(embeddings_file, rank)
    valid_clip_ids = []
    damaged_shard_ids = set()
    all_embeddings = None

    if regenerate_embeddings:
        model = AutoModel.from_pretrained(model_name).to(device)
        if distributed and device.type == "cuda":
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        model.eval()
        processor = AutoProcessor.from_pretrained(model_name)

        local_shard_ids = all_shard_ids[rank::world_size]
        local_clip_ids, local_damaged_shard_ids, local_embeddings = generate_embeddings(
            local_shard_ids,
            shards_to_paths,
            model,
            processor,
            device,
            batch_size,
            num_frames,
        )

        valid_clip_ids, damaged_shard_ids, all_embeddings = gather_embedding_results(
            local_clip_ids,
            local_damaged_shard_ids,
            local_embeddings,
            distributed,
            rank,
            world_size,
        )

        if rank == 0:
            np.save(embeddings_file, all_embeddings)
            np.save(clip_ids_file, np.asarray(valid_clip_ids))
            corrupted_file.write_text(json.dumps(sorted(damaged_shard_ids), indent=2))
            all_embeddings = np.load(embeddings_file, mmap_mode="r")
    elif rank == 0:
        valid_clip_ids = np.load(clip_ids_file).tolist()
        damaged_shard_ids = set(json.loads(corrupted_file.read_text())) if corrupted_file.exists() else set()
        all_embeddings = np.load(embeddings_file, mmap_mode="r")

    if distributed:
        dist.barrier()

    if rank != 0:
        cleanup_distributed(distributed)
        return

    if len(all_embeddings) < 2:
        print("Not enough valid clips to cluster.")
        write_cluster_manifests(image_directory, {}, damaged_shard_ids)
        cleanup_distributed(distributed)
        return

    print("Applying k-means clustering...")
    labels = apply_clustering(all_embeddings, num_clusters)

    clip_id_clusters = build_image_clusters(valid_clip_ids, labels)
    write_cluster_manifests(image_directory, clip_id_clusters, damaged_shard_ids)
    cleanup_distributed(distributed)

# Check for existing embeddings file and load it if found, otherwise generate new embeddings
def check_and_load_embeddings(embeddings_file, rank):
    reuse_embeddings = False
    if rank == 0 and embeddings_file.exists():
        use_existing_embeddings = input("Embeddings file found. Do you want to use existing embeddings? (Y/N) ").strip().lower()
        reuse_embeddings = use_existing_embeddings in ("", "y", "yes")

    if dist.is_available() and dist.is_initialized():
        decisions = [reuse_embeddings]
        dist.broadcast_object_list(decisions, src=0)
        reuse_embeddings = decisions[0]

    if reuse_embeddings:
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

# Generate one embedding per video clip found inside the assigned tar shards.
def generate_embeddings(all_image_ids, images_to_paths, model, processor, device, batch_size, num_frames):
    valid_clip_ids, damaged_image_ids, all_embeddings = [], set(), []
    show_progress = not (dist.is_available() and dist.is_initialized() and dist.get_rank() != 0)
    progress_bar = tqdm(total=len(all_image_ids), desc="Generating clip embeddings", disable=not show_progress)

    for shard_id in all_image_ids:
        clip_records = build_shard_embedding(
            shard_id,
            images_to_paths[shard_id],
            model,
            processor,
            device,
            batch_size,
            num_frames,
        )
        if not clip_records:
            print(f"\nError processing shard {images_to_paths[shard_id]}, marking as corrupted or empty.")
            damaged_image_ids.add(shard_id)
        else:
            for clip_id, clip_embedding in clip_records:
                valid_clip_ids.append(clip_id)
                all_embeddings.append(clip_embedding)
        progress_bar.update(1)

    progress_bar.close()
    return valid_clip_ids, damaged_image_ids, all_embeddings

def build_shard_embedding(shard_id, shard_path, model, processor, device, batch_size, num_frames):
    clip_records = []

    for clip_name, clip_frames in iter_shard_videos(shard_path, num_frames):
        clip_embedding = build_clip_embedding(clip_frames, model, processor, device, batch_size)
        if clip_embedding is not None:
            clip_records.append((f"{shard_id}::{clip_name}", clip_embedding.numpy()))

    return clip_records


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

    return clip_tensor.reshape(-1).to(dtype=torch.float32)


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
                        yield member.name, clip_frames
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
    if isinstance(model, DDP):
        model = model.module

    if hasattr(model, "get_image_features"):
        return model.get_image_features(**model_inputs)

    outputs = model(**model_inputs)

    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        return outputs.image_embeds

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output

    raise AttributeError("Model does not expose image embeddings via get_image_features, image_embeds, or pooler_output.")

# Apply k-means directly on clip embeddings.
def apply_clustering(all_embeddings, num_clusters):
    embeddings = np.asarray(all_embeddings, dtype=np.float32)
    effective_clusters = min(num_clusters, len(embeddings))
    if effective_clusters < 1:
        return np.array([], dtype=np.int32)

    model = KMeans(n_clusters=effective_clusters, random_state=0, n_init="auto")
    return model.fit_predict(embeddings)

# Build clusters of clip ids based on the clustering labels.
def build_image_clusters(all_image_ids, labels):
    image_id_clusters = defaultdict(set)

    for image_id, cluster_label in zip(all_image_ids, labels):
        image_id_clusters[int(cluster_label)].add(image_id)

    return image_id_clusters

def write_cluster_manifests(image_directory, image_id_clusters, damaged_image_ids):
    output_directory = image_directory / "clip_clusters"
    output_directory.mkdir(parents=True, exist_ok=True)

    for cluster_idx, clip_ids in image_id_clusters.items():
        manifest_path = output_directory / f"cluster_{cluster_idx}.txt"
        manifest_path.write_text("\n".join(sorted(clip_ids)) + "\n" if clip_ids else "")

    corrupted_manifest = output_directory / "corrupted_shards.txt"
    corrupted_manifest.write_text("\n".join(sorted(damaged_image_ids)) + "\n" if damaged_image_ids else "")


def init_distributed():
    env_keys = {"RANK", "WORLD_SIZE", "LOCAL_RANK"}
    if not dist.is_available() or not env_keys.issubset(os.environ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, 0, device

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return True, rank, world_size, local_rank, device


def cleanup_distributed(distributed):
    if distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def gather_embedding_results(local_clip_ids, local_damaged_shard_ids, local_embeddings, distributed, rank, world_size):
    if not distributed:
        return local_clip_ids, local_damaged_shard_ids, np.asarray(local_embeddings, dtype=np.float32)

    gathered_results = [None] * world_size
    dist.all_gather_object(
        gathered_results,
        {
            "clip_ids": local_clip_ids,
            "damaged_ids": list(local_damaged_shard_ids),
            "embeddings": local_embeddings,
        },
    )

    if rank != 0:
        return [], set(), None

    clip_ids = []
    embeddings = []
    damaged_ids = set()
    for result in gathered_results:
        damaged_ids.update(result["damaged_ids"])
        clip_ids.extend(result["clip_ids"])
        embeddings.extend(result["embeddings"])

    return clip_ids, damaged_ids, np.asarray(embeddings, dtype=np.float32)



def main():
    parser = argparse.ArgumentParser(description="Finding conceptually similar video clips from webdataset tar shards using SigLIP/CLIP-style clip embeddings and k-means clustering.")
    parser.add_argument("--image_directory", type=str, required=True, help="Path to the directory containing the webdataset .tar shards to cluster.")
    parser.add_argument(
        "--model_name",
        "--clip_model",
        dest="model_name",
        type=str,
        default="google/siglip2-so400m-patch14-384",
        help="Hugging Face vision-language model used to generate per-frame embeddings. SigLIP 2 models such as google/siglip2-so400m-patch14-384 are supported.",
    )
    parser.add_argument("--num_clusters", type=int, default=100, help="Number of k-means clusters to fit over shard embeddings. (default: 10)")
    parser.add_argument("--batch_size", type=int, default=192, help="Batch size for encoding sampled video frames. Higher values will require more VRAM. (default: 192)")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of uniformly sampled frames per clip. Each clip embedding is the concatenation of these frame embeddings. (default: 8)")
    args = parser.parse_args()

    process_images(args.image_directory, args.model_name, args.num_clusters, args.batch_size, args.num_frames)

if __name__ == "__main__":
    main()

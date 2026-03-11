import argparse
import json
import tarfile
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor

# Main function to process webdataset shards and save video-clip embeddings.
def process_images(image_directory, model_name, batch_size, num_frames):
    image_directory = Path(image_directory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddings_file = image_directory / "clip_embeddings.npy"
    clip_ids_file = image_directory / "clip_ids.npy"
    corrupted_file = image_directory / "corrupted_shards.json"
    allowed_extensions = {".tar"}
    shards_to_paths, all_shard_ids = get_images_to_paths(image_directory, allowed_extensions)

    regenerate_embeddings = check_and_load_embeddings(embeddings_file)
    if not regenerate_embeddings:
        return

    model = AutoModel.from_pretrained(model_name).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)

    clip_ids, damaged_shard_ids, all_embeddings = generate_embeddings(
        all_shard_ids,
        shards_to_paths,
        model,
        processor,
        device,
        batch_size,
        num_frames,
    )

    np.save(embeddings_file, np.asarray(all_embeddings, dtype=np.float32))
    np.save(clip_ids_file, np.asarray(clip_ids))
    corrupted_file.write_text(json.dumps(sorted(damaged_shard_ids), indent=2))

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

# Generate one embedding per video clip found inside the assigned tar shards.
def generate_embeddings(all_image_ids, images_to_paths, model, processor, device, batch_size, num_frames):
    valid_clip_ids, damaged_image_ids, all_embeddings = [], set(), []
    progress_bar = tqdm(total=len(all_image_ids), desc="Generating clip embeddings")

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
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    if hasattr(model, "get_image_features"):
        return model.get_image_features(**model_inputs)

    outputs = model(**model_inputs)

    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        return outputs.image_embeds

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output

    raise AttributeError("Model does not expose image embeddings via get_image_features, image_embeds, or pooler_output.")

def main():
    parser = argparse.ArgumentParser(description="Extract clip embeddings from webdataset tar shards using SigLIP/CLIP-style frame encoders.")
    parser.add_argument("--image_directory", type=str, required=True, help="Path to the directory containing the webdataset .tar shards to process.")
    parser.add_argument(
        "--model_name",
        "--clip_model",
        dest="model_name",
        type=str,
        default="google/siglip2-so400m-patch14-384",
        help="Hugging Face vision-language model used to generate per-frame embeddings. SigLIP 2 models such as google/siglip2-so400m-patch14-384 are supported.",
    )
    parser.add_argument("--batch_size", type=int, default=192, help="Batch size for encoding sampled video frames. Higher values will require more VRAM. (default: 192)")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of uniformly sampled frames per clip. Each clip embedding is the concatenation of these frame embeddings. (default: 8)")
    args = parser.parse_args()

    process_images(args.image_directory, args.model_name, args.batch_size, args.num_frames)

if __name__ == "__main__":
    main()

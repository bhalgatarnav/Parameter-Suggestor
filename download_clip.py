import os
import requests
import hashlib
from tqdm import tqdm

CLIP_MODEL_URLS = {
    'config.json': 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32-config.json',
    'pytorch_model.bin': 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt',
    'merges.txt': 'https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz',
    'vocab.json': 'https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz'
}

CLIP_MODEL_HASHES = {
    'config.json': '4be9880e09d7c8790a7f6ae8c1663a35a9d3b6223a3f6f5d34da27ff53362b54',
    'pytorch_model.bin': '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af',
    'merges.txt': '9fd0aad4ff415e043683d25327725b3d',
    'vocab.json': '9fd0aad4ff415e043683d25327725b3d'
}

def calculate_hash(filepath):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_file(url, filepath):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file, tqdm(
        desc=os.path.basename(filepath),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def main():
    # Create clip-vit-b32 directory if it doesn't exist
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'clip-vit-b32')
    os.makedirs(model_dir, exist_ok=True)
    
    print("Downloading CLIP model files...")
    
    for filename, url in CLIP_MODEL_URLS.items():
        filepath = os.path.join(model_dir, filename)
        print(f"\nDownloading {filename}...")
        
        # Download file
        download_file(url, filepath)
        
        # Verify hash
        print(f"Verifying {filename}...")
        file_hash = calculate_hash(filepath)
        expected_hash = CLIP_MODEL_HASHES[filename]
        
        if file_hash != expected_hash:
            print(f"Hash mismatch for {filename}!")
            print(f"Expected: {expected_hash}")
            print(f"Got: {file_hash}")
            os.remove(filepath)
            print(f"Deleted corrupted file: {filename}")
        else:
            print(f"Successfully downloaded and verified: {filename}")

if __name__ == "__main__":
    main() 
import os
import urllib.request
from tqdm import tqdm
import zipfile
from torch_utils import distributed as dist

urls = {
    "cifar10": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl",
    "ffhq": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl",
    "afhqv2": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-vp.pkl",
    "imagenet64": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl",
    "lsun_bedroom": "https://openaipublic.blob.core.windows.net/consistency/edm_bedroom256_ema.pt",
    "imagenet256": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt",
    "imagenet256-classifier": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt",
    "lsun_bedroom_ldm": "https://ommer-lab.com/files/latent-diffusion/lsun_bedrooms.zip",
    "ffhq_ldm": "https://ommer-lab.com/files/latent-diffusion/ffhq.zip",
    "vq-f4": "https://ommer-lab.com/files/latent-diffusion/vq-f4.zip",
    "ms_coco": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt",
    "prompts": "https://github.com/boomb0om/text2image-benchmark/releases/download/v0.0.1/MS-COCO_val2014_30k_captions.csv",
}

#----------------------------------------------------------------------------
# Search the model file

def search_local_model(key, subsubdir="src", key_extra=None):
    contents = os.listdir('../')
    subdirs = [item for item in contents if os.path.isdir(os.path.join('../', item))]
    url = urls[key] if key_extra is None else urls[key_extra]

    exist_local_model = False
    for subdir in subdirs:
        if key_extra == 'vq-f4':
            target_dir = os.path.join('../', subdir, subsubdir)
        else:
            target_dir = os.path.join('../', subdir, subsubdir, key)

        if os.path.exists(target_dir):
            download_path = model_path = os.path.join(target_dir, url.split("/")[-1])
            if download_path.endswith(".zip"):        # for lsun_bedroom_ldm and ffhq_ldm
                model_path = os.path.join(target_dir, 'model.ckpt')

            if os.path.exists(model_path):
                exist_local_model = True
                return exist_local_model, download_path, model_path 

    download_path = os.path.join('./', subsubdir, key, url.split("/")[-1])
    return exist_local_model, download_path, None

#----------------------------------------------------------------------------
# Download the model file and unzip it if it is a zip file

def download_model(url, download_path):
    target_dir = os.path.dirname(download_path)
    os.makedirs(target_dir, exist_ok=True)
    download_with_url(url, download_path)
    if download_path.endswith(".zip"):
        try:
            unzip_file(download_path, target_dir)
            os.remove(download_path)
        except:
            raise ValueError(f"Fail to unzip the file: {download_path}")

#----------------------------------------------------------------------------
        
def download_with_url(url, target_path):
    req = urllib.request.urlopen(url)
    total_size = int(req.getheader('Content-Length').strip())
    with open(target_path, 'wb') as file, tqdm(unit='B', unit_scale=True, unit_divisor=1024, total=total_size, desc=target_path) as bar:
        urllib.request.urlretrieve(url, target_path, reporthook=lambda block_num, block_size, total_size: bar.update(block_size))

#----------------------------------------------------------------------------
        
def unzip_file(file_path, target_dir):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

#----------------------------------------------------------------------------
# Check the existence of the model file and download it if it does not exist

def check_file_by_key(key, subsubdir="src"):

    if key not in urls:
        raise ValueError(f"Unknown key: {key}")

    exist_local_model, download_path, model_path = search_local_model(key, subsubdir)
    if exist_local_model:
        dist.print0(f'Model already exists: {model_path}')
    else:
        url = urls[key]
        dist.print0(f'File does not exist, downloading from {url}')
        download_model(url, download_path)

    # Check addtional models such as the classifier and vq_f4 model
    model_path_extra = None
    if key == "imagenet256":    # check the classifier
        key_extra = "imagenet256-classifier"
        exist_local_model, download_path, model_path_extra = search_local_model(key, subsubdir, key_extra)
        if exist_local_model:
            dist.print0(f'The classifier already exists: {model_path_extra}')
        else:
            url = urls[key_extra]
            dist.print0(f'The classifier does not exist, downloading from {url}')
            download_model(url, download_path)
    elif key in ["lsun_bedroom_ldm", "ffhq_ldm"]:    # check the vq_f4 model
        key_extra = "vq-f4"
        subsubdir = "models/ldm_models/first_stage_models/vq-f4"
        exist_local_model, download_path, model_path_extra = search_local_model(key, subsubdir, key_extra)
        if exist_local_model:
            dist.print0(f'The vq-f4 model already exists: {model_path_extra}')
        else:
            url = urls[key_extra]
            dist.print0(f'The vq-f4 model does not exist, downloading from {url}')
            download_model(url, download_path)

    return model_path, model_path_extra

#----------------------------------------------------------------------------
# Search and download Inception-v3 model

def find_inception_model(max_depth=5):
    """
    Search for inception-2015-12-05.pkl in current directory and parent directories.
    
    Args:
        max_depth: Maximum depth to search (default: 3)
        
    Returns:
        Path to the inception model file
    """
    import pickle
    
    model_filename = 'inception-2015-12-05.pkl'
    inception_url = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/inception-2015-12-05.pkl'
    
    # Search in current and parent directories up to max_depth levels
    current_dir = os.path.abspath('.')
    
    # Search strategy: check current dir and its subdirectories, then go up
    search_paths = []
    
    # Add current directory and subdirectories
    for root, dirs, files in os.walk(current_dir):
        depth = root[len(current_dir):].count(os.sep)
        if depth <= max_depth:
            if model_filename in files:
                found_path = os.path.join(root, model_filename)
                dist.print0(f'Found Inception model at: {found_path}')
                return found_path
    
    # Search parent directories
    for level in range(1, max_depth + 1):
        parent_dir = os.path.abspath(os.path.join(current_dir, '../' * level))
        for root, dirs, files in os.walk(parent_dir):
            depth = root[len(parent_dir):].count(os.sep)
            if depth <= max_depth:
                if model_filename in files:
                    found_path = os.path.join(root, model_filename)
                    dist.print0(f'Found Inception model at: {found_path}')
                    return found_path
    
    # If not found, download it
    dist.print0(f'Inception model not found locally. Downloading from {inception_url}')
    download_dir = os.path.join(current_dir, 'pretrained_models')
    os.makedirs(download_dir, exist_ok=True)
    download_path = os.path.join(download_dir, model_filename)
    
    if not os.path.exists(download_path):
        download_with_url(inception_url, download_path)
        
        # Verify the downloaded file is valid
        try:
            with open(download_path, 'rb') as f:
                pickle.load(f)
            dist.print0(f'Successfully downloaded and verified Inception model at: {download_path}')
        except Exception as e:
            os.remove(download_path)
            raise ValueError(f"Downloaded file is corrupted: {e}")
    
    return download_path

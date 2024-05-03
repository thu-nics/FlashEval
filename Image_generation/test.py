from huggingface_hub import snapshot_download
proxies={'http': 'http://127.0.0.1:10807', 'https': 'http://127.0.0.1:10807'}
snapshot_download(repo_id="stabilityai/stable-diffusion-xl-base-0.9",cache_dir="./cache", local_dir="~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-0.9/snapshots/ccb3e0a2bfc06b2c27b38c54684074972c365258", proxies=proxies)
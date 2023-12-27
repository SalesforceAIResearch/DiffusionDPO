from huggingface_hub import upload_folder, upload_file
import sys

d = 'path/to/local/ckpt/unet/folder'
for fname in ['config.json', 'diffusion_pytorch_model.safetensors']:
    print(f"Uploading {fname}...")
    upload_file(path_or_fileobj=d + fname,
                path_in_repo=fname,
                repo_id = "<hf-org>/<make a repo and add it here>", # please don't overwrite other files in repo
                token=sys.argv[1] # Add your HF token here or just feed in via command line
               )

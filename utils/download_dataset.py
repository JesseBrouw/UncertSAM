import argparse
import os
import subprocess
import tarfile

DATASET_URL = "https://huggingface.co/datasets/JesseBrouw/UncertSAM/resolve/main/"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, default='./data')
    return parser.parse_args()

def main(cache_dir:str):
    
    os.makedirs(cache_dir, exist_ok=True)
    
    datasets = ['ISTD', 'MSD_Spleen', 'Trans10K', 'COD10K-v3', 'COIFT', 'BIG', 'SBU-shadow', 'Flare7K']
    hf_token = os.environ['HF_TOKEN']
    
    for d in datasets:
        tar_names = [f'{d}/train.tar.gz', f'{d}/test.tar.gz', f'{d}/val.tar.gz']
        
        os.makedirs(os.path.join(cache_dir, d), exist_ok=True)
        
        for tar_name in tar_names:
            target_location = os.path.join(cache_dir, tar_name) 

            subprocess.run(['wget', '--header', f'Authorization: Bearer {hf_token}', DATASET_URL+tar_name, '-O', target_location])

        print(f"File downloaded successfully to: {target_location}")
        
        with tarfile.open(os.path.join(cache_dir, d, 'train.tar.gz'), "r:gz") as tar:
            tar.extractall(path=os.path.join(cache_dir, d))
        
        with tarfile.open(os.path.join(cache_dir, d, 'val.tar.gz'), "r:gz") as tar:
            tar.extractall(path=os.path.join(cache_dir, d))
        
        with tarfile.open(os.path.join(cache_dir, d, 'test.tar.gz'), "r:gz") as tar:
            tar.extractall(path=os.path.join(cache_dir, d))
    
if __name__ == '__main__':
    args = parse_args()
    main(args.cache_dir)

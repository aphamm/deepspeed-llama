## Finetune a LLM with DeepSpeed on Kubernetes Cluster 🚀

### Create a VM in GCP

```bash
$ python3 create_vm.py --project_id="high-performance-ml" --vm_name="sleds" --disk_size=200 --gpu_type="nvidia-tesla-t4" --gpu_count=4 --machine_type="n1-standard-8"
```

### Install Dependencies

```bash
sudo apt-get install libaio-dev
pip install -r requirements.txt
```

### Convert GitHub Repo to Hugging Face Instruct Dataset

Create a personal access token with the necessary scopes (e.g., repo for accessing repositories) on [GitHub Developer settings](https://github.com/settings/tokens). Save to a file named `config.py`. 

```python
# config.py
ACCESS_TOKEN = "ghp_Br6gWv..."
HF_USERNAME = "aphamm"
```

Use the `extract_repo.py` script to save the contents and respective metadata to a text file, which will be used for finetuning.

```bash
python extract_repo.py --repo_url="https://github.com/modal-labs/modal-client" --create=True
```

### Finetune Llama-2 

```bash
wandb login
git config --global credential.helper store
huggingface-cli login
python finetune.py --repo="modal-client" --batch_size=16 --num_steps=400 --ds_config="config/stage1.json"
```

### Inference

```bash
python inference.py --model_path="aphamm/finetune2"
```

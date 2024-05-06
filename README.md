## Finetune a LLM with DeepSpeed on Kubernetes Cluster 🚀

### Create a VM in GCP

```bash
gcloud auth login
python3 create_vm.py --project_id="high-performance-ml" --vm_name="sleds" --disk_size=200 --gpu_type="nvidia-tesla-t4" --gpu_count=4 --machine_type="n1-standard-8"
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
python finetune.py --repo="modal-client" --batch_size=16 --num_steps=400 --ds_config="config/stage2.json"
python finetune.py --repo="modal-client" --batch_size=16 --num_steps=400 --ds_config="config/stage3.json"
```

### Inference

```bash
python inference.py --model_path="aphamm/stage1" --user_input="In file tasks.py, create a function with declaration: @task def protoc(ctx)."
python inference.py --model_path="arnifm/stage2" --user_input="In file tasks.py, create a PyTorch container image to do ResNet training on CIFAR10."
```

### Test Deployment

```bash
python app.py
```

### Inference using Kubernetes

Install kubectl on macOS

```bash
brew install kubernetes-cli
```

Install Google Cloud SDK

```bash
$ brew install --cask google-cloud-sdk
```

[Install Docker](https://docs.docker.com/desktop/install/mac-install/)


### Create Cluster via Google Kubernetes Engine

Launch Kubernetes Cluster with GPUS

```bash
gcloud container clusters create sleds --num-nodes=2 \
 --zone=asia-east1-c --machine-type="n1-standard-8" \
 --accelerator="type=nvidia-tesla-t4,count=1,gpu-sharing-strategy=time-sharing,max-shared-clients-per-gpu=2" \
 --scopes="gke-default,storage-rw"
```
Set activate project ID to GCP project

```bash
gcloud config set project high-performance-ml
```

Get the cluster credentials and configure kubectl

```bash
gcloud container clusters get-credentials sleds --zone asia-east1-c
```

Verify kubectl properly connected to your cluster

```bash
kubectl get nodes
```

Install the NVIDIA Drivers

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml
```

### Build Docker Image

Log in to Docker Hub

```bash
docker login
```

Build image and push to Hub

```bash
docker build -t austinphamm/deepspeed-inference:v1 -f ./Dockerfile .
docker push austinphamm/deepspeed-inference:v1
```

### Deploy K8s Service

```bash
kubectl apply -f inference/inference.yaml
```

deployment.apps/inference-app created
service/inference-service created

```bash
kubectl logs -f inference-app-7996dbbd86-t8nmn
kubectl get services  
```

NAME TYPE CLUSTER-IP EXTERNAL-IP PORT(S) AGE
inference-service LoadBalancer 10.22.88.22 104.199.180.199 80:30560/TCP 56s
kubernetes ClusterIP 10.22.80.1 <none> 443/TCP 10h

### Cleanup resources

```bash
kubectl delete deployment inference-app && kubectl delete service inference-service
gcloud container clusters delete sleds --zone=asia-east1-c
```
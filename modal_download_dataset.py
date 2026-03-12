import json
import modal
import os
from pathlib import Path
import subprocess

# 1. Build an image with the Google Cloud CLI installed
image = (
    modal.Image.debian_slim()
    .apt_install("curl", "apt-transport-https", "ca-certificates", "gnupg")
    .run_commands(
        "echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main' | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list",
        "curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -",
        "apt-get update && apt-get install -y google-cloud-cli"
    )
)

app = modal.App("dataset-downloader")
dataset_volume = modal.Volume.from_name("llama3-dataset-vol", create_if_missing=True)

# 2. Force the function into a US region to get the $0.01/GB routing rate
@app.function(
    image=image,
    region="us-central1", # Matches the continent of the multi-region bucket
    volumes={"/data": dataset_volume},
    secrets=[modal.Secret.from_name("gcp-requester-pays-secret")], 
    timeout=86400 # Give it 24 hours to ensure it doesn't timeout
)
def download_dataset():
    # 3. Materialize credentials inside the container from Modal Secret env vars.
    credentials_json = os.environ["GCP_CREDENTIALS_JSON"]
    billing_project = os.environ["GCP_BILLING_PROJECT"]
    credentials_path = Path("/tmp/gcp-credentials.json")
    credentials_path.write_text(credentials_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)

    creds = json.loads(credentials_json)
    boto_path = Path("/tmp/.boto")
    if creds.get("type") == "service_account":
        subprocess.run([
            "gcloud", "auth", "activate-service-account",
            f"--key-file={credentials_path}"
        ], check=True)
        boto_path.write_text("[GSUtil]\nsliced_object_download_threshold = 0\n")
    else:
        boto_path.write_text(
            f"[Credentials]\ngs_oauth2_refresh_token = {creds['refresh_token']}\n\n"
            f"[OAuth2]\nclient_id = {creds['client_id']}\nclient_secret = {creds['client_secret']}\n\n"
            f"[GSUtil]\nsliced_object_download_threshold = 0\n"
        )
    os.environ["BOTO_CONFIG"] = str(boto_path)

    print("Starting 1.1 TB transfer...")
    subprocess.run([
        "gsutil", "-m",
        "-u", billing_project,
        "cp", "-R",
        "gs://llama3-dclm-filter-8k/data.zarr", "/data",
    ], check=True)
    
    # 5. Save the state of the volume
    dataset_volume.commit()
    print("Download complete and persisted to Modal!")
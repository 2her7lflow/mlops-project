# Run on your machine
pip install "dvc[gcs]"
dvc init
dvc remote add -d gcsremote gs://YOUR_BUCKET/mlops-pet-rag
# Optional:
# dvc remote modify gcsremote credentialpath C:\path\to\gcp-sa.json
dvc add knowledge_base/raw
dvc add knowledge_base/processed
dvc add knowledge_base/feedback
git add .dvc .dvcignore dvc.yaml params.yaml *.dvc
git commit -m "chore: enable dvc + gcs remote"
dvc push

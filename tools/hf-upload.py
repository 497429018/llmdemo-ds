from huggingface_hub import HfApi

api = HfApi(token="*******")
api.upload_folder(
    folder_path="output/qwen2.5-14b-ds-demo",
    repo_id="wu1124/ds-100",
    repo_type="model"
)
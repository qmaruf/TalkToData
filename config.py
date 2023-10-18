from pathlib import Path


class Config:
    chunk_size = 1000
    chunk_overlap = 100
    vectorstore_dir = 'docs'
    vectorstore_path = f'{vectorstore_dir}/vectorstore.pkl'
    chatgpt_model_name = 'gpt-3.5-turbo'


Path(Config.vectorstore_dir).unlink(missing_ok=True)

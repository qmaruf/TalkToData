from pathlib import Path
Path('docs').mkdir(parents=True, exist_ok=True)

class Config:
	chunk_size = 1000
	chunk_overlap = 50
	vectordb_persist_directory = 'docs/chroma'
	vectorstore_path = 'docs/vectorstore.pkl'
	chatgpt_model_name = 'gpt-3.5-turbo'
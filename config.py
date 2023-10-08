from pathlib import Path
Path('docs').mkdir(parents=True, exist_ok=True)

class Config:
	chunk_size = 500
	chunk_overlap = 50
	vectordb_persist_directory = 'docs/chroma'
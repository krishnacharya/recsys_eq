from pathlib import Path

def project_root():
    current_dir = Path(__file__).absolute().parent.parent
    return current_dir

def emb_root():
    res = project_root()/"saved_embeddings"
    res.mkdir(parents=True, exist_ok=True)
    return res

def emb_dataset(dataset:str):
    res = emb_root() / dataset
    res.mkdir(parents=True, exist_ok=True)
    return res

def emb_dataset_alg(dataset:str, alg='nmf'):
    res = emb_root() / dataset / alg
    res.mkdir(parents=True, exist_ok=True)
    return res

def game_dir():
    return project_root() / "game"


import yaml

def load_config(name:str) -> dict:
    with open(name, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded

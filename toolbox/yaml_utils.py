import yaml


def dump_yaml(data, file_path):
    with open(file_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data

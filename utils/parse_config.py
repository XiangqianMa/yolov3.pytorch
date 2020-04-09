import json


def parse_config(config_file):
    with open(config_file, 'r') as json_file:
        config = json.load(json_file)

    return config


if __name__ == '__main__':
    config_file = "config.json"
    parse_config(config_file)

import json

_test_options = {
    "use_sample": False
}

def _load_config():
    if _test_options["use_sample"]:
        config_file = open("conf/analysis/labels.conf.json.sample")
    else:
        try:
            config_file = open("conf/analysis/labels.conf.json")
        except FileNotFoundError:
            print("labels.conf.json not configured, falling back to sample, default configuration")
            config_file = open("conf/analysis/labels.conf.json.sample")
    config_data = json.load(config_file)
    config_file.close()
    return config_data

labels = _load_config()

def reload_config():
    global labels
    labels = _load_config()

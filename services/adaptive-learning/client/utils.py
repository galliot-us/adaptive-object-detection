from configparser import ConfigParser
from collections import defaultdict
import json


def ini2json(config_path):
    '''
    Convert .ini config file to json
    :param config_path: The path of .ini config file
    :return:
    '''
    config = ConfigParser()
    config.optionxform = str
    config.read_file(open(config_path))
    config_dict = defaultdict(dict)
    for sections in config.sections():
        for key, value in config.items(sections):
            config_dict[sections][key] = value

    return json.dumps(config_dict)


def json2ini(cfg_json):
    '''
    Convert json to .ini file formats
    :param cfg_json: Json format content
    :return:
    '''
    config_parser = ConfigParser()
    config_parser.optionxform = str
    for section in cfg_json.keys():
        section_json = cfg_json[section]
        config_parser[section] = {}
        for key in section_json.keys():
            config_parser[section][key] = section_json[key]

    return config_parser

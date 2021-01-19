from configs.config_engine import ConfigEngine
from argparse import ArgumentParser
import json
import urllib.request
from pprint import pprint

def send_config(config):
    config_json = json.dumps(config.get_section_dict('Teacher')).encode('utf-8')
    headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    }
    url = 'https://httpbin.org/post ' #config.get_section_dict('Server')['ServerPath']
    try:
        req = urllib.request.Request(url, config_json, headers)
        with urllib.request.urlopen(req) as f:
            res = f.read()
        print(res.decode())
    except Exception as e:
        print(e)
    
def get_model(config):
    url = 'https://httpbin.org/get'
    with urllib.request.urlopen(url) as response:
        html = response.read()
    pprint(html)

def main():
    argparse = ArgumentParser()
    argparse.add_argument('--config', type=str, help='config file path', default='configs/config-x86.ini')
    args = argparse.parse_args()
    config_path = args.config
    config = ConfigEngine(config_path)
    #send_config(config)
    #get_config(config)
    get_model(config)


if __name__ == "__main__":
    main()

import json
import urllib.request
from pprint import pprint


class Request():
    def __init__(self, config):
        self.config = self.config

    def send_config(self):
        config_json = json.dumps(self.config.get_section_dict('Teacher')).encode('utf-8')
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        url = 'https://httpbin.org/post '  # self.config.get_section_dict('Server')['ServerPath']
        try:
            req = urllib.request.Request(url, config_json, headers)
            with urllib.request.urlopen(req) as f:
                res = f.read()
            print(res.decode())
        except Exception as e:
            print(e)

    def get_model(self):
        url = 'https://httpbin.org/get'
        with urllib.request.urlopen(url) as response:
            html = response.read()
        pprint(html)

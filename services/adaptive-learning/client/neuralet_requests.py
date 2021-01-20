import json
import urllib.request
from pprint import pprint


class NeuraletRequest():

    @staticmethod
    def send_config():
        config_json = json.dumps(config.get_section_dict('Teacher')).encode('utf-8')
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

    @staticmethod
    def get_model(task_id):
        url = 'https://httpbin.org/get' + task_id
        with urllib.request.urlopen(url) as response:
            html = response.read()
        pprint(html)

    @staticmethod
    def get_task_status(task_id):
        url = 'https://httpbin.org/get' + task_id
        with urllib.request.urlopen(url) as response:
            html = response.read()
        pprint(html)

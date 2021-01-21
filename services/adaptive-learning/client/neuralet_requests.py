import json
import urllib.request
from pprint import pprint
import requests


class NeuraletRequest():

    @staticmethod
    def send_config(config):
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

    @staticmethod
    def send_file(file, file_name):
        url = "https://api.neuralet.io/api/v1/upload/"
        headers = {
            "file": str(file_name)
        }
        print("Uploading file, please wait it may take a while ...!")
        r = requests.post(url, headers=headers, files=file)
        if r.status_code == 200
            uploaded_file_name = r.json()['filename']
            print(f'The file is uploaded successfully. Uploaded filename = {uploaded_file_name}')
            return uploaded_file_name
        else:
            print(f'ERROR! ({r.status_code})')
            return None

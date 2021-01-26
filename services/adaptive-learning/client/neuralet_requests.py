import requests
from utils import ini2json
import os
import json


class NeuraletRequest():
    """
    Provides a wrapper for Neuralet Adaptive Learning API
    """

    @staticmethod
    def send_config(config_path, server_address='https://api.neuralet.io'):
        config_json = ini2json(config_path)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        url = server_address + '/api/v1/train/'
        print(f"Waiting for {url} ...")
        r = requests.post(url, headers=headers, data=config_json)
        if r.status_code == 200:
            job_id = r.json()['job_id']
            print(f'The job is successfully initiated, job_id= {job_id}')
            return job_id
        else:
            print(f'ERROR! ({r.status_code})')
            return None

    @staticmethod
    def get_model(job_id, server_address='https://api.neuralet.io'):
        url = server_address + "/api/v1/download/"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        data = {"job_id": str(job_id)}
        data = json.dumps(data)
        print(f"Waiting for {url} ...")
        r = requests.post(url, headers=headers, data=data)
        if r.status_code == 200:
            download_link = r.json()['download_link']
            print("Downloading model/output file...")
            command = 'curl "' + download_link + '" --output output.zip'
            os.system(command)
            print("The model is saved at output.zip file.")
        else:
            print(f'ERROR! ({r.status_code})')

    @staticmethod
    def get_task_status(job_id, server_address='https://api.neuralet.io'):
        url = server_address + "/api/v1/status/"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        data = {"job_id": str(job_id)}
        data = json.dumps(data)
        print(f"Waiting for {url} ...")
        r = requests.post(url, headers=headers, data=data)
        if r.status_code == 200:
            return r.json()  # TODO: parse and return
        else:
            print(f'ERROR! ({r.status_code})')
            return None

    @staticmethod
    def send_file(file_path, server_address='https://api.neuralet.io'):
        url = server_address + "/api/v1/upload/"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        print(f"Waiting for {url} ...")
        print("Uploading file, please wait it may take a while ...!")
        r = requests.get(url=url, headers=headers)
        if r.status_code == 200:
            upload_link = r.json()['upload_link']
            command = 'curl "' + str(upload_link) + '" --upload-file ' + str(file_path)
            os.system(command)
            uploade_file_dir = r.json()['name']
            return uploade_file_dir
        else:
            print(f'ERROR! ({r.status_code})')
            return None

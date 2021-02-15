import requests
from utils import token_reader, json_reader
import os
import json


class NeuraletRequest():
    """
    Provides a wrapper for Neuralet Adaptive Learning API
    """

    @staticmethod
    def send_config(config_path, token_file, server_address='https://api.neuralet.io'):
        '''
        Send a config to server for starting a new job
        :param config_path: Path of .json config file
        :param server_address: Server address
        :param token_file: Path of token file
        :return:
        '''
        config_json = json_reader(config_path)
        url = server_address + '/api/v1/model/train/'
        token = token_reader(token_file)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}"
        }
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
    def get_model(job_id, token_file, server_address='https://api.neuralet.io'):
        '''
        Download the frozen graph of trained model
        :param job_id: Job id of initiated task
        :param token_file: Path of token file
        :param server_address: Server Address
        :return:
        '''
        url = server_address + "/api/v1/file/download/"
        token = token_reader(token_file)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}"
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
    def get_task_status(job_id, token_file, server_address='https://api.neuralet.io'):
        '''
        Get the status of initiated job
        :param job_id: Job id of initiated task
        :param token_file: The path of token file
        :param server_address: Server Address
        :return:
        '''
        url = server_address + "/api/v1/model/status/"
        token = token_reader(token_file)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}"
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
    def send_file(file_path, token_file, server_address='https://api.neuralet.io'):
        '''
        Upload a file to the server and provide a uuid for further using
        :param file_path: .zip file path
        :param server_address: Server Address
        :return: uuid
        '''
        url = server_address + "/api/v1/file/upload/"
        token = token_reader(token_file)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}"
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

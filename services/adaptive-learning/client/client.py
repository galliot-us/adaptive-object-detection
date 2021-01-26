from neuralet_requests import NeuraletRequest
from argparse import ArgumentParser
from configs.config_engine import ConfigEngine
import os

req = NeuraletRequest()


def init_task(cfg_path, server_address='https://api.neuralet.io'):
    job_id = req.send_config(cfg_path, server_address)
    return job_id


def get_task_status(job_id, server_address='https://api.neuralet.io'):
    result = req.get_task_status(job_id, server_address)
    if result is not None:
        print(result['status'])
    else:
        print("Something went wrong. Please retry in a few minutes or contact the administrator!")
    return result


def download_model(job_id, server_address='https://api.neuralet.io'):
    req.get_model(job_id, server_address)


def upload_file(file_path, server_address='https://api.neuralet.io'):
    uuid = None
    if os.path.isfile(file_path):
        uuid = req.send_file(file_path, server_address)
        print("UploadUUID: ", uuid)
    else:
        raise Exception("Invalid Arguments, expected a file that exists not %r" % (file_path))
    return uuid


def main():
    argparse = ArgumentParser()
    argparse.add_argument('--config', type=str, help='config file path', default='configs/iterdet.ini')
    argparse.add_argument('--job_id', type=str, help='running job id', default=0)
    argparse.add_argument('--task_type', type=int, help=''
                                                        '0: initialize a new adaptive learning request'
                                                        '1: get the status of initialized task'
                                                        '2: download the trained model'
                                                        '3: upload file',
                          default=0)
    argparse.add_argument('--file_path', type=str, help='file path for uploading', default=None)
    args = argparse.parse_args()
    config_path = args.config
    job_id = args.job_id
    task_type = args.task_type
    file_path = args.file_path

    cfg_engine = ConfigEngine(config_path)
    server_adr = str(cfg_engine.get_section_dict('Server')['ServerPath'])
    print(f"Config file: {config_path}")
    print(f"Server Address: {server_adr}")
    print("---------------------------------")

    if task_type == 0:
        print(f"Task type {task_type}: Initialize a new task.")
        init_task(config_path, server_adr)
    elif task_type == 1:
        print(f"Task type {task_type}: Get task status from server.")
        get_task_status(job_id, server_adr)
    elif task_type == 2:
        print(f"Task type {task_type}: Download the trained model from server.")
        download_model(job_id, server_adr)
    elif task_type == 3:
        print(f"Task type {task_type}: Upload file to server.")
        upload_file(file_path, server_adr)
    else:
        raise ValueError(f"Task type {task_type}: Is not supported")


if __name__ == "__main__":
    main()

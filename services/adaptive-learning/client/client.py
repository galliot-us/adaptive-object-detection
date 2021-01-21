from configs.config_engine import ConfigEngine
from neuralet_requests import NeuraletRequest
from argparse import ArgumentParser
import os

req = NeuraletRequest()


def init_task(cfg_path):
    config = ConfigEngine(cfg_path)
    req.send_config(config)
    # TODO: should return task_id


def get_task_status(task_id):
    req.get_task_status(task_id)
    # TODO: should return metadata or someting like that


def download_model(task_id):
    req.get_model(task_id)


def upload_file(file_path):
    uploaded_file_name = None
    if os.path.isfile(file_path):
        file_name = file_path.split('/')[-1]
        file = (file_name, open(file_path, 'rb'))
        uploaded_file_name = req.send_file(file, file_name)
    else:
        raise Exception("Invalid Arguments, expected a file that exists not %r" % (file_path))
    return uploaded_file_name


def main():
    argparse = ArgumentParser()
    argparse.add_argument('--config', type=str, help='config file path', default='configs/config-x86.ini')
    argparse.add_argument('--task_id', type=int, halp='running task id', default=0)
    argparse.add_argument('--task_type', type=int, help=''
                                                        '0: initialize a new adaptive learning request'
                                                        '1: get the status of initialized task'
                                                        '2: download the trained model'
                                                        '3: upload file',
                          default=0)
    argparse.add_argument('--file_path', type=str, help='file path for uploading', default=None)

    args = argparse.parse_args()
    config_path = args.config
    task_id = args.task_id
    task_type = args.task_type
    file_path = args.file_path

    if task_type == 0:
        print(f"Task type {task_type}: Initialize a new task.")
        init_task(config_path)

    elif task_type == 1:
        print(f"Task type {task_type}: Get task status from server.")
        get_task_status(task_id)

    elif task_type == 2:
        print(f"Task type {task_type}: Download the trained model from server.")
        download_model(task_id)
    elif task_type == 3:
        print(f"Task type {task_type}: Upload file to server.")
        upload_file(file_path)
    else:
        raise ValueError(f"Task type {task_type}: Is not supported")


if __name__ == "__main__":
    main()

from configs.config_engine import ConfigEngine
from neuralet_requests import NeuraletRequest
from argparse import ArgumentParser

req = NeuraletRequest()


def init_task(cfg_path):
    config = ConfigEngine(cfg_path)
    req.send_config(config)
    # TODO: should return task_id


def get_task_status(task_id):
    req.get_task_status(task_id)


def download_model(task_id):
    req.get_model(task_id)


def main():
    argparse = ArgumentParser()
    argparse.add_argument('--config', type=str, help='config file path', default='configs/config-x86.ini')
    argparse.add_argument('--task_id', type=int, halp='running task id', default=0)
    argparse.add_argument('--task_type', type=int, help=''
                                                        '0: initialize a new adaptive learning request'
                                                        '1: get the status of initialized task'
                                                        '2: download the trained model',
                          default=0)

    args = argparse.parse_args()
    config_path = args.config
    task_id = args.task_id
    task_type = args.task_type

    if task_type == 0:
        print(f"Task type {task_type}: Initialize a new task.")
        init_task(config_path)

    elif task_type == 1:
        print(f"Task type {task_type}: Get task status from server.")
        get_task_status(task_id)

    elif task_type == 2:
        print(f"Task type {task_type}: Download the trained model from server.")
        download_model(task_id)
    else:
        raise ValueError(f"TaskID {task_type}: Is not supported")


if __name__ == "__main__":
    main()

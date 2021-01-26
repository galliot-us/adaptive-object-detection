from neuralet_requests import NeuraletRequest
from argparse import ArgumentParser
import os

req = NeuraletRequest()


def init_task(cfg_path):
    task_id = req.send_config(cfg_path)
    return task_id


def get_task_status(task_id):
    out = req.get_task_status(task_id)
    return out


def download_model(task_id):
    req.get_model(task_id)


def upload_file(file_path):
    uploaded_file_name = None
    if os.path.isfile(file_path):
        uploaded_file_name = req.send_file(file_path)
    else:
        raise Exception("Invalid Arguments, expected a file that exists not %r" % (file_path))
    return uploaded_file_name


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()
    upload_parser = subparsers.add_parser("upload_file")
    upload_parser.set_defaults(action="upload_file")
    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(action="train")
    status_parser = subparsers.add_parser("get_status")
    status_parser.set_defaults(action="get_status")
    download_parser = subparsers.add_parser("downlaod_file")
    download_parser.set_defaults(action="download_file")

    upload_parser.add_argument('--file_path', type=str, help='file path for uploading', required=True)
    train_parser.add_argument('--config', type=str, help='config file path', required=True)
    status_parser.add_argument('--task_id', type=str, help='running task id', required=True)
    download_parser.add_argument('--task_id', type=str, help='running task id', required=True)


    args = parser.parse_args()
    
    if args.action == "upload_file":
        print("Upload file to server.")
        upload_file(args.file_path)

    elif args.action == "train":
        print("Initialize a new task.")
        init_task(args.config_path)

    elif args.action == "get_status":
        print("Get task status from server.")
        get_task_status(args.task_id)

    elif args.action == "download_file":
        print(f"Download the trained model from server.")
        download_model(args.task_id)

if __name__ == "__main__":
    main()

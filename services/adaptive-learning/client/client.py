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

    parser = ArgumentParser()
    subparsers = parser.add_subparsers()
    upload_parser = subparsers.add_parser("upload_file")
    upload_parser.set_defaults(action="upload_file")
    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(action="train")
    status_parser = subparsers.add_parser("get_status")
    status_parser.set_defaults(action="get_status")
    download_parser = subparsers.add_parser("download_file")
    download_parser.set_defaults(action="download_file")

    upload_parser.add_argument('--file_path', type=str, help='file path for uploading', required=True)
    train_parser.add_argument('--config', type=str, help='config file path', required=True)
    status_parser.add_argument('--job_id', type=str, help='running job id', required=True)
    download_parser.add_argument('--job_id', type=str, help='running job id', required=True)


    args = parser.parse_args()
    
    if args.action == "upload_file":
        print("Upload file to server.")
        upload_file(args.file_path)

    elif args.action == "train":
        print("Initialize a new job.")
        init_task(args.config_path)

    elif args.action == "get_status":
        print("Get job status from server.")
        get_task_status(args.job_id)


    elif args.action == "download_file":
        print(f"Download the trained model from server.")
        download_model(args.job_id)

if __name__ == "__main__":
    main()

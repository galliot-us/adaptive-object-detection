from os.path import join
from neuralet_requests import NeuraletRequest
from argparse import ArgumentParser
import os

req = NeuraletRequest()


def init_task(cfg_path, token_file, server_address="https://api.neuralet.io"):
    job_id = req.send_config(cfg_path, token_file, server_address)
    return job_id


def get_task_status(job_id, token_file, server_address="https://api.neuralet.io"):
    result = req.get_task_status(job_id, token_file, server_address)
    if result is not None:
        print(result)
    else:
        print(
            "Something went wrong. Please retry in a few minutes or contact the administrator!"
        )
    return result


def download_model(job_id, token_file, server_address="https://api.neuralet.io"):
    req.get_model(job_id, token_file, server_address)


def kill_job(job_id, token_file, server_address="https://api.neuralet.io"):
    req.kill_job(job_id, token_file, server_address)


def kill_job(job_id, token_file, server_address="https://api.neuralet.io"):
    req.kill_job(job_id, token_file, server_address)


def user_jobs(token_file, server_address="https://api.neuralet.io", page=1):
    req.user_jobs(token_file, server_address, page)


def user_uploads(token_file, server_address="https://api.neuralet.io", page=1):
    req.user_uploads(token_file, server_address, page)


def user_detail(token_file, server_address="https://api.neuralet.io"):
    req.user_detail(token_file, server_address)


def upload_file(
    file_path, token_file, server_address="https://api.neuralet.io", params=dict
):
    uuid = None
    if os.path.isfile(file_path):
        uuid = req.send_file(file_path, token_file, server_address, params)
        print("UploadUUID: ", uuid)
    else:
        raise Exception(
            "Invalid Arguments, expected a file that exists not %r" % (file_path)
        )
    return uuid


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()
    parser.add_argument("--url", type=str, default="https://api.neuralet.io")
    parser.add_argument("--token", type=str)
    upload_parser = subparsers.add_parser("upload_file")
    upload_parser.set_defaults(action="upload_file")
    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(action="train")
    status_parser = subparsers.add_parser("get_status")
    status_parser.set_defaults(action="get_status")
    download_parser = subparsers.add_parser("download_file")
    download_parser.set_defaults(action="download_file")
    kill_parser = subparsers.add_parser("kill_job")
    kill_parser.set_defaults(action="kill_job")
    jobs_parser = subparsers.add_parser("user_jobs")
    jobs_parser.set_defaults(action="user_jobs")
    uploads_parser = subparsers.add_parser("user_uploads")
    uploads_parser.set_defaults(action="user_uploads")
    user_detail_parser = subparsers.add_parser("user_detail")
    user_detail_parser.set_defaults(action="user_detail")

    upload_parser.add_argument(
        "--file_path", type=str, help="file path for uploading", required=True
    )
    upload_parser.add_argument(
        "--label", type=str, help="label to distinguish uploads", required=False
    )
    upload_parser.add_argument(
        "--video_names",
        type=str,
        help="comma-separated video names included in zip file",
        required=False,
    )
    train_parser.add_argument(
        "--config_path", type=str, help="json config file path", required=True
    )
    status_parser.add_argument(
        "--job_id", type=str, help="running job id", required=True
    )
    download_parser.add_argument(
        "--job_id", type=str, help="running job id", required=True
    )
    kill_parser.add_argument("--job_id", type=str, help="running job id", required=True)
    jobs_parser.add_argument(
        "--page", type=int, help="page number", required=False, default=1
    )
    uploads_parser.add_argument(
        "--page", type=int, help="page number", required=False, default=1
    )

    args = parser.parse_args()
    server_adr = args.url
    token_file = args.token

    if args.action == "upload_file":
        print("Upload file to server.")
        params = {}
        if args.label:
            params["label"] = args.label

        if args.video_names:
            try:
                args.video_names.split(",")
                params["video_names"] = args.video_names
            except:
                pass

        upload_file(args.file_path, token_file, server_adr, params)

    elif args.action == "train":
        print("Initialize a new job.")
        init_task(args.config_path, token_file, server_adr)

    elif args.action == "get_status":
        print("Get job status from server.")
        get_task_status(args.job_id, token_file, server_adr)

    elif args.action == "download_file":
        print("Download the trained model from server.")
        download_model(args.job_id, token_file, server_adr)

    elif args.action == "kill_job":
        print("Kill job")
        kill_job(args.job_id, token_file, server_adr)

    elif args.action == "user_jobs":
        print("User jobs")
        user_jobs(token_file, server_adr, args.page)

    elif args.action == "user_uploads":
        print("User uploads")
        user_uploads(token_file, server_adr, args.page)

    elif args.action == "user_detail":
        print("User detail")
        user_detail(token_file, server_adr)


if __name__ == "__main__":
    main()

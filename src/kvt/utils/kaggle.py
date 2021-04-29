import datetime
import json
import os
import time
from datetime import timezone

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("kaggle is not installed yet.")
    KaggleApi = None
except OSError:
    print("kaggle.json is not found.")
    KaggleApi = None


def upload_dataset(title, dirname, user_id, dir_mode="zip"):
    dataset_metadata = {}
    dataset_metadata["id"] = f"{user_id}/{title}"
    dataset_metadata["licenses"] = [{"name": "CC0-1.0"}]
    dataset_metadata["title"] = title

    with open(os.path.join(dirname, "dataset-metadata.json"), "w") as f:
        json.dump(dataset_metadata, f, indent=4)

    api = KaggleApi()
    api.authenticate()

    if dataset_metadata["id"] not in [
        str(d) for d in api.dataset_list(user=user_id, search=f'"{title}"')
    ]:
        api.dataset_create_new(
            folder=dirname,
            convert_to_csv=False,
            dir_mode=dir_mode,
        )
    else:
        api.dataset_create_version(
            folder=dirname,
            version_notes="update",
            convert_to_csv=False,
            delete_old_versions=True,
            dir_mode=dir_mode,
        )


def monitor_submission_time(competition_name):
    api = KaggleApi()
    api.authenticate()

    submissions = api.competition_submissions(competition_name)
    if len(submissions):
        result_ = submissions[0]
        latest_ref = str(result_)  # latest submission number
        submit_time = result_.date

        status = ""

        while status != "complete":
            list_of_submission = api.competition_submissions(competition_name)
            for result in list_of_submission:
                if str(result.ref) == latest_ref:
                    break
            status = result.status

            now = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
            elapsed_time = int((now - submit_time).seconds / 60) + 1
            if status == "complete":
                print("\r", f"run-time: {elapsed_time} min, LB: {result.publicScore}")
                print(f"{result.fileName}: {result.url}")
            else:
                print("\r", f"elapsed time: {elapsed_time} min", end="")
                time.sleep(60)
    else:
        print("You have no submissions.")

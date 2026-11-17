"""
Where approved files are kept.

The rest of the server only talks to the Storage interface, so moving from the local
disk to the cloud means adding one class here, not changing the API or the app.

  local   (default)  files in DATA_DIR/archive, laid out as year/month/type/file.pdf
  s3      S3-compatible object storage (MinIO on your own server, or any cloud). TODO
  gdrive  a Google Drive Shared Drive (good if the college already has Google Workspace). TODO

Choose with the STORAGE_BACKEND environment variable.
"""
import os
import shutil
from pathlib import Path

from fastapi.responses import FileResponse, Response

from settings import DATA_DIR


class Storage:
    name = "base"

    def save(self, local_file: Path, key: str) -> None:
        raise NotImplementedError

    def serve(self, key: str, download_name: str) -> Response:
        raise NotImplementedError


class LocalStorage(Storage):
    name = "local"

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root not in path.parents:  # blocks ../ tricks
            raise ValueError("Invalid storage key")
        return path

    def save(self, local_file: Path, key: str) -> None:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(local_file), str(target))

    def serve(self, key: str, download_name: str) -> Response:
        path = self._path(key)
        if not path.exists():
            raise FileNotFoundError(key)
        return FileResponse(path, media_type="application/pdf", filename=download_name)


class S3Storage(Storage):
    """
    TODO for cloud: use boto3 with endpoint_url (works with MinIO, AWS S3, Wasabi, Backblaze).
      save():  client.upload_file(str(local_file), BUCKET, key)
      serve(): return RedirectResponse(client.generate_presigned_url("get_object", ...))
    """
    name = "s3"

    def __init__(self):
        raise NotImplementedError("S3 storage is not implemented yet. See the TODO in storage.py.")


class GoogleDriveStorage(Storage):
    """
    TODO for cloud: use the Google Drive API with a service account that is a member of a
    Shared Drive. Create the year/month/type folders on demand and keep the Drive file id
    as the key. serve() can stream the file or redirect to a view link.
    """
    name = "gdrive"

    def __init__(self):
        raise NotImplementedError("Google Drive storage is not implemented yet. See the TODO in storage.py.")


def get_storage() -> Storage:
    backend = os.environ.get("STORAGE_BACKEND", "local").lower()
    if backend == "local":
        return LocalStorage(Path(os.environ.get("ARCHIVE_DIR", DATA_DIR / "archive")))
    if backend == "s3":
        return S3Storage()
    if backend == "gdrive":
        return GoogleDriveStorage()
    raise RuntimeError(f"Unknown STORAGE_BACKEND: {backend}")

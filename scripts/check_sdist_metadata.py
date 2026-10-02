"""Verify that a source distribution contains its declared license files."""

import argparse
import glob
import tarfile
from email.parser import BytesParser
from email.policy import default
from pathlib import PurePosixPath


def check_sdist(path: str) -> None:
    with tarfile.open(path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        metadata_files = [name for name in members if PurePosixPath(name).name == "PKG-INFO"]
        if len(metadata_files) != 1:
            raise ValueError(f"{path}: expected one PKG-INFO, found {len(metadata_files)}")

        metadata_name = metadata_files[0]
        metadata_file = archive.extractfile(metadata_name)
        if metadata_file is None:
            raise ValueError(f"{path}: cannot read {metadata_name}")
        metadata = BytesParser(policy=default).parsebytes(metadata_file.read())

        license_files = metadata.get_all("License-File", [])
        if not license_files:
            raise ValueError(f"{path}: PKG-INFO declares no license files")

        root = PurePosixPath(metadata_name).parent
        for license_file in license_files:
            relative_path = PurePosixPath(license_file)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"{path}: invalid license path {license_file!r}")
            archive_path = (root / relative_path).as_posix()
            member = members.get(archive_path)
            if member is None or not member.isfile():
                raise ValueError(f"{path}: declared license file {archive_path} is missing")

    print(f"{path}: all {len(license_files)} declared license files are present")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", help="source archives or glob patterns")
    args = parser.parse_args()
    archives = [archive for pattern in args.archives for archive in glob.glob(pattern)]
    if not archives:
        parser.error("no source archives found")
    for archive in archives:
        check_sdist(archive)


if __name__ == "__main__":
    main()

"""

RenderPipeline

Copyright (c) 2014-2016 tobspr <tobias.springer1@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

"""

from __future__ import print_function

import os
import sys
import zipfile
import shutil

from metadrive.render_pipeline.rplibs.six.moves import urllib  # pylint: disable=import-error
from metadrive.render_pipeline.rplibs.six import BytesIO, binary_type


def download_file(url, chunk_size=100 * 1024):
    """ Helper method to download a file displaying a progress bar """
    print("Fetching:", url)
    file_content = None
    progressbar = None

    if sys.version_info.major <= 2:

        # Import progressbar library
        from metadrive.render_pipeline.rplibs.progressbar import FileTransferSpeed, ETA, ProgressBar, Percentage
        from metadrive.render_pipeline.rplibs.progressbar import Bar
        widgets = ['\tDownloading: ', FileTransferSpeed(), ' ', Bar(), Percentage(), '   ', ETA()]
        file_content = []
        bytes_read = 0

        # Progressively download the file
        try:
            usock = urllib.request.urlopen(url)
            file_size = int(usock.headers.get("Content-Length", 1e10))
            print("File size is", round(file_size / (1024**2), 2), "MB")
            progressbar = ProgressBar(widgets=widgets, maxval=file_size).start()
            while True:
                data = usock.read(chunk_size)
                file_content.append(data)
                bytes_read += len(data)
                progressbar.update(bytes_read)
                if not data:
                    break
            usock.close()
        except Exception:
            print("ERROR: Could not fetch", url, "!", file=sys.stderr)
            raise
    else:
        # Don't use progressbar in python 3
        print("Downloading .. (progressbar disabled due to python 3 build)")
        try:
            usock = urllib.request.urlopen(url)
            file_content = []
            while True:
                data = usock.read(chunk_size)
                file_content.append(data)
                if not data:
                    break
            usock.close()
        except Exception:
            print("ERROR: Could not fetch", url, "!", file=sys.stderr)
            raise

    if progressbar:
        progressbar.finish()

    return binary_type().join(file_content)


def download_submodule(author, module_name, dest_path, ignore_list):
    """ Downloads a submodule from the given author and module name, and extracts
    all files which are not on the ignore_list to the dest_path.

    Example: download_submodule("tobspr", "RenderPipeline", ".", ["README.md", "LICENSE"])
    """

    # Make directory, if it does not exist yet
    if not os.path.isdir(dest_path):
        os.makedirs(dest_path)

    # Construct download url
    source_url = "https://github.com/" + author + "/" + module_name + "/archive/master.zip"
    prefix = module_name + "-master"

    # Extract the zip
    zip_ptr = BytesIO(download_file(source_url))
    print("Extracting ZIP ...")

    try:
        zip_handle = zipfile.ZipFile(zip_ptr)
    except zipfile.BadZipfile:
        print("ERROR: Invalid zip file!", file=sys.stderr)
        sys.exit(3)

    if zip_handle.testzip() is not None:
        print("ERROR: Invalid zip file checksums!", file=sys.stderr)
        sys.exit(1)

    num_files, num_dirs = 0, 0

    for fname in zip_handle.namelist():
        rel_name = fname.replace(prefix, "").replace("\\", "/").lstrip("/")
        if not rel_name:
            continue

        is_file = not rel_name.endswith("/")
        rel_name = dest_path.rstrip("/\\") + "/" + rel_name

        # Files
        if is_file:
            for ignore in ignore_list:
                if ignore in rel_name:
                    break
            else:
                with zip_handle.open(fname, "r") as source, open(rel_name, "wb") as dest:
                    shutil.copyfileobj(source, dest)
                num_files += 1

        # Directories
        else:
            if not os.path.isdir(rel_name):
                os.makedirs(rel_name)
            num_dirs += 1

    print("Extracted", num_files, "files and", num_dirs, "directories")

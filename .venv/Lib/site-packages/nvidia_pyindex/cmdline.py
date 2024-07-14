#! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

from nvidia_pyindex.utils import clean_nvidia_pyindex_from_config
from nvidia_pyindex.utils import get_configuration_files_by_priority


def uninstall():
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Uninstalling NVIDIA Pip Configuration ...")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    for config_file in get_configuration_files_by_priority():
        clean_nvidia_pyindex_from_config(config_file)


def total_clean():
    config_files = get_configuration_files_by_priority()

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Uninstalling NVIDIA Pip Configuration ...")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    for config_file in config_files:
        for file in [config_file, "{}.old".format(config_file)]:
            print("Removing configuration file: %s ..." % file)
            try:
                print("Deleting file: `{}` ...".format(file))
                os.remove(file)
                print("Success !\n")
            except:
                print("Pass ...\n")
                pass


def main():
    if len(sys.argv) >= 2:
        if sys.argv[1] == "uninstall":
            uninstall()
        elif sys.argv[1] == "total_clean":
            total_clean()
        else:
            raise RuntimeError("Unknown commandline argument: %s" % sys.argv[1])
    else:
        raise RuntimeError(
            "Incorrect usage: `nvidia_pyindex uninstall`\n."
            "Received: %s" % sys.argv
        )


if __name__ == "__main__":
    sys.exit(main())

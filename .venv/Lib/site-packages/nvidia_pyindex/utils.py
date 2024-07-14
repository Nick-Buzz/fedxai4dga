import subprocess
import shlex
import json
import sys


def get_configuration_files():
    command = '{py_binary_path} -c "from pip._internal.configuration import ' \
              'get_configuration_files; ' \
              'print(get_configuration_files())"'.format(
                  py_binary_path=sys.executable
              )

    command = shlex.split(command)

    if sys.platform == "win32":
        command[0] = sys.executable  # otherwise there's a bug that removes `\`

    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        output = output.decode()
    except subprocess.CalledProcessError as e:
        output = e.output.decode()
        raise RuntimeError(output) from e

    output = output.replace("'", '"')
    output = json.loads(output)

    return output


def get_configuration_files_by_priority():
    config_files = get_configuration_files()

    config_priority_list = []
    config_priority_list += config_files["site"][::-1]
    config_priority_list += config_files["user"][::-1]
    config_priority_list += config_files["global"][::-1]

    return config_priority_list


def is_nvidia_pyindex(line):
    return any([
        pattern in line
        for pattern in [
            "sqrl.nvidia.com",
            "pypi.ngc.nvidia.com",
        ]
    ])


def clean_nvidia_pyindex_from_config(filepath):
    try:
        print("==============================================================")
        print("Cleaning previous NVIDIA PyIndex Configuration ...")
        print("File: `{}`".format(filepath))

        with open(filepath, 'r') as _file:
            file_content = _file.readlines()

        def edit_config_file(env_name, _file_content):
            # Removing all NVIDIA changes
            temp = []
            try:
                while _file_content:
                    line = _file_content.pop(0)
                    if env_name in line:
                        if is_nvidia_pyindex(line):
                            line = "{} = \n".format(env_name)
                        temp.append(line)
                        while True:
                            line = _file_content.pop(0)
                            if line[0] != " ":
                                temp.append(line)
                                break
                            elif not is_nvidia_pyindex(line):
                                temp.append(line)
                    else:
                        temp.append(line)
            except IndexError:
                pass

            return temp

        file_content = edit_config_file("extra-index-url", file_content)
        file_content = edit_config_file("trusted-host", file_content)
        file_content = edit_config_file("no-cache-dir", file_content)

        with open(filepath, 'w') as _file:
            _file.writelines(file_content)

        print("Previous NVIDIA PyIndex Configuration cleaned with success ...")
        print("==============================================================")

    except (FileNotFoundError, PermissionError) as e:
        print("Error while removing the old NVIDIA PyINDEX configuration:", e)

    print("==============================================================")

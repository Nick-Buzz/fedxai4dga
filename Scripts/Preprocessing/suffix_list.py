VERBOSE = True

import requests

def debug(output):
    if VERBOSE == True:
        print(output)
    return None

def download_list():
    debug("Will download the public suffixes list")
    suffix_list_url = "https://publicsuffix.org/list/public_suffix_list.dat"
    r = requests.get(suffix_list_url, allow_redirects = True)
    open("public_suffixes_list.csv", "wb").write(r.content)
    debug("The list of public suffixes has been downloaded")
    return None

def delete_unwanted_lines():
    debug("Will delete unwanted lines: empty lines and lines starting with slashes")
    fdw = open("public_suffixes_list_v2.csv", "w")
    with open("public_suffixes_list.csv", "r") as file:
        for line in file:
            if line != "\n" and line[0] != "/":
                fdw.write(line)
    fdw.close()

    debug("Unwanted lines have been deleted")
    return None

if __name__ == "__main__":
    download_list()
    delete_unwanted_lines()

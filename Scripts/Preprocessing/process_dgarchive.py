# The script assumes that "wget" command is installed

DEBUG = True

dgarchive_username = "ntua_gr"
dgarchive_password = "XXXXXXXXXXXXXXXXXXx"
filename = "2020-06-19-dgarchive_full.tgz"
dgarchive_link = "https://dgarchive.caad.fkie.fraunhofer.de/datasets/" + str(filename)
number_of_names = 30000

import os, subprocess

def debug(output):
    if DEBUG == True:
        print(output)
    return None

def return_command_output(command):
    proc = subprocess.Popen(command, stdout = subprocess.PIPE, shell = True)
    (out, err) = proc.communicate()
    output = out.rstrip('\n'.encode('utf8'))
    return output

def download_dataset():
    debug("DGArchive dataset will be downloaded")

    command = "wget --user " + str(dgarchive_username) + " --password " + str(dgarchive_password) + " " + str(dgarchive_link)
    os.system(command)
    

    debug("The DGArchive dataset has been downloaded")

    return None

def untar_dataset():
    debug("Time to untar the downloaded dataset")

    command = "tar -zxvf " + str(filename)
    os.system(command)

    debug("The dataset has been untarred")
    debug("Erase p2p file because we want to keep the DGA names")

    os.system("rm ./*_p2p.csv")

    return None

def list_dga_files():
    debug("retrieve list of DGA files")
    command = "ls *.csv"
    dga_files_terminal = return_command_output(command).decode('utf-8')
    dga_files_list = dga_files_terminal.split("\n")
    return dga_files_list

def determine_dga_families(remaining_dga_families):
    fdw = open("dga_families.txt", "w")
    for dga in remaining_dga_families:
        dga_family = dga.split("_")[0]
        fdw.write(dga_family + "\n")
    fdw.close()
    return None

def all_dga_files(dga_files_list):
    debug("Will append all DGA files to one")
    os.system("touch ./dgarchive_full.csv")
    for dga_family in dga_files_list:
        command = "cat " + str(dga_family) + " >> dgarchive_full.csv"
        os.system(command)
        debug("Copied " + str(dga_family) + " to the full DGA file")
    return None

def keep_large_dga_files(dga_files_list):
    debug("Will delete files with less than the specified names")
    remaining_dga_families = set()
    for dga_family in dga_files_list:
        command = "cat " + str(dga_family) + " | wc -l"
        lines = int(return_command_output(command).decode('utf-8'))
        if lines < number_of_names:
            command = "rm " + str(dga_family)
            os.system(command)
            debug("Will delete" + str(dga_family))
        else:
            command = "tail -n " + str(number_of_names) + " " + str(dga_family) + " > " + str(dga_family)[:-4] + "-top.csv"
            os.system(command)
            command = "rm " + str(dga_family)
            os.system(command)
            debug("Kept " + str(dga_family))
            remaining_dga_families.add(dga_family)
    return remaining_dga_families

if __name__ == "__main__":
    download_dataset()
    untar_dataset()
    dga_files_list = list_dga_files()
    all_dga_files(dga_files_list)
    remaining_dga_families = keep_large_dga_files(dga_files_list)
    determine_dga_families(remaining_dga_families)

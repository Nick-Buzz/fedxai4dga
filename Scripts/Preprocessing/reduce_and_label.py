VERBOSE = True
tranco_filename = "tranco_full_list.csv"
# The filename where everything will be stored
filename_out = "./labeled_dataset.csv"

# Numbers for each DGA
maximum_size = 15000

import sys
import os

def debug(output):
    if VERBOSE == True:
        print(output)
    return None

def determine_families():
    debug("Will determine DGA families in the final dataset")
    dgas = set()
    fdr = open("dga_families.txt", "r")
    for line in fdr:
        dga = line.strip()
        dgas.add(dga)
    fdr.close()
    debug("These name categories will be included in the final dataset")
    debug(dgas)
    return dgas

# Assign a number to each entry of the dgas list so that they can be discerned
def assign_sequence_number(dgas):    
    dga_dict = {}
    dga_dict["tranco"] = 0
    sequence = 1
    for dga in dgas:
        dga_dict[dga] = sequence
        sequence += 1

    return dga_dict

def load_suffixes():
    # Load Mozilla Firefox libraries to exclude valid domain name suffixes
    debug("Loading the Mozilla Firefox suffixes in a set")
    suffix_list = "./public_suffixes_list_v2.csv"
    fdr = open(suffix_list, "r")
    suffixes = set()
    for line in fdr:
        suffix = line.strip()
        suffixes.add(suffix)
    
    fdr.close()
    return suffixes

remaining_families = set()

dgas = determine_families()
dga_dict = assign_sequence_number(dgas)

fdw = open(filename_out, "w")

total_sizes = []

suffixes = load_suffixes()

for dga in dga_dict:
    debug("Now working on: " + dga)

    if dga == "tranco":
        # Load tranco names
        filename = "./tranco_remaining.txt"
    else:
        # Load names of a specific DGA family
        filename = "./" + str(dga) + "_dga-top.csv"

    fdr = open(filename, "r")
    
    # Find the appropriate prefix
    prefix_set = set()
    for line in fdr:
        line = line.strip()
        if dga != "tranco":
            name = line.split(",")[0].replace('"', '')
        else:
            name = line
        labels = name.split(".")
        labels.reverse()
        candidate_suffix = labels[0]
        index = 0
        try:
            while (candidate_suffix) in suffixes:
                index += 1
                candidate_suffix = labels[index] + "." + candidate_suffix
            labels.reverse()
            prefix = ".".join(labels[0:(len(labels) - index)])
            to_add = str(prefix) + "," + str(name)
            prefix_set.add(to_add)
        except:
            pass
        if (dga == "tranco" and len(prefix_set) == 900000) or (dga != "tranco" and len(prefix_set) == maximum_size):
            break
    
    fdr.close()

    total_names = len(prefix_set)
    if total_names < maximum_size:
        continue

    remaining_families.add(dga)

    # Labeling for binary classifiers: 0 for tranco (legitimate) and 1 for DGA's
    for item in prefix_set:
        if dga == "tranco":
            item = item + ",0," + str(dga) 
        else:
            item = item + ",1," + str(dga)
        fdw.write(item + "\n")
    
    print(total_names)
    total_sizes.append(total_names)

compound_list = []
start_value = 0
for item in total_sizes:
    start_value += int(item)
    compound_list.append(start_value)

fdw.close()

debug("Remaining families")
debug(remaining_families)
debug(len(remaining_families))

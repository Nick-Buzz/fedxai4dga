#!/bin/sh
# Build the labeled dataset, its 50 features and the model-ready train/test files.
# Run from the repository root. DGArchive credentials: see README.
set -e
# reduce_and_label.py iterates Python sets; a fixed hash seed makes the row
# order, and therefore the train/test split, deterministic.
export PYTHONHASHSEED=0
P="$(pwd)/Scripts/Preprocessing"

# process_dgarchive.py treats every *.csv in its working directory as a DGA
# family and deletes the small ones, so it runs in its own folder.
mkdir -p Data/Raw/dgarchive
cd Data/Raw/dgarchive
python3 "$P/process_dgarchive.py"  # -> dgarchive_full.csv, dga_families.txt, <family>_dga-top.csv
mv ./*_dga-top.csv dga_families.txt dgarchive_full.csv ..
cd ..

# The two Tranco files and the public-suffix list used in the paper are shipped
# in Data/Raw. Uncomment to rebuild them instead (filter_tranco.py needs
# dgarchive_full.csv and downloads the Tranco list it names).
# python3 "$P/filter_tranco.py"    # -> tranco_top100k.txt, tranco_remaining.txt
# python3 "$P/suffix_list.py"      # -> public_suffixes_list_v2.csv
python3 "$P/reduce_and_label.py"   # -> labeled_dataset.csv
python3 "$P/feature_extractor.py"  # -> labeled_dataset_features.csv (50 features)

cd ../..
python3 -m Scripts.Preprocessing.build_dataset   # -> Data/Processed/{train,test}_data.csv

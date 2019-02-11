#!/bin/bash

global_root=$(git rev-parse --show-toplevel)
repos_folder="$global_root/logsModel/"
dump_folder="$global_root/dump/logsModelCompressed"

# clean folder
rm -R -- $repos_folder/*/

# UNZIP
cd $dump_folder
for filename in *.tar.gz
do
  tar zxf $filename -C $repos_folder
done

exit 0
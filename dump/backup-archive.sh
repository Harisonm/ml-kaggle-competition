#!/bin/bash

global_root=$(git rev-parse --show-toplevel)
repos_folder="$global_root/logsModel/"
dump_folder="$global_root/dump/logsModelCompressed"

# clean folder
rm -R -- $dump_folder/*/

# ZIP
cd $repos_folder
for dir in */
do
  base=$(basename "$dir")
  tar -cvf "${base}.tar.gz" "$dir"
done

# MOVE AFTER ZIP
mv *.tar.gz $dump_folder

exit 0
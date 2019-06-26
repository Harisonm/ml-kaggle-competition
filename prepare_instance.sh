#!/usr/bin/env bash
sudo apt-get -y update
sudo apt-get -y install nfs-common
sudo mount 10.0.0.2:/vol1 /mnt/project_ml



#!/usr/bin/env bash

if [[ "$1" == "-f" ]] ; then
    force=1
else
    force=0
fi

cert_dir=~/.config/certs
cert_file=aics-ca.crt
cert_file_path=${cert_dir}/${cert_file}

echo "Ensuring the certificate folder exists - ${cert_dir}"
mkdir -p ${cert_dir}

echo "Ensuring the aics certificate exists - ${cert_file}"
if [[ ! -e ${cert_file_path} ]] ; then
    url="https://s3-us-west-2.amazonaws.com/cacerts.allencell.org/${cert_file}"
    wget $url -O ${cert_file_path}
fi

pip_dir=~/.config/pip
pip_file=pip.conf
pip_file_path=${pip_dir}/${pip_file}

echo "Ensuring the user pip config folder exists - ${pip_dir}"
mkdir -p ${pip_dir}

pip_conf_content=$(cat <<PIPCONTENT
[global]
cert = ${cert_file_path}
index-url = https://artifactory.corp.alleninstitute.org/artifactory/api/pypi/pypi-virtual/simple

PIPCONTENT
)

echo "Ensuring the user pip config file exists - ${pip_file_path}"
if [[ ! -e ${pip_file_path} || "${force}" == "1" ]] ; then
    echo "${pip_conf_content}" > ${pip_file_path}
    echo "Complete"
else
    echo "You already have the file ${pip_file_path}"
    echo "Run this again with the -f flag or remove the existing file."
    echo "Exiting..."
fi

#!/usr/bin/env python
import os
import yaml
import requests

DIR = os.path.dirname(__file__)
DOWNLOADS = [{"manifest":"https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/refs/heads/master/models/intel/face-detection-retail-0005/model.yml","files":("FP32/face-detection-retail-0005.xml","FP32/face-detection-retail-0005.bin")},
    {"manifest":"https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/refs/heads/master/models/intel/face-reidentification-retail-0095/model.yml", "files":("FP32/face-reidentification-retail-0095.xml","FP32/face-reidentification-retail-0095.bin")},
    {"manifest":"https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/refs/heads/master/models/intel/landmarks-regression-retail-0009/model.yml", "files":("FP32/landmarks-regression-retail-0009.xml","FP32/landmarks-regression-retail-0009.bin")}]

def download(url, dest):
    """Helper to download files"""
    with requests.get(url, stream=True, timeout=10) as r:
        total_length = int(r.headers.get('content-length'))
        downloaded=0
        with open(dest,'wb') as fp:
            for chunk in r.iter_content(chunk_size=65536):
                fp.write(chunk)
                downloaded+=65536
                print("\r",round(downloaded/total_length*100),
                    "% of",round(total_length/1000000,1),"Mo downloaded...",end="")

def download_models():
    """
    Download the needed models according to DOWNLOADS global variable
    """
    os.makedirs(os.path.join(DIR,"models"),exist_ok=True)
    for model in DOWNLOADS:
        manifest=yaml.safe_load(requests.get(model["manifest"], timeout=10).content)
        for entry in manifest["files"]:
            if entry["name"] in model["files"]:
                filename = entry["name"].split("/")[-1]
                output_path=os.path.join(DIR,"models",filename)
                download(entry["source"], output_path)

download_models()

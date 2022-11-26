import os
import platform
#unpack
import gzip
import tarfile
import zipfile
#internet streaming
from urllib import request
from io import StringIO, FileIO
import requests
#flags
from config import SAVE

def create_dir(dirpath):
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)

def unpickle(batchfile):
    if platform.python_implementation() == 'PyPy':
        from zodbpickle import pickle
    else: 
        import _pickle as cPickle
    with open(batchfile, 'rb') as fd:
        dict_ = cPickle.load(fd, encoding='latin1')
    return dict_

def download_file_from_google_drive(fid, destination):
    from google_drive_downloader import GoogleDriveDownloader as gdd
    gdd.download_file_from_google_drive(file_id=fid,dest_path=destination)

def download_file_from_aws(s3path, destination):
    from os import system
    system("aws s3 cp {} {} --request-payer requester".format(s3path, destination))

def exists_or_download(filename, url, unpack=False, google_drive=False, amazon_aws=False):
    if not os.path.exists(filename):
        print("Download:", filename, "from:",url)
        if google_drive:
            download_file_from_google_drive(url,filename)
        elif amazon_aws:
            download_file_from_aws(url,filename)
        else:
            response = request.urlopen(url)
            with open(filename,'wb') as output:
                CHUNK_SIZE = 16 * 1024
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    else:
                        output.write(chunk)
    
    if unpack:
        print("unpack:", filename)
        if filename.endswith("zip"):
            if not os.path.exists(filename[:filename.rfind(".zip")]): 
                zip_ = zipfile.ZipFile(filename, "r")
                zip_.extractall(SAVE)
                zip_.close()
        elif filename.endswith("tar.gz"):
            if not os.path.exists(filename[:filename.rfind(".tar.gz")]): 
                tar = tarfile.open(filename, "r:gz")
                tar.extractall(SAVE)
                tar.close()
        elif filename.endswith("tar"):
            if not os.path.exists(filename[:filename.rfind(".tar")]): 
                tar = tarfile.open(filename, "r:")
                tar.extractall(SAVE)
                tar.close()
        elif filename.endswith("gz"):
            outputpath = os.path.splitext(filename)[0]
            if not os.path.exists(outputpath): 
                with open(filename,'wb') as ingzfile:
                    decompressedFile = gzip.GzipFile(fileobj=ingzfile, mode='rb')
                    with open(outputpath,'wb') as output:
                        output.write(decompressedFile.read())


def exists_or_download_list(filename, urlinfo_list):
    """
        urlinfo_list = [
            ( url, unpack, type )   
        ]
        where type is a string which could be: None/download (standard), google or amazon 
    """
    for info in urlinfo_list:
        if not os.path.exists(filename):
            url, unpack, dtype = info
            dw_google = dw_amazon = False
            if dtype in ('g',"google"):
                dw_google = True
            if dtype in ('aws', "amazon"):
                dw_amazon = True
            exists_or_download(filename, url, unpack=unpack,google_drive=dw_google, amazon_aws=dw_amazon)
# Check Python module installation
print("Checking requirements...")

print("Is numpy installed?")
try:
    import numpy
    print("Yes")
except ImportError:
    print("No. Please install numpy before continuing.")

print("Is scikit-learn installed?")
try:
    import sklearn
    print("Yes")
except ImportError:
    print("No. Please install scikit-learn before continuing.")


####################################################################
# Downloading dataset
print("\nDownloading Apache SpamAssassin's dataset. This may take a while...")
import urllib.request
import sys
import tarfile
import os

DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM1_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
HAM2_URL = DOWNLOAD_ROOT + "20030228_easy_ham_2.tar.bz2"
HAM3_URL = DOWNLOAD_ROOT + "20030228_hard_ham.tar.bz2"
HAM_URL = [HAM1_URL, HAM2_URL, HAM3_URL]
HAM_FILENAME = ["ham1.tar.bz2", "ham2.tar.bz2", "ham3.tar.bz2"]
SPAM1_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM2_URL = DOWNLOAD_ROOT + "20030228_spam_2.tar.bz2"
SPAM_URL = [SPAM1_URL, SPAM2_URL]
SPAM_FILENAME = ["spam1.tar.bz2", "spam2.tar.bz2"]
PATH = "./data/"

# Show % download
def report(count, blockSize, totalSize):
    percent = int(count*blockSize*100/totalSize)
    sys.stdout.write("\r%d%%" % percent + ' complete')
    sys.stdout.flush()

# Fetch data
def fetch_data(ham_url=HAM_URL, ham_fn = HAM_FILENAME,  spam_url=SPAM_URL, spam_fn = SPAM_FILENAME,  data_path=PATH):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    print("\nDownloading Spam messages")
    download_data(spam_fn, spam_url, data_path, "spam")
    print("\nDownloading Ham messages")
    download_data(ham_fn, ham_url, data_path, "ham")

def download_data(fns, urls, p, type):
    for (filename, url) in zip(fns, urls):
        path = os.path.join(p, filename)
        print("Downloading from:", url)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path, reporthook=report)
        print("\nUnzipping dataset...")
        tar_bz2_file = tarfile.open(path)
        # remove the path by reset it
        members = []
        for member in tar_bz2_file.getmembers():
            if member.isreg():
                member.name = os.path.basename(member.name)  # remove the path by reset it
                members.append(member)
        tar_bz2_file.extractall(path=p+type, members=members)
        tar_bz2_file.close()
    print("All successful! Yeah!")


fetch_data()


print("\nDone. Have fun using ML!")




from setuptools import find_packages, setup

# read the contents of README file
from os import path
from io import open
import sys, re

# get __version__ from version.py  
try:
    verpath = path.join('pythresh', 'version.py')
    version_file = open(verpath)
    __version__ = str(re.findall(r'\b\d+(?:\.\d+)+', version_file.read())[0])
    version_file.close() 

except Exception as error:
    __version__ = "0.0.1"
    sys.stderr.write("Warning: Could not open '%s' due %s\n" % (verpath, error))
     

this_directory = path.abspath(path.dirname(__file__))


# read the contents of README.rst
def readme():
    with open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
        return f.read()


# read the contents of requirements.txt
with open(path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='pythresh',
    version=__version__,
    description='A Python Toolbox for Outlier Detection Thresholding',
    long_description=readme(),
    long_description_content_type='text/x-rst',
    author='D Kulik',
    url='https://github.com/KulikDM/pythresh',
    download_url='https://github.com/KulikDM/pythresh/archive/master.zip',
    keywords=['outlier detection', 'anomaly detection', 'thresholding', 'cutoff', 
              'contamintion level', 'data science', 'machine learning'],
    project_urls={"Documentation": 'https://pythresh.readthedocs.io/en/latest/'},
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    install_requires=requirements,
    setup_requires=['setuptools>=38.6.0'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)

from setuptools import find_packages, setup

# read the contents of README file
from os import path
from io import open  # for Python 3 compatibility

# get __version__ from _version.py
ver_file = path.join('pythresh', 'version.py')
with open(ver_file) as f:
    exec(f.read())

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
    keywords=['outlier detection', 'anomaly detection', 'outlier ensembles',
              'thresholding', 'cutoff'],
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
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)

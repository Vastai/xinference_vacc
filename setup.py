import setuptools
import subprocess
from pathlib import Path
import datetime
from distutils.sysconfig import get_python_lib
import os
import sys
from setuptools import find_packages

relative_site_packages = os.path.relpath(get_python_lib(), sys.prefix)

def get_local_version_suffix() -> str:
    if not (Path(__file__).parent / ".git").is_dir():
        # Most likely installing from a source distribution
        return ""
    date_suffix = datetime.datetime.now().strftime("%Y%m%d")
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent
    ).decode("ascii")[:-1]
    return f"{git_hash}.d{date_suffix}"


setuptools.setup(
    name="xinference_vacc",
    version="1.5.1",
    author="Vastai AIS",
    author_email="ais@vastaitech.com",
    description=f"An adaptor for align toolkits patch on VACC GPU with version info: {get_local_version_suffix()}",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=[
        "setuptools>=61.0.0",
        "wheel",
        "xoscar==0.6.2",
        "sentence-transformers==4.1.0",
        "wrapt",
        "loguru",
        "xinference==1.5.1"
    ],
    extras_require={
        "dev": [
            "black>=23.0",
            "flake8>=6.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "xinference=xinference_vacc.adaptor:cli",
            "xinference-local=xinference_vacc.adaptor:local",
            "xinference-supervisor=xinference_vacc.adaptor:supervisor",
            "xinference-worker=xinference_vacc.adaptor:worker",
        ]
    }
)


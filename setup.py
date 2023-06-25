from setuptools import find_packages,setup
from typing import List
from pathlib import Path

Root_dir = Path(__file__).resolve().parent
requirements_file = Root_dir/"requirements.txt"

def get_requirements(file_path:str)->List[str]:
    """_summary_

    Parameters
    ----------
    file_path : str
        get the file path of requirements.txt file from root directory

    Returns
    -------
    List[str]
        Produces a list of all packages that need to be installed
    """
    Hyphen_e_dot = "-e ."
    with open(file_path) as file_obj:
        packages = file_obj.read().splitlines()
        if Hyphen_e_dot in packages:
            packages.remove(Hyphen_e_dot)
        
    return packages


setup(  
    name= "ML_deployment",
    version ="0.0.1",
    author="abhinav",
    author_email="abhinavthorat15@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements(requirements_file)
)

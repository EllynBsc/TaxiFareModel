from setuptools import find_packages
from setuptools import setup

# REQUIRED_PACKAGES = [
#     'pandas==1.1.3',
#     'scikit-learn==0.23.2'
# ]

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(
    name='TaxiFareModel',
    version='1.0',
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    description='Taxi Fare Prediction Pipeline'
)

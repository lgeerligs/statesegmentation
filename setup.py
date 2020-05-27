from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='statesegmentation',
    version='0.0.1',
    packages=['statesegmentation'],
    url='https://github.com/lgeerligs/statesegmentation',
    license='MIT',
    author='Linda Geerligs, Umut Güçlü',
    description='Detecting neural state transitions underlying event segmentation',
    long_description=long_description,
    long_description_content_type="text/markdown"
)

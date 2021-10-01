from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='statesegmentation',
    version='0.0.4',
    packages=['statesegmentation'],
    url='https://github.com/lgeerligs/statesegmentation',
    license='MIT',
    author='Linda Geerligs, Umut Güçlü',
    description='Detecting neural state transitions underlying event segmentation',
    long_description=long_description,
    long_description_content_type="text/markdown"
)

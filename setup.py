from setuptools import setup, find_packages
import gan_utils

packages = find_packages(
        where='.',
        include=['gan_utils*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='gan_utils',
    version=gan_utils.__version__,
    description='GAN testing',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cm107/gan_utils",
    author='Clayton Mork',
    author_email='mork.clayton3@gmail.com',
    license='MIT License',
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'pylint==2.4.4',
        'torch',
        'torchvision',
        'pyclay-annotation_utils @ https://github.com/cm107/annotation_utils/archive/development.zip'
    ],
    python_requires='>=3.7'
)

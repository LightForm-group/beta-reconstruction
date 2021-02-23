from setuptools import setup, find_packages

setup(
    name='beta_reconstruction',
    version='0.1',
    packages=find_packages(exclude=["tests"]),
    url='github.com/LightForm-group/beta-reconstruction',
    license='MIT',
    author='Michael D. Atkinson',
    author_email='michael.atkinson@manchester.ac.uk',
    install_requires=[
        'numpy',
        'networkx',
        'defdap>=0.93.0',
        'tqdm',
    ],
    extras_require={
        'testing': ['pytest']
    }
)

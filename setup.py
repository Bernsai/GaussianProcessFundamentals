from setuptools import setup

INSTALL_REQUIRES = [
    'numpy >= 1.18',
    'pandas >= 1.0.5',
    'tensorflow >= 2.4',
    'tensorflow-probability >= 0.12',
    'matplotlib >= 3.2.2',
]

setup(
    name='gpbasics',
    version='2.0.0',
    packages=['gpbasics', 'gpbasics.Metrics', 'gpbasics.Auxiliary', 'gpbasics.Optimizer', 'gpbasics.Statistics',
              'gpbasics.DataHandling', 'gpbasics.DataHandling.Normalization', 'gpbasics.KernelBasics',
              'gpbasics.MeanFunctionBasics'],
    package_dir={'': 'main'},
    url='URL',
    license='MIT License',
    author='Fabian Berns',
    author_email='fabian.berns@googlemail.com',
    description='',
    install_requires=INSTALL_REQUIRES,
)

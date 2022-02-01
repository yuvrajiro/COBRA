from setuptools import setup

setup(name='COBRA',
    version='0.1',
    description='COBRA',
    author='Rahul Goswami',
    author_email='yuvrajiro@gmail.com',
    license='MIT',
    packages=['COBRA'],
    install_requires=['scikit-learn', 'numpy','tqdm'],
    zip_safe=False)
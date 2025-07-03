# optimization_library/setup.py

from setuptools import setup, find_packages

def readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

setup(
    name='optimization-library',
    version='1.0.0',
    author='UwuFlames',
    author_email='uwaterflames@gmail.com',
    description='A Python library for linear, nonlinear, and integer programming',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/UWFms/optimization-library',
    packages=find_packages(),
    install_requires=[
        'numpy~=2.3.0',
        'cvxpy~=1.6.5',
        'pandas~=2.3.0',
        'matplotlib~=3.10.3',
        'PuLP~=3.2.1',
        'scipy~=1.15.3',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    keywords='optimization linear-programming nonlinear-programming integer-programming',
    project_urls={
        'Documentation': 'https://disk.yandex.ru/i/k7wqcUCCx1Ym9Q',
        'Source': 'https://github.com/UWFms/optimization-library',
    },
    python_requires='>=3.7',
)
from setuptools import setup, find_packages

setup(
    name='HamiltonianSnippets',
    version='0.1.0',
    description='A Python package for sampling with Hamiltonian Snippets',
    author='Mauro Camara Escudero',
    author_email='maurocamaraescudero@gmail.com',
    url='https://github.com/MauroCE/HamiltonianSnippets',
    packages=find_packages(),
    install_requires=[
        'matplotlib==3.9.2',
        'numpy==2.0.2',
        'scipy==1.14.1',
        'setuptools==75.2.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

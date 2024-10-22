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
        'contourpy==1.3.0',
        'cycler==0.12.1',
        'fonttools==4.54.1',
        'kiwisolver==1.4.7',
        'llvmlite==0.43.0',
        'matplotlib==3.9.2',
        'numba==0.60.0',
        'numpy==2.0.2',
        'packaging==24.1',
        'pandas==2.2.3',
        'pillow==10.4.0',
        'pyparsing==3.1.4',
        'python-dateutil==2.9.0.post0',
        'pytz==2024.2',
        'scipy==1.14.1',
        'seaborn==0.13.2',
        'setuptools==75.2.0',
        'six==1.16.0',
        'tzdata==2024.2'
    ],
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

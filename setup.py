"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')


setup(
    name='imripy',  # Required
    version='0.5.0',  # Required

    description='A python project to calulcate the physics of an Intermediate Mass Ratio Inspiral',  # Optional

    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)

#    url='https://github.com/LauraSagunski/DM-density-spikes-GWs-BH-shadows',  # Optional

    author='Niklas Becker',  # Optional

    author_email='nbecker@itp.uni-frankfurt.de',  # Optional

    classifiers=[
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Researchers',
        'Topic :: Physics Sim',

        # Pick your license as you wish
        'License ::  MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],


    keywords='simulation, physics, gravitational waves',  # Optional

    # When your source code is in a subdirectory under the project root, e.g.
    # `src/`, it is necessary to specify the `package_dir` argument.
    package_dir={'': 'src'},  # Optional


    packages=find_packages(where='src'),  # Required

    python_requires='>=3.6, <4',

    install_requires=['numpy', 'scipy', 'matplotlib'],  # Optional

)


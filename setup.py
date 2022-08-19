from setuptools import find_packages, setup

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setup(
    name='dempy',
    version='0.1.0',
    author='Johan Medrano',
    python_requires='>=3.4',
    author_email='',
    description='',
    platforms='Linux',
    packages=find_packages(),
    install_requires=(
        "numpy", 
        "scipy", 
        "symengine",
        "tqdm",
        "plotly",
        'plotly_express'
    ),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)

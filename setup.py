from setuptools import setup

setup(
    name='ClarityAI',
    version='1.0.0',
    author='Xiyue Zhang',
    description='ClarityAI is a Python package designed to empower machine learning practitioners with a wide range of interpretability methods to enhance the transparency and explainability of their ML models.',
    url='https://github.com/JasmineZhangxyz/clarityai-pypkg',
    packages=['clarityai'],
    install_requires=[
        'dependency1',
        'dependency2',
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
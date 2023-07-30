from setuptools import setup, find_packages

setup(
    name='ClarityAI',
    version='1.0.0',
    author=['Xiyue Zhang', 'Anna Teresa Lai'],
    author_email='uoft.clarityai@gmail.com'
    description='ClarityAI is a Python package designed to empower machine learning practitioners with a wide range of interpretability methods to enhance the transparency and explainability of their ML models.',
    url='https://github.com/JasmineZhangxyz/clarityai-pypkg',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'numpy',
        'opencv-python',
        'matplotlib',
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
    keywords=['python package', 'machine learning', 'cnn', 'interpretability', 'xai', 'attention maps', 'saliency maps']
    python_requires='>=3.6',
    project_urls={
        'Bug Reports': 'https://github.com/JasmineZhangxyz/clarityai-pypkg/issues',
        'Source': 'https://github.com/JasmineZhangxyz/clarityai-pypkg',
    },
)
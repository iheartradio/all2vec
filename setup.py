from setuptools import setup, find_packages


setup(
    name='all2vec',
    version='0.4.0',
    author='Ravi Mody, Jon Banafato',
    author_email='datascience@iheartmedia.com',
    description='Store and compare high dimensional vectors',
    packages=find_packages(exclude=['tests']),
    zip_safe=True,
    install_requires=[
        'annoy',
        'boto3',
        'dill',
        'numpy',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: POSIX',

        'Programming Language :: Python',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3 :: Only',
    ]
)

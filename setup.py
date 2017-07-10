from setuptools import setup, find_packages


setup(
    name='all2vec',
    version='0.5.0',
    author='Ravi Mody, Jon Banafato',
    author_email='datascience@iheartmedia.com',
    description='Store and compare high dimensional vectors',
    packages=find_packages(exclude=['tests']),
    zip_safe=True,
    install_requires=[
        'all2vec==0.3.0'
        , 'annoy==1.8.3'
        , 'appdirs==1.4.3'
        , 'boto==2.47.0'
        , 'boto3==1.4.4'
        , 'botocore==1.5.55'
        , 'cookies==2.2.1'
        , 'coverage==4.4.1'
        , 'dicttoxml==1.7.4'
        , 'dill==0.2.6'
        , 'docutils==0.13.1'
        , 'Jinja2==2.9.6'
        , 'jmespath==0.9.2'
        , 'MarkupSafe==1.0'
        , 'mock==2.0.0'
        , 'moto==1.0.0'
        , 'numpy==1.12.1'
        , 'packaging==16.8'
        , 'pbr==3.0.1'
        , 'py==1.4.33'
        , 'pyaml==16.12.2'
        , 'pyparsing==2.2.0'
        , 'pytest==3.1.0'
        , 'python-dateutil==2.6.0'
        , 'pytz==2017.2'
        , 'PyYAML==3.12'
        , 'requests==2.14.2'
        , 's3transfer==0.1.10'
        , 'six==1.10.0'
        , 'Werkzeug==0.12.2'
        , 'xmltodict==0.11.0'
        
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

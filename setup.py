from distutils.core import setup

with open('requirements.txt') as fp:
    install_requires = fp.read()

setup(
    name='EndToEndClassification',
    version='0.1',
    packages=['EndToEndClassification', 'EndToEndClassification/Dataset', 'EndToEndClassification/EnvClassification',
              'EndToEndClassification/MSTmodel', 'EndToEndClassification/Utilities',
              'EndToEndClassification/EnvClassification/Models'],
    url='',
    license='Apache License Version 2.0',
    author='Tycho_Tax_Corti',
    author_email='tt@cortilabs.com',
    description='',
    install_requires=install_requires
)

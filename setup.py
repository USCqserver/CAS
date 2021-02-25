from setuptools import setup, find_packages

VERSION = '0.1'
DESCRIPTION = 'CAS: Customized Annealing Schedules'
LONG_DESCRIPTION = open('README.md').read()

REQUIREMENTS = open('requirements.txt').readlines()

setup(
	name='cas',
	url='https://github.com/USCqserver/CAS',
	version=VERSION,
	author='Mostafa Khezri',
	author_email='mostafanodet@gmail.com',
	description=DESCRIPTION,
	long_description=LONG_DESCRIPTION,
	long_description_content_type="text/markdown",
	packages=find_packages(),
	python_requires=">=3.6",
	install_requires=REQUIREMENTS,
	license='LICENSE.txt'
)

from setuptools import find_packages, setup

setup(
	name='neuralpy',
	version='1.0.0',
	description='simple and easy fully-connected multilayer feedforward neural network model',
	author='Jonathan N. Lee',
	author_email='jonathan_lee@berkeley.edu',
	url='https://github.com/jon--lee/neuralpy',
	license='MIT',
	classifiers=[
		'Development Status :: 3 - Alpha',
		'Intended Audience :: Developers',
		'Topic :: Software Development :: Build Tools',
		'Programming Language :: Python :: 2.7',
		],
	install_requires=['numpy==1.9.2', 'matplotlib==1.4.3'],
	packages=find_packages()
	)



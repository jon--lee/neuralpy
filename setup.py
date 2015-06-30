from setuptools import setup

setup(
	name='py-net',
	version='1.0.0',
	description='neural network model',
	author='Jonathan N. Lee',
	author_email='jonathan_lee@berkeley.edu',
	url='https://github.com/jon--lee/py-net',
	license='MIT',
	classifiers=[
		'Development Status :: 3 - Alpha',
		'Intended Audience :: Developers',
		'Topic :: Software Development :: Build Tools',
		'Programming Language :: Python :: 2.7',
		],
	install_requires=['numpy', 'matplotlib'],
	packages=['py-net']
	)

from setuptools import find_packages, setup
print find_packages(exclude=['DEPneuralpy'])
setup(
	name='neuralpy',
	version='1.3.0',
        description='neuralpy - The most intuitive Neural Network Model',
	author='Jonathan N. Lee',
        keywords='neuralpy neural networks',
	author_email='jonathan_lee@berkeley.edu',
	url='https://github.com/jon--lee/neuralpy',
	license='MIT',
	classifiers=[
		'Development Status :: 3 - Alpha',
		'Intended Audience :: Developers',
		'Intended Audience :: Science/Research',
		'Topic :: Software Development',
		'Topic :: Scientific/Engineering',
		'Topic :: Scientific/Engineering :: Artificial Intelligence',
		'Programming Language :: Python',
		'Programming Language :: Python :: 2.7',
		],
	install_requires=['numpy==1.22.0', 'matplotlib==1.4.3'],
        packages=find_packages(exclude=['DEPneuralpy'])
)

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

install_requires = ['dipy', 'numpy', 'scikit-image', 'scipy']
docs_requires = ['Sphinx', 'sphinx-rtd-theme']

setuptools.setup(
    name='dmriphantomutils',
    version='0.0.1',
    author='Tristan Kuehn',
    author_email='tkuehn@uwo.ca',
    url='https://github.com/tkkuehn/dmri-phantom-utilities',
    license='BSD-3-Clause',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering'
    ],
    description='3D printed diffusion phantom analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(exclude='test'),
    install_requires=install_requires,
    extras_require={'docs': docs_requires}
)


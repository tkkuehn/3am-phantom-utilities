import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dmriphantomutils",
    version="0.0.1",
    author="Tristan Kuehn",
    author_email="tkuehn@uwo.ca",
    description="DKI analysis for diffusion phantoms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude="test"),
    install_requires=['dipy', 'matplotlib', 'numpy', 'scikit-image', 'scipy']
)


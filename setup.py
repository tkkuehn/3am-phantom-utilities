import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="phantomdki",
    version="0.0.1",
    author="Tristan Kuehn",
    author_email="tkuehn@uwo.ca",
    description="DKI analysis for diffusion phantoms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=['nibabel', 'dipy', 'skimage']
)


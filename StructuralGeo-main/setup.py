from setuptools import find_packages, setup

setup(
    name="GeoGen",
    version="0.1.0",
    description="A package for creating, visualizing, and exporting 3D structural geology models. \
    Allows either user specified, or randomized generation of models.",
    packages=find_packages(where="src"),  # Look for packages in the 'src' directory
    package_dir={"": "src"},  # Root package directory is 'src'
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pyvista[all]",
        "pyvistaqt",
        "ipywidgets",
        "ipykernel",
        "trame",
        "trame-vuetify",
        "trame-vtk",
        "tqdm",
        "PyDTMC",
    ],
    package_data={
        "geogen.generation.markov_matrix": ["default_markov_matrix.csv"],
    },
    include_package_data=True,
)

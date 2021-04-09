from setuptools import setup, find_packages

packages = find_packages(
    where='.',
    include=['ai_practice*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ai_practice",
    version="0.1.0",
    author="Jitesh Gosar",
    author_email="gosar95@gmail.com",
    description="Tools for RL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jitesh17/ai_practice",
    py_modules=["ai_practice"],
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "printj",
        "pyjeasy",
        "torch",
        "torchvision",
        "opencv-python",
        "numpy",
        "pandas",
        "seaborn",
        "funcy",
        "stable_baselines3"
    ],
    python_requires='>=3.6',
)

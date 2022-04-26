from setuptools import setup

setup(
    name="dqn-pacman",
    python_requires=">=3.8.5",
    install_requires=[
        "matplotlib==3.4.2",
        "opencv-python==4.1.2.30",
        "numpy>=1.20.3",
        "torch>=1.11.0",
        "torchaudio>=0.11.0",
        "torchvision>=0.12.0",
    ],
    license="GNU LGPL v3",
    author="Benjamin Bourbon",
    author_email="ben.bourbon06@gmail.com",
    description="Deep Q-Network on the Atari Game Ms-Pacman",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bourbonut/dqn-pacman",
)

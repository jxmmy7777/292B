from setuptools import setup, find_packages

setup(
    name='me292b',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A brief description of the 292B project',
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here.
        # They will be installed by pip when your project is installed.
        'pytorch-lightning==1.8.3.post0',
        "torchvision",
        "protobuf==3.20.1",
        "tensorboard",
        # Add other dependencies as needed
    ],
)

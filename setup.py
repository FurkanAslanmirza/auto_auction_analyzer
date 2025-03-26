from setuptools import setup, find_packages

setup(
    name="auto_auction_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Lesen Sie Abhängigkeiten aus der requirements.txt
        line.strip() for line in open("requirements.txt")
        if not line.startswith("#")
    ],
    author="EasyTech",
    author_email="easytech.furkan@gmail.com",
    description="Tool zur Analyse von Fahrzeugauktionen und Marktwerten",
)
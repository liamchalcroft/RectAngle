import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RectAngle", # Replace with your own username
    version="0.0.1",
    author="Liam Chalcroft",
    author_email="liam.chalcroft.20@ucl.ac.uk",
    description="Ultrasound prostate image analysis using CNNs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liamchalcroft/RectAngle",
    project_urls={
        "Bug Tracker": "https://github.com/liamchalcroft/RectAngle/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "rectangle"},
    packages=[setuptools.find_packages(where="rectangle"),
    python_requires=">=3.6",
)
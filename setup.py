from setuptools import setup, find_packages


# requires = [
#     'numpy>=1.22.3',
#     'tifffile>=2023.2.3',
#     'scikit-image>=0.21.0',
#     'opencv-python>=4.8.0.76',
#     'onnxruntime>=1.15.1',
#     'tqdm>=4.65.0',
#     'scikit-learn>=1.3.0',
#     'imagecodecs>=2023.3.16',
#     'pandas>=1.5.3',
#     'matplotlib>=3.7.1',
#     'pyvips>=2.2.1',
#     'numba==0.56.4',
# ]

with open('requirements.txt') as f:
    requires = f.read().splitlines()

# Parse version number from cellbin/__init__.py:
with open('cellbin/__init__.py') as f:
    info = {}
    for line in f:
        if line.startswith('version'):
            exec(line, info)
            break

print(f"Version: {info['version']}")


setup(
    name='cell bin',
    version=info['version'],
    description='A framework for generating single-cell gene expression data',
    long_description_content_type="text/markdown",
    long_description=open('README.md').read(),
    packages=find_packages(),
    author='cell bin research group',
    author_email='bgi@genomics.cn',
    url='https://gitlab.genomics.cn/biointelligence/implab/stero-rnd/cellbin/algorithms/cellbin/-/tree/dev',
    install_requires=requires,
    python_requires='==3.8.*',
    include_package_data=True,
    data_files = [
        ('', ['cellbin/modules/chip_file/ori.S6.6.8fov.txt',
              'cellbin/modules/chip_file/ori.S13.6.8fov.txt',
              'cellbin/modules/chip_file/S13_track_points.txt',
              'cellbin/modules/chip_file/SN_fov_extend.txt'])
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
    ],

  )

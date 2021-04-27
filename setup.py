from setuptools import setup



VERSION="1.0.0"

setup(
    name = 'simple_tsne',
    version = VERSION,
    packages = ['simple_tsne'],
    license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description = 'numpy implementation of the original T-SNE paper',
    author = 'Abdullah AlOthman, Ethan Swartzentruber',
    author_email = 'eswartzen@gmail.com',
    url = 'https://github.com/epswartz/t-SNE',
    download_url=f'https://github.com/epswartz/t-SNE/archive/{VERSION}.zip',
    keywords = ['t-sne', 'manifold', 'visualization', 'numpy'],
    install_requires=[
        'numpy',
        'tqdm'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable', # "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
        'License :: OSI Approved :: MIT License'
    ],
    python_requires='>=3.6',
)
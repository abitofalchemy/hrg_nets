import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = []

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name='phoenix',
    version='0.0.001',
    url='https://github.com/nddsg/phoenix',
    license='MIT',
    description="A lossy graph compression and decompression algorithm",
    packages=['phoenix'],
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    install_requires=[],
    keywords='markov chain chains graph compression and decompression algorithm',
    author='Rodrigo Palacios',
    author_email='rodrigopala91@gmail.com',
    scripts=[],
    package_data={},
    tests_require=['pytest'],
    cmdclass={'test': PyTest},
)
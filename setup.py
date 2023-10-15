from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize

try:
    import sage.env
except ImportError:
    raise ValueError("this package requires SageMath")

def do_cythonize():
    return cythonize(
            [Extension(
                "*",
                ["gen/*.pyx"],
            )],
            aliases = sage.env.cython_aliases(),
            compiler_directives={'language_level' : "3"},
        )

try:
    from sage.misc.package_dir import cython_namespace_package_support
    with cython_namespace_package_support():
        extensions = do_cythonize()
except ImportError:
    extensions = do_cythonize()

setup(
    name = "test",
    packages = [
        "gen",
    ],
    ext_modules = extensions,
    include_dirs = sage.env.sage_include_directories(),
    zip_safe=False,
)

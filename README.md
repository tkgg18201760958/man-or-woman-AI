# 神经网络实现性别识别

## 文件介绍

### 文件结构

```
G:.
├─.venv
│  ├─Include
│  ├─Lib
│  │  └─site-packages
│  │      ├─cv2
│  │      │  ├─aruco
│  │      │  ├─barcode
│  │      │  ├─cuda
│  │      │  ├─data
│  │      │  │  └─__pycache__
│  │      │  ├─detail
│  │      │  ├─dnn
│  │      │  ├─Error
│  │      │  ├─fisheye
│  │      │  ├─flann
│  │      │  ├─gapi
│  │      │  │  ├─core
│  │      │  │  │  ├─cpu
│  │      │  │  │  ├─fluid
│  │      │  │  │  └─ocl
│  │      │  │  ├─ie
│  │      │  │  │  └─detail
│  │      │  │  ├─imgproc
│  │      │  │  │  └─fluid
│  │      │  │  ├─oak
│  │      │  │  ├─onnx
│  │      │  │  │  └─ep
│  │      │  │  ├─ot
│  │      │  │  │  └─cpu
│  │      │  │  ├─ov
│  │      │  │  ├─own
│  │      │  │  │  └─detail
│  │      │  │  ├─render
│  │      │  │  │  └─ocv
│  │      │  │  ├─streaming
│  │      │  │  ├─video
│  │      │  │  ├─wip
│  │      │  │  │  ├─draw
│  │      │  │  │  ├─gst
│  │      │  │  │  └─onevpl
│  │      │  │  └─__pycache__
│  │      │  ├─ipp
│  │      │  ├─mat_wrapper
│  │      │  │  └─__pycache__
│  │      │  ├─misc
│  │      │  │  └─__pycache__
│  │      │  ├─ml
│  │      │  ├─ocl
│  │      │  ├─ogl
│  │      │  ├─parallel
│  │      │  ├─samples
│  │      │  ├─segmentation
│  │      │  ├─typing
│  │      │  │  └─__pycache__
│  │      │  ├─utils
│  │      │  │  ├─fs
│  │      │  │  ├─nested
│  │      │  │  └─__pycache__
│  │      │  ├─videoio_registry
│  │      │  └─__pycache__
│  │      ├─jax
│  │      │  ├─example_libraries
│  │      │  │  └─__pycache__
│  │      │  ├─experimental
│  │      │  │  ├─array_api
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─array_serialization
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─compilation_cache
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─export
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─jax2tf
│  │      │  │  │  ├─examples
│  │      │  │  │  │  ├─serving
│  │      │  │  │  │  │  └─__pycache__
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─tests
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─key_reuse
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─mosaic
│  │      │  │  │  ├─gpu
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─pallas
│  │      │  │  │  ├─ops
│  │      │  │  │  │  ├─gpu
│  │      │  │  │  │  │  └─__pycache__
│  │      │  │  │  │  ├─tpu
│  │      │  │  │  │  │  ├─megablox
│  │      │  │  │  │  │  │  └─__pycache__
│  │      │  │  │  │  │  ├─paged_attention
│  │      │  │  │  │  │  │  └─__pycache__
│  │      │  │  │  │  │  ├─splash_attention
│  │      │  │  │  │  │  │  └─__pycache__
│  │      │  │  │  │  │  └─__pycache__
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─sparse
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─extend
│  │      │  │  ├─core
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─mlir
│  │      │  │  │  ├─dialects
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─image
│  │      │  │  └─__pycache__
│  │      │  ├─interpreters
│  │      │  │  └─__pycache__
│  │      │  ├─lax
│  │      │  │  └─__pycache__
│  │      │  ├─lib
│  │      │  │  └─__pycache__
│  │      │  ├─nn
│  │      │  │  └─__pycache__
│  │      │  ├─numpy
│  │      │  │  └─__pycache__
│  │      │  ├─ops
│  │      │  │  └─__pycache__
│  │      │  ├─scipy
│  │      │  │  ├─cluster
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─interpolate
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─optimize
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─sparse
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─spatial
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─stats
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─tools
│  │      │  │  └─__pycache__
│  │      │  ├─_src
│  │      │  │  ├─clusters
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─cudnn
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─debugger
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─export
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─extend
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─image
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─internal_test_util
│  │      │  │  │  ├─lazy_loader_module
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─interpreters
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─lax
│  │      │  │  │  ├─control_flow
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─lib
│  │      │  │  │  ├─mlir
│  │      │  │  │  │  ├─dialects
│  │      │  │  │  │  │  └─__pycache__
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─nn
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─numpy
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─ops
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─pallas
│  │      │  │  │  ├─mosaic
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─mosaic_gpu
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─triton
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─scipy
│  │      │  │  │  ├─cluster
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─interpolate
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─optimize
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─sparse
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─spatial
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─stats
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─state
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─third_party
│  │      │  │  │  ├─scipy
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  └─__pycache__
│  │      ├─jax-0.4.35.dist-info
│  │      ├─jaxlib
│  │      │  ├─cpu
│  │      │  ├─include
│  │      │  │  └─xla
│  │      │  │      └─ffi
│  │      │  │          └─api
│  │      │  ├─mlir
│  │      │  │  ├─dialects
│  │      │  │  │  ├─gpu
│  │      │  │  │  │  ├─passes
│  │      │  │  │  │  │  └─__pycache__
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─extras
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─_mlir_libs
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─mosaic
│  │      │  │  └─python
│  │      │  │      └─__pycache__
│  │      │  ├─triton
│  │      │  │  └─__pycache__
│  │      │  ├─xla_extension
│  │      │  └─__pycache__
│  │      ├─jaxlib-0.4.35.dist-info
│  │      ├─ml_dtypes
│  │      │  └─__pycache__
│  │      ├─ml_dtypes-0.5.0.dist-info
│  │      ├─numpy
│  │      │  ├─char
│  │      │  │  └─__pycache__
│  │      │  ├─compat
│  │      │  │  ├─tests
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─core
│  │      │  │  └─__pycache__
│  │      │  ├─doc
│  │      │  │  └─__pycache__
│  │      │  ├─f2py
│  │      │  │  ├─src
│  │      │  │  ├─tests
│  │      │  │  │  ├─src
│  │      │  │  │  │  ├─abstract_interface
│  │      │  │  │  │  ├─array_from_pyobj
│  │      │  │  │  │  ├─assumed_shape
│  │      │  │  │  │  ├─block_docstring
│  │      │  │  │  │  ├─callback
│  │      │  │  │  │  ├─cli
│  │      │  │  │  │  ├─common
│  │      │  │  │  │  ├─crackfortran
│  │      │  │  │  │  ├─f2cmap
│  │      │  │  │  │  ├─isocintrin
│  │      │  │  │  │  ├─kind
│  │      │  │  │  │  ├─mixed
│  │      │  │  │  │  ├─modules
│  │      │  │  │  │  │  ├─gh25337
│  │      │  │  │  │  │  └─gh26920
│  │      │  │  │  │  ├─negative_bounds
│  │      │  │  │  │  ├─parameter
│  │      │  │  │  │  ├─quoted_character
│  │      │  │  │  │  ├─regression
│  │      │  │  │  │  ├─return_character
│  │      │  │  │  │  ├─return_complex
│  │      │  │  │  │  ├─return_integer
│  │      │  │  │  │  ├─return_logical
│  │      │  │  │  │  ├─return_real
│  │      │  │  │  │  ├─size
│  │      │  │  │  │  ├─string
│  │      │  │  │  │  └─value_attrspec
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─_backends
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─fft
│  │      │  │  ├─tests
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─lib
│  │      │  │  ├─tests
│  │      │  │  │  ├─data
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─linalg
│  │      │  │  ├─tests
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─ma
│  │      │  │  ├─tests
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─matrixlib
│  │      │  │  ├─tests
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─polynomial
│  │      │  │  ├─tests
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─random
│  │      │  │  ├─lib
│  │      │  │  ├─tests
│  │      │  │  │  ├─data
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─_examples
│  │      │  │  │  ├─cffi
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─cython
│  │      │  │  │  └─numba
│  │      │  │  │      └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─rec
│  │      │  │  └─__pycache__
│  │      │  ├─strings
│  │      │  │  └─__pycache__
│  │      │  ├─testing
│  │      │  │  ├─tests
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─_private
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─tests
│  │      │  │  └─__pycache__
│  │      │  ├─typing
│  │      │  │  ├─tests
│  │      │  │  │  ├─data
│  │      │  │  │  │  ├─fail
│  │      │  │  │  │  ├─misc
│  │      │  │  │  │  ├─pass
│  │      │  │  │  │  │  └─__pycache__
│  │      │  │  │  │  └─reveal
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─_core
│  │      │  │  ├─include
│  │      │  │  │  └─numpy
│  │      │  │  │      └─random
│  │      │  │  ├─lib
│  │      │  │  │  ├─npy-pkg-config
│  │      │  │  │  └─pkgconfig
│  │      │  │  ├─tests
│  │      │  │  │  ├─data
│  │      │  │  │  ├─examples
│  │      │  │  │  │  ├─cython
│  │      │  │  │  │  │  └─__pycache__
│  │      │  │  │  │  └─limited_api
│  │      │  │  │  │      └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─_pyinstaller
│  │      │  │  ├─tests
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─_typing
│  │      │  │  └─__pycache__
│  │      │  ├─_utils
│  │      │  │  └─__pycache__
│  │      │  └─__pycache__
│  │      ├─numpy-2.1.3.dist-info
│  │      ├─numpy.libs
│  │      ├─opencv_python-4.10.0.84.dist-info
│  │      ├─opt_einsum
│  │      │  ├─backends
│  │      │  │  └─__pycache__
│  │      │  ├─tests
│  │      │  │  └─__pycache__
│  │      │  └─__pycache__
│  │      ├─opt_einsum-3.4.0.dist-info
│  │      │  └─licenses
│  │      ├─pip
│  │      │  ├─_internal
│  │      │  │  ├─cli
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─commands
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─distributions
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─index
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─locations
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─metadata
│  │      │  │  │  ├─importlib
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─models
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─network
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─operations
│  │      │  │  │  ├─build
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─install
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─req
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─resolution
│  │      │  │  │  ├─legacy
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─resolvelib
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─utils
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─vcs
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─_vendor
│  │      │  │  ├─cachecontrol
│  │      │  │  │  ├─caches
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─certifi
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─distlib
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─distro
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─idna
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─msgpack
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─packaging
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─pkg_resources
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─platformdirs
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─pygments
│  │      │  │  │  ├─filters
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─formatters
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─lexers
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─styles
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─pyproject_hooks
│  │      │  │  │  ├─_in_process
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─requests
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─resolvelib
│  │      │  │  │  ├─compat
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─rich
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─tomli
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─truststore
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─urllib3
│  │      │  │  │  ├─contrib
│  │      │  │  │  │  ├─_securetransport
│  │      │  │  │  │  │  └─__pycache__
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─packages
│  │      │  │  │  │  ├─backports
│  │      │  │  │  │  │  └─__pycache__
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─util
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  └─__pycache__
│  │      ├─pip-24.3.1.dist-info
│  │      ├─scipy
│  │      │  ├─cluster
│  │      │  │  ├─tests
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─constants
│  │      │  │  ├─tests
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─datasets
│  │      │  │  ├─tests
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─fft
│  │      │  │  ├─tests
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─_pocketfft
│  │      │  │  │  ├─tests
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─fftpack
│  │      │  │  ├─tests
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─integrate
│  │      │  │  ├─tests
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─_ivp
│  │      │  │  │  ├─tests
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─interpolate
│  │      │  │  ├─tests
│  │      │  │  │  ├─data
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─io
│  │      │  │  ├─arff
│  │      │  │  │  ├─tests
│  │      │  │  │  │  ├─data
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─matlab
│  │      │  │  │  ├─tests
│  │      │  │  │  │  ├─data
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─tests
│  │      │  │  │  ├─data
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─_fast_matrix_market
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─_harwell_boeing
│  │      │  │  │  ├─tests
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─linalg
│  │      │  │  ├─tests
│  │      │  │  │  ├─data
│  │      │  │  │  ├─_cython_examples
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─misc
│  │      │  │  ├─tests
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─ndimage
│  │      │  │  ├─tests
│  │      │  │  │  ├─data
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─odr
│  │      │  │  ├─tests
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─optimize
│  │      │  │  ├─cython_optimize
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─tests
│  │      │  │  │  ├─_cython_examples
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─_highs
│  │      │  │  │  ├─src
│  │      │  │  │  │  └─cython
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─_lsq
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─_shgo_lib
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─_trlib
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─_trustregion_constr
│  │      │  │  │  ├─tests
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─signal
│  │      │  │  ├─tests
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─windows
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─sparse
│  │      │  │  ├─csgraph
│  │      │  │  │  ├─tests
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─linalg
│  │      │  │  │  ├─tests
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─_dsolve
│  │      │  │  │  │  ├─tests
│  │      │  │  │  │  │  └─__pycache__
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─_eigen
│  │      │  │  │  │  ├─arpack
│  │      │  │  │  │  │  ├─tests
│  │      │  │  │  │  │  │  └─__pycache__
│  │      │  │  │  │  │  └─__pycache__
│  │      │  │  │  │  ├─lobpcg
│  │      │  │  │  │  │  ├─tests
│  │      │  │  │  │  │  │  └─__pycache__
│  │      │  │  │  │  │  └─__pycache__
│  │      │  │  │  │  ├─tests
│  │      │  │  │  │  │  └─__pycache__
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─_isolve
│  │      │  │  │  │  ├─tests
│  │      │  │  │  │  │  └─__pycache__
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─_propack
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─tests
│  │      │  │  │  ├─data
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─spatial
│  │      │  │  ├─qhull_src
│  │      │  │  ├─tests
│  │      │  │  │  ├─data
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─transform
│  │      │  │  │  ├─tests
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─special
│  │      │  │  ├─special
│  │      │  │  │  └─cephes
│  │      │  │  ├─tests
│  │      │  │  │  ├─data
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─_cython_examples
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─_precompute
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─stats
│  │      │  │  ├─tests
│  │      │  │  │  ├─data
│  │      │  │  │  │  ├─levy_stable
│  │      │  │  │  │  ├─nist_anova
│  │      │  │  │  │  ├─nist_linregress
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─_levy_stable
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─_rcont
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─_unuran
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  ├─_lib
│  │      │  │  ├─array_api_compat
│  │      │  │  │  ├─common
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─cupy
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─dask
│  │      │  │  │  │  ├─array
│  │      │  │  │  │  │  └─__pycache__
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─numpy
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─torch
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─cobyqa
│  │      │  │  │  ├─subsolvers
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  ├─utils
│  │      │  │  │  │  └─__pycache__
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─tests
│  │      │  │  │  └─__pycache__
│  │      │  │  ├─_uarray
│  │      │  │  │  └─__pycache__
│  │      │  │  └─__pycache__
│  │      │  └─__pycache__
│  │      ├─scipy-1.14.1.dist-info
│  │      └─scipy.libs
│  └─Scripts
├─linux-venv
│  ├─bin
│  ├─include
│  │  └─python3.12
│  └─lib
│      └─python3.12
│          └─site-packages
│              ├─pip
│              │  ├─_internal
│              │  │  ├─cli
│              │  │  │  └─__pycache__
│              │  │  ├─commands
│              │  │  │  └─__pycache__
│              │  │  ├─distributions
│              │  │  │  └─__pycache__
│              │  │  ├─index
│              │  │  │  └─__pycache__
│              │  │  ├─locations
│              │  │  │  └─__pycache__
│              │  │  ├─metadata
│              │  │  │  ├─importlib
│              │  │  │  │  └─__pycache__
│              │  │  │  └─__pycache__
│              │  │  ├─models
│              │  │  │  └─__pycache__
│              │  │  ├─network
│              │  │  │  └─__pycache__
│              │  │  ├─operations
│              │  │  │  ├─build
│              │  │  │  │  └─__pycache__
│              │  │  │  ├─install
│              │  │  │  │  └─__pycache__
│              │  │  │  └─__pycache__
│              │  │  ├─req
│              │  │  │  └─__pycache__
│              │  │  ├─resolution
│              │  │  │  ├─legacy
│              │  │  │  │  └─__pycache__
│              │  │  │  ├─resolvelib
│              │  │  │  │  └─__pycache__
│              │  │  │  └─__pycache__
│              │  │  ├─utils
│              │  │  │  └─__pycache__
│              │  │  ├─vcs
│              │  │  │  └─__pycache__
│              │  │  └─__pycache__
│              │  ├─_vendor
│              │  │  ├─cachecontrol
│              │  │  │  ├─caches
│              │  │  │  │  └─__pycache__
│              │  │  │  └─__pycache__
│              │  │  ├─certifi
│              │  │  │  └─__pycache__
│              │  │  ├─distlib
│              │  │  │  └─__pycache__
│              │  │  ├─distro
│              │  │  │  └─__pycache__
│              │  │  ├─idna
│              │  │  │  └─__pycache__
│              │  │  ├─msgpack
│              │  │  │  └─__pycache__
│              │  │  ├─packaging
│              │  │  │  └─__pycache__
│              │  │  ├─pkg_resources
│              │  │  │  └─__pycache__
│              │  │  ├─platformdirs
│              │  │  │  └─__pycache__
│              │  │  ├─pygments
│              │  │  │  ├─filters
│              │  │  │  │  └─__pycache__
│              │  │  │  ├─formatters
│              │  │  │  │  └─__pycache__
│              │  │  │  ├─lexers
│              │  │  │  │  └─__pycache__
│              │  │  │  ├─styles
│              │  │  │  │  └─__pycache__
│              │  │  │  └─__pycache__
│              │  │  ├─pyproject_hooks
│              │  │  │  ├─_in_process
│              │  │  │  │  └─__pycache__
│              │  │  │  └─__pycache__
│              │  │  ├─requests
│              │  │  │  └─__pycache__
│              │  │  ├─resolvelib
│              │  │  │  ├─compat
│              │  │  │  │  └─__pycache__
│              │  │  │  └─__pycache__
│              │  │  ├─rich
│              │  │  │  └─__pycache__
│              │  │  ├─tenacity
│              │  │  │  └─__pycache__
│              │  │  ├─tomli
│              │  │  │  └─__pycache__
│              │  │  ├─truststore
│              │  │  │  └─__pycache__
│              │  │  ├─urllib3
│              │  │  │  ├─contrib
│              │  │  │  │  ├─_securetransport
│              │  │  │  │  │  └─__pycache__
│              │  │  │  │  └─__pycache__
│              │  │  │  ├─packages
│              │  │  │  │  ├─backports
│              │  │  │  │  │  └─__pycache__
│              │  │  │  │  └─__pycache__
│              │  │  │  ├─util
│              │  │  │  │  └─__pycache__
│              │  │  │  └─__pycache__
│              │  │  └─__pycache__
│              │  └─__pycache__
│              └─pip-24.1.1.dist-info
├─数学算法
└─测试
```

1. ### main.py

2. ### 测试

* 测试使用，可以忽略

3. ### 数学算法

* 存放神经网络所需算法，如sigmoid、softmax等

4. ### test.py

* 测试模型

5. ### predict.py

* 预测模型

## python 环境

### 安装所需要的依赖库

```
pip install numpy
pip install matplotlib
```

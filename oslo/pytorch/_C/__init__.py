from oslo.pytorch._C.binder import Binder

try:
    # make binding object
    binding = Binder().bind()

    # c++ objects
    CompileCache = binding.CompileCache

except Exception as e:
    print(
        "Failed compiling the C++ source code. Please check your C++ environments.\n"
        f"Error message: {e}"
    )

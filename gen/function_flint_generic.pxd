from sage.libs.flint.types cimport gr_ptr, gr_srcptr, gr_ctx_t

cdef extern from "flint_wrap.h":
    ctypedef int ((*gr_method_unary_op)(gr_ptr, gr_srcptr, gr_ctx_ptr))


cdef class FlintFunction:
    cdef str name


cdef class FlintConstant(FlintFunction):
    cdef int(* ptr)(gr_ptr, gr_ctx_t)


cdef class FlintUnaryOperator(FlintFunction):
    # NOTE: using the typedef gr_method_unary_op does not seem to work
    cdef int(* ptr)(gr_ptr, gr_srcptr, gr_ctx_t)

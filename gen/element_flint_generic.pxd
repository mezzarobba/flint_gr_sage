from sage.libs.flint.types cimport *
from sage.structure.element cimport Element
from sage.structure.sage_object cimport SageObject

from .parent_flint cimport FlintParent

cdef class FlintElement(Element):

    cdef gr_ptr ptr

    cdef FlintElement _new(self)
    cdef inline gr_ctx_struct * ctx(self)

    cpdef _add_(self, other)
    cpdef _add_long(self, long other)
    cpdef _add_flint(self, FlintElement other)
    cpdef _div_(self, other)
    cpdef _mul_(self, other)
    cpdef _mul_long(self, long other)
    cpdef _pow_(self, other)
    cpdef _floordiv_(self, other)

cdef class FlintVector(SageObject):

    cdef FlintParent _universe
    cdef gr_vec_t vec

    cdef inline gr_ctx_struct * ctx(self)

from sage.libs.flint.gr cimport *
from sage.libs.flint.types cimport *
from sage.structure.parent cimport Parent

from .missing cimport *

from .element_flint_generic cimport FlintElement


cdef class FlintParent(Parent):

    cdef gr_ctx_t ctx

    cdef FlintElement _new_element(self)
    cdef void _set_flint_names(self)


cdef class FlintVectorParent(FlintParent):
    # XXX Element = FlintVector
    pass


# Here the idea is to first do explicit checks in client code if one wants,
# e.g., to raise more meaningful exceptions, and then call check_status if not
# all cases have been covered. (We don't want to construct exceptions
# unconditionally and pass them to this function!)
cdef inline check_status(int status):
    if status == GR_SUCCESS:
        return
    # XXX FlintDomainError, FlintUnableException?
    elif status == GR_DOMAIN:
        raise ValueError("Invalid argument for generic FLINT operation (GR_DOMAIN)")
    elif status == GR_UNABLE:
        raise RuntimeError("Operation not supported (GR_UNABLE)")
    else:
        raise AssertionError  # ?


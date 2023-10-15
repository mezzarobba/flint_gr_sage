r"""
Parents based on Flint generic rings
"""

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from sage.cpython.string cimport char_to_str
from sage.libs.flint.flint cimport flint_free

from .element_flint_generic cimport FlintElement, FlintVector

cdef class FlintParent(Parent):
    r"""
    Base class for parents associated to Flint generic rings.

    This is just a base class that typically should not be used directly. In
    particular, it does not guarantee parent uniqueness.
    """

    Element = FlintElement

    def __cinit__(self):
        r"""
        Disable auto-pickling.
        """
        # ctx initialized by subclasses, but let's not segfault if the parent
        # is deallocated before it happens
        (<ulong*>self.ctx)[0] = 0xdeadbeef

    def __dealloc__(self):
        if (<ulong*>self.ctx)[0] != 0xdeadbeef:
            gr_ctx_clear(self.ctx)

    def __init__(self, base=None, *, category=None, names=None, normalize=True):

        if category is None:
            category = self._category_from_flint()

        if base is None:
            base = self

        # XXX Morphisms for fast conversions between FLINT rings?
        # XXX Register Actions for some/all coercions from FLINT rings (so that
        # they use gr_OP_other rather than a conversion)?

        Parent.__init__(self,
                        base=base,
                        category=category,
                        names=names,
                        normalize=normalize)

        self._set_flint_names()

    def _category_from_flint(self):
        if gr_ctx_is_field(self.ctx) == T_TRUE:
            import sage.categories.fields
            category = sage.categories.fields.Fields()
        elif gr_ctx_is_unique_factorization_domain(self.ctx) == T_TRUE:
            import sage.categories.unique_factorization_domains
            category = sage.categories.unique_factorization_domains.UniqueFactorizationDomains()
        elif gr_ctx_is_integral_domain(self.ctx) == T_TRUE:
            import sage.categories.integral_domains
            category = sage.categories.integral_domains.IntegralDomains()
        elif gr_ctx_is_commutative_ring(self.ctx) == T_TRUE:
            import sage.categories.commutative_rings
            category = sage.categories.commutative_rings.CommutativeRings()
        elif gr_ctx_is_ring(self.ctx) == T_TRUE:
            import sage.categories.rings
            category = sage.categories.rings.Rings()
            # XXX join Posets if ordered?
        elif gr_ctx_is_multiplicative_group(self.ctx) == T_TRUE:
            import sage.categories.groups
            category = sage.categories.groups.Groups()
        else:
            import sage.categories.sets_cat
            category = sage.categories.sets_cat.Sets()

        is_finite = gr_ctx_is_finite(self.ctx)
        if is_finite == T_TRUE:
            category = category.Finite()
        elif is_finite == T_FALSE:
            category = category.Infinite()

        return category

    cdef void _set_flint_names(self):
        # XXX check that we have the correct number of names?
        if self._names is None:
            return
        cdef slong ngens = len(self._names)
        cdef char **names_c = <char**> PyMem_Malloc(sizeof(char *)*ngens)
        for i in range(ngens):
            name = self._names[i].encode('ascii')
            names_c[i] = name
        # ignoring errors for now, is this the right thing to do?
        gr_ctx_set_gen_name(self.ctx, names_c[0])
        gr_ctx_set_gen_names(self.ctx, names_c)
        PyMem_Free(names_c)

    cpdef _coerce_map_from_(self, other):
        r"""
        Some default generic coercions
        """
        if other is self:
            return
        # XXX: should we have coercions, e.g., from ZZ to FlintZZ,
        # from FlintZZ to ZZ, or both?
        if gr_ctx_is_ring(self.ctx) == T_TRUE:
            from sage.rings.integer_ring import ZZ
            from .flint_parents import IntegerRing
            if other is ZZ or other is IntegerRing():
                return True
        if gr_ctx_is_field(self.ctx) == T_TRUE:
            from sage.rings.rational_field import QQ
            from .flint_parents import RationalField
            if other is QQ or other is RationalField():
                return True

    def _repr_(self):
        cdef char* c_string
        check_status(gr_ctx_get_str(&c_string, self.ctx))
        try:
            py_string = char_to_str(c_string)
        finally:
            flint_free(c_string)
        return py_string

    cpdef bint is_exact(self) except -2:
        return gr_ctx_is_exact(self.ctx) == T_TRUE

    # gr_ctx_is_algebraically_closed
    # gr_ctx_is_finite_characteristic
    # gr_ctx_is_ordered_ring
    # gr_ctx_is_canonical
    # gr_ctx_has_real_prec
    # gr_ctx_set_real_prec
    # gr_ctx_get_real_prec
    # gr_ctx_sizeof_elem

    cdef FlintElement _new_element(self):
        return FlintElement.__new__(FlintElement, self)

    def zero(self):
        cdef FlintElement g = self._new_element()
        check_status(gr_zero(g.ptr, self.ctx))
        return g

    def one(self):
        cdef FlintElement g = self._new_element()
        check_status(gr_one(g.ptr, self.ctx))
        return g

    # neg_one

    def gen(self):
        cdef FlintElement g = self._new_element()
        check_status(gr_gen(g.ptr, self.ctx))
        return g

    def gens(self):
        cdef FlintVector gens = FlintVector.__new__(FlintVector, self)
        check_status(gr_gens(gens.vec, self.ctx))
        return gens

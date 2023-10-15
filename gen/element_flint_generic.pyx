from cpython.long cimport PyLong_AsLong
from cpython.object cimport Py_EQ, Py_NE
from cysignals.signals cimport sig_on, sig_off
from sage.arith.long cimport is_small_python_int
from sage.cpython.string cimport char_to_str
from sage.ext.stdsage cimport PY_NEW
from sage.libs.flint.flint cimport flint_free
from sage.libs.flint.fmpz cimport *
from sage.libs.flint.fmpq cimport *
from sage.libs.flint.gr cimport *
from sage.libs.flint.gr_vec cimport *
from sage.rings.integer cimport Integer
from sage.rings.rational cimport Rational
from sage.structure.coerce cimport coercion_model
from sage.structure.element cimport parent
from sage.structure.richcmp cimport rich_to_bool

from .parent_flint cimport FlintParent, check_status

from .missing cimport *

cdef operator_add, operator_mul
from operator import add as operator_add
from operator import mul as operator_mul


cdef class FlintElement(Element):
    r"""
    Generic FLINT element
    """

    # XXX my understanding of the cython docs is that one can have cdef
    # inline properties: why doesn't this work?
    # @property
    cdef inline gr_ctx_struct * ctx(self):
        return (<FlintParent> (self._parent)).ctx

    def __cinit__(self, parent, data=None):
        self._parent = <FlintParent ?> parent
        # It would be nice to avoid the indirection, but, since elements
        # can have arbitrary size, that would require some hackery.
        self.ptr = gr_heap_init((<FlintParent ?> parent).ctx)

    def __dealloc__(self):
        # XXX Apparently python does not guarantee that the finalizer of self
        # will be called before that of ctx, but I haven't seen the
        # issue in real life. How is it with Cython? Special care may be
        # required if some parents hold references to their elements.
        gr_heap_clear(self.ptr, self.ctx())

    cdef FlintElement _new(self):
        return FlintElement.__new__(FlintElement, self._parent)

    def __init__(self, parent, data):
        cdef int status
        cdef FlintElement _elt
        cdef fmpz_t _fmpz
        cdef fmpq_t _fmpq
        if isinstance(data, FlintElement):
            _elt = data
            if _elt.parent is self._parent:
                status = gr_set(self.ptr, _elt.ptr, self.ctx())
            else:
                status = gr_set_other(self.ptr, _elt.ptr, _elt.ctx(), self.ctx())
        elif is_small_python_int(data):
            status = gr_set_si(self.ptr, PyLong_AsLong(data), self.ctx())
        elif isinstance(data, Integer):
            fmpz_init_set_readonly(_fmpz, (<Integer> data).value)
            status = gr_set_fmpz(self.ptr, _fmpz, self.ctx())
            fmpz_clear_readonly(_fmpz)
        elif isinstance(data, Rational):
            fmpq_init_set_readonly(_fmpq, (<Rational> data).value)
            status = gr_set_fmpq(self.ptr, _fmpq, self.ctx())
            fmpq_clear_readonly(_fmpq)
        # XXX python float, RDF
        # perhaps mpfr via gr_set_fmpz_2exp_fmpz
        # decimal
        else:
            raise TypeError(f"unable to convert {data} to an element of {parent}")
        if status != GR_SUCCESS:
            raise ValueError(f"unable to convert {data} to an element of {parent}")

    # XXX what can we do with gr_set_fexpr? convert from string???

    def _repr_(self):
        cdef char* c_string
        check_status(gr_get_str(&c_string, self.ptr, self.ctx()))
        try:
            py_string = char_to_str(c_string)
        finally:
            flint_free(c_string)
        return py_string

    def is_zero(self):
        return gr_is_zero(self.ptr, self.ctx()) == T_TRUE

    def is_nonzero(self):
        return gr_is_zero(self.ptr, self.ctx()) == T_FALSE

    def __bool__(self):
        return gr_is_zero(self.ptr, self.ctx()) != T_TRUE  # (?)

    def is_one(self):
        return gr_is_one(self.ptr, self.ctx()) == T_TRUE

    cpdef _richcmp_(left, right, int op):
        cdef FlintElement _left, _right
        _left = left
        _right = right
        if op == Py_EQ:
            return gr_equal(_left.ptr, _right.ptr, _left.ctx()) == T_TRUE
        elif op == Py_NE:
            return gr_equal(_left.ptr, _right.ptr, _left.ctx()) == T_FALSE
        cdef int res
        cdef int status = gr_cmp(&res, _left.ptr, _right.ptr, _left.ctx())
        if status == GR_SUCCESS:
            return rich_to_bool(op, res)
        elif status == GR_DOMAIN:
            raise ValueError(f"{left} and {right} are not comparable")
        else:
            raise RuntimeError(f"comparison of {left} to {right} failed")

    # is_integer
    # is_rational

    # Conversions to other types

    def __float__(self):
        cdef double res
        check_status(gr_get_d(&res, self.ptr, self.ctx()))
        return res

    def _integer_(self, _):
        cdef Integer res
        cdef fmpz_t tmp
        # Fredrik says that promote + mpz_swap may not always be safe,
        # but that it may be okay to swap the limb pointers
        fmpz_init(tmp)
        try:
            check_status(gr_get_fmpz(tmp, self.ptr, self.ctx()))
            res = PY_NEW(Integer)
            fmpz_get_mpz(res.value, tmp)
        finally:
            fmpz_clear(tmp)
        return res

    def _rational_(self):
        cdef Rational res
        cdef fmpq_t tmp
        fmpq_init(tmp)
        try:
            check_status(gr_get_fmpq(tmp, self.ptr, self.ctx()))
            res = PY_NEW(Rational)
            fmpq_get_mpq(res.value, tmp)
        finally:
            fmpq_clear(tmp)
        return res

    # Basic arithmetic

    def __add__(left, right):
        cdef FlintElement _left, _right
        if type(left) is type(right):
            _left = left
            _right = right
            if _left._parent is _right._parent:
                return _left._add_(_right)
            # XXX how would something like this interact wit the Sage coercion
            # system?
            # elif gr_ctx_cmp_coercion(_left._parent, _right._parent):
            #     return _left._add_flint(_right)
        return coercion_model.bin_op(left, right, operator_add)

    cpdef _add_(self, other):
        cdef FlintElement res = self._new()
        cdef FlintElement _other = other
        cdef int status
        sig_on()  # ?
        status = gr_add(res.ptr, self.ptr, _other.ptr, self.ctx())
        sig_off()
        check_status(status)
        return res

    cpdef _add_long(self, long other):
        cdef FlintElement res = self._new()
        sig_on()
        status = gr_add_si(res.ptr, self.ptr, other, self.ctx())
        sig_off()
        check_status(status)
        return res

    cpdef _add_flint(self, FlintElement other):
        cdef FlintElement res = self._new()
        cdef FlintElement _other = other
        cdef int status
        sig_on()  # ?
        status = gr_add_other(res.ptr, self.ptr, _other.ptr, self.ctx(), other.ctx())
        sig_off()
        check_status(status)
        return res

    def __neg__(self):
        cdef FlintElement res = self._new()
        cdef int status
        status = gr_neg(res.ptr, self.ptr, self.ctx())
        check_status(status)
        return res

    cpdef _sub_(self, other):
        cdef FlintElement res = self._new()
        cdef FlintElement _other = other
        cdef int status
        sig_on()  # ?
        status = gr_sub(res.ptr, self.ptr, _other.ptr, self.ctx())
        sig_off()
        check_status(status)
        return res

    cpdef _mul_(self, other):
        cdef FlintElement res = self._new()
        cdef FlintElement _other = other
        cdef int status
        sig_on()  # ?
        status = gr_mul(res.ptr, self.ptr, _other.ptr, self.ctx())
        sig_off()
        check_status(status)
        return res

    cpdef _mul_long(self, long other):
        cdef FlintElement res = self._new()
        cdef FlintElement _other = other
        cdef int status
        if gr_ctx_is_commutative_ring(self.ctx()) == T_TRUE:  # XXX too slow?
            sig_on()  # ?
            status = gr_mul(res.ptr, self.ptr, _other.ptr, self.ctx())
            sig_off()
        else:
            coercion_model.bin_op(self, other, operator_mul)
        check_status(status)
        return res

    # is_invertible

    def __invert__(self):
        cdef FlintElement res = self._new()
        cdef int status
        status = gr_neg(res.ptr, self.ptr, self.ctx())
        if status == GR_DOMAIN:
            raise ArithmeticError(self)
        check_status(status)
        return res

    cpdef _div_(self, other):
        cdef FlintElement res = self._new()
        cdef FlintElement _other = other
        cdef int status
        # gr_div may still work as soon as the quotient exists and is unique,
        # but in Sage, when the parent is not a field, the division should fail
        # or go to the fraction field

        if gr_ctx_is_field(self.ctx()) == T_TRUE:
            sig_on()
            status = gr_div(res.ptr, self.ptr, _other.ptr, self.ctx())
            sig_off()
            if status == GR_DOMAIN:
                raise ZeroDivisionError(self, other)
        elif gr_ctx_is_integral_domain(self.ctx()) == T_TRUE:
            # XXX integral domain => divide in the fraction field, either by
            # inheriting from RingElement and falling back to its generic
            # _div_, or directly
            raise NotImplementedError
        elif gr_ctx_is_multiplicative_group(self.ctx()) == T_TRUE:
            # XXX commutative only?
            sig_on()
            status = gr_div(res.ptr, self.ptr, _other.ptr, self.ctx())
            sig_off()
            if status == GR_DOMAIN:
                raise ArithmeticError(f"{other} is not invertible")
        else:
            # XXX use division_parent? including for integral domains etc.
            status = GR_UNABLE
        check_status(status)
        return res

    # # XXX __div__, _div_flint

    cpdef _pow_(self, expo):
        cdef int status
        cdef FlintElement _expo = <FlintElement> expo
        cdef FlintElement res = self._new()
        sig_on()
        status = gr_pow(res.ptr, self.ptr, _expo.ptr, self.ctx())
        sig_off()
        check_status(status)
        return res

    cdef _pow_long(self, long expo):
        cdef int status
        cdef FlintElement res = self._new()
        sig_on()
        status = gr_pow_si(res.ptr, self.ptr, expo, self.ctx())
        sig_off()
        check_status(status)
        return res


    # def __pow__(base, expo, _):
    #     pass
    #     # XXX optimize some cases using gr_pow_other?
    #     # I don't think we have a use for gr_other_pow?

    # Euclidean arithmetic

    cpdef _floordiv_(self, other):
        cdef FlintElement res = self._new()
        cdef FlintElement _other = other
        cdef int status
        sig_on()
        status = gr_euclidean_div(res.ptr, self.ptr, _other.ptr, self.ctx())
        sig_off()
        if status == GR_DOMAIN:
            raise ZeroDivisionError
        check_status(status)
        return res

    # _mod_ = euclidean_rem
    # euclidean_divrem

    # Specialized arithmetic

    # divexact
    # addmul, submul
    # add_si, mul_si...
    # mul_two, sqr, ...

    # XXX many more increasingly specialized methods... do we want to keep
    # a structure corresponding to that implemented in flint (elt, poly,
    # mat...), at the risk of polluting the namespace with unsupported methods,
    # or do we want a finer class hierarchy?


cdef class FlintElementWithSpecialFunctions(FlintElement):
    pass


# XXX vectors are generic objects!
# making them Elements has pros and cons...
cdef class FlintVector(SageObject):
    r"""
    Mutable list of Flint generic objects with a common parent

    (Something like a Sage Sequence, but stores the parent only once and can be
    used directly by Flint.)
    """

    cdef inline gr_ctx_struct * ctx(self):
        return (<FlintParent> (self._universe)).ctx

    # XXX initialization from list with automatic coercion
    # maybe also guess the parent

    def __cinit__(self, FlintParent universe, slong initial_length=0):
        self._universe = universe
        gr_vec_init(self.vec, initial_length, universe.ctx)

    def dealloc(self):
        gr_vec_clear(self.vec, self.ctx())

    def _repr_(self):
        return repr(list(self))

    def universe(self):
        return self._universe

    def __len__(self):
        return gr_vec_length(self.vec, self.ctx())

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise NotImplementedError
        cdef slong i = index
        if not 0 <= i < gr_vec_length(self.vec, self.ctx()):
            raise IndexError("vector index out of range")
        cdef FlintElement item = self._universe._new_element()
        gr_set(item.ptr, gr_vec_entry_ptr(self.vec, i, self.ctx()), self.ctx())
        return item

    def __setitem__(self, index, elt):
        if isinstance(index, slice):
            raise NotImplementedError
        cdef slong i = index
        if not 0 <= i < gr_vec_length(self.vec, self.ctx()):
            raise IndexError("vector index out of range")
        if parent(elt) is not self._universe:
            elt = self._universe.coerce(elt)
        cdef FlintElement _elt = elt
        gr_set(gr_vec_entry_ptr(self.vec, i, self.ctx()), _elt.ptr, self.ctx())

    def steal_item(self, slong i):
        r"""
        Efficiently retrieve the i-th element and set the corresponding
        position in the vector to zero.
        """
        cdef FlintElement item = self._universe._new_element()
        gr_swap(item.ptr, gr_vec_entry_ptr(self.vec, i, self.ctx()), self.ctx())
        return item

    def append(self, elt):
        if parent(elt) is not self._universe:
            elt = self._universe.coerce(elt)
        cdef FlintElement _elt = elt
        gr_vec_append(self.vec, _elt.ptr, self.ctx())

    def extend(self, elts):
        cdef slong i
        cdef slong length = gr_vec_length(self.vec, self.ctx())
        gr_vec_set_length(self.vec, length + len(elts), self.ctx())
        for i, elt in enumerate(elts):
            self[length + i] = elt

    def set_length(self, slong length):
        gr_vec_set_length(self.vec, length, self.ctx())

    def copy(self):
        cdef FlintVector res = FlintVector.__new__(FlintVector, self._universe,
                                                   len(self))
        gr_vec_set(res.vec, self.vec, self.ctx())
        return res

    # XXX add etc. -- or via gr_add etc.


r"""
Flint built-in domains

EXAMPLES::

    sage: import gen.flint_parents as F

    sage: F.SymmetricGroup(5).one()
    [0, 1, 2, 3, 4]
    sage: F.PSL2Z().one()
    [[1, 0], [0, 1]]
    sage: F.PSL2Z()(F.MatrixRing(F.IntegerRing(), 2).one())
    [[1, 0], [0, 1]]
    sage: F.DirichletGroup(42).one()
    chi_42(1, .)

    sage: F.IntegerRing()(-1)
    -1
    sage: F.RationalField()(1/3)
    1/3
    sage: I = F.GaussianIntegerRing().gen()
    sage: I + 1
    (1+I)

    sage: F.IntegerModRing(7, type="fmpz")
    Integers mod 7 (fmpz)
    sage: F.IntegerModRing(7, type="nmod")
    Integers mod 7 (_gr_nmod)
    sage: F.IntegerModRing(7, type="nmod32")
    Integers mod 7 (nmod32)
    sage: F.IntegerModRing(7, type="nmod8")
    Integers mod 7 (nmod8)
    sage: p = 2^61 - 1
    sage: F.IntegerModRing(p) in Fields()
    True
    sage: p = 2^1279 - 1
    sage: F.IntegerModRing(p) in Fields()
    False
    sage: F.IntegerModRing(p, prime=True) in Fields()
    True
    sage: F.IntegerModRing(p*p, prime=False) in Fields()
    False
    sage: p = 2^61 - 1
    sage: F.IntegerModRing(p)
    Integers mod 2305843009213693951 (_gr_nmod)

    sage: F.FiniteField(3, 2).gen() + 4
    a+1
    sage: F.FiniteField(7, 2, type="zech")
    Finite field (fq_zech)
    sage: F.FiniteField(2^1279-1, 2, check=False)
    Finite field (fq)
    sage: F.FiniteField(2^61-1, 3)
    Finite field (fq_nmod)
    sage: F.FiniteField(2, 1, type="fmpz")
    Finite field (fq)
    sage: F.FiniteField(4, 3)
    Traceback (most recent call last):
    ...
    ValueError: p must be a prime number

    sage: F.FiniteField(2^61-1, 0)
    Traceback (most recent call last):
    ...
    ValueError: d must be positive

    sage: t = F.UnivariatePolynomialRing(F.IntegerRing(), 't').gen()
    sage: F.NumberField(t^2+1, 'b', check=False).gen()^3
    -b

    sage: FAA = F.AlgebraicRealField_qqbar(); FAA
    Real algebraic numbers (qqbar)
    sage: rt = FAA(2)^FAA(1/3); rt
    Root a = 1.25992 of a^3-2
    sage: FAA(-1)^FAA(1/3)
    Traceback (most recent call last):
    ...
    ValueError: Invalid argument for generic FLINT operation (GR_DOMAIN)
    sage: FQQbar = F.AlgebraicField_qqbar()
    sage: alg = FQQbar(I) + FQQbar(rt); alg
    Root a = 1.25992 + 1.00000*I of a^6+3*a^4-4*a^3+3*a^2+12*a+5

    sage: F.RealBallField(10)(rt)
    [1.26 +/- 2.19e-3]
    sage: F.ComplexBallField(30)(alg)
    ([1.25992105 +/- 2.17e-9] + [1.000000000 +/- 1e-14]*I)

    sage: FSRF = F.RealSymbolicField(); FSRF
    Real numbers (ca)
    sage: FSRF(rt)
    1.25992 {a where a = 1.25992 [a^3-2=0]}
    sage: FSRF(I)
    Traceback (most recent call last):
    ...
    ValueError: unable to convert I to an element of Real numbers (ca)
    sage: FSCF = F.ComplexSymbolicField(); FSCF
    Complex numbers (ca)
    sage: FSCF.base_ring()
    Real numbers (ca)
    sage: FSCF(rt) + FSCF(I)
    1.25992 + 1.00000*I {a+b where a = 1.25992 [a^3-2=0], b = I [b^2+1=0]}
    sage: FSCF(-rt)^FSCF(rt)
    -0.916100 - 0.975062*I {a where a = -0.916100 - 0.975062*I [Pow(-1.25992 {b}, 1.25992 {c})], b = -1.25992 [b^3+2=0], c = 1.25992 [c^3-2=0]}

    sage: FCaAA = F.AlgebraicRealField_calcium(); FCaAA
    Real algebraic numbers (ca)
    sage: FCaAA(rt)
    1.25992 {a where a = 1.25992 [a^3-2=0]}
    sage: FCaAA(-rt)^FCaAA(1/2)
    Traceback (most recent call last):
    ...
    ValueError: Invalid argument for generic FLINT operation (GR_DOMAIN)
    sage: FCaQQbar = F.AlgebraicField_calcium(); FCaQQbar
    Complex algebraic numbers (ca)
    sage: FCaQQbar.base_ring()
    Real algebraic numbers (qqbar)
    sage: FCaQQbar(rt)^FCaQQbar(1/2)
    1.12246 {a where a = 1.12246 [Sqrt(1.25992 {b})], b = 1.25992 [b^3-2=0]}
    sage: F.SymbolicExtendedComplexField()
    Complex numbers + extended values (ca)

    sage: F.RealFloatingPointField(30)(rt)
    1.259921050
    sage: F.RealFloatingPointField(30)(F.RealBallField(30)(rt))
    1.259921050
    sage: F.ComplexFloatingPointField(20)(I)
    1.000000*I
    sage: F.ComplexFloatingPointField(20).base_ring()
    Floating-point numbers (arf, prec = 20)

    sage: FSeq = F.FiniteSequences(F.RationalField()); FSeq
    Vectors (any length) over Rational field (fmpq)
    sage: F.VectorSpace(F.RationalField(), 3).zero()
    [0, 0, 0]
    sage: vec = F.VectorSpace(F.RationalField(), 3).zero()
    sage: FSeq(vec)
    [0, 0, 0]

    sage: F.Matrices(F.SymmetricGroup(3))
    Matrices (any shape) over Symmetric group S_3 (perm)
    sage: AllMat = F.Matrices(F.RationalField())
    sage: Mat = F.MatrixSpace(F.IntegerRing(), 3, 4)
    sage: Mat(AllMat(Mat.one()))
    [[1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]]
    sage: F.MatrixRing(F.RationalField(), 2)()
    [[0, 0],
    [0, 0]]

    sage: FZZX = F.UnivariatePolynomialRing(F.IntegerRing(), 'x')
    sage: FZZX.gens()
    [x]
    sage: FZZXY = F.UnivariatePolynomialRing(FZZX, 'y')
    sage: FZZXY.gens()
    [y]
    sage: FZZXY.gen() + FZZXY(FZZX.gen())
    x + y
    sage: F.UnivariatePolynomialRing(F.RationalField(), 'x')
    Polynomials over rationals (fmpq_poly)
    sage: F.UnivariatePolynomialRing(F.RationalField(), 'x', force_generic=True)
    Ring of polynomials over Rational field (fmpq)

    sage: R = F.MultivariatePolynomialRing(F.IntegerRing(), 'a,b,c'); R
    Ring of multivariate polynomials over Integer ring (fmpz) in 3 variables, degrevlex order
    sage: a, b, c = R.gens()
    sage: a + b + 2
    a + b + 2
    sage: R.is_commutative()
    True
    sage: F.MultivariatePolynomialRing(F.RationalField(), 'a', order='lex')
    Ring of multivariate polynomials over Rational field (fmpq) in 1 variables, lex order
    sage: R = F.MultivariatePolynomialRing(F.MatrixSpace(F.IntegerRing(), 2, 2), (), order='DEGLEX'); R
    Ring of multivariate polynomials over Ring of 2 x 2 matrices over Integer ring (fmpz) in 0 variables, deglex order
    sage: R.is_commutative()
    False

    sage: x, y = F.MultivariateRationalFunctionField(F.IntegerRing(), 'x,y').gens()
    sage: (x + y)/(x - y)
    (x+y)/(x-y)

    sage: F.SymbolicRing()(FSCF(rt)^FSCF(rt))
    Where(a_1, Def(a_1, Pow(a_2, a_2)), Def(a_2, PolynomialRootNearest(List(-2, 0, 0, 1), Decimal("1.25992"))))
"""


from cysignals.signals cimport sig_on, sig_off
from sage.structure.category_object cimport normalize_names

from .missing cimport *

from .element_flint_generic cimport FlintElement
from .parent_flint cimport FlintParent, FlintVectorParent

from sage.misc.fast_methods import Singleton
from sage.structure.unique_representation import UniqueRepresentation


cdef FlintParent _to_flint_parent(X):
    if isinstance(X, FlintParent):
        return X
    else:
        # XXX use adapter
        raise NotImplementedError


# XXX normalize arguments when using UniqueRepresentation


class SymmetricGroup(UniqueRepresentation, FlintParent):
    r"""
    sage: import gen.flint_parents
    sage: gen.flint_parents.SymmetricGroup(5).one()
    [0, 1, 2, 3, 4]
    """

    def __init__(self, ulong n):
        gr_ctx_init_perm((<FlintParent> self).ctx, n)
        super().__init__()


class PSL2Z(Singleton, FlintParent):

    def __init__(self):
        gr_ctx_init_psl2z((<FlintParent> self).ctx)
        super().__init__()


class DirichletGroup(UniqueRepresentation, FlintParent):

    def __init__(self, ulong q):
        gr_ctx_init_dirichlet_group((<FlintParent> self).ctx, q)
        super().__init__()


class IntegerRing(Singleton, FlintParent):

    def __init__(self):
        gr_ctx_init_fmpz((<FlintParent> self).ctx)
        super().__init__()


class RationalField(Singleton, FlintParent):

    def __init__(self):
        gr_ctx_init_fmpq((<FlintParent> self).ctx)
        super().__init__(base=IntegerRing())
        self._populate_coercion_lists_([
            IntegerRing(),
        ])

class GaussianIntegerRing(Singleton, FlintParent):

    def __init__(self):
        gr_ctx_init_fmpzi((<FlintParent> self).ctx)
        super().__init__(base=IntegerRing())


class IntegerModRing(UniqueRepresentation, FlintParent):
    r"""
    EXAMPLES::

        sage: import gen.flint_parents as F
        sage: R = F.IntegerModRing(7)
        sage: R in Fields()
        True
        sage: F.IntegerModRing(2^100, prime=False)
        Integers mod 1267650600228229401496703205376 (fmpz)
        sage: F.IntegerModRing(2^20)
        Integers mod 1048576 (nmod32)
        sage: F.IntegerModRing(2^63-1)
        Integers mod 9223372036854775807 (_gr_nmod)

    TESTS::

        sage: F.IntegerModRing(-1)
        Traceback (most recent call last):
        ...
        ValueError: n should be >= 0
    """

    def __init__(self, n, *, type=None, prime=None):
        cdef truth_t _prime
        cdef FlintElement _n = IntegerRing()(n)  # XXX delay?
        if _n <= 0:
            raise ValueError("n should be >= 0")
        if type is None:
            if _n < 1 << 8:
                type = "nmod8"
            elif n < 1 << 32:
                type = "nmod32"
            elif n < UWORD_MAX:
                type = "nmod"
            else:
                type = "fmpz"
        if type == "nmod8":
            gr_ctx_init_nmod8((<FlintParent> self).ctx, n)
        elif type == "nmod32":
            gr_ctx_init_nmod32((<FlintParent> self).ctx, n)
        elif type == "nmod":
            gr_ctx_init_nmod((<FlintParent> self).ctx, n)
        elif type == "fmpz":
            gr_ctx_init_fmpz_mod((<FlintParent> self).ctx, <fmpz_t> _n.ptr)
        else:
            raise ValueError(f"unknown coefficient type: {type}")
        if prime is not None:
            _prime = T_TRUE if prime else T_FALSE
            # For newer versions of flint
            # if type == "nmod":
            #     gr_ctx_nmod_set_primality((<FlintParent> self).ctx, _prime)
            if type == "fmpz":
                gr_ctx_fmpz_mod_set_primality((<FlintParent> self).ctx, _prime)
        super().__init__()


class FiniteField(UniqueRepresentation, FlintParent):
    r"""
    EXAMPLES::

        sage: import gen.flint_parents as F
        sage: F.FiniteField(3, 2).gen() + 4
        a+1
    """

    def __init__(self, p, d, names=None, *, type=None, check=True):
        cdef FlintElement _p
        from sage.rings.integer_ring import ZZ
        if check and not ZZ(p).is_prime():
            raise ValueError("p must be a prime number")
        if d <= 0:
            raise ValueError("d must be positive")
        if names is not None:
            names = normalize_names(1, names)
        if type is None:
            if p < UWORD_MAX:
                type = "nmod"
            else:
                type = "fmpz"
        # XXX: set name of make set_gen_name do it
        # XXX: sig_on()
        if type == "fmpz":
            _p = IntegerRing()(p)
            sig_on()
            gr_ctx_init_fq((<FlintParent> self).ctx, <fmpz_t> _p.ptr, d, NULL)
            sig_off()
        elif type == "nmod":
            sig_on()
            gr_ctx_init_fq_nmod((<FlintParent> self).ctx, p, d, NULL)
            sig_off()
        elif type == "zech":
            if p**d >= 1 << 64:
                raise ValueError("type='zech' requires p^d < 2^64")
            sig_on()
            gr_ctx_init_fq_zech((<FlintParent> self).ctx, p, d, NULL)
            sig_off()
        else:
            raise ValueError(f"unknown implementation type: {type}")
        # XXX base should probably depend on the implementation type
        super().__init__(base=IntegerModRing(p, prime=True), names=names, normalize=False)


class NumberField(UniqueRepresentation, FlintParent):

    def __init__(self, poly, names, *, check=True):
        FQQ = RationalField()
        if poly.base_ring() is not FQQ:
            Pol = UnivariatePolynomialRing(FQQ, poly.parent().variable_name())
            poly = Pol.coerce(poly)
        assert isinstance(poly, FlintElement)
        assert (<gr_which_structure*>((<char*>((<FlintParent>(<FlintElement> poly)._parent).ctx)) + 6 * sizeof(ulong)))[0] == GR_CTX_FMPQ_POLY
        if check:
            # TODO once polynomials have all necessary features...
            raise NotImplementedError
        # no real point in using init_nf_fmpz_poly
        gr_ctx_init_nf((<FlintParent> self).ctx,
                       <fmpq_poly_t> (<FlintElement> poly).ptr)  # too fragile?
        super().__init__(base=RationalField(), names=names)


class AlgebraicRealField_qqbar(Singleton, FlintParent):

    def __init__(self):
        gr_ctx_init_real_qqbar((<FlintParent> self).ctx)
        super().__init__(base=RationalField())


class AlgebraicField_qqbar(Singleton, FlintParent):

    def __init__(self):
        gr_ctx_init_complex_qqbar((<FlintParent> self).ctx)
        super().__init__(base=AlgebraicRealField_qqbar())


class RealBallField(UniqueRepresentation, FlintParent):

    def __init__(self, prec):
        gr_ctx_init_real_arb((<FlintParent> self).ctx, prec)
        super().__init__()


class ComplexBallField(UniqueRepresentation, FlintParent):

    def __init__(self, prec):
        gr_ctx_init_complex_acb((<FlintParent> self).ctx, prec)
        super().__init__(base=RealBallField(prec))
        self._populate_coercion_lists_([
            GaussianIntegerRing(),
        ])


class _CalciumField(FlintParent):

    def __init__(self, base=None):
        super().__init__(base=base)


# XXX make it possible to create several calcium fields?

class RealSymbolicField(Singleton, _CalciumField):

    def __init__(self):
        gr_ctx_init_real_ca((<FlintParent> self).ctx)
        super().__init__()


class ComplexSymbolicField(Singleton, _CalciumField):

    def __init__(self):
        gr_ctx_init_complex_ca((<FlintParent> self).ctx)
        super().__init__(base=RealSymbolicField())
        self._populate_coercion_lists_([
            GaussianIntegerRing(),
        ])


class AlgebraicRealField_calcium(Singleton, _CalciumField):

    def __init__(self):
        gr_ctx_init_real_algebraic_ca((<FlintParent> self).ctx)
        super().__init__(base=RationalField())


class AlgebraicField_calcium(Singleton, _CalciumField):

    def __init__(self):
        gr_ctx_init_complex_algebraic_ca((<FlintParent> self).ctx)
        super().__init__(base=AlgebraicRealField_qqbar())
        self._populate_coercion_lists_([
            GaussianIntegerRing(),
        ])


class SymbolicExtendedComplexField(Singleton, _CalciumField):

    def __init__(self):
        gr_ctx_init_complex_extended_ca((<FlintParent> self).ctx)
        super().__init__(base=ComplexSymbolicField())
        self._populate_coercion_lists_([
            GaussianIntegerRing(),
        ])


class RealFloatingPointField(UniqueRepresentation, FlintParent):

    def __init__(self, prec):
        gr_ctx_init_real_float_arf((<FlintParent> self).ctx, prec)
        super().__init__()


class ComplexFloatingPointField(UniqueRepresentation, FlintParent):

    def __init__(self, prec):
        gr_ctx_init_complex_float_acf((<FlintParent> self).ctx, prec)
        super().__init__(base=RealFloatingPointField(prec))


class FiniteSequences(UniqueRepresentation, FlintVectorParent):
    r"""
    Flint vectors of arbitrary length.
    """

    def __init__(self, base):
        cdef FlintParent _base = _to_flint_parent(base)
        gr_ctx_init_vector_gr_vec((<FlintParent> self).ctx, _base.ctx)
        super().__init__(base=base)


class VectorSpace(UniqueRepresentation, FlintVectorParent):
    r"""
    sage: import gen.flint_parents as F
    sage: F.VectorSpace(F.IntegerRing(), 5).one()
    [1, 1, 1, 1, 1]
    """

    # XXX works even when base is not a field => better name or multiple
    # classes?

    def __init__(self, base, slong n):
        cdef FlintParent _base = _to_flint_parent(base)
        gr_ctx_init_vector_space_gr_vec((<FlintParent> self).ctx, _base.ctx, n)
        super().__init__(base=base)


# FLINT matrices, as generic FLINT objects not supporting the matrix API


class Matrices(UniqueRepresentation, FlintParent):

    def __init__(self, base):
        cdef FlintParent _base = _to_flint_parent(base)
        gr_ctx_init_matrix_domain((<FlintParent> self).ctx, _base.ctx)
        super().__init__(base=base)


class MatrixSpace(UniqueRepresentation, FlintParent):
    r"""
    sage: import gen.flint_parents as F
    sage: F.MatrixSpace(F.RationalField(), 3, 4)(1)
    [[1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]]
    """

    def __init__(self, base, slong n, slong m):
        cdef FlintParent _base = _to_flint_parent(base)
        gr_ctx_init_matrix_space((<FlintParent> self).ctx, _base.ctx, n, m)
        super().__init__(base=base)


class MatrixRing(UniqueRepresentation, FlintParent):

    def __init__(self, base, slong n):
        cdef FlintParent _base = _to_flint_parent(base)
        gr_ctx_init_matrix_ring((<FlintParent> self).ctx, _base.ctx, n)
        super().__init__(base=base)


# FLINT polynomials, as generic FLINT objects not supporting the polynomial API


class UnivariatePolynomialRing(UniqueRepresentation, FlintParent):

    def __init__(self, base, names, *, force_generic=False):
        cdef FlintParent _base
        names = normalize_names(1, names)
        if not force_generic:
            if isinstance(base, IntegerRing):
                gr_ctx_init_fmpz_poly((<FlintParent> self).ctx)
            elif isinstance(base, RationalField):
                gr_ctx_init_fmpq_poly((<FlintParent> self).ctx)
            else:
                force_generic = True
        if force_generic:
            _base = _to_flint_parent(base)
            gr_ctx_init_gr_poly((<FlintParent> self).ctx, _base.ctx)
        super().__init__(base=base, names=names, normalize=False)

    def _coerce_map_from_(self, Source):
        import sage.rings.polynomial.polynomial_ring
        if (isinstance(Source,
                      (sage.rings.polynomial.polynomial_ring.PolynomialRing_general,
                       UnivariatePolynomialRing))
                and Source.variable_name() == self.variable_name()):
            return True
        # XXX return a better morphism
        if self.base_ring().has_coerce_map_from(Source):
            return True


_flint_term_order = {
    "lex": ORD_LEX,
    "deglex": ORD_DEGLEX,
    "degrevlex": ORD_DEGREVLEX,
}


# XXX gr_series and gr_series_mod now exist but are not documented in 3.1 and
# have changed since


class MultivariatePolynomialRing(UniqueRepresentation, FlintParent):

    # XXX require nvars?
    def __init__(self, base, names, order='degrevlex', *,
                 force_generic_implementation=False):
        names = normalize_names(-1, names)
        cdef slong nvars = len(names)
        from sage.rings.polynomial.term_order import TermOrder
        cdef ordering_t _order
        try:
            _order = _flint_term_order[TermOrder(order).name()]
        except KeyError:
            raise NotImplementedError(f"unsupported term order: {order}")
        if not force_generic_implementation:
            if isinstance(base, IntegerRing):
                gr_ctx_init_fmpz_mpoly((<FlintParent> self).ctx, nvars,
                                       _order)
        cdef FlintParent _base = _to_flint_parent(base)
        gr_ctx_init_gr_mpoly((<FlintParent> self).ctx, _base.ctx, nvars,
                             _order)
        super().__init__(base=base, names=names, normalize=False)


class MultivariateRationalFunctionField(UniqueRepresentation, FlintParent):

    def __init__(self, base, names, order='degrevlex'):
        names = normalize_names(-1, names)
        cdef slong nvars = len(names)
        from sage.rings.polynomial.term_order import TermOrder
        cdef ordering_t _order
        try:
            _order = _flint_term_order[TermOrder(order).name()]
        except KeyError:
            raise NotImplementedError(f"unsupported term order: {order}")
        if not isinstance(base, IntegerRing):
            raise NotImplementedError("unsupported base ring")
        gr_ctx_init_fmpz_mpoly_q((<FlintParent> self).ctx, nvars, _order)
        super().__init__(base=base, names=names, normalize=False)


class SymbolicRing(Singleton, FlintParent):

    def __init__(self):
        gr_ctx_init_fexpr((<FlintParent> self).ctx)
        super().__init__()

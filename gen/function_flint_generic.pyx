r"""
EXAMPLES::

    sage: from gen.flint_parents import *
    sage: from gen.function_flint_generic import *
    sage: FSRF = RealSymbolicField()
    sage: RBF = F.RealBallField(64)

We list constants::

    sage: pi_function(FSRF)
    3.14159 {a where a = 3.14159 [Pi]

    sage: pi_function(RBF)
    [3.141592653589793239 +/- 5.96e-19]
    sage: euler_function(RBF)
    [0.5772156649015328606 +/- 4.35e-20]
    sage: catalan_function(RBF)
    [0.915965594177219015 +/- 1.02e-19]
    sage: khinchin_function(RBF)
    [2.685452001065306445 +/- 3.65e-19]
    sage: glaisher_function(RBF)
    [1.282427129100622637 +/- 3.01e-19]

Unary functions::

    sage: a = FSRF(2)
    sage: log(a)
    0.693147 {a where a = 0.693147 [Log(2)]}
    sage: log(log(a))
    -0.366513 {a where a = -0.366513 [Log(0.693147 {b})], b = 0.693147 [Log(2)]}
    sage: exp(FSRF(2))
    7.38906 {a where a = 7.38906 [Exp(2)]}
    sage: log(exp(FSRF(2))) == FSRF(2)
    True

    sage: log(RBF(2))
    [0.6931471805599453094 +/- 8.66e-20]

Bug or not implemented (it would better raise an error rather than returning
zero silently)::

    sage: euler_function(FSRF)  # TODO: bug? or not implemented?
    0
    sage: zeta(FSRF(2))
    0
"""

from sage.libs.flint.gr_special cimport *
from sage.libs.flint.types cimport *

from sage.structure.element cimport Element
from sage.structure.sage_object cimport SageObject

from .element_flint_generic cimport FlintElement
from .parent_flint cimport FlintParent


cdef class FlintFunction:
    def __repr__(self):
        return self.name


cdef class FlintConstant(FlintFunction):
    r"""
    Wrapper of a generic flint constant function.
    """
    def __call__(self, parent):
        if not isinstance(parent, FlintParent):
            raise TypeError
        cdef FlintElement y = (<FlintParent> parent)._new_element()
        self.ptr(y.ptr, y.ctx())
        return y


cdef FlintConstant make_flint_constant(int(* func)(gr_ptr, gr_ctx_t), str name):
    cdef FlintConstant function = FlintConstant.__new__(FlintConstant)
    function.ptr = func
    function.name = name
    return function


pi_function = make_flint_constant(gr_pi, "pi")
euler_function = make_flint_constant(gr_euler, "euler")
catalan_function = make_flint_constant(gr_catalan, "catalan")
khinchin_function = make_flint_constant(gr_khinchin, "khinchin")
glaisher_function = make_flint_constant(gr_glaisher, "glaisher")


cdef class FlintUnaryOperator(FlintFunction):
    r"""
    Wrapper of a generic flint unary function.
    """
    def __call__(self, x):
        if not isinstance(x, FlintElement):
            raise NotImplementedError("not a flint element")
        cdef FlintElement y = (<FlintElement> x)._new()
        self.ptr(y.ptr, (<FlintElement> x).ptr, (<FlintElement> x).ctx())
        return y


cdef FlintUnaryOperator make_flint_unary(int(* func)(gr_ptr, gr_srcptr, gr_ctx_t), str name):
    cdef FlintUnaryOperator function = FlintUnaryOperator.__new__(FlintUnaryOperator)
    function.ptr = func
    function.name = name
    return function


exp = make_flint_unary(gr_exp, "exp")
expm1 = make_flint_unary(gr_expm1, "expm1")
exp2 = make_flint_unary(gr_exp2, "exp2")
exp10 = make_flint_unary(gr_exp10, "exp10")
exp_pi_i = make_flint_unary(gr_exp_pi_i, "exp_pi_i")
log = make_flint_unary(gr_log, "log")
log1p = make_flint_unary(gr_log1p, "log1p")
log2 = make_flint_unary(gr_log2, "log2")
log10 = make_flint_unary(gr_log10, "log10")
log_pi_i = make_flint_unary(gr_log_pi_i, "log_pi_i")
sin = make_flint_unary(gr_sin, "sin")
cos = make_flint_unary(gr_cos, "cos")
tan = make_flint_unary(gr_tan, "tan")
cot = make_flint_unary(gr_cot, "cot")
sec = make_flint_unary(gr_sec, "sec")
csc = make_flint_unary(gr_csc, "csc")
sin_pi = make_flint_unary(gr_sin_pi, "sin_pi")
cos_pi = make_flint_unary(gr_cos_pi, "cos_pi")
tan_pi = make_flint_unary(gr_tan_pi, "tan_pi")
cot_pi = make_flint_unary(gr_cot_pi, "cot_pi")
sec_pi = make_flint_unary(gr_sec_pi, "sec_pi")
csc_pi = make_flint_unary(gr_csc_pi, "csc_pi")
sinc = make_flint_unary(gr_sinc, "sinc")
sinc_pi = make_flint_unary(gr_sinc_pi, "sinc_pi")
sinh = make_flint_unary(gr_sinh, "sinh")
cosh = make_flint_unary(gr_cosh, "cosh")
tanh = make_flint_unary(gr_tanh, "tanh")
coth = make_flint_unary(gr_coth, "coth")
sech = make_flint_unary(gr_sech, "sech")
csch = make_flint_unary(gr_csch, "csch")
asin = make_flint_unary(gr_asin, "asin")
acos = make_flint_unary(gr_acos, "acos")
atan = make_flint_unary(gr_atan, "atan")
acot = make_flint_unary(gr_acot, "acot")
asec = make_flint_unary(gr_asec, "asec")
acsc = make_flint_unary(gr_acsc, "acsc")
asin_pi = make_flint_unary(gr_asin_pi, "asin_pi")
acos_pi = make_flint_unary(gr_acos_pi, "acos_pi")
atan_pi = make_flint_unary(gr_atan_pi, "atan_pi")
acot_pi = make_flint_unary(gr_acot_pi, "acot_pi")
asec_pi = make_flint_unary(gr_asec_pi, "asec_pi")
asinh = make_flint_unary(gr_asinh, "asinh")
acsc_pi = make_flint_unary(gr_acsc, "acsc_pi")
asinh = make_flint_unary(gr_asinh, "asinh")
acosh = make_flint_unary(gr_acosh, "acosh")
atanh = make_flint_unary(gr_atanh, "atanh")
acoth = make_flint_unary(gr_acoth, "acoth")
asech = make_flint_unary(gr_asech, "asech")
acsch = make_flint_unary(gr_acsch, "acsch")
lambertw = make_flint_unary(gr_lambertw, "lambertw")
fac = make_flint_unary(gr_fac, "fac")
rfac = make_flint_unary(gr_rfac, "rfac")
gamma = make_flint_unary(gr_gamma, "gamma")
rgamma = make_flint_unary(gr_rgamma, "rgamma")
lgamma = make_flint_unary(gr_lgamma, "lgamma")
digamma = make_flint_unary(gr_digamma, "digamma")
barnes_g = make_flint_unary(gr_barnes_g, "barnes_g")
log_barnes_g = make_flint_unary(gr_log_barnes_g, "log_barnes_g")
doublefac = make_flint_unary(gr_doublefac, "doublefac")
harmonic = make_flint_unary(gr_harmonic, "harmonic")
erf = make_flint_unary(gr_erf, "erf")
erfc = make_flint_unary(gr_erfc, "erfc")
erfcx = make_flint_unary(gr_erfcx, "erfcx")
erfi = make_flint_unary(gr_erfi, "erfi")
erfinv = make_flint_unary(gr_erfinv, "erfinv")
erfcinv = make_flint_unary(gr_erfcinv, "erfcinv")
exp_integral_ei = make_flint_unary(gr_exp_integral_ei, "exp_integral_ei")
sin_integral = make_flint_unary(gr_sin_integral, "sin_integral")
cos_integral = make_flint_unary(gr_cos_integral, "cos_integral")
sinh_integral = make_flint_unary(gr_sinh_integral, "sinh_integral")
cosh_integral = make_flint_unary(gr_cosh_integral, "cosh_integral")
dilog = make_flint_unary(gr_dilog, "dilog")
airy_ai = make_flint_unary(gr_airy_ai, "airy_ai")
airy_bi = make_flint_unary(gr_airy_bi, "airy_bi")
airy_ai_prime = make_flint_unary(gr_airy_ai_prime, "airy_ai_prime")
airy_bi_prime = make_flint_unary(gr_airy_bi_prime, "airy_bi_prime")
zeta = make_flint_unary(gr_zeta, "zeta")
dirichlet_eta = make_flint_unary(gr_dirichlet_eta, "dirichlet_eta")
riemann_xi = make_flint_unary(gr_riemann_xi, "riemann_xi")
zeta_nzeros = make_flint_unary(gr_zeta_nzeros, "zeta_nzeros")
agm1 = make_flint_unary(gr_agm1, "agm1")
elliptic_k = make_flint_unary(gr_elliptic_k, "elliptic_k")
elliptic_e = make_flint_unary(gr_elliptic_e, "elliptic_e")
dedekind_eta = make_flint_unary(gr_dedekind_eta, "dedekind_eta")
dedekind_eta_q = make_flint_unary(gr_dedekind_eta_q, "dedekind_eta_q")
modular_j = make_flint_unary(gr_modular_j, "modular_j")
modular_lambda = make_flint_unary(gr_modular_lambda, "modular_lambda")
modular_delta = make_flint_unary(gr_modular_delta, "modular_delta")

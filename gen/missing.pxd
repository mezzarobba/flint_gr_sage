from sage.libs.flint.types cimport *


cdef extern from "flint_wrap.h":

    cdef ulong UWORD_MAX

    cdef char GR_SUCCESS
    cdef char GR_DOMAIN
    cdef char GR_UNABLE
    cdef char GR_TEST_FAIL

    void gr_ctx_init_random(gr_ctx_t ctx, flint_rand_t state) noexcept

    void gr_ctx_init_fmpz(gr_ctx_t ctx) noexcept
    void gr_ctx_init_fmpq(gr_ctx_t ctx) noexcept
    void gr_ctx_init_fmpzi(gr_ctx_t ctx) noexcept

    void gr_ctx_init_fmpz_mod(gr_ctx_t ctx, const fmpz_t n) noexcept
    void gr_ctx_fmpz_mod_set_primality(gr_ctx_t ctx, truth_t is_prime) noexcept

    void gr_ctx_init_nmod(gr_ctx_t ctx, ulong n) noexcept
    void _gr_ctx_init_nmod(gr_ctx_t ctx, void * nmod_t_ref) noexcept
    void gr_ctx_nmod_set_primality(gr_ctx_t ctx, truth_t is_prime) noexcept

    void gr_ctx_init_nmod8(gr_ctx_t ctx, unsigned char n) noexcept
    void gr_ctx_init_nmod32(gr_ctx_t ctx, unsigned int n) noexcept

    void gr_ctx_init_real_qqbar(gr_ctx_t ctx) noexcept
    void gr_ctx_init_complex_qqbar(gr_ctx_t ctx) noexcept
    void _gr_ctx_qqbar_set_limits(gr_ctx_t ctx, slong deg_limit, slong bits_limit) noexcept

    void gr_ctx_init_real_arb(gr_ctx_t ctx, slong prec) noexcept
    void gr_ctx_init_complex_acb(gr_ctx_t ctx, slong prec) noexcept

    void gr_ctx_init_real_float_arf(gr_ctx_t ctx, slong prec) noexcept
    void gr_ctx_init_complex_float_acf(gr_ctx_t ctx, slong prec) noexcept

    void gr_ctx_init_real_ca(gr_ctx_t ctx) noexcept
    void gr_ctx_init_complex_ca(gr_ctx_t ctx) noexcept
    void gr_ctx_init_real_algebraic_ca(gr_ctx_t ctx) noexcept
    void gr_ctx_init_complex_algebraic_ca(gr_ctx_t ctx) noexcept
    void gr_ctx_init_complex_extended_ca(gr_ctx_t ctx) noexcept
    void _gr_ctx_init_ca_from_ref(gr_ctx_t ctx, int which_ring, void * ca_ctx) noexcept
    void gr_ctx_ca_set_option(gr_ctx_t ctx, slong option, slong value) noexcept
    slong gr_ctx_ca_get_option(gr_ctx_t ctx, slong option) noexcept

    void gr_ctx_init_fq(gr_ctx_t ctx, const fmpz_t p, slong d, const char * var) noexcept
    void gr_ctx_init_fq_nmod(gr_ctx_t ctx, ulong p, slong d, const char * var) noexcept
    void gr_ctx_init_fq_zech(gr_ctx_t ctx, ulong p, slong d, const char * var) noexcept

    void _gr_ctx_init_fq_from_ref(gr_ctx_t ctx, const void * fq_ctx) noexcept
    void _gr_ctx_init_fq_nmod_from_ref(gr_ctx_t ctx, const void * fq_nmod_ctx) noexcept
    void _gr_ctx_init_fq_zech_from_ref(gr_ctx_t ctx, const void * fq_zech_ctx) noexcept

    void gr_ctx_init_fmpz_poly(gr_ctx_t ctx) noexcept
    void gr_ctx_init_fmpq_poly(gr_ctx_t ctx) noexcept

    void gr_ctx_init_nf(gr_ctx_t ctx, const fmpq_poly_t poly) noexcept
    void gr_ctx_init_nf_fmpz_poly(gr_ctx_t ctx, const fmpz_poly_t poly) noexcept
    void _gr_ctx_init_nf_from_ref(gr_ctx_t ctx, const void * nfctx) noexcept

    void gr_ctx_init_perm(gr_ctx_t ctx, ulong n) noexcept
    void gr_ctx_init_psl2z(gr_ctx_t ctx) noexcept
    int gr_ctx_init_dirichlet_group(gr_ctx_t ctx, ulong q) noexcept

    void gr_ctx_init_gr_poly(gr_ctx_t ctx, gr_ctx_t base_ring) noexcept
    void gr_ctx_init_fmpz_mpoly(gr_ctx_t ctx, slong nvars, const ordering_t ord) noexcept
    void gr_ctx_init_gr_mpoly(gr_ctx_t ctx, gr_ctx_t base_ring, slong nvars, const ordering_t ord) noexcept
    void gr_ctx_init_fmpz_mpoly_q(gr_ctx_t ctx, slong nvars, const ordering_t ord) noexcept
    void gr_ctx_init_gr_series(gr_ctx_t ctx, gr_ctx_t base_ring, slong prec) noexcept
    void gr_ctx_init_gr_series_mod(gr_ctx_t ctx, gr_ctx_t base_ring, slong mod) noexcept

    void gr_ctx_init_vector_gr_vec(gr_ctx_t ctx, gr_ctx_t base_ring) noexcept
    void gr_ctx_init_vector_space_gr_vec(gr_ctx_t ctx, gr_ctx_t base_ring, slong n) noexcept

    void gr_ctx_init_matrix_domain(gr_ctx_t ctx, gr_ctx_t base_ring) noexcept
    void gr_ctx_init_matrix_space(gr_ctx_t ctx, gr_ctx_t base_ring, slong nrows, slong ncols) noexcept
    void gr_ctx_init_matrix_ring(gr_ctx_t ctx, gr_ctx_t base_ring, slong n)

    void gr_ctx_init_fexpr(gr_ctx_t ctx)

    truth_t gr_ctx_is_finite(gr_ctx_t ctx)
    truth_t gr_ctx_is_multiplicative_group(gr_ctx_t ctx)
    truth_t gr_ctx_is_ring(gr_ctx_t ctx)
    truth_t gr_ctx_is_commutative_ring(gr_ctx_t ctx)
    truth_t gr_ctx_is_integral_domain(gr_ctx_t ctx)
    truth_t gr_ctx_is_unique_factorization_domain(gr_ctx_t ctx)
    truth_t gr_ctx_is_field(gr_ctx_t ctx)
    truth_t gr_ctx_is_algebraically_closed(gr_ctx_t ctx)
    truth_t gr_ctx_is_finite_characteristic(gr_ctx_t ctx)
    truth_t gr_ctx_is_ordered_ring(gr_ctx_t ctx)
    truth_t gr_ctx_is_zero_ring(gr_ctx_t ctx)

    truth_t gr_ctx_is_exact(gr_ctx_t ctx)

    truth_t gr_ctx_is_canonical(gr_ctx_t ctx)

    truth_t gr_ctx_has_real_prec(gr_ctx_t ctx)

    int gr_ctx_set_real_prec(gr_ctx_t ctx, slong prec)
    int gr_ctx_get_real_prec(slong *prec, gr_ctx_t ctx)

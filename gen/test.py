r"""
TESTS::

    sage: import gen.flint_parents
    sage: from gen.element_flint_generic import FlintElement, FlintVector

    sage: FZZ = gen.flint_parents.IntegerRing()
    sage: FZZ is gen.flint_parents.IntegerRing()
    True
    sage: FZZ
    Integer ring (fmpz)
    sage: FZZ.gens()
    []
    sage: type(FZZ.gens())
    <class 'gen.element_flint_generic.FlintVector'>
    sage: FZZ.category()
    Category of infinite unique factorization domains
    sage: FZZ.is_commutative()
    True
    sage: FZZ.zero()
    0
    sage: FZZ.zero().is_one()
    False
    sage: FZZ.one().is_one()
    True
    sage: FZZ.zero() == FZZ.zero()
    True
    sage: FZZ.zero() == FZZ.one()
    False
    sage: FZZ.zero() + FZZ.one() + FZZ.one()
    2
    sage: FlintElement(FZZ, -1)
    -1
    sage: FlintElement(FZZ, 1r)
    1
    sage: FlintElement(FZZ, QQ(2))
    2
    sage: FlintElement(FZZ, 1/3)
    Traceback (most recent call last):
    ...
    ValueError: unable to convert 1/3 to an element of Integer ring (fmpz)
    sage: FlintElement(FZZ, FZZ)
    Traceback (most recent call last):
    ...
    TypeError: unable to convert Integer ring (fmpz) to an element of Integer ring (fmpz)
    sage: one = FZZ(1)
    sage: two = one + one
    sage: -two
    -2
    sage: two/one
    Traceback (most recent call last):
    ...
    NotImplementedError
    sage: (one + two)//two
    1
    sage: float(FZZ(1))
    1.0
    sage: ZZ(FZZ(1)).parent()
    Integer Ring

    sage: C = gen.flint_parents.ComplexSymbolicField()
    sage: zero = C.zero()
    sage: one = C.one()
    sage: two = one + one
    sage: one/two
    0.500000 {1/2}
    sage: two/zero
    Traceback (most recent call last):
    ...
    ZeroDivisionError: (2, 0)
    sage: two = C.one() + C.one()
    sage: two^(-C.one()/two)
    0.707107 {(a)/2 where a = 1.41421 [a^2-2=0]}

    sage: ZZi = gen.flint_parents.GaussianIntegerRing()
    sage: ZZi.gen()
    I
    sage: ZZi.base_ring()
    Integer ring (fmpz)
    sage: C(ZZi.gen() + ZZi(1))
    1.00000 + 1.00000*I {a+1 where a = I [a^2+1=0]}
    sage: (ZZi.one() + 1).parent()
    Gaussian integer ring (fmpzi)
    sage: ZZi.one() + 1
    2
    sage: C.one() + ZZi.gen() + 1/3
    1.33333 + 1.00000*I {(3*a+4)/3 where a = I [a^2+1=0]}
    sage: float(C(1/3))
    0.3333333333333333
    sage: ZZ(C(1))
    1
    sage: ZZ(C(1/2))
    Traceback (most recent call last):
    ...
    ValueError: Invalid argument for generic FLINT operation (GR_DOMAIN)

    sage: vec =  FlintVector(FZZ)
    sage: vec.universe()
    Integer ring (fmpz)
    sage: len(vec)
    0
    sage: vec.append(FZZ(1))
    sage: len(vec)
    1
    sage: list(vec)
    [1]
    sage: vec[0]
    1
    sage: vec[-1]
    Traceback (most recent call last):
    ...
    IndexError: vector index out of range
    sage: vec[1]
    Traceback (most recent call last):
    ...
    IndexError: vector index out of range
    sage: vec1 = vec.copy()
    sage: vec[0] = 42
    sage: vec[0] = 1/2
    Traceback (most recent call last):
    ...
    TypeError: no canonical coercion from Rational Field to Integer ring (fmpz)
    sage: vec[1] = 1
    Traceback (most recent call last):
    ...
    IndexError: vector index out of range
    sage: vec.append(1)
    sage: vec.steal_item(0)
    42
    sage: vec
    [0, 1]
    sage: vec1.set_length(3)
    sage: vec1
    [1, 0, 0]
"""

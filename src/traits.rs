pub(crate) mod sealed {
    pub trait Sealed {}
}

/// A common interface for modular arithmetic.
///
/// This trait is implemented by all types in this crate.
pub trait Mod: Sized + sealed::Sealed {
    /// The underlying native type.
    type Native;

    /// The value of the modulus, or `0` if it is a power of two.
    ///
    /// `0` is used because the full value `2^k` does not fit in a `k`-bit integer, so it overflows
    /// to `0`.
    const MODULUS: Self::Native;

    /// A constant `0` value.
    const ZERO: Self;

    /// A constant `1` value.
    const ONE: Self;

    /// Create a value corresponding to `x mod m`.
    #[must_use]
    fn new(x: Self::Native) -> Self;

    /// Create a value corresponding to `x`, assuming `x` is a correct representation for this type.
    ///
    /// This function is most useful for prime moduli, since it's faster than [`new`](Self::new) for
    /// those types.
    ///
    /// # Safety
    ///
    /// This function is safe to call if either of the two conditions holds:
    /// - `x` is less than the modulus, or
    /// - `x` was produced by [`to_raw`](Self::to_raw) on the same type.
    #[must_use]
    unsafe fn new_unchecked(x: Self::Native) -> Self;

    /// Get the normalized residue `x mod m`.
    #[must_use]
    fn remainder(self) -> Self::Native;

    /// Get the internal optimized representation of the number.
    ///
    /// This returns some value equivalent to `x` modulo `m`, but not necessarily `x mod m` itself.
    /// This is more efficient than [`remainder`](Self::remainder) for prime moduli. Passing this
    /// value to [`new`](Self::new) or [`new_unchecked`](Self::new_unchecked)
    /// is guaranteed to produce the same value as `self`.
    #[must_use]
    fn to_raw(self) -> Self::Native;

    /// Compare for equality with a constant.
    ///
    /// For fast moduli, this is more efficient than `x == FastK::new(C)`. `C` must be a valid
    /// reminder, i.e. `C < m`.
    ///
    /// `C` is typed `u64` instead of [`Native`](Self::Native) due to Rust having issues with
    /// associated types in `const`.
    #[must_use]
    fn is<const C: u64>(self) -> bool;

    /// Compare for equality with zero.
    ///
    /// This is equivalent to `is::<0>()`.
    #[must_use]
    fn is_zero(&self) -> bool;

    /// Compute `x^n mod m`.
    ///
    /// The current implementation uses iterative binary exponentiation, combining it with [the
    /// Carmichael function][1] to reduce exponents. It works in `O(log n)`.
    ///
    /// [1]: https://en.wikipedia.org/wiki/Carmichael_function
    #[must_use]
    fn pow(self, n: u64) -> Self;

    /// Check if the value is invertible, i.e. if `x` is coprime with `m`.
    ///
    /// The current implementation checks `x % p != 0` for all prime factors `p` of the modulus.
    /// This is very fast for `PowerK`, relatively fast for `PrimeK`/`BigPrimeK` and slightly slow
    /// for `FastK`.
    #[must_use]
    fn is_invertible(&self) -> bool;

    /// Compute the multiplicative inverse.
    ///
    /// Returns `None` if `x` is not coprime with `m`.
    ///
    /// For power-of-two moduli, the current implementation uses [an algorithm by
    /// Hurchalla][hurchalla], which works in `O(log k)`. For other moduli, it uses a variation of
    /// [Pornin's algorithm][pornin], which works in `O(k)`.
    ///
    /// [hurchalla]: https://arxiv.org/abs/2204.04342
    /// [pornin]: https://eprint.iacr.org/2020/972.pdf
    #[must_use]
    fn inverse(self) -> Option<Self>;
}

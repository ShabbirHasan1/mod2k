//! Arithmetic modulo `2^8`, `2^16`, `2^32`, and `2^64`.
//!
//! Most combinations of operations are compiled as efficiently as possible. Note that these types
//! behave more like modular arithmetic than `uK`:
//!
//! - Left shifts are unbounded.
//! - Right shifts are not implemented because `2` is not invertible, and left shifts by negative
//!   amounts panic.
//! - Arithmetic doesn't panic on overflow in debug.
//! - Negation is implemented with wrapping semantics.

use core::ops::{Add, Mul, Neg, Shl, Sub};

macro_rules! define_type {
    (
        #[$meta:meta]
        $ty:ident as $native:ident, $signed:ident,
        test in $test_mod:ident
    ) => {
        #[$meta]
        ///
        /// See [module-level documentation](self) for more information.
        #[derive(Clone, Copy, Default)]
        pub struct $ty {
            /// Stores the remainder exactly.
            value: $native,
        }

        crate::macros::define_type_basics!($ty as $native, shr = false);

        impl $ty {
            #[allow(unused, reason = "used by tests")]
            const MODULUS: $native = 0;
            #[allow(unused, reason = "used by tests")]
            const CARMICHAEL: u64 = 1 << ($native::BITS - 2);

            /// Create a value corresponding to `x mod 2^k`.
            #[inline]
            pub const fn new(x: $native) -> Self {
                Self { value: x }
            }

            /// Create a value corresponding to `x`, assuming `x < 2^k`.
            ///
            /// This function behaves exactly like [`new`](Self::new). It is present for
            /// compatibility with prime moduli, for which `from_remainder_unchecked` is faster.
            ///
            /// # Safety
            ///
            /// This function is always valid to call, since any `k`-bit number is less than `2^k`.
            #[inline]
            pub unsafe fn from_remainder_unchecked(x: $native) -> Self {
                Self { value: x }
            }

            /// Get the normalized residue `x mod 2^k`.
            #[inline]
            pub const fn remainder(self) -> $native {
                self.value
            }

            /// Get the internal optimized representation of the number.
            ///
            /// This is the same thing as [`remainder`](Self::remainder). This is present for
            /// compatibility with fast moduli, where `to_raw` is faster.
            #[inline]
            pub const fn to_raw(self) -> $native {
                self.value
            }

            /// Compare for equality with a constant.
            ///
            /// This is the same thing as `x == ModK::new(C)`. This function is present for
            /// compatibility with fast moduli, for which `is` is faster than `==`.
            #[inline]
            pub const fn is<const C: $native>(self) -> bool {
                self.value == C
            }

            /// Compare for equality with zero.
            ///
            /// This is equialvent to `x.is::<0>()` or `x == ModK::ZERO`.
            #[inline]
            pub const fn is_zero(self) -> bool {
                self.value == 0
            }

            /// Compute `x^n mod 2^k`.
            ///
            /// The current implementation uses iterative binary exponentiation, combining it with
            /// [the Carmichael function][1] to reduce exponents. It works in `O(log n)`.
            ///
            /// [1]: https://en.wikipedia.org/wiki/Carmichael_function#Exponential_cycle_length
            pub fn pow(self, mut n: u64) -> Self {
                if n >= Self::CARMICHAEL {
                    if self.value & 1 == 0 {
                        // `2^(big number) = 0 (mod 2^k)`.
                        return Self::ZERO;
                    }
                    // `x` is invertible, so `x^lambda = 1 (mod 2^k)`.
                    n %= Self::CARMICHAEL;
                }
                self.pow_internal(n)
            }

            /// Check if the value is invertible, i.e. if `x` is odd.
            ///
            /// This method is provided for compatibility with fast moduli, for which the check is
            /// more complicated.
            pub fn is_invertible(self) -> bool {
                self.value & 1 == 1
            }

            /// Compute multiplicative inverse.
            ///
            /// Returns `None` if `x` is even.
            ///
            /// The current implementation uses Hensel lifting and works in `O(log k)`.
            pub fn inverse(self) -> Option<Self> {
                if self.value & 1 == 0 {
                    return None;
                }

                // https://en.wikipedia.org/wiki/Modular_multiplicative_inverse#Inverses_modulo_prime_powers_(including_powers_of_2)
                let mut x = self;
                for _ in 0..$native::BITS.ilog2() {
                    x *= Self::new(2) - self * x;
                }
                Some(x)
            }
        }

        impl Add for $ty {
            type Output = Self;

            #[inline]
            fn add(self, other: Self) -> Self {
                Self::new(self.value.wrapping_add(other.value))
            }
        }

        impl Sub for $ty {
            type Output = Self;

            #[inline]
            fn sub(self, other: Self) -> Self {
                Self::new(self.value.wrapping_sub(other.value))
            }
        }

        impl Mul for $ty {
            type Output = Self;

            #[inline]
            #[allow(clippy::suspicious_arithmetic_impl, reason = "2^k mod (2^k - 1) = 1")]
            fn mul(self, other: Self) -> Self {
                Self::new(self.value.wrapping_mul(other.value))
            }
        }

        impl Neg for $ty {
            type Output = Self;

            #[inline]
            fn neg(self) -> Self {
                Self::new(self.value.wrapping_neg())
            }
        }

        impl Shl<i64> for $ty {
            type Output = Self;

            #[inline]
            fn shl(self, n: i64) -> Self {
                assert!(n >= 0, "shift by negative amount");
                self << (n as u64)
            }
        }

        impl Shl<u64> for $ty {
            type Output = Self;

            #[inline]
            fn shl(self, n: u64) -> Self {
                if n >= $native::BITS.into() {
                    Self::ZERO
                } else {
                    Self::new(self.value << n as u32)
                }
            }
        }

        impl PartialEq for $ty {
            #[inline]
            fn eq(&self, other: &$ty) -> bool {
                self.value == other.value
            }
        }

        #[cfg(test)]
        mod $test_mod {
            use super::$ty;

            crate::macros::test_ty!($ty as $native, $signed, shr = false);
            crate::macros::test_exact_raw!($ty as $native);
        }
    };
}

define_type! {
    /// Arithmetic modulo `2^8 = 256`.
    Mod8 as u8, i8, test in test8
}

define_type! {
    /// Arithmetic modulo `2^16 = 65536`.
    Mod16 as u16, i16, test in test16
}

define_type! {
    /// Arithmetic modulo `2^32 = 4294967296`.
    Mod32 as u32, i32, test in test32
}

define_type! {
    /// Arithmetic modulo `2^64 = 18446744073709551616`.
    Mod64 as u64, i64, test in test64
}

#[cfg(doctest)]
#[allow(dead_code, reason = "ad-hoc compile-fail test")]
/// ```compile_fail
/// mod2km1::power::Mod8::ZERO >> 0;
/// ```
///
/// ```compile_fail
/// mod2km1::power::Mod16::ZERO >> 0;
/// ```
///
/// ```compile_fail
/// mod2km1::power::Mod32::ZERO >> 0;
/// ```
///
/// ```compile_fail
/// mod2km1::power::Mod64::ZERO >> 0;
/// ```
fn test_shr() {}

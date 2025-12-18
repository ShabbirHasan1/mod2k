// This implementation of extended Stein's GCD algorithm is a mixture of:
// - Highly optimized Stein's GCD from https://en.algorithmica.org/hpc/algorithms/gcd/
// - Constant-time extended GCD from https://eprint.iacr.org/2020/972.pdf
// ...with a few further performance optimizations.
//
// Stein's GCD operates on values `a, b` with subtraction or exact right-shifting, and stops when
// reaching `a' = 0`, returning `b'` as the GCD. The transforms are linear, so there exist some
// `x, y` dependent exclusively on the performed operations such that `b' = ax + by`. For invertible
// values, this produces `b' = 1`, and thus `ax = 1 (mod y)`. By applying the same operations in
// parallel to a pair `(u, v) = (1, 0)`, we get the modular inverse `v' = 1x + 0y = x`.
//
// The only issue with this is that since the operations include division by powers of two, `x, y`
// are not integers, so `u, v` require a more complicated data type. There are three options
// depending on the modular type:
// 1. If modular right-shifts are cheap, we can use the modular types for `u` and `v`.
// 2. Alternatively, if the total amount of right-shifts is small, we can store `u, v` in
//    fixed-point, allocating enough fractional bits that the right-shifts are exact, and then
//    convert fixed-point to modular once at the end.
// 3. Alternatively, .
// as implicitly multiplied by some `2^t`, replace `u >>= q` with
//   `v <<= q; t += q`, and .
// - Otherwise, we can store `u, v` as implicitly multiplied by some `2^t`, replace `u >>= q` with
//   `v <<= q; t += q`, and .

// - If it's expensive, we can work in fixed point and convert to modular at the end. We prove that
//   a signed numeric type twice as long as the input is sufficient below.

macro_rules! define_exgcd_inverse {
    (prime = $prime:literal, strategy = builtin) => {
        fn inverse(self) -> Option<Self> {
            if self.is_zero() {
                return None;
            }

            let mut a = self.value as u64;
            let mut b = Self::MODULUS as u64;
            let mut u = Self::ONE;
            let mut v = Self::ZERO;

            // At the start of each iteration, `a` is non-zero and `b` is odd.
            let mut q = a.trailing_zeros();
            let mut total_q = 0;
            while a != 0 {
                // Teach the optimizer that `u` can be right-shifted efficiently.
                // SAFETY: At any point, `current a <= max(initial a, initial b)`, thus
                // `q < len(x) <= len(max(initial x, initial y))`. Both the stored value and the
                // modulus are guaranteed to fit in `k` bits, even if the value is above the
                // modulus, so `q <= k - 1`.
                unsafe {
                    core::hint::assert_unchecked(q <= Self::MODULUS.ilog2());
                }
                a >>= q;
                v <<= q;
                total_q += q;

                // (a, b) -> (|y - x|, min(a, b))
                let diff_ba = b.wrapping_sub(a);
                q = diff_ba.trailing_zeros(); // `|y - x|` has the same ctz as `y - x`
                (a, b, u, v) = core::hint::select_unpredictable(
                    a < b,
                    (diff_ba, a, v, u),
                    (diff_ba.wrapping_neg(), b, u, v),
                );
                u -= v;
            }

            unsafe {
                core::hint::assert_unchecked(total_q <= 2 * Self::MODULUS.ilog2());
            }
            ($prime || b == 1).then_some(v >> total_q)
        }
    };

    // `$modulus_inv = MODULUS^-1 mod 2^62`.
    (prime = $prime:literal, strategy = short with $modulus_inv:literal) => {
        fn inverse(self) -> Option<Self> {
            if self.is_zero() {
                return None;
            }

            let fp_to_modular = |x: u64| -> Self {
                // Get 1 out of the way quickly, since it makes handling of signed numbers difficult
                // and REDC doesn't handle it correctly.
                if x == 1 << 63 {
                    return Self::ONE;
                }
                let x = x as i64;
                // Compute `x / 2^63 mod MODULUS` with REDC.
                //
                // For non-negative `x`, take
                //     x' = x - ((x * MODULUS^-1) mod 2^63) * MODULUS
                // Then `x' = x (mod MODULUS)` and `x' = 0 (mod 2^63)`, so
                //     x / 2^63 = x' >> 63 (mod MODULUS)
                // Since the bottom 63 bits of `x` and the subtrahend are equal, and the bits above
                // that position in `x` are zero (unless `x = 2^63`, which we handle explicitly),
                // that's equal to
                //     -(((x * MODULUS^-1) mod 2^63) * MODULUS) >> 63
                // which can be computed more efficiently as
                //     -(((x * (MODULUS^-1 << 1)) mod 2^64) * MODULUS) >> 64
                // This is a value between `0` and `-(MODULUS - 1)`, so this can be
                // straightforwardly translated to a remainder.
                let factor = x.unsigned_abs().wrapping_mul($modulus_inv << 1);
                let neg_rem = factor.carrying_mul(Self::MODULUS as u64, 0).1 as Self::Native;
                Self::new(if x >= 0 {
                    Self::MODULUS - neg_rem
                } else {
                    neg_rem
                })
            };

            let mut a = self.value as u64;
            let mut b = Self::MODULUS as u64;

            let mut q = a.trailing_zeros();

            let mut u: u64 = 1;
            let mut v: u64 = 0;
            let mut precision_left = 63;

            while a != 0 {
                precision_left -= q;
                a >>= q;
                v <<= q;

                // (a, b) -> (|y - x|, min(a, b))
                let diff_ba = b.wrapping_sub(a);
                q = diff_ba.trailing_zeros(); // `|y - x|` has the same ctz as `y - x`
                (a, b, u, v) = core::hint::select_unpredictable(
                    a < b,
                    (diff_ba, a, v, u),
                    (diff_ba.wrapping_neg(), b, u, v),
                );
                u = u.wrapping_sub(v);
            }

            v <<= precision_left;

            ($prime || b == 1).then_some(fp_to_modular(v))
        }
    };

    // `$modulus_inv = MODULUS^-1 mod 2^63`.
    (prime = $prime:literal, strategy = long with $modulus_inv:literal) => {
        fn inverse(self) -> Option<Self> {
            if self.is_zero() {
                return None;
            }

            let fp_to_modular = |x: u64| -> Self {
                // Get 1 out of the way quickly, since it makes handling of signed numbers difficult
                // and REDC doesn't handle it correctly.
                if x == 1 << 63 {
                    return Self::ONE;
                }
                let x = x as i64;
                // Compute `x / 2^63 mod MODULUS` with REDC.
                //
                // For non-negative `x`, take
                //     x' = x - ((x * MODULUS^-1) mod 2^63) * MODULUS
                // Then `x' = x (mod MODULUS)` and `x' = 0 (mod 2^63)`, so
                //     x / 2^63 = x' >> 63 (mod MODULUS)
                // Since the bottom 63 bits of `x` and the subtrahend are equal, and the bits above
                // that position in `x` are zero (unless `x = 2^63`, which we handle explicitly),
                // that's equal to
                //     -(((x * MODULUS^-1) mod 2^63) * MODULUS) >> 63
                // which can be computed more efficiently as
                //     -(((x * (MODULUS^-1 << 1)) mod 2^64) * MODULUS) >> 64
                // This is a value between `0` and `-(MODULUS - 1)`, so this can be
                // straightforwardly translated to a remainder.
                let factor = x.unsigned_abs().wrapping_mul($modulus_inv << 1);
                let neg_rem = factor.carrying_mul(Self::MODULUS as u64, 0).1 as Self::Native;
                Self::new(if x >= 0 {
                    Self::MODULUS - neg_rem
                } else {
                    neg_rem
                })
            };

            let mut a = self.value;
            let mut b = Self::MODULUS;

            // The values are implicitly multiplied by `2^(63 - precision_left)`, so that they can
            // be stored as integers. They are signed, but stored in an unusual format that
            // represents values `-2^63 + 1..=2^63` instead of the usual `-2^63..=2^63 - 1`; that
            // is, the bit pattern `100..000` represents `2^63` and not `-2^63`.

            let mut is_first_iteration = true;
            let mut u_acc = Self::ONE;
            let mut v_acc = Self::ZERO;
            let mut q = a.trailing_zeros();

            loop {
                // The matrix transforming `(u, v)` to `(u', v')`:
                // (u') = (f0 g0) (u)
                // (v')   (f1 g1) (v)
                let mut u: u64 = 1;
                let mut v: u64 = if is_first_iteration { 0 } else { 1 << 32 };
                let mut precision_left = if is_first_iteration { 63 } else { 31 };

                // At the start of each iteration, `x` is non-zero and `y` is odd.
                while a != 0 && q < precision_left {
                    precision_left -= q;
                    a >>= q;
                    v <<= q;

                    // (a, b) -> (|y - x|, min(a, b))
                    let diff_ba = b.wrapping_sub(a);
                    q = diff_ba.trailing_zeros(); // `|y - x|` has the same ctz as `y - x`
                    (a, b, u, v) = core::hint::select_unpredictable(
                        a < b,
                        (diff_ba, a, v, u),
                        (diff_ba.wrapping_neg(), b, u, v),
                    );
                    u = u.wrapping_sub(v);
                }

                q -= precision_left;
                a >>= precision_left;
                v <<= precision_left;

                let parse_coeffs = |coeffs: u64| -> [Self; 2] {
                    [coeffs & ((1 << 32) - 1), (coeffs + (1 << 31) - 1) >> 32]
                        .map(|x| fp_to_modular(x << 32))
                };

                if a == 0 {
                    v_acc = if is_first_iteration {
                        fp_to_modular(v)
                    } else {
                        let [f1, g1] = parse_coeffs(v);
                        f1 * u_acc + g1 * v_acc
                    };
                    return ($prime || b == 1).then_some(v_acc);
                }

                if is_first_iteration {
                    u_acc = fp_to_modular(u);
                    v_acc = fp_to_modular(v);
                } else {
                    let [f0, g0] = parse_coeffs(u);
                    let [f1, g1] = parse_coeffs(v);
                    (u_acc, v_acc) = (f0 * u_acc + g0 * v_acc, f1 * u_acc + g1 * v_acc);
                }

                is_first_iteration = false;
            }
        }
    };
}
pub(crate) use define_exgcd_inverse;

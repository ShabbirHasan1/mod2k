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
// are not integers, so `u, v` require a more complicated data type. From a high-level perspective,
// there are two options:
// 1. If modular arithmetic is cheap, we can use the modular type for `u, v`.
// 2. Alternatively, we can store `u, v` in fixed-point, allocating enough fractional bits that the
//    shifts are exact, and then convert fixed-point to modular once at the end.
//
// When we're talking about cheap arithmetic, we mean *really* cheap: if modular subtraction + shift
// have a total latency of more than ~3 ticks, the core loop will get slowed down, which is worse
// than wasting constant time on conversion. Option 1 is useful almost exclusively for `Fast64`, and
// then only because the handling of 64-bit numbers in exgcd is a bit slower than of smaller ones.
//
// A straightforward high-level implementation of exgcd would look like:
//     u = 1
//     v = 0
//     while a != 0:
//         q = a.trailing_zeros()
//         a >>= q
//         u >>= q
//         if a < b:
//             swap(a, b)
//             swap(u, v)
//         a -= b
//         u -= v
// ...but we write it slightly differently. First, we reduce the latency of the critical path with
// the optimization from algorithmica, computing `trailing_zeros` of `a - b` immediately after the
// right shift, since `trailing_zeros(|a - b|) = trailing_zeros(a - b)`. But in exgcd, this is not
// the only critical path: there's also the path `swap(u, v)` -> `u -= v` -> `u >>= q`, which is
// dangerously long for non-trivial implementations of arithmetic. But since we don't actually use
// `u, v` until later, we can replacing `u >>= q` with `v <<= q` and fix that up in post. This
// replacement allows `u -= v` and `v <<= q` to be performed in parallel.

macro_rules! define_exgcd_inverse {
    // Option 1 from the list above.
    (prime = $prime:literal, builtin) => {
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
                // `q < len(x) <= len(max(initial x, initial y))`. In our data model, both the
                // stored value and the modulus always fit in `k` bits, so `q <= k - 1`.
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

            ($prime || b == 1).then_some(v >> total_q)
        }
    };

    // Option 2 from the list above.
    //
    // "Fixed-point" in the high-level explanation does a lot of heavy lifting. Since we replace
    // `u >>= q` with `v <<= q`, the scale effectively changes with time, so it's closer to floating
    // point.
    //
    // How many bits do we need? Suppose that, at the beginning of some iteration, we have
    //     -2^(t+1) < u < 2^(t+1)
    //     -2^t < v <= 2^t
    // If this is not the first iteration, we're guaranteed to have `q > 0`, so after the shift we
    // get
    //     -2^(t+1) < u < 2^(t+1)
    //     -2^(t+q) < v <= 2^(t+q)
    // and thus
    //     -2^(t+q) < u < 2^(t+q)
    //     -2^(t+q) < v <= 2^(t+q)
    // After the conditional swap and `u -= v` we get
    //     -2^(t+q+1) < u < 2^(t+q+1)
    //     -2^(t+q) < v <= 2^(t+q)
    // ...i.e. the original statement with `t' = t + q`. We can manually check that the original
    // statement holds after the first iteration for `t = 0`, so we can infer
    //     -2^(total_q+1) < u < 2^(total_q+1)
    //     -2^total_q < v <= 2^total_q
    // So `u, v` can be represented by `total_q + 2`-bit integers.
    //
    // Now, what is `total_q`? `len(a) + len(b)` starts at `<= 2k` and, right after the last shift,
    // ends at `>= 2`, so `total_q <= 2k - 2` and `2k` bits are necessary. This bound is reachable.
    //
    // This provides an acceptable solution for `k <= 32`. For `k > 32`, it would require long
    // arithmetic, completely tanking performance, so we use something slightly different.
    //
    // We start with the same idea, operating on `u, v` directly until all `64` bits are used. When
    // that happens, we convert `u, v` to modular and keep going using *symbolic representations* of
    // `u, v`. Instead of storing the exact values, each of `u` and `v` becomes a pair of
    // coefficients, such that `u = (f0, g0)` indicates `f0 * old_u + g0 * old_v` and similarly for
    // `v`. We pack these pairs into 64-bit values, allocating `32` bits to each linear coefficient,
    // so that the exact same looping logic can be used. When we run out of `32` bits, we parse
    // `f0, g0` out of the pair, apply these modifications to `u, v`, and repeat the process.
    //
    // The packing is just `u = f0 + g0 * 2^32`. Note that `f0` can be negative, and it's
    // effectively sign-extended so that vector subtraction works consistently, so `u` is not just
    // a bitwise concatenation of `f0, g0`. This is an improvement over Pornin.
    //
    // In general, handling 64-bit numbers requires at most 126 iterations, which will be split into
    // 62 + 30 + 30 + 4. The `+ 4` part is rarely reached, and in fact on average just one of `+ 30`
    // typically suffices.
    (prime = $prime:literal, long = $long:literal) => {
        fn inverse(self) -> Option<Self> {
            if self.is_zero() {
                return None;
            }

            let mut a = self.value as u64;
            let mut b = Self::MODULUS as u64;
            let mut u_acc = Self::ONE;
            let mut v_acc = Self::ZERO;
            let mut is_first_iteration = true;

            let mut q = a.trailing_zeros();
            loop {
                // Depending on whether this is the first iteration, `u, v` is either the upscaled
                // exact value, or a representation of the 2x2 matrix in
                //     (u') = (f0 g0) (u)
                //     (v')   (f1 g1) (v)
                // In the latter case, `u = f0 + g0 * 2^32` and `v = f1 + g1 * 2^32`.
                let mut u: u64 = 1;
                let mut v: u64 = if is_first_iteration { 0 } else { 1 << 32 };

                // Assuming 1 bit is devoted to the sign, how many bits are left for data? IOW, how
                // many times can we shift to the left without losing data?
                let mut precision_left = if is_first_iteration { 64 } else { 32 } - 1;

                // At the start of each iteration, `x` is non-zero and `y` is odd.
                // Note the spelling `q < precision_left` instead of `q <= precision_left`. While we
                // shift by just `q`, we then subtract `u -= v`, which needs one more bit. This is
                // evidenced by the asymmetry in the two inequalities from above:
                //     -2^(t+1) < u < 2^(t+1)
                //     -2^t < v <= 2^t
                while a != 0 && (!$long || q < precision_left) {
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

                // These three lines look deceptively simple. They can in fact perform two
                // completely things:
                //
                // - If the previous loop quit due to lack of precision, this performs the amount of
                //   shift that still fits and adjusts `q` so that the operation is completed by the
                //   next iteration. This effectively decreases `precision_left` to `0`, though we
                //   don't do this explicitly because this is the last use of `precision_left`. This
                //   effectively increases `total_q` to `63`/`31`.
                //
                //   You might ask why this even works: since we've noted that `total_q + 2` bits
                //   are required, you might expect `total_q = 63` to require `65` bits, while we
                //   only have `64` bits here. That's not quite true: the set of inequalities below
                //   only holds at the beginning/end of the iteration:
                //       -2^(t+1) < u < 2^(t+1)
                //       -2^t < v <= 2^t
                //   In the middle of the iteration, i.e. after the shift (which we effectively
                //   perform here), we obtain
                //       -2^(t+q) < u < 2^(t+q)
                //       -2^t < v <= 2^t
                //   ...and thus
                //       -2^total_q < u < 2^total_q
                //       -2^(total_q-1) < v <= 2^(total_q-1)
                //   ...so `total_q + 1` bits are sufficient.
                //
                // - If the previous loop quit due to `a = 0`, the algorithm is complete and we just
                //   want to compute `v / 2^total_q (mod MODULUS)`. Since the first case already
                //   handles this for `total_q = 63/31` specifically, it seems reasonable to shift
                //   `v` to the left by `63 - total_q` (which is what `v <<= precision_left` is
                //   responsible for) and reuse the implementation of `v / 2^63 (mod MODULUS)`.
                //
                //   The modified values of `q` and `a` are ignored. Those operations are still
                //   valid in the sense that, on the last iteration, `q = trailing_zeros(0) = 64`,
                //   and so subtraction doesn't overflow.
                //
                //   You may wonder why `total_q + 1` bits are sufficient this time, since we're not
                //   in the middle of the iteration and so the reasoning from case 1 doesn't apply.
                //   However, notice that we don't need `u` on the last iteration, only `v`, which
                //   barely fits in `total_q + 1` bits:
                //       -2^(total_q+1) < u < 2^(total_q+1)
                //       -2^total_q < v <= 2^total_q
                //   It's not exactly `i64`, but it's very close: the only difference is that the
                //   bit pattern `100...000` represents `2^63`, not `-2^63`. Same thing for the
                //   32-bit case.
                q -= precision_left;
                a >>= precision_left;
                v <<= precision_left;

                // Compute `x / 2^63 (mod MODULUS)`, where `x` is in the modified signed format.
                let fp_to_modular = |x: u64| -> Self {
                    if x == 1 << 63 {
                        Self::ONE
                    } else {
                        let xm = Self::redc64((x as i64).unsigned_abs() << 1);
                        if x as i64 >= 0 { xm } else { -xm }
                    }
                };

                let apply = |x: u64| -> Self {
                    if is_first_iteration {
                        fp_to_modular(x)
                    } else {
                        // `+ (2^31 - 1)` fixes off-by-one for `g` if `f` is negative.
                        let [f, g] =
                            [x << 32, (x + (1 << 31) - 1) & (u64::MAX << 32)].map(fp_to_modular);
                        f * u_acc + g * v_acc
                    }
                };

                let new_v_acc = apply(v);
                if a == 0 {
                    return ($prime || b == 1).then_some(new_v_acc);
                }

                u_acc = apply(u);
                v_acc = new_v_acc;
                is_first_iteration = false;
            }
        }
    };
}
pub(crate) use define_exgcd_inverse;

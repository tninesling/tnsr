# Host readback of virtual layouts

Fusion timing that calls `execute_compiled` includes both device execution and
host output materialization. `PtxExecutor::get_value` and the normal execution
return path download physical storage and call `materialize_host_view`.

The previous strided-copy compiler required every source coordinate to be an
affine function of output coordinates. A normal shape-changing row-major reshape
usually fails that condition: its decoded coordinates contain floor division
and remainder, although the final flattened source address is the original
linear address. Falling back to the reference implementation then allocates
coordinates and interprets checked `IndexExpr` trees for every output element.
This is CPU readback work, not evidence of a slow CUDA kernel.

After predicate and shape/storage validation, exact full-rank identity maps now
return the downloaded vector immediately, avoiding proof work for common outputs.
Prefix or vacuous identity maps with a wrong rank do not take that fast path.

`src/ptx/readback.rs` otherwise compiles the flattened address symbolically. It:

1. Proves bounds for every original checked-i64 subexpression and verifies each
   source coordinate independently, before replacing its evaluator.
2. Represents affine terms and quotient atoms with compile-time i128
   coefficients, using Euclidean remainder reconstruction and positive nested
   division identities. Affine numerators are split into divisor multiples and
   a signed residual: floor/mod are affine when the residual quotient is constant
   over the full rectangular domain. This handles bounded mixed-radix digits,
   reversed low digits and negative coefficients without assuming divisibility
   of every coefficient. Affine quotient arguments are canonicalized so composed
   reshape decode/reconstruction terms can cancel.
3. Accepts the result only when all nonlinear atoms cancel and the final offsets
   and strides fit checked i64. Remaining nonlinear maps retain the reference
   fallback.
4. Reuses the downloaded vector for a contiguous full-storage reshape. Positive
   two-dimensional scalar-stride maps validate their complete address range once
   and extend exact-size rows, avoiding per-element slice copies and odometer
   bookkeeping. Negative or more general affine maps retain the checked strided
   contiguous-block copying loop.

The compile-time expansion is never executed as checked i64 arithmetic. This
matters near negative i64 boundaries, where a valid Euclidean remainder can have
an out-of-range intermediate if naively rewritten as runtime multiplication and
subtraction. Original overflow and invalid-coordinate errors are not suppressed.

Tests compare compiled maps with the reference for shape-changing and composed
reshapes, permutations, broadcasts and nonlinear gathers. Storage-pointer reuse
and exact float bit patterns cover signed zero, infinities and NaN payloads.
Additional negative tests ensure a flattened identity cannot hide an invalid
source coordinate or overflow in an original subexpression.

For GPU profiling, separate CUDA event/kernel measurements from host readback,
or call `get_value` with `TNSR_PROFILE_READBACK=1` to report its copy and view
materialization times separately. That diagnostic currently wraps `get_value`,
not the normal `execute_compiled` return path.
No floating-point operations, GPU reduction association or dtype boundaries are
changed by this host materialization improvement.


Focused verification: `cargo nextest run --features cuda --lib -E
'test(ptx::readback)'` passed all 20 selected device-free tests. These host-only tests are separate from GPU end-to-end benchmark validation.


Mixed-radix property checks compare every accepted affine floor/mod against the
original Euclidean evaluator across approximately 68,000 signed-coefficient
expressions. Exact carry boundaries that are not affine remain on the checked
fallback, and overflow in an original multiply cannot be hidden by later
cancellation or division.

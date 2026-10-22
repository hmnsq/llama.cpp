#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/broadcast.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/divide.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/gather.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/loop.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/matmul.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/scatter_update.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/shape_of.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/subtract.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

// GGML SOLVE_TRI: solve Ax = B for lower-triangular A via forward substitution.
// Currently only lower, right, non-unitriangular variant is implemented.
//
// ggml layout: A [n, n, B1, B2], B [k, n, B1, B2] → X [k, n, B1, B2]
// OV layout:   A [B2, B1, n, n], B [B2, B1, n, k] → X [B2, B1, n, k]
//
// Forward substitution row i:
//   x[i] = (b[i] - sum_{t<i} A[i,t]*x[t]) / A[i,i]
//
// Implemented as an OV Loop op iterating n times with a carried X accumulator.
// Key insight: A is lower-triangular and X starts as zeros, so the full matmul
//   A_row_i @ X_partial = sum_{t<i} A[i,t]*x[t] exactly (upper triangle of A
//   is zero; unfilled rows of X are zero).
Output_ਸਦਿਸ਼ translate_solve_tri(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 2, 2);

    ਸਵੈਚਾਲਿਤ A = context.get_ਨਿਵੇਸ਼(0);  // [B2, B1, n, n]
    ਸਵੈਚਾਲਿਤ B = context.get_ਨਿਵੇਸ਼(1);  // [B2, B1, n, k]

    ਸਵੈਚਾਲਿਤ A_shape = context.get_ਨਿਵੇਸ਼_shape(0).ਮੰਜ਼ਿਲ_shape();
    int64_t n = static_cast<int64_t>(A_shape[2]);

    // Initial X: zeros with shape of B
    ਸਵੈਚਾਲਿਤ B_shape_ਗੰਢ = std::make_shared<ov::op::v3::Shape_ਦਾ>(B, ov::element::i64);
    ਸਵੈਚਾਲਿਤ zero_f32  = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
    ਸਵੈਚਾਲਿਤ X_init    = std::make_shared<ov::op::v3::ਪ੍ਰਸਾਰਣ>(zero_f32, B_shape_ਗੰਢ);

    // --- Loop body parameters ---
    // body_iter: iteration counter injected by the Loop op (i64, shape {1})
    ਸਵੈਚਾਲਿਤ body_iter = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
    ਸਵੈਚਾਲਿਤ body_X    = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    ਸਵੈਚਾਲਿਤ body_A    = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    ਸਵੈਚਾਲਿਤ body_B_p  = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));

    ਸਵੈਚਾਲਿਤ c_axis2        = ov::op::v0::Constant::create(ov::element::i64, {1}, {int64_t(2)});
    ਸਵੈਚਾਲਿਤ c_axis3        = ov::op::v0::Constant::create(ov::element::i64, {1}, {int64_t(3)});
    ਸਵੈਚਾਲਿਤ c_axis2_scalar = ov::op::v0::Constant::create(ov::element::i64, {}, {int64_t(2)});

    // b_i      = B[..., i, :]      [B2, B1, 1, k]
    ਸਵੈਚਾਲਿਤ b_i = std::make_shared<ov::op::v8::Gather>(body_B_p, body_iter, c_axis2);

    // A_row_i  = A[..., i, :]      [B2, B1, 1, n]
    ਸਵੈਚਾਲਿਤ A_ਪੰਕਤੀ_i = std::make_shared<ov::op::v8::Gather>(body_A, body_iter, c_axis2);

    // sum_i    = A_row_i @ X       [B2, B1, 1, k]
    // (lower-tri zeros + unfilled-X zeros make this equal to the partial sum)
    ਸਵੈਚਾਲਿਤ ਜੋੜ_i = std::make_shared<ov::op::v0::MatMul>(A_ਪੰਕਤੀ_i, body_X, ਝੂਠ, ਝੂਠ);

    // diag_i   = A[..., i, i]      [B2, B1, 1, 1]
    ਸਵੈਚਾਲਿਤ diag_i = std::make_shared<ov::op::v8::Gather>(A_ਪੰਕਤੀ_i, body_iter, c_axis3);

    // x_i      = (b_i - sum_i) / diag_i    [B2, B1, 1, k]
    ਸਵੈਚਾਲਿਤ x_i = std::make_shared<ov::op::v1::Divide>(
        std::make_shared<ov::op::v1::ਘਟਾਓ>(b_i, ਜੋੜ_i), diag_i);

    // X_updated: scatter x_i into body_X at row i along axis 2
    ਸਵੈਚਾਲਿਤ X_updated = std::make_shared<ov::op::v3::ScatterUpdate>(body_X, body_iter, x_i, c_axis2_scalar);

    ਸਵੈਚਾਲਿਤ body_cond = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, {ਸੱਚ});

    ਸਵੈਚਾਲਿਤ body = std::make_shared<ov::Model>(
        ov::Output_ਸਦਿਸ਼{body_cond, X_updated},
        ov::Parameter_ਸਦਿਸ਼{body_iter, body_X, body_A, body_B_p});

    // --- Assemble Loop ---
    ਸਵੈਚਾਲਿਤ trip_ਗਿਣਤੀ = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{n});
    ਸਵੈਚਾਲਿਤ exec_cond  = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, {ਸੱਚ});

    ਸਵੈਚਾਲਿਤ loop = std::make_shared<ov::op::v5::Loop>(trip_ਗਿਣਤੀ, exec_cond);
    loop->ਸੈੱਟ_function(body);
    // iter_counter_body_param_idx=0 (body_iter), exec_condition_body_result_idx=0 (body_cond)
    loop->ਸੈੱਟ_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{0, 0});

    // Carried state: X feeds back from X_updated each iteration
    loop->ਸੈੱਟ_merged_ਨਿਵੇਸ਼(body_X, X_init, X_updated);
    // Invariant inputs passed through unchanged
    loop->ਸੈੱਟ_invariant_ਨਿਵੇਸ਼(body_A,   A);
    loop->ਸੈੱਟ_invariant_ਨਿਵੇਸ਼(body_B_p, B);

    // Final output: value of X_updated after the last iteration
    ਸਵੈਚਾਲਿਤ X_final = loop->get_iter_ਮੁੱਲ(X_updated, -1);

    ਵਾਪਸ rename_outputs_with_suffix({X_final}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

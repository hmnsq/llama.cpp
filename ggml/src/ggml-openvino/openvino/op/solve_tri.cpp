#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <openvino/op/broadcast.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/divide.hpp>
#شامل <openvino/op/gather.hpp>
#شامل <openvino/op/loop.hpp>
#شامل <openvino/op/matmul.hpp>
#شامل <openvino/op/scatter_update.hpp>
#شامل <openvino/op/shape_of.hpp>
#شامل <openvino/op/subtract.hpp>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

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
OutputVector translate_solve_tri(ثابت گره_Context & context) {
    رقم_inputs_check(context, 2, 2);

    auto A = context.get_درون_داد(0);  // [B2, B1, n, n]
    auto B = context.get_درون_داد(1);  // [B2, B1, n, k]

    auto A_shape = context.get_درون_داد_shape(0).to_shape();
    int64_t n = static_cast<int64_t>(A_shape[2]);

    // Initial X: zeros with shape of B
    auto B_shape_گره = std::make_shared<ov::op::v3::ShapeOf>(B, ov::element::i64);
    auto zero_f32  = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
    auto X_init    = std::make_shared<ov::op::v3::Broadcast>(zero_f32, B_shape_گره);

    // --- Loop body parameters ---
    // body_iter: iteration counter injected by the Loop op (i64, shape {1})
    auto body_iter = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
    auto body_X    = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    auto body_A    = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    auto body_B_p  = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));

    auto c_axis2        = ov::op::v0::Constant::create(ov::element::i64, {1}, {int64_t(2)});
    auto c_axis3        = ov::op::v0::Constant::create(ov::element::i64, {1}, {int64_t(3)});
    auto c_axis2_scalar = ov::op::v0::Constant::create(ov::element::i64, {}, {int64_t(2)});

    // b_i      = B[..., i, :]      [B2, B1, 1, k]
    auto b_i = std::make_shared<ov::op::v8::Gather>(body_B_p, body_iter, c_axis2);

    // A_row_i  = A[..., i, :]      [B2, B1, 1, n]
    auto A_row_i = std::make_shared<ov::op::v8::Gather>(body_A, body_iter, c_axis2);

    // sum_i    = A_row_i @ X       [B2, B1, 1, k]
    // (lower-tri zeros + unfilled-X zeros make this equal to the partial sum)
    auto sum_i = std::make_shared<ov::op::v0::MatMul>(A_row_i, body_X, نادرست, نادرست);

    // diag_i   = A[..., i, i]      [B2, B1, 1, 1]
    auto diag_i = std::make_shared<ov::op::v8::Gather>(A_row_i, body_iter, c_axis3);

    // x_i      = (b_i - sum_i) / diag_i    [B2, B1, 1, k]
    auto x_i = std::make_shared<ov::op::v1::Divide>(
        std::make_shared<ov::op::v1::Subtract>(b_i, sum_i), diag_i);

    // X_updated: scatter x_i into body_X at row i along axis 2
    auto X_updated = std::make_shared<ov::op::v3::ScatterUpdate>(body_X, body_iter, x_i, c_axis2_scalar);

    auto body_cond = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, {درست});

    auto body = std::make_shared<ov::Model>(
        ov::OutputVector{body_cond, X_updated},
        ov::ParameterVector{body_iter, body_X, body_A, body_B_p});

    // --- Assemble Loop ---
    auto trip_شمار = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{n});
    auto exec_cond  = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, {درست});

    auto loop = std::make_shared<ov::op::v5::Loop>(trip_شمار, exec_cond);
    loop->set_function(body);
    // iter_counter_body_param_idx=0 (body_iter), exec_condition_body_result_idx=0 (body_cond)
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{0, 0});

    // Carried state: X feeds back from X_updated each iteration
    loop->set_merged_درون_داد(body_X, X_init, X_updated);
    // Invariant inputs passed through unchanged
    loop->set_invariant_درون_داد(body_A,   A);
    loop->set_invariant_درون_داد(body_B_p, B);

    // Final output: value of X_updated after the last iteration
    auto X_final = loop->get_iter_مقدار(X_updated, -1);

    بازگشت rename_outputs_with_suffix({X_final}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

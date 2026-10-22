#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <openvino/op/broadcast.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/divide.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/loop.hpp>
#ضمن <openvino/op/matmul.hpp>
#ضمن <openvino/op/scatter_update.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <openvino/op/subtract.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

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
Output_ويڪٽر translate_solve_tri(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 2, 2);

    خودڪار A = context.get_داخلا(0);  // [B2, B1, n, n]
    خودڪار B = context.get_داخلا(1);  // [B2, B1, n, k]

    خودڪار A_shape = context.get_داخلا_shape(0).منزل_shape();
    int64_t n = static_cast<int64_t>(A_shape[2]);

    // Initial X: zeros with shape of B
    خودڪار B_shape_ڳنڍ = std::make_shared<ov::op::v3::Shape_جو>(B, ov::element::i64);
    خودڪار zero_f32  = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
    خودڪار X_init    = std::make_shared<ov::op::v3::نشر>(zero_f32, B_shape_ڳنڍ);

    // --- Loop body parameters ---
    // body_iter: iteration counter injected by the Loop op (i64, shape {1})
    خودڪار body_iter = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
    خودڪار body_X    = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    خودڪار body_A    = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    خودڪار body_B_p  = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));

    خودڪار c_axis2        = ov::op::v0::Constant::create(ov::element::i64, {1}, {int64_t(2)});
    خودڪار c_axis3        = ov::op::v0::Constant::create(ov::element::i64, {1}, {int64_t(3)});
    خودڪار c_axis2_scalar = ov::op::v0::Constant::create(ov::element::i64, {}, {int64_t(2)});

    // b_i      = B[..., i, :]      [B2, B1, 1, k]
    خودڪار b_i = std::make_shared<ov::op::v8::Gather>(body_B_p, body_iter, c_axis2);

    // A_row_i  = A[..., i, :]      [B2, B1, 1, n]
    خودڪار A_صف_i = std::make_shared<ov::op::v8::Gather>(body_A, body_iter, c_axis2);

    // sum_i    = A_row_i @ X       [B2, B1, 1, k]
    // (lower-tri zeros + unfilled-X zeros make this equal to the partial sum)
    خودڪار جوڙ_i = std::make_shared<ov::op::v0::MatMul>(A_صف_i, body_X, خطأ, خطأ);

    // diag_i   = A[..., i, i]      [B2, B1, 1, 1]
    خودڪار diag_i = std::make_shared<ov::op::v8::Gather>(A_صف_i, body_iter, c_axis3);

    // x_i      = (b_i - sum_i) / diag_i    [B2, B1, 1, k]
    خودڪار x_i = std::make_shared<ov::op::v1::Divide>(
        std::make_shared<ov::op::v1::ڪٽيو>(b_i, جوڙ_i), diag_i);

    // X_updated: scatter x_i into body_X at row i along axis 2
    خودڪار X_updated = std::make_shared<ov::op::v3::ScatterUpdate>(body_X, body_iter, x_i, c_axis2_scalar);

    خودڪار body_cond = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, {سچ});

    خودڪار body = std::make_shared<ov::Model>(
        ov::Output_ويڪٽر{body_cond, X_updated},
        ov::Parameter_ويڪٽر{body_iter, body_X, body_A, body_B_p});

    // --- Assemble Loop ---
    خودڪار trip_ڳڻپ = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{n});
    خودڪار exec_cond  = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, {سچ});

    خودڪار loop = std::make_shared<ov::op::v5::Loop>(trip_ڳڻپ, exec_cond);
    loop->سيٽ_function(body);
    // iter_counter_body_param_idx=0 (body_iter), exec_condition_body_result_idx=0 (body_cond)
    loop->سيٽ_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{0, 0});

    // Carried state: X feeds back from X_updated each iteration
    loop->سيٽ_merged_داخلا(body_X, X_init, X_updated);
    // Invariant inputs passed through unchanged
    loop->سيٽ_invariant_داخلا(body_A,   A);
    loop->سيٽ_invariant_داخلا(body_B_p, B);

    // Final output: value of X_updated after the last iteration
    خودڪار X_final = loop->get_iter_قدر(X_updated, -1);

    أرجع rename_outputs_with_suffix({X_final}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

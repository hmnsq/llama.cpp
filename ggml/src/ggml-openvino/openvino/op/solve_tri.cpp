#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <openvino/op/broadcast.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/divide.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/gather.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/loop.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/matmul.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/scatter_update.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/shape_of.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/subtract.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

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
Output_ಸದಿಶ translate_solve_tri(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);

    ಸ್ವಯಂಚಾಲಿತ A = context.get_ಒಳಹರಿವು(0);  // [B2, B1, n, n]
    ಸ್ವಯಂಚಾಲಿತ B = context.get_ಒಳಹರಿವು(1);  // [B2, B1, n, k]

    ಸ್ವಯಂಚಾಲಿತ A_shape = context.get_ಒಳಹರಿವು_shape(0).ಗಮ್ಯ_shape();
    int64_t n = static_cast<int64_t>(A_shape[2]);

    // Initial X: zeros with shape of B
    ಸ್ವಯಂಚಾಲಿತ B_shape_ಗಂಟು = std::make_shared<ov::op::v3::Shape_ದ>(B, ov::element::i64);
    ಸ್ವಯಂಚಾಲಿತ zero_f32  = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
    ಸ್ವಯಂಚಾಲಿತ X_init    = std::make_shared<ov::op::v3::ಪ್ರಸಾರ>(zero_f32, B_shape_ಗಂಟು);

    // --- Loop body parameters ---
    // body_iter: iteration counter injected by the Loop op (i64, shape {1})
    ಸ್ವಯಂಚಾಲಿತ body_iter = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
    ಸ್ವಯಂಚಾಲಿತ body_X    = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    ಸ್ವಯಂಚಾಲಿತ body_A    = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    ಸ್ವಯಂಚಾಲಿತ body_B_p  = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));

    ಸ್ವಯಂಚಾಲಿತ c_axis2        = ov::op::v0::Constant::create(ov::element::i64, {1}, {int64_t(2)});
    ಸ್ವಯಂಚಾಲಿತ c_axis3        = ov::op::v0::Constant::create(ov::element::i64, {1}, {int64_t(3)});
    ಸ್ವಯಂಚಾಲಿತ c_axis2_scalar = ov::op::v0::Constant::create(ov::element::i64, {}, {int64_t(2)});

    // b_i      = B[..., i, :]      [B2, B1, 1, k]
    ಸ್ವಯಂಚಾಲಿತ b_i = std::make_shared<ov::op::v8::Gather>(body_B_p, body_iter, c_axis2);

    // A_row_i  = A[..., i, :]      [B2, B1, 1, n]
    ಸ್ವಯಂಚಾಲಿತ A_ಅಡ್ಡಸಾಲು_i = std::make_shared<ov::op::v8::Gather>(body_A, body_iter, c_axis2);

    // sum_i    = A_row_i @ X       [B2, B1, 1, k]
    // (lower-tri zeros + unfilled-X zeros make this equal to the partial sum)
    ಸ್ವಯಂಚಾಲಿತ ಮೊತ್ತ_i = std::make_shared<ov::op::v0::MatMul>(A_ಅಡ್ಡಸಾಲು_i, body_X, ಸುಳ್ಳು, ಸುಳ್ಳು);

    // diag_i   = A[..., i, i]      [B2, B1, 1, 1]
    ಸ್ವಯಂಚಾಲಿತ diag_i = std::make_shared<ov::op::v8::Gather>(A_ಅಡ್ಡಸಾಲು_i, body_iter, c_axis3);

    // x_i      = (b_i - sum_i) / diag_i    [B2, B1, 1, k]
    ಸ್ವಯಂಚಾಲಿತ x_i = std::make_shared<ov::op::v1::Divide>(
        std::make_shared<ov::op::v1::ಕಳೆ>(b_i, ಮೊತ್ತ_i), diag_i);

    // X_updated: scatter x_i into body_X at row i along axis 2
    ಸ್ವಯಂಚಾಲಿತ X_updated = std::make_shared<ov::op::v3::ScatterUpdate>(body_X, body_iter, x_i, c_axis2_scalar);

    ಸ್ವಯಂಚಾಲಿತ body_cond = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, {ನಿಜ});

    ಸ್ವಯಂಚಾಲಿತ body = std::make_shared<ov::Model>(
        ov::Output_ಸದಿಶ{body_cond, X_updated},
        ov::Parameter_ಸದಿಶ{body_iter, body_X, body_A, body_B_p});

    // --- Assemble Loop ---
    ಸ್ವಯಂಚಾಲಿತ trip_ಎಣಿಕೆ = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{n});
    ಸ್ವಯಂಚಾಲಿತ exec_cond  = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, {ನಿಜ});

    ಸ್ವಯಂಚಾಲಿತ loop = std::make_shared<ov::op::v5::Loop>(trip_ಎಣಿಕೆ, exec_cond);
    loop->ಗಣ_function(body);
    // iter_counter_body_param_idx=0 (body_iter), exec_condition_body_result_idx=0 (body_cond)
    loop->ಗಣ_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{0, 0});

    // Carried state: X feeds back from X_updated each iteration
    loop->ಗಣ_merged_ಒಳಹರಿವು(body_X, X_init, X_updated);
    // Invariant inputs passed through unchanged
    loop->ಗಣ_invariant_ಒಳಹರಿವು(body_A,   A);
    loop->ಗಣ_invariant_ಒಳಹರಿವು(body_B_p, B);

    // Final output: value of X_updated after the last iteration
    ಸ್ವಯಂಚಾಲಿತ X_final = loop->get_iter_ಮೌಲ್ಯ(X_updated, -1);

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({X_final}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

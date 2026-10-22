#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <openvino/op/broadcast.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/divide.hpp>
#అన్తర్భావ <openvino/op/gather.hpp>
#అన్తర్భావ <openvino/op/loop.hpp>
#అన్తర్భావ <openvino/op/matmul.hpp>
#అన్తర్భావ <openvino/op/scatter_update.hpp>
#అన్తర్భావ <openvino/op/shape_of.hpp>
#అన్తర్భావ <openvino/op/subtract.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

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
Output_సదిశ translate_solve_tri(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 2, 2);

    స్వయంచాలక A = context.get_ఆదానం(0);  // [B2, B1, n, n]
    స్వయంచాలక B = context.get_ఆదానం(1);  // [B2, B1, n, k]

    స్వయంచాలక A_shape = context.get_ఆదానం_shape(0).గమ్యం_shape();
    int64_t n = static_cast<int64_t>(A_shape[2]);

    // Initial X: zeros with shape of B
    స్వయంచాలక B_shape_కణుపు = std::make_shared<ov::op::v3::Shape_యొక్క>(B, ov::element::i64);
    స్వయంచాలక zero_f32  = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
    స్వయంచాలక X_init    = std::make_shared<ov::op::v3::ప్రసారం>(zero_f32, B_shape_కణుపు);

    // --- Loop body parameters ---
    // body_iter: iteration counter injected by the Loop op (i64, shape {1})
    స్వయంచాలక body_iter = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
    స్వయంచాలక body_X    = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    స్వయంచాలక body_A    = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    స్వయంచాలక body_B_p  = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));

    స్వయంచాలక c_axis2        = ov::op::v0::Constant::create(ov::element::i64, {1}, {int64_t(2)});
    స్వయంచాలక c_axis3        = ov::op::v0::Constant::create(ov::element::i64, {1}, {int64_t(3)});
    స్వయంచాలక c_axis2_scalar = ov::op::v0::Constant::create(ov::element::i64, {}, {int64_t(2)});

    // b_i      = B[..., i, :]      [B2, B1, 1, k]
    స్వయంచాలక b_i = std::make_shared<ov::op::v8::Gather>(body_B_p, body_iter, c_axis2);

    // A_row_i  = A[..., i, :]      [B2, B1, 1, n]
    స్వయంచాలక A_అడ్డువరుస_i = std::make_shared<ov::op::v8::Gather>(body_A, body_iter, c_axis2);

    // sum_i    = A_row_i @ X       [B2, B1, 1, k]
    // (lower-tri zeros + unfilled-X zeros make this equal to the partial sum)
    స్వయంచాలక కూడిక_i = std::make_shared<ov::op::v0::MatMul>(A_అడ్డువరుస_i, body_X, అబద్ధం, అబద్ధం);

    // diag_i   = A[..., i, i]      [B2, B1, 1, 1]
    స్వయంచాలక diag_i = std::make_shared<ov::op::v8::Gather>(A_అడ్డువరుస_i, body_iter, c_axis3);

    // x_i      = (b_i - sum_i) / diag_i    [B2, B1, 1, k]
    స్వయంచాలక x_i = std::make_shared<ov::op::v1::Divide>(
        std::make_shared<ov::op::v1::వ్యవకలనం>(b_i, కూడిక_i), diag_i);

    // X_updated: scatter x_i into body_X at row i along axis 2
    స్వయంచాలక X_updated = std::make_shared<ov::op::v3::ScatterUpdate>(body_X, body_iter, x_i, c_axis2_scalar);

    స్వయంచాలక body_cond = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, {నిజం});

    స్వయంచాలక body = std::make_shared<ov::Model>(
        ov::Output_సదిశ{body_cond, X_updated},
        ov::Parameter_సదిశ{body_iter, body_X, body_A, body_B_p});

    // --- Assemble Loop ---
    స్వయంచాలక trip_లెక్క = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{n});
    స్వయంచాలక exec_cond  = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, {నిజం});

    స్వయంచాలక loop = std::make_shared<ov::op::v5::Loop>(trip_లెక్క, exec_cond);
    loop->సమితి_function(body);
    // iter_counter_body_param_idx=0 (body_iter), exec_condition_body_result_idx=0 (body_cond)
    loop->సమితి_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{0, 0});

    // Carried state: X feeds back from X_updated each iteration
    loop->సమితి_merged_ఆదానం(body_X, X_init, X_updated);
    // Invariant inputs passed through unchanged
    loop->సమితి_invariant_ఆదానం(body_A,   A);
    loop->సమితి_invariant_ఆదానం(body_B_p, B);

    // Final output: value of X_updated after the last iteration
    స్వయంచాలక X_final = loop->get_iter_విలువ(X_updated, -1);

    తిరిగి rename_outputs_with_suffix({X_final}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

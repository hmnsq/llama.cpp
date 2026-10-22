#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <climits>
#అన్తర్భావ <openvino/core/node.hpp>
#అన్తర్భావ <openvino/core/node_output.hpp>
#అన్తర్భావ <openvino/op/add.hpp>
#అన్తర్భావ <openvino/op/broadcast.hpp>
#అన్తర్భావ <openvino/op/concat.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/gather.hpp>
#అన్తర్భావ <openvino/op/shape_of.hpp>
#అన్తర్భావ <openvino/op/slice.hpp>
#అన్తర్భావ <openvino/op/squeeze.hpp>
#అన్తర్భావ <openvino/op/unsqueeze.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_get_అడ్డువరుసలు(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 2, 2);

    Output<Node> res;
    Output<Node> దత్తాంశం;
    int64_t అడ్డువరుస_స్థానభ్రంశం = 0;
    ఒకవేళ (context.get_view_ఆదానం_కొలత(0) > 0 && context.get_ఆదానం(0).get_partial_shape().rank() == 2) {
        // A row range view over a 2D weight Constant folds into the gather indices,
        // which keeps the dequantization subgraph intact for the plugins.
        స్థిరరాశి స్వయంచాలక view_shape = context.get_view_ఆదానం_ggml_shape(0, 0);
        స్థిరరాశి స్వయంచాలక view_stride = context.get_view_ఆదానం_stride(0, 0);
        స్థిరరాశి size_t view_స్థానభ్రంశం = context.get_view_ఆదానం_స్థానభ్రంశం(0, 0);
        స్థిరరాశి size_t అడ్డువరుస_బైట్లు = view_stride[2];
        దత్తాంశం = context.get_ఆదానం(0);
        FRONT_చివర_OP_CONVERSION_CHECK(అడ్డువరుస_బైట్లు > 0 && view_స్థానభ్రంశం % అడ్డువరుస_బైట్లు == 0 &&
                                          దత్తాంశం.get_partial_shape()[1].compatible(view_shape[3]),
                                      "GET_ROWS: view over a weight must be a row range");
        అడ్డువరుస_స్థానభ్రంశం = static_cast<int64_t>(view_స్థానభ్రంశం / అడ్డువరుస_బైట్లు);
    } లేకపోతే {
        దత్తాంశం = process_view_ఆదానం_new(context, 0);
    }

    స్వయంచాలక op_case = context.get_op_case();
    ఒకవేళ (op_case == 3 || op_case == 4) {
        తిరిగి {దత్తాంశం};
    }

    ov::Output<ov::Node> indices;
    ఒకవేళ ((op_case == 1 || op_case == 2) && context.has_ఆదానం("s_copy_active_slot_len")) {
        // Recurrent state reorder (inp->s_copy): slice the active (op_case 1) or extra (op_case 2)
        // segment from the s_copy index list at runtime, instead of baking the static view offset,
        // so the cached IR works for any number of active sequences.
        స్వయంచాలక s_copy = context.get_ఆదానం(1);
        స్వయంచాలక len = context.get_ఆదానం("s_copy_active_slot_len");
        స్వయంచాలక అడుగు = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        స్వయంచాలక axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        ఒకవేళ (op_case == 1) {
            స్వయంచాలక begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            indices = std::make_shared<ov::op::v8::ముక్క>(s_copy, begin, len, అడుగు, axis);
        } లేకపోతే {
            స్వయంచాలక end = ov::op::v0::Constant::create(ov::element::i64, {1}, {గరిష్ఠ_పూర్ణాంకం});
            indices = std::make_shared<ov::op::v8::ముక్క>(s_copy, len, end, అడుగు, axis);
        }
    } లేకపోతే {
        indices = process_view_ఆదానం_new(context, 1);
    }

    // data[1,b,x,y] ind[1,1,b,x'] test-backend-ops case
    // data[x,y] ind[1,1,1,x'] normal case
    // Squeeze the leading dims down to [b,x']. Stateful models drop one rank, so a hardcoded
    // {0,1} would also strip the batch dim whenever b == 1 (every decode step).
    స్థిరరాశి స్వయంచాలక indices_rank = indices.get_partial_shape().rank();
    FRONT_చివర_OP_CONVERSION_CHECK(indices_rank.is_static(), "Expected static rank for GET_ROWS indices");
    std::vector<int64_t> indices_squeeze_axes;
    కొరకు (int64_t i = 0; i + 2 < indices_rank.get_నిడివి(); ++i) {
        indices_squeeze_axes.push_back(i);
    }
    ఒకవేళ (!indices_squeeze_axes.ఖాళీలు()) {
        indices = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {indices_squeeze_axes.size()},
                                                  indices_squeeze_axes));
    }
    ఒకవేళ (అడ్డువరుస_స్థానభ్రంశం != 0) {
        indices = std::make_shared<ov::op::v1::కూడు>(
            indices, ov::op::v0::Constant::create(indices.get_element_type(), {}, {అడ్డువరుస_స్థానభ్రంశం}));
    }
    ఒకవేళ (దత్తాంశం.get_partial_shape().rank() == 4) {
        ఒకవేళ (!(దత్తాంశం.get_partial_shape()[1].is_dynamic()) && దత్తాంశం.get_partial_shape()[1].get_నిడివి() == 1) {
            // Work-around for a bug in ov cpu plugin for test-backend-ops
            దత్తాంశం = std::make_shared<ov::op::v0::Squeeze>(దత్తాంశం,
                                                         ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
            స్వయంచాలక axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
            res = std::make_shared<ov::op::v8::Gather>(దత్తాంశం, indices, axis);
        } లేకపోతే {
            స్వయంచాలక axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
            దత్తాంశం =
                std::make_shared<ov::op::v0::Squeeze>(దత్తాంశం, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
            // data: [batch, rows, ...], indices: [batch, n] - this is a batched gather
            // (batch_dims=1) along the rows axis. The data and indices batch dims are
            // logically equal (both == n_tokens) but reach this node through independent
            // reshapes, so the GPU plugin's gather shape inference cannot prove
            // data.shape[0] == indices.shape[0] and rejects the node. We must tie both
            // batch dims to the SAME value, and crucially that value must stay DYNAMIC.
            స్థిరరాశి స్వయంచాలక దత్తాంశం_ps = దత్తాంశం.get_partial_shape();
            స్థిరరాశి స్వయంచాలక idx_ps = indices.get_partial_shape();
            స్థిరరాశి తార్కిక దత్తాంశం_batch_static = దత్తాంశం_ps.rank().is_static() && దత్తాంశం_ps[0].is_static();
            స్థిరరాశి తార్కిక idx_batch_dynamic = idx_ps.rank().is_dynamic() || idx_ps[0].is_dynamic();

            ఒకవేళ (దత్తాంశం_batch_static && idx_batch_dynamic) {
                // MoE per-expert-scale path: `data` is a statically-tiled REPEAT
                // (ggml_repeat_4d(scale, 1, n_expert, n_tokens, 1)) whose batch dim is a
                // compile-time-constant n_tokens, and every batch slice is IDENTICAL (it was
                // tiled from a single [1, n_expert, 1] scale). `indices` (selected_experts)
                // carries the genuinely dynamic token dim. Broadcasting indices up to the
                // static data batch (the naive fix) would freeze the token dim to the
                // captured prefill length, and that static value then flows through the
                // gather into the residual stream, making every following decoder layer
                // static -> triggers the GPU in-place-concat KV-cache corruption (only
                // layer 0 stays dynamic). A static->dynamic Broadcast cannot expand, so
                // instead collapse the redundant data batch to 1 and broadcast 1->dynamic to
                // match the indices batch. Mathematically identical (the slices are equal),
                // and the whole graph stays dynamic.
                స్వయంచాలక zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                స్వయంచాలక one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
                స్వయంచాలక axis0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                స్వయంచాలక దత్తాంశం_b1 = std::make_shared<ov::op::v8::ముక్క>(దత్తాంశం, zero, one, one, axis0);  // [1, rows, ...]

                స్వయంచాలక idx_shape = std::make_shared<ov::op::v3::Shape_యొక్క>(indices, ov::element::i64);
                స్వయంచాలక idx_batch = get_dimensions(idx_shape, {0});  // [batch] (dynamic)
                స్వయంచాలక దత్తాంశం_b1_shape = std::make_shared<ov::op::v3::Shape_యొక్క>(దత్తాంశం_b1, ov::element::i64);
                స్థిరరాశి స్వయంచాలక rank = దత్తాంశం_ps.rank().get_నిడివి();
                std::vector<పూర్ణాంకం> rest_axes;
                కొరకు (పూర్ణాంకం a = 1; a < rank; ++a) {
                    rest_axes.push_back(a);
                }
                స్వయంచాలక దత్తాంశం_rest = get_dimensions(దత్తాంశం_b1_shape, rest_axes);  // [rows, ...]
                స్వయంచాలక దత్తాంశం_లక్ష్యం = std::make_shared<ov::op::v0::Concat>(ov::Output_సదిశ{idx_batch, దత్తాంశం_rest}, 0);
                దత్తాంశం =
                    std::make_shared<ov::op::v3::ప్రసారం>(దత్తాంశం_b1, దత్తాంశం_లక్ష్యం, ov::op::ప్రసారం_Type::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(దత్తాంశం, indices, axis, 1);
            } లేకపోతే {
                // General case: tie the indices batch to the data batch (the data batch is
                // already dynamic, e.g. the routing-weights gather whose data comes from the
                // activations). Broadcast indices to [data_batch, indices_n].
                స్వయంచాలక దత్తాంశం_shape = std::make_shared<ov::op::v3::Shape_యొక్క>(దత్తాంశం, ov::element::i64);
                స్వయంచాలక దత్తాంశం_batch = get_dimensions(దత్తాంశం_shape, {0});  // [batch]
                స్వయంచాలక idx_shape = std::make_shared<ov::op::v3::Shape_యొక్క>(indices, ov::element::i64);
                స్వయంచాలక idx_n = get_dimensions(idx_shape, {1});  // [n]
                స్వయంచాలక idx_లక్ష్యం = std::make_shared<ov::op::v0::Concat>(ov::Output_సదిశ{దత్తాంశం_batch, idx_n}, 0);
                indices = std::make_shared<ov::op::v3::ప్రసారం>(indices, idx_లక్ష్యం,
                                                                  ov::op::ప్రసారం_Type::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(దత్తాంశం, indices, axis, 1);
            }
        }
    } లేకపోతే ఒకవేళ (context.is_stateful() && దత్తాంశం.get_partial_shape().rank() == 3) {
        స్వయంచాలక axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
        res = std::make_shared<ov::op::v8::Gather>(దత్తాంశం, indices, axis, 1);
    } లేకపోతే {
        స్వయంచాలక axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
        res = std::make_shared<ov::op::v8::Gather>(దత్తాంశం, indices, axis);
    }

    ఒకవేళ (res.get_element_type() != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
    }
    ఒకవేళ (!(context.is_stateful())) {
        res = std::make_shared<ov::op::v0::Unsqueeze>(res, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
    }
    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

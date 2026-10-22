#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <climits>
#अन्तर्भाव <openvino/core/node.hpp>
#अन्तर्भाव <openvino/core/node_output.hpp>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/broadcast.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/gather.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/squeeze.hpp>
#अन्तर्भाव <openvino/op/unsqueeze.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

Output_सदिशम् translate_get_पङ्क्तिकाः(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 2, 2);

    Output<Node> res;
    Output<Node> दत्त;
    int64_t row_विस्थापनम् = 0;
    यदि (context.get_view_निवेशः_आकार(0) > 0 && context.get_निवेशः(0).get_partial_shape().rank() == 2) {
        // A row range view over a 2D weight Constant folds into the gather indices,
        // which keeps the dequantization subgraph intact for the plugins.
        ध्रुव स्वचालित view_shape = context.get_view_निवेशः_ggml_shape(0, 0);
        ध्रुव स्वचालित view_stride = context.get_view_निवेशः_stride(0, 0);
        ध्रुव size_t view_विस्थापनम् = context.get_view_निवेशः_विस्थापनम्(0, 0);
        ध्रुव size_t row_अष्टकानि = view_stride[2];
        दत्त = context.get_निवेशः(0);
        FRONT_अन्तः_OP_CONVERSION_CHECK(row_अष्टकानि > 0 && view_विस्थापनम् % row_अष्टकानि == 0 &&
                                          दत्त.get_partial_shape()[1].compatible(view_shape[3]),
                                      "GET_ROWS: view over a weight must be a row range");
        row_विस्थापनम् = static_cast<int64_t>(view_विस्थापनम् / row_अष्टकानि);
    } अन्यथा {
        दत्त = process_view_निवेशः_new(context, 0);
    }

    स्वचालित op_case = context.get_op_case();
    यदि (op_case == 3 || op_case == 4) {
        प्रत्यागम {दत्त};
    }

    ov::Output<ov::Node> indices;
    यदि ((op_case == 1 || op_case == 2) && context.has_निवेशः("s_copy_active_slot_len")) {
        // Recurrent state reorder (inp->s_copy): slice the active (op_case 1) or extra (op_case 2)
        // segment from the s_copy index list at runtime, instead of baking the static view offset,
        // so the cached IR works for any number of active sequences.
        स्वचालित s_copy = context.get_निवेशः(1);
        स्वचालित len = context.get_निवेशः("s_copy_active_slot_len");
        स्वचालित चरणः = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        स्वचालित axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        यदि (op_case == 1) {
            स्वचालित begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            indices = std::make_shared<ov::op::v8::खण्डिका>(s_copy, begin, len, चरणः, axis);
        } अन्यथा {
            स्वचालित end = ov::op::v0::Constant::create(ov::element::i64, {1}, {अधिकतम_पूर्णाङ्कः});
            indices = std::make_shared<ov::op::v8::खण्डिका>(s_copy, len, end, चरणः, axis);
        }
    } अन्यथा {
        indices = process_view_निवेशः_new(context, 1);
    }

    // data[1,b,x,y] ind[1,1,b,x'] test-backend-ops case
    // data[x,y] ind[1,1,1,x'] normal case
    // Squeeze the leading dims down to [b,x']. Stateful models drop one rank, so a hardcoded
    // {0,1} would also strip the batch dim whenever b == 1 (every decode step).
    ध्रुव स्वचालित indices_rank = indices.get_partial_shape().rank();
    FRONT_अन्तः_OP_CONVERSION_CHECK(indices_rank.is_static(), "Expected static rank for GET_ROWS indices");
    std::vector<int64_t> indices_squeeze_axes;
    प्रति (int64_t i = 0; i + 2 < indices_rank.get_दीर्घता(); ++i) {
        indices_squeeze_axes.push_back(i);
    }
    यदि (!indices_squeeze_axes.रिक्तानि()) {
        indices = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {indices_squeeze_axes.size()},
                                                  indices_squeeze_axes));
    }
    यदि (row_विस्थापनम् != 0) {
        indices = std::make_shared<ov::op::v1::सङ्कलय>(
            indices, ov::op::v0::Constant::create(indices.get_element_type(), {}, {row_विस्थापनम्}));
    }
    यदि (दत्त.get_partial_shape().rank() == 4) {
        यदि (!(दत्त.get_partial_shape()[1].is_dynamic()) && दत्त.get_partial_shape()[1].get_दीर्घता() == 1) {
            // Work-around for a bug in ov cpu plugin for test-backend-ops
            दत्त = std::make_shared<ov::op::v0::Squeeze>(दत्त,
                                                         ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
            स्वचालित axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
            res = std::make_shared<ov::op::v8::Gather>(दत्त, indices, axis);
        } अन्यथा {
            स्वचालित axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
            दत्त =
                std::make_shared<ov::op::v0::Squeeze>(दत्त, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
            // data: [batch, rows, ...], indices: [batch, n] - this is a batched gather
            // (batch_dims=1) along the rows axis. The data and indices batch dims are
            // logically equal (both == n_tokens) but reach this node through independent
            // reshapes, so the GPU plugin's gather shape inference cannot prove
            // data.shape[0] == indices.shape[0] and rejects the node. We must tie both
            // batch dims to the SAME value, and crucially that value must stay DYNAMIC.
            ध्रुव स्वचालित दत्त_ps = दत्त.get_partial_shape();
            ध्रुव स्वचालित idx_ps = indices.get_partial_shape();
            ध्रुव तार्किक दत्त_batch_static = दत्त_ps.rank().is_static() && दत्त_ps[0].is_static();
            ध्रुव तार्किक idx_batch_dynamic = idx_ps.rank().is_dynamic() || idx_ps[0].is_dynamic();

            यदि (दत्त_batch_static && idx_batch_dynamic) {
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
                स्वचालित zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                स्वचालित one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
                स्वचालित axis0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                स्वचालित दत्त_b1 = std::make_shared<ov::op::v8::खण्डिका>(दत्त, zero, one, one, axis0);  // [1, rows, ...]

                स्वचालित idx_shape = std::make_shared<ov::op::v3::Shape_सम्बन्धि>(indices, ov::element::i64);
                स्वचालित idx_batch = get_dimensions(idx_shape, {0});  // [batch] (dynamic)
                स्वचालित दत्त_b1_shape = std::make_shared<ov::op::v3::Shape_सम्बन्धि>(दत्त_b1, ov::element::i64);
                ध्रुव स्वचालित rank = दत्त_ps.rank().get_दीर्घता();
                std::vector<पूर्णाङ्क> rest_axes;
                प्रति (पूर्णाङ्क a = 1; a < rank; ++a) {
                    rest_axes.push_back(a);
                }
                स्वचालित दत्त_rest = get_dimensions(दत्त_b1_shape, rest_axes);  // [rows, ...]
                स्वचालित दत्त_target = std::make_shared<ov::op::v0::Concat>(ov::Output_सदिशम्{idx_batch, दत्त_rest}, 0);
                दत्त =
                    std::make_shared<ov::op::v3::प्रसारणम्>(दत्त_b1, दत्त_target, ov::op::प्रसारणम्_Type::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(दत्त, indices, axis, 1);
            } अन्यथा {
                // General case: tie the indices batch to the data batch (the data batch is
                // already dynamic, e.g. the routing-weights gather whose data comes from the
                // activations). Broadcast indices to [data_batch, indices_n].
                स्वचालित दत्त_shape = std::make_shared<ov::op::v3::Shape_सम्बन्धि>(दत्त, ov::element::i64);
                स्वचालित दत्त_batch = get_dimensions(दत्त_shape, {0});  // [batch]
                स्वचालित idx_shape = std::make_shared<ov::op::v3::Shape_सम्बन्धि>(indices, ov::element::i64);
                स्वचालित idx_n = get_dimensions(idx_shape, {1});  // [n]
                स्वचालित idx_target = std::make_shared<ov::op::v0::Concat>(ov::Output_सदिशम्{दत्त_batch, idx_n}, 0);
                indices = std::make_shared<ov::op::v3::प्रसारणम्>(indices, idx_target,
                                                                  ov::op::प्रसारणम्_Type::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(दत्त, indices, axis, 1);
            }
        }
    } अन्यथा यदि (context.is_stateful() && दत्त.get_partial_shape().rank() == 3) {
        स्वचालित axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
        res = std::make_shared<ov::op::v8::Gather>(दत्त, indices, axis, 1);
    } अन्यथा {
        स्वचालित axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
        res = std::make_shared<ov::op::v8::Gather>(दत्त, indices, axis);
    }

    यदि (res.get_element_type() != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
    }
    यदि (!(context.is_stateful())) {
        res = std::make_shared<ov::op::v0::Unsqueeze>(res, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
    }
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

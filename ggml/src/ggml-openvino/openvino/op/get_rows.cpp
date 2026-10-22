#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <climits>
#समावेश <openvino/core/node.hpp>
#समावेश <openvino/core/node_output.hpp>
#समावेश <openvino/op/add.hpp>
#समावेश <openvino/op/broadcast.hpp>
#समावेश <openvino/op/concat.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/convert.hpp>
#समावेश <openvino/op/gather.hpp>
#समावेश <openvino/op/shape_of.hpp>
#समावेश <openvino/op/slice.hpp>
#समावेश <openvino/op/squeeze.hpp>
#समावेश <openvino/op/unsqueeze.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_get_rows(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 2, 2);

    Output<Node> res;
    Output<Node> तथ्य;
    int64_t row_offset = 0;
    अगर (context.get_view_निवेश_आकार(0) > 0 && context.get_निवेश(0).get_partial_shape().rank() == 2) {
        // A row range view over a 2D weight Constant folds into the gather indices,
        // which keeps the dequantization subgraph intact for the plugins.
        स्थिर स्वतः view_shape = context.get_view_निवेश_ggml_shape(0, 0);
        स्थिर स्वतः view_stride = context.get_view_निवेश_stride(0, 0);
        स्थिर माप_प्रकार view_offset = context.get_view_निवेश_offset(0, 0);
        स्थिर माप_प्रकार row_bytes = view_stride[2];
        तथ्य = context.get_निवेश(0);
        FRONT_END_OP_CONVERSION_CHECK(row_bytes > 0 && view_offset % row_bytes == 0 &&
                                          तथ्य.get_partial_shape()[1].compatible(view_shape[3]),
                                      "GET_ROWS: view over a weight must be a row range");
        row_offset = static_cast<int64_t>(view_offset / row_bytes);
    } अन्यथा {
        तथ्य = process_view_निवेश_new(context, 0);
    }

    स्वतः op_case = context.get_op_case();
    अगर (op_case == 3 || op_case == 4) {
        वापस {तथ्य};
    }

    ov::Output<ov::Node> indices;
    अगर ((op_case == 1 || op_case == 2) && context.has_निवेश("s_copy_active_slot_len")) {
        // Recurrent state reorder (inp->s_copy): slice the active (op_case 1) or extra (op_case 2)
        // segment from the s_copy index list at runtime, instead of baking the static view offset,
        // so the cached IR works for any number of active sequences.
        स्वतः s_copy = context.get_निवेश(1);
        स्वतः len = context.get_निवेश("s_copy_active_slot_len");
        स्वतः step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        स्वतः axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        अगर (op_case == 1) {
            स्वतः begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            indices = std::make_shared<ov::op::v8::Slice>(s_copy, begin, len, step, axis);
        } अन्यथा {
            स्वतः end = ov::op::v0::Constant::create(ov::element::i64, {1}, {पूर्णांक_उच्च});
            indices = std::make_shared<ov::op::v8::Slice>(s_copy, len, end, step, axis);
        }
    } अन्यथा {
        indices = process_view_निवेश_new(context, 1);
    }

    // data[1,b,x,y] ind[1,1,b,x'] test-backend-ops case
    // data[x,y] ind[1,1,1,x'] normal case
    // Squeeze the leading dims down to [b,x']. Stateful models drop one rank, so a hardcoded
    // {0,1} would also strip the batch dim whenever b == 1 (every decode step).
    स्थिर स्वतः indices_rank = indices.get_partial_shape().rank();
    FRONT_END_OP_CONVERSION_CHECK(indices_rank.is_static(), "Expected static rank for GET_ROWS indices");
    std::vector<int64_t> indices_squeeze_axes;
    क्रम (int64_t i = 0; i + 2 < indices_rank.get_लंबाई(); ++i) {
        indices_squeeze_axes.push_back(i);
    }
    अगर (!indices_squeeze_axes.empty()) {
        indices = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {indices_squeeze_axes.size()},
                                                  indices_squeeze_axes));
    }
    अगर (row_offset != 0) {
        indices = std::make_shared<ov::op::v1::Add>(
            indices, ov::op::v0::Constant::create(indices.get_element_type(), {}, {row_offset}));
    }
    अगर (तथ्य.get_partial_shape().rank() == 4) {
        अगर (!(तथ्य.get_partial_shape()[1].is_dynamic()) && तथ्य.get_partial_shape()[1].get_लंबाई() == 1) {
            // Work-around for a bug in ov cpu plugin for test-backend-ops
            तथ्य = std::make_shared<ov::op::v0::Squeeze>(तथ्य,
                                                         ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
            स्वतः axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
            res = std::make_shared<ov::op::v8::Gather>(तथ्य, indices, axis);
        } अन्यथा {
            स्वतः axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
            तथ्य =
                std::make_shared<ov::op::v0::Squeeze>(तथ्य, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
            // data: [batch, rows, ...], indices: [batch, n] - this is a batched gather
            // (batch_dims=1) along the rows axis. The data and indices batch dims are
            // logically equal (both == n_tokens) but reach this node through independent
            // reshapes, so the GPU plugin's gather shape inference cannot prove
            // data.shape[0] == indices.shape[0] and rejects the node. We must tie both
            // batch dims to the SAME value, and crucially that value must stay DYNAMIC.
            स्थिर स्वतः तथ्य_ps = तथ्य.get_partial_shape();
            स्थिर स्वतः idx_ps = indices.get_partial_shape();
            स्थिर bool तथ्य_batch_static = तथ्य_ps.rank().is_static() && तथ्य_ps[0].is_static();
            स्थिर bool idx_batch_dynamic = idx_ps.rank().is_dynamic() || idx_ps[0].is_dynamic();

            अगर (तथ्य_batch_static && idx_batch_dynamic) {
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
                स्वतः zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                स्वतः one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
                स्वतः axis0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                स्वतः तथ्य_b1 = std::make_shared<ov::op::v8::Slice>(तथ्य, zero, one, one, axis0);  // [1, rows, ...]

                स्वतः idx_shape = std::make_shared<ov::op::v3::ShapeOf>(indices, ov::element::i64);
                स्वतः idx_batch = get_dimensions(idx_shape, {0});  // [batch] (dynamic)
                स्वतः तथ्य_b1_shape = std::make_shared<ov::op::v3::ShapeOf>(तथ्य_b1, ov::element::i64);
                स्थिर स्वतः rank = तथ्य_ps.rank().get_लंबाई();
                std::vector<पूर्णांक> rest_axes;
                क्रम (पूर्णांक a = 1; a < rank; ++a) {
                    rest_axes.push_back(a);
                }
                स्वतः तथ्य_rest = get_dimensions(तथ्य_b1_shape, rest_axes);  // [rows, ...]
                स्वतः तथ्य_target = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{idx_batch, तथ्य_rest}, 0);
                तथ्य =
                    std::make_shared<ov::op::v3::Broadcast>(तथ्य_b1, तथ्य_target, ov::op::BroadcastType::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(तथ्य, indices, axis, 1);
            } अन्यथा {
                // General case: tie the indices batch to the data batch (the data batch is
                // already dynamic, e.g. the routing-weights gather whose data comes from the
                // activations). Broadcast indices to [data_batch, indices_n].
                स्वतः तथ्य_shape = std::make_shared<ov::op::v3::ShapeOf>(तथ्य, ov::element::i64);
                स्वतः तथ्य_batch = get_dimensions(तथ्य_shape, {0});  // [batch]
                स्वतः idx_shape = std::make_shared<ov::op::v3::ShapeOf>(indices, ov::element::i64);
                स्वतः idx_n = get_dimensions(idx_shape, {1});  // [n]
                स्वतः idx_target = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{तथ्य_batch, idx_n}, 0);
                indices = std::make_shared<ov::op::v3::Broadcast>(indices, idx_target,
                                                                  ov::op::BroadcastType::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(तथ्य, indices, axis, 1);
            }
        }
    } अन्यथा अगर (context.is_stateful() && तथ्य.get_partial_shape().rank() == 3) {
        स्वतः axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
        res = std::make_shared<ov::op::v8::Gather>(तथ्य, indices, axis, 1);
    } अन्यथा {
        स्वतः axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
        res = std::make_shared<ov::op::v8::Gather>(तथ्य, indices, axis);
    }

    अगर (res.get_element_type() != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
    }
    अगर (!(context.is_stateful())) {
        res = std::make_shared<ov::op::v0::Unsqueeze>(res, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
    }
    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

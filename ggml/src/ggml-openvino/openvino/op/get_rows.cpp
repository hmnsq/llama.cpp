#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <climits>
#અન્તર્ભાવ <openvino/core/node.hpp>
#અન્તર્ભાવ <openvino/core/node_output.hpp>
#અન્તર્ભાવ <openvino/op/add.hpp>
#અન્તર્ભાવ <openvino/op/broadcast.hpp>
#અન્તર્ભાવ <openvino/op/concat.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/convert.hpp>
#અન્તર્ભાવ <openvino/op/gather.hpp>
#અન્તર્ભાવ <openvino/op/shape_of.hpp>
#અન્તર્ભાવ <openvino/op/slice.hpp>
#અન્તર્ભાવ <openvino/op/squeeze.hpp>
#અન્તર્ભાવ <openvino/op/unsqueeze.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_get_rows(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 2, 2);

    Output<Node> res;
    Output<Node> data;
    int64_t row_offset = 0;
    જો (context.get_view_આગત_size(0) > 0 && context.get_આગત(0).get_partial_shape().rank() == 2) {
        // A row range view over a 2D weight Constant folds into the gather indices,
        // which keeps the dequantization subgraph intact for the plugins.
        અચલ auto view_shape = context.get_view_આગત_ggml_shape(0, 0);
        અચલ auto view_stride = context.get_view_આગત_stride(0, 0);
        અચલ size_t view_offset = context.get_view_આગત_offset(0, 0);
        અચલ size_t row_bytes = view_stride[2];
        data = context.get_આગત(0);
        FRONT_END_OP_CONVERSION_CHECK(row_bytes > 0 && view_offset % row_bytes == 0 &&
                                          data.get_partial_shape()[1].compatible(view_shape[3]),
                                      "GET_ROWS: view over a weight must be a row range");
        row_offset = static_cast<int64_t>(view_offset / row_bytes);
    } નહીંતર {
        data = process_view_આગત_new(context, 0);
    }

    auto op_case = context.get_op_case();
    જો (op_case == 3 || op_case == 4) {
        પાછા {data};
    }

    ov::Output<ov::Node> indices;
    જો ((op_case == 1 || op_case == 2) && context.has_આગત("s_copy_active_slot_len")) {
        // Recurrent state reorder (inp->s_copy): slice the active (op_case 1) or extra (op_case 2)
        // segment from the s_copy index list at runtime, instead of baking the static view offset,
        // so the cached IR works for any number of active sequences.
        auto s_copy = context.get_આગત(1);
        auto len = context.get_આગત("s_copy_active_slot_len");
        auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        જો (op_case == 1) {
            auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            indices = std::make_shared<ov::op::v8::Slice>(s_copy, begin, len, step, axis);
        } નહીંતર {
            auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {INT_MAX});
            indices = std::make_shared<ov::op::v8::Slice>(s_copy, len, end, step, axis);
        }
    } નહીંતર {
        indices = process_view_આગત_new(context, 1);
    }

    // data[1,b,x,y] ind[1,1,b,x'] test-backend-ops case
    // data[x,y] ind[1,1,1,x'] normal case
    // Squeeze the leading dims down to [b,x']. Stateful models drop one rank, so a hardcoded
    // {0,1} would also strip the batch dim whenever b == 1 (every decode step).
    અચલ auto indices_rank = indices.get_partial_shape().rank();
    FRONT_END_OP_CONVERSION_CHECK(indices_rank.is_static(), "Expected static rank for GET_ROWS indices");
    std::vector<int64_t> indices_squeeze_axes;
    માટે (int64_t i = 0; i + 2 < indices_rank.get_length(); ++i) {
        indices_squeeze_axes.push_back(i);
    }
    જો (!indices_squeeze_axes.empty()) {
        indices = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {indices_squeeze_axes.size()},
                                                  indices_squeeze_axes));
    }
    જો (row_offset != 0) {
        indices = std::make_shared<ov::op::v1::Add>(
            indices, ov::op::v0::Constant::create(indices.get_element_type(), {}, {row_offset}));
    }
    જો (data.get_partial_shape().rank() == 4) {
        જો (!(data.get_partial_shape()[1].is_dynamic()) && data.get_partial_shape()[1].get_length() == 1) {
            // Work-around for a bug in ov cpu plugin for test-backend-ops
            data = std::make_shared<ov::op::v0::Squeeze>(data,
                                                         ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
            auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
            res = std::make_shared<ov::op::v8::Gather>(data, indices, axis);
        } નહીંતર {
            auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
            data =
                std::make_shared<ov::op::v0::Squeeze>(data, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
            // data: [batch, rows, ...], indices: [batch, n] - this is a batched gather
            // (batch_dims=1) along the rows axis. The data and indices batch dims are
            // logically equal (both == n_tokens) but reach this node through independent
            // reshapes, so the GPU plugin's gather shape inference cannot prove
            // data.shape[0] == indices.shape[0] and rejects the node. We must tie both
            // batch dims to the SAME value, and crucially that value must stay DYNAMIC.
            અચલ auto data_ps = data.get_partial_shape();
            અચલ auto idx_ps = indices.get_partial_shape();
            અચલ તાર્કિક data_batch_static = data_ps.rank().is_static() && data_ps[0].is_static();
            અચલ તાર્કિક idx_batch_dynamic = idx_ps.rank().is_dynamic() || idx_ps[0].is_dynamic();

            જો (data_batch_static && idx_batch_dynamic) {
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
                auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
                auto axis0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                auto data_b1 = std::make_shared<ov::op::v8::Slice>(data, zero, one, one, axis0);  // [1, rows, ...]

                auto idx_shape = std::make_shared<ov::op::v3::ShapeOf>(indices, ov::element::i64);
                auto idx_batch = get_dimensions(idx_shape, {0});  // [batch] (dynamic)
                auto data_b1_shape = std::make_shared<ov::op::v3::ShapeOf>(data_b1, ov::element::i64);
                અચલ auto rank = data_ps.rank().get_length();
                std::vector<પૂર્ણાંક> rest_axes;
                માટે (પૂર્ણાંક a = 1; a < rank; ++a) {
                    rest_axes.push_back(a);
                }
                auto data_rest = get_dimensions(data_b1_shape, rest_axes);  // [rows, ...]
                auto data_target = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{idx_batch, data_rest}, 0);
                data =
                    std::make_shared<ov::op::v3::Broadcast>(data_b1, data_target, ov::op::BroadcastType::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(data, indices, axis, 1);
            } નહીંતર {
                // General case: tie the indices batch to the data batch (the data batch is
                // already dynamic, e.g. the routing-weights gather whose data comes from the
                // activations). Broadcast indices to [data_batch, indices_n].
                auto data_shape = std::make_shared<ov::op::v3::ShapeOf>(data, ov::element::i64);
                auto data_batch = get_dimensions(data_shape, {0});  // [batch]
                auto idx_shape = std::make_shared<ov::op::v3::ShapeOf>(indices, ov::element::i64);
                auto idx_n = get_dimensions(idx_shape, {1});  // [n]
                auto idx_target = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{data_batch, idx_n}, 0);
                indices = std::make_shared<ov::op::v3::Broadcast>(indices, idx_target,
                                                                  ov::op::BroadcastType::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(data, indices, axis, 1);
            }
        }
    } નહીંતર જો (context.is_stateful() && data.get_partial_shape().rank() == 3) {
        auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
        res = std::make_shared<ov::op::v8::Gather>(data, indices, axis, 1);
    } નહીંતર {
        auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
        res = std::make_shared<ov::op::v8::Gather>(data, indices, axis);
    }

    જો (res.get_element_type() != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
    }
    જો (!(context.is_stateful())) {
        res = std::make_shared<ov::op::v0::Unsqueeze>(res, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
    }
    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

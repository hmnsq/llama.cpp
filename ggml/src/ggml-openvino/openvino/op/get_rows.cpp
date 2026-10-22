#অন্তর্ভাব "../node_context.h"
#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"

#অন্তর্ভাব <climits>
#অন্তর্ভাব <openvino/core/node.hpp>
#অন্তর্ভাব <openvino/core/node_output.hpp>
#অন্তর্ভাব <openvino/op/add.hpp>
#অন্তর্ভাব <openvino/op/broadcast.hpp>
#অন্তর্ভাব <openvino/op/concat.hpp>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/convert.hpp>
#অন্তর্ভাব <openvino/op/gather.hpp>
#অন্তর্ভাব <openvino/op/shape_of.hpp>
#অন্তর্ভাব <openvino/op/slice.hpp>
#অন্তর্ভাব <openvino/op/squeeze.hpp>
#অন্তর্ভাব <openvino/op/unsqueeze.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_get_rows(ধ্রুবক গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 2, 2);

    Output<Node> res;
    Output<Node> তথ্য;
    int64_t row_offset = 0;
    যদি (context.get_view_প্রবেশ_আকার(0) > 0 && context.get_প্রবেশ(0).get_partial_shape().rank() == 2) {
        // A row range view over a 2D weight Constant folds into the gather indices,
        // which keeps the dequantization subgraph intact for the plugins.
        ধ্রুবক auto view_shape = context.get_view_প্রবেশ_ggml_shape(0, 0);
        ধ্রুবক auto view_stride = context.get_view_প্রবেশ_stride(0, 0);
        ধ্রুবক size_t view_offset = context.get_view_প্রবেশ_offset(0, 0);
        ধ্রুবক size_t row_bytes = view_stride[2];
        তথ্য = context.get_প্রবেশ(0);
        FRONT_END_OP_CONVERSION_CHECK(row_bytes > 0 && view_offset % row_bytes == 0 &&
                                          তথ্য.get_partial_shape()[1].compatible(view_shape[3]),
                                      "GET_ROWS: view over a weight must be a row range");
        row_offset = static_cast<int64_t>(view_offset / row_bytes);
    } অন্যথা {
        তথ্য = process_view_প্রবেশ_new(context, 0);
    }

    auto op_case = context.get_op_case();
    যদি (op_case == 3 || op_case == 4) {
        ফেরত {তথ্য};
    }

    ov::Output<ov::Node> indices;
    যদি ((op_case == 1 || op_case == 2) && context.has_প্রবেশ("s_copy_active_slot_len")) {
        // Recurrent state reorder (inp->s_copy): slice the active (op_case 1) or extra (op_case 2)
        // segment from the s_copy index list at runtime, instead of baking the static view offset,
        // so the cached IR works for any number of active sequences.
        auto s_copy = context.get_প্রবেশ(1);
        auto len = context.get_প্রবেশ("s_copy_active_slot_len");
        auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        যদি (op_case == 1) {
            auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            indices = std::make_shared<ov::op::v8::Slice>(s_copy, begin, len, step, axis);
        } অন্যথা {
            auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {INT_MAX});
            indices = std::make_shared<ov::op::v8::Slice>(s_copy, len, end, step, axis);
        }
    } অন্যথা {
        indices = process_view_প্রবেশ_new(context, 1);
    }

    // data[1,b,x,y] ind[1,1,b,x'] test-backend-ops case
    // data[x,y] ind[1,1,1,x'] normal case
    // Squeeze the leading dims down to [b,x']. Stateful models drop one rank, so a hardcoded
    // {0,1} would also strip the batch dim whenever b == 1 (every decode step).
    ধ্রুবক auto indices_rank = indices.get_partial_shape().rank();
    FRONT_END_OP_CONVERSION_CHECK(indices_rank.is_static(), "Expected static rank for GET_ROWS indices");
    std::vector<int64_t> indices_squeeze_axes;
    জন্য (int64_t i = 0; i + 2 < indices_rank.get_দৈর্ঘ্য(); ++i) {
        indices_squeeze_axes.push_back(i);
    }
    যদি (!indices_squeeze_axes.empty()) {
        indices = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {indices_squeeze_axes.size()},
                                                  indices_squeeze_axes));
    }
    যদি (row_offset != 0) {
        indices = std::make_shared<ov::op::v1::Add>(
            indices, ov::op::v0::Constant::create(indices.get_element_type(), {}, {row_offset}));
    }
    যদি (তথ্য.get_partial_shape().rank() == 4) {
        যদি (!(তথ্য.get_partial_shape()[1].is_dynamic()) && তথ্য.get_partial_shape()[1].get_দৈর্ঘ্য() == 1) {
            // Work-around for a bug in ov cpu plugin for test-backend-ops
            তথ্য = std::make_shared<ov::op::v0::Squeeze>(তথ্য,
                                                         ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
            auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
            res = std::make_shared<ov::op::v8::Gather>(তথ্য, indices, axis);
        } অন্যথা {
            auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
            তথ্য =
                std::make_shared<ov::op::v0::Squeeze>(তথ্য, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
            // data: [batch, rows, ...], indices: [batch, n] - this is a batched gather
            // (batch_dims=1) along the rows axis. The data and indices batch dims are
            // logically equal (both == n_tokens) but reach this node through independent
            // reshapes, so the GPU plugin's gather shape inference cannot prove
            // data.shape[0] == indices.shape[0] and rejects the node. We must tie both
            // batch dims to the SAME value, and crucially that value must stay DYNAMIC.
            ধ্রুবক auto তথ্য_ps = তথ্য.get_partial_shape();
            ধ্রুবক auto idx_ps = indices.get_partial_shape();
            ধ্রুবক যৌক্তিক তথ্য_batch_static = তথ্য_ps.rank().is_static() && তথ্য_ps[0].is_static();
            ধ্রুবক যৌক্তিক idx_batch_dynamic = idx_ps.rank().is_dynamic() || idx_ps[0].is_dynamic();

            যদি (তথ্য_batch_static && idx_batch_dynamic) {
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
                auto তথ্য_b1 = std::make_shared<ov::op::v8::Slice>(তথ্য, zero, one, one, axis0);  // [1, rows, ...]

                auto idx_shape = std::make_shared<ov::op::v3::ShapeOf>(indices, ov::element::i64);
                auto idx_batch = get_dimensions(idx_shape, {0});  // [batch] (dynamic)
                auto তথ্য_b1_shape = std::make_shared<ov::op::v3::ShapeOf>(তথ্য_b1, ov::element::i64);
                ধ্রুবক auto rank = তথ্য_ps.rank().get_দৈর্ঘ্য();
                std::vector<পূর্ণসংখ্যা> rest_axes;
                জন্য (পূর্ণসংখ্যা a = 1; a < rank; ++a) {
                    rest_axes.push_back(a);
                }
                auto তথ্য_rest = get_dimensions(তথ্য_b1_shape, rest_axes);  // [rows, ...]
                auto তথ্য_target = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{idx_batch, তথ্য_rest}, 0);
                তথ্য =
                    std::make_shared<ov::op::v3::Broadcast>(তথ্য_b1, তথ্য_target, ov::op::BroadcastType::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(তথ্য, indices, axis, 1);
            } অন্যথা {
                // General case: tie the indices batch to the data batch (the data batch is
                // already dynamic, e.g. the routing-weights gather whose data comes from the
                // activations). Broadcast indices to [data_batch, indices_n].
                auto তথ্য_shape = std::make_shared<ov::op::v3::ShapeOf>(তথ্য, ov::element::i64);
                auto তথ্য_batch = get_dimensions(তথ্য_shape, {0});  // [batch]
                auto idx_shape = std::make_shared<ov::op::v3::ShapeOf>(indices, ov::element::i64);
                auto idx_n = get_dimensions(idx_shape, {1});  // [n]
                auto idx_target = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{তথ্য_batch, idx_n}, 0);
                indices = std::make_shared<ov::op::v3::Broadcast>(indices, idx_target,
                                                                  ov::op::BroadcastType::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(তথ্য, indices, axis, 1);
            }
        }
    } অন্যথা যদি (context.is_stateful() && তথ্য.get_partial_shape().rank() == 3) {
        auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
        res = std::make_shared<ov::op::v8::Gather>(তথ্য, indices, axis, 1);
    } অন্যথা {
        auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
        res = std::make_shared<ov::op::v8::Gather>(তথ্য, indices, axis);
    }

    যদি (res.get_element_type() != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
    }
    যদি (!(context.is_stateful())) {
        res = std::make_shared<ov::op::v0::Unsqueeze>(res, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
    }
    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

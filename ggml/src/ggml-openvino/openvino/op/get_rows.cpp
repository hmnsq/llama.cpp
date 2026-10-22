#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <climits>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node_output.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/add.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/broadcast.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/concat.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/gather.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/shape_of.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/slice.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/squeeze.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/unsqueeze.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_get_ਪੰਕਤੀਆਂ(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 2, 2);

    Output<Node> res;
    Output<Node> ਅੰਕੜੇ;
    int64_t ਪੰਕਤੀ_ਵਿੱਥ = 0;
    ਜੇ (context.get_view_ਨਿਵੇਸ਼_ਮਾਪ(0) > 0 && context.get_ਨਿਵੇਸ਼(0).get_partial_shape().rank() == 2) {
        // A row range view over a 2D weight Constant folds into the gather indices,
        // which keeps the dequantization subgraph intact for the plugins.
        ਅਚਲ ਸਵੈਚਾਲਿਤ view_shape = context.get_view_ਨਿਵੇਸ਼_ggml_shape(0, 0);
        ਅਚਲ ਸਵੈਚਾਲਿਤ view_stride = context.get_view_ਨਿਵੇਸ਼_stride(0, 0);
        ਅਚਲ size_t view_ਵਿੱਥ = context.get_view_ਨਿਵੇਸ਼_ਵਿੱਥ(0, 0);
        ਅਚਲ size_t ਪੰਕਤੀ_ਬਾਈਟਾਂ = view_stride[2];
        ਅੰਕੜੇ = context.get_ਨਿਵੇਸ਼(0);
        FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(ਪੰਕਤੀ_ਬਾਈਟਾਂ > 0 && view_ਵਿੱਥ % ਪੰਕਤੀ_ਬਾਈਟਾਂ == 0 &&
                                          ਅੰਕੜੇ.get_partial_shape()[1].compatible(view_shape[3]),
                                      "GET_ROWS: view over a weight must be a row range");
        ਪੰਕਤੀ_ਵਿੱਥ = static_cast<int64_t>(view_ਵਿੱਥ / ਪੰਕਤੀ_ਬਾਈਟਾਂ);
    } ਨਹੀਂਤਾਂ {
        ਅੰਕੜੇ = process_view_ਨਿਵੇਸ਼_new(context, 0);
    }

    ਸਵੈਚਾਲਿਤ op_case = context.get_op_case();
    ਜੇ (op_case == 3 || op_case == 4) {
        ਵਾਪਸ {ਅੰਕੜੇ};
    }

    ov::Output<ov::Node> indices;
    ਜੇ ((op_case == 1 || op_case == 2) && context.has_ਨਿਵੇਸ਼("s_copy_active_slot_len")) {
        // Recurrent state reorder (inp->s_copy): slice the active (op_case 1) or extra (op_case 2)
        // segment from the s_copy index list at runtime, instead of baking the static view offset,
        // so the cached IR works for any number of active sequences.
        ਸਵੈਚਾਲਿਤ s_copy = context.get_ਨਿਵੇਸ਼(1);
        ਸਵੈਚਾਲਿਤ len = context.get_ਨਿਵੇਸ਼("s_copy_active_slot_len");
        ਸਵੈਚਾਲਿਤ ਕਦਮ = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        ਸਵੈਚਾਲਿਤ axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        ਜੇ (op_case == 1) {
            ਸਵੈਚਾਲਿਤ begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            indices = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(s_copy, begin, len, ਕਦਮ, axis);
        } ਨਹੀਂਤਾਂ {
            ਸਵੈਚਾਲਿਤ end = ov::op::v0::Constant::create(ov::element::i64, {1}, {ਵੱਧ_ਤੋਂ_ਵੱਧ_ਅੰਕ});
            indices = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(s_copy, len, end, ਕਦਮ, axis);
        }
    } ਨਹੀਂਤਾਂ {
        indices = process_view_ਨਿਵੇਸ਼_new(context, 1);
    }

    // data[1,b,x,y] ind[1,1,b,x'] test-backend-ops case
    // data[x,y] ind[1,1,1,x'] normal case
    // Squeeze the leading dims down to [b,x']. Stateful models drop one rank, so a hardcoded
    // {0,1} would also strip the batch dim whenever b == 1 (every decode step).
    ਅਚਲ ਸਵੈਚਾਲਿਤ indices_rank = indices.get_partial_shape().rank();
    FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(indices_rank.is_static(), "Expected static rank for GET_ROWS indices");
    std::vector<int64_t> indices_squeeze_axes;
    ਲਈ (int64_t i = 0; i + 2 < indices_rank.get_ਲੰਮਾਈ(); ++i) {
        indices_squeeze_axes.push_back(i);
    }
    ਜੇ (!indices_squeeze_axes.ਖਾਲੀ_ਥਾਵਾਂ()) {
        indices = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {indices_squeeze_axes.size()},
                                                  indices_squeeze_axes));
    }
    ਜੇ (ਪੰਕਤੀ_ਵਿੱਥ != 0) {
        indices = std::make_shared<ov::op::v1::ਜੋੜੋ>(
            indices, ov::op::v0::Constant::create(indices.get_element_type(), {}, {ਪੰਕਤੀ_ਵਿੱਥ}));
    }
    ਜੇ (ਅੰਕੜੇ.get_partial_shape().rank() == 4) {
        ਜੇ (!(ਅੰਕੜੇ.get_partial_shape()[1].is_dynamic()) && ਅੰਕੜੇ.get_partial_shape()[1].get_ਲੰਮਾਈ() == 1) {
            // Work-around for a bug in ov cpu plugin for test-backend-ops
            ਅੰਕੜੇ = std::make_shared<ov::op::v0::Squeeze>(ਅੰਕੜੇ,
                                                         ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
            ਸਵੈਚਾਲਿਤ axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
            res = std::make_shared<ov::op::v8::Gather>(ਅੰਕੜੇ, indices, axis);
        } ਨਹੀਂਤਾਂ {
            ਸਵੈਚਾਲਿਤ axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
            ਅੰਕੜੇ =
                std::make_shared<ov::op::v0::Squeeze>(ਅੰਕੜੇ, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
            // data: [batch, rows, ...], indices: [batch, n] - this is a batched gather
            // (batch_dims=1) along the rows axis. The data and indices batch dims are
            // logically equal (both == n_tokens) but reach this node through independent
            // reshapes, so the GPU plugin's gather shape inference cannot prove
            // data.shape[0] == indices.shape[0] and rejects the node. We must tie both
            // batch dims to the SAME value, and crucially that value must stay DYNAMIC.
            ਅਚਲ ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ_ps = ਅੰਕੜੇ.get_partial_shape();
            ਅਚਲ ਸਵੈਚਾਲਿਤ idx_ps = indices.get_partial_shape();
            ਅਚਲ ਤਾਰਕਿਕ ਅੰਕੜੇ_batch_static = ਅੰਕੜੇ_ps.rank().is_static() && ਅੰਕੜੇ_ps[0].is_static();
            ਅਚਲ ਤਾਰਕਿਕ idx_batch_dynamic = idx_ps.rank().is_dynamic() || idx_ps[0].is_dynamic();

            ਜੇ (ਅੰਕੜੇ_batch_static && idx_batch_dynamic) {
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
                ਸਵੈਚਾਲਿਤ zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                ਸਵੈਚਾਲਿਤ one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
                ਸਵੈਚਾਲਿਤ axis0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ_b1 = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(ਅੰਕੜੇ, zero, one, one, axis0);  // [1, rows, ...]

                ਸਵੈਚਾਲਿਤ idx_shape = std::make_shared<ov::op::v3::Shape_ਦਾ>(indices, ov::element::i64);
                ਸਵੈਚਾਲਿਤ idx_batch = get_dimensions(idx_shape, {0});  // [batch] (dynamic)
                ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ_b1_shape = std::make_shared<ov::op::v3::Shape_ਦਾ>(ਅੰਕੜੇ_b1, ov::element::i64);
                ਅਚਲ ਸਵੈਚਾਲਿਤ rank = ਅੰਕੜੇ_ps.rank().get_ਲੰਮਾਈ();
                std::vector<ਪੂਰਨ> rest_axes;
                ਲਈ (ਪੂਰਨ a = 1; a < rank; ++a) {
                    rest_axes.push_back(a);
                }
                ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ_rest = get_dimensions(ਅੰਕੜੇ_b1_shape, rest_axes);  // [rows, ...]
                ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ_ਨਿਸ਼ਾਨਾ = std::make_shared<ov::op::v0::Concat>(ov::Output_ਸਦਿਸ਼{idx_batch, ਅੰਕੜੇ_rest}, 0);
                ਅੰਕੜੇ =
                    std::make_shared<ov::op::v3::ਪ੍ਰਸਾਰਣ>(ਅੰਕੜੇ_b1, ਅੰਕੜੇ_ਨਿਸ਼ਾਨਾ, ov::op::ਪ੍ਰਸਾਰਣ_Type::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(ਅੰਕੜੇ, indices, axis, 1);
            } ਨਹੀਂਤਾਂ {
                // General case: tie the indices batch to the data batch (the data batch is
                // already dynamic, e.g. the routing-weights gather whose data comes from the
                // activations). Broadcast indices to [data_batch, indices_n].
                ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ_shape = std::make_shared<ov::op::v3::Shape_ਦਾ>(ਅੰਕੜੇ, ov::element::i64);
                ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ_batch = get_dimensions(ਅੰਕੜੇ_shape, {0});  // [batch]
                ਸਵੈਚਾਲਿਤ idx_shape = std::make_shared<ov::op::v3::Shape_ਦਾ>(indices, ov::element::i64);
                ਸਵੈਚਾਲਿਤ idx_n = get_dimensions(idx_shape, {1});  // [n]
                ਸਵੈਚਾਲਿਤ idx_ਨਿਸ਼ਾਨਾ = std::make_shared<ov::op::v0::Concat>(ov::Output_ਸਦਿਸ਼{ਅੰਕੜੇ_batch, idx_n}, 0);
                indices = std::make_shared<ov::op::v3::ਪ੍ਰਸਾਰਣ>(indices, idx_ਨਿਸ਼ਾਨਾ,
                                                                  ov::op::ਪ੍ਰਸਾਰਣ_Type::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(ਅੰਕੜੇ, indices, axis, 1);
            }
        }
    } ਨਹੀਂਤਾਂ ਜੇ (context.is_stateful() && ਅੰਕੜੇ.get_partial_shape().rank() == 3) {
        ਸਵੈਚਾਲਿਤ axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
        res = std::make_shared<ov::op::v8::Gather>(ਅੰਕੜੇ, indices, axis, 1);
    } ਨਹੀਂਤਾਂ {
        ਸਵੈਚਾਲਿਤ axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
        res = std::make_shared<ov::op::v8::Gather>(ਅੰਕੜੇ, indices, axis);
    }

    ਜੇ (res.get_element_type() != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
    }
    ਜੇ (!(context.is_stateful())) {
        res = std::make_shared<ov::op::v0::Unsqueeze>(res, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
    }
    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

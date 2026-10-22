#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <climits>
#ಅನ್ತರ್ಭಾವ <openvino/core/node.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/node_output.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/add.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/broadcast.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/concat.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/gather.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/shape_of.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/squeeze.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/unsqueeze.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

Output_ಸದಿಶ translate_get_ಅಡ್ಡಸಾಲುಗಳು(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);

    Output<Node> res;
    Output<Node> ದತ್ತಾಂಶ;
    int64_t ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ = 0;
    ಆದರೆ (context.get_view_ಒಳಹರಿವು_ಗಾತ್ರ(0) > 0 && context.get_ಒಳಹರಿವು(0).get_partial_shape().rank() == 2) {
        // A row range view over a 2D weight Constant folds into the gather indices,
        // which keeps the dequantization subgraph intact for the plugins.
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ view_shape = context.get_view_ಒಳಹರಿವು_ggml_shape(0, 0);
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ view_stride = context.get_view_ಒಳಹರಿವು_stride(0, 0);
        ಸ್ಥಿರಾಂಕ size_t view_ಸ್ಥಾನಾಂತರ = context.get_view_ಒಳಹರಿವು_ಸ್ಥಾನಾಂತರ(0, 0);
        ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = view_stride[2];
        ದತ್ತಾಂಶ = context.get_ಒಳಹರಿವು(0);
        FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು > 0 && view_ಸ್ಥಾನಾಂತರ % ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು == 0 &&
                                          ದತ್ತಾಂಶ.get_partial_shape()[1].compatible(view_shape[3]),
                                      "GET_ROWS: view over a weight must be a row range");
        ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ = static_cast<int64_t>(view_ಸ್ಥಾನಾಂತರ / ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ದತ್ತಾಂಶ = process_view_ಒಳಹರಿವು_new(context, 0);
    }

    ಸ್ವಯಂಚಾಲಿತ op_case = context.get_op_case();
    ಆದರೆ (op_case == 3 || op_case == 4) {
        ಹಿಂತಿರುಗಿ {ದತ್ತಾಂಶ};
    }

    ov::Output<ov::Node> indices;
    ಆದರೆ ((op_case == 1 || op_case == 2) && context.has_ಒಳಹರಿವು("s_copy_active_slot_len")) {
        // Recurrent state reorder (inp->s_copy): slice the active (op_case 1) or extra (op_case 2)
        // segment from the s_copy index list at runtime, instead of baking the static view offset,
        // so the cached IR works for any number of active sequences.
        ಸ್ವಯಂಚಾಲಿತ s_copy = context.get_ಒಳಹರಿವು(1);
        ಸ್ವಯಂಚಾಲಿತ len = context.get_ಒಳಹರಿವು("s_copy_active_slot_len");
        ಸ್ವಯಂಚಾಲಿತ ಹೆಜ್ಜೆ = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        ಸ್ವಯಂಚಾಲಿತ axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        ಆದರೆ (op_case == 1) {
            ಸ್ವಯಂಚಾಲಿತ begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            indices = std::make_shared<ov::op::v8::ತುಂಡು>(s_copy, begin, len, ಹೆಜ್ಜೆ, axis);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ವಯಂಚಾಲಿತ end = ov::op::v0::Constant::create(ov::element::i64, {1}, {ಗರಿಷ್ಠ_ಪೂರ್ಣಾಂಕ});
            indices = std::make_shared<ov::op::v8::ತುಂಡು>(s_copy, len, end, ಹೆಜ್ಜೆ, axis);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        indices = process_view_ಒಳಹರಿವು_new(context, 1);
    }

    // data[1,b,x,y] ind[1,1,b,x'] test-backend-ops case
    // data[x,y] ind[1,1,1,x'] normal case
    // Squeeze the leading dims down to [b,x']. Stateful models drop one rank, so a hardcoded
    // {0,1} would also strip the batch dim whenever b == 1 (every decode step).
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ indices_rank = indices.get_partial_shape().rank();
    FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(indices_rank.is_static(), "Expected static rank for GET_ROWS indices");
    std::vector<int64_t> indices_squeeze_axes;
    ಗಾಗಿ (int64_t i = 0; i + 2 < indices_rank.get_ಉದ್ದಳತೆ(); ++i) {
        indices_squeeze_axes.push_back(i);
    }
    ಆದರೆ (!indices_squeeze_axes.ಖಾಲಿಗಳು()) {
        indices = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {indices_squeeze_axes.size()},
                                                  indices_squeeze_axes));
    }
    ಆದರೆ (ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ != 0) {
        indices = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(
            indices, ov::op::v0::Constant::create(indices.get_element_type(), {}, {ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ}));
    }
    ಆದರೆ (ದತ್ತಾಂಶ.get_partial_shape().rank() == 4) {
        ಆದರೆ (!(ದತ್ತಾಂಶ.get_partial_shape()[1].is_dynamic()) && ದತ್ತಾಂಶ.get_partial_shape()[1].get_ಉದ್ದಳತೆ() == 1) {
            // Work-around for a bug in ov cpu plugin for test-backend-ops
            ದತ್ತಾಂಶ = std::make_shared<ov::op::v0::Squeeze>(ದತ್ತಾಂಶ,
                                                         ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
            ಸ್ವಯಂಚಾಲಿತ axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
            res = std::make_shared<ov::op::v8::Gather>(ದತ್ತಾಂಶ, indices, axis);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ವಯಂಚಾಲಿತ axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
            ದತ್ತಾಂಶ =
                std::make_shared<ov::op::v0::Squeeze>(ದತ್ತಾಂಶ, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
            // data: [batch, rows, ...], indices: [batch, n] - this is a batched gather
            // (batch_dims=1) along the rows axis. The data and indices batch dims are
            // logically equal (both == n_tokens) but reach this node through independent
            // reshapes, so the GPU plugin's gather shape inference cannot prove
            // data.shape[0] == indices.shape[0] and rejects the node. We must tie both
            // batch dims to the SAME value, and crucially that value must stay DYNAMIC.
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ_ps = ದತ್ತಾಂಶ.get_partial_shape();
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ idx_ps = indices.get_partial_shape();
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ದತ್ತಾಂಶ_batch_static = ದತ್ತಾಂಶ_ps.rank().is_static() && ದತ್ತಾಂಶ_ps[0].is_static();
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ idx_batch_dynamic = idx_ps.rank().is_dynamic() || idx_ps[0].is_dynamic();

            ಆದರೆ (ದತ್ತಾಂಶ_batch_static && idx_batch_dynamic) {
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
                ಸ್ವಯಂಚಾಲಿತ zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                ಸ್ವಯಂಚಾಲಿತ one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
                ಸ್ವಯಂಚಾಲಿತ axis0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ_b1 = std::make_shared<ov::op::v8::ತುಂಡು>(ದತ್ತಾಂಶ, zero, one, one, axis0);  // [1, rows, ...]

                ಸ್ವಯಂಚಾಲಿತ idx_shape = std::make_shared<ov::op::v3::Shape_ದ>(indices, ov::element::i64);
                ಸ್ವಯಂಚಾಲಿತ idx_batch = get_dimensions(idx_shape, {0});  // [batch] (dynamic)
                ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ_b1_shape = std::make_shared<ov::op::v3::Shape_ದ>(ದತ್ತಾಂಶ_b1, ov::element::i64);
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ rank = ದತ್ತಾಂಶ_ps.rank().get_ಉದ್ದಳತೆ();
                std::vector<ಪೂರ್ಣಾಂಕ> rest_axes;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ a = 1; a < rank; ++a) {
                    rest_axes.push_back(a);
                }
                ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ_rest = get_dimensions(ದತ್ತಾಂಶ_b1_shape, rest_axes);  // [rows, ...]
                ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ_ಗುರಿ = std::make_shared<ov::op::v0::Concat>(ov::Output_ಸದಿಶ{idx_batch, ದತ್ತಾಂಶ_rest}, 0);
                ದತ್ತಾಂಶ =
                    std::make_shared<ov::op::v3::ಪ್ರಸಾರ>(ದತ್ತಾಂಶ_b1, ದತ್ತಾಂಶ_ಗುರಿ, ov::op::ಪ್ರಸಾರ_Type::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(ದತ್ತಾಂಶ, indices, axis, 1);
            } ಇಲ್ಲದಿದ್ದರೆ {
                // General case: tie the indices batch to the data batch (the data batch is
                // already dynamic, e.g. the routing-weights gather whose data comes from the
                // activations). Broadcast indices to [data_batch, indices_n].
                ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ_shape = std::make_shared<ov::op::v3::Shape_ದ>(ದತ್ತಾಂಶ, ov::element::i64);
                ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ_batch = get_dimensions(ದತ್ತಾಂಶ_shape, {0});  // [batch]
                ಸ್ವಯಂಚಾಲಿತ idx_shape = std::make_shared<ov::op::v3::Shape_ದ>(indices, ov::element::i64);
                ಸ್ವಯಂಚಾಲಿತ idx_n = get_dimensions(idx_shape, {1});  // [n]
                ಸ್ವಯಂಚಾಲಿತ idx_ಗುರಿ = std::make_shared<ov::op::v0::Concat>(ov::Output_ಸದಿಶ{ದತ್ತಾಂಶ_batch, idx_n}, 0);
                indices = std::make_shared<ov::op::v3::ಪ್ರಸಾರ>(indices, idx_ಗುರಿ,
                                                                  ov::op::ಪ್ರಸಾರ_Type::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(ದತ್ತಾಂಶ, indices, axis, 1);
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (context.is_stateful() && ದತ್ತಾಂಶ.get_partial_shape().rank() == 3) {
        ಸ್ವಯಂಚಾಲಿತ axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
        res = std::make_shared<ov::op::v8::Gather>(ದತ್ತಾಂಶ, indices, axis, 1);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ವಯಂಚಾಲಿತ axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
        res = std::make_shared<ov::op::v8::Gather>(ದತ್ತಾಂಶ, indices, axis);
    }

    ಆದರೆ (res.get_element_type() != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
    }
    ಆದರೆ (!(context.is_stateful())) {
        res = std::make_shared<ov::op::v0::Unsqueeze>(res, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
    }
    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

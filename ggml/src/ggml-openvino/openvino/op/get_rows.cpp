#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <climits>
#ضمن <openvino/core/node.hpp>
#ضمن <openvino/core/node_output.hpp>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/broadcast.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/squeeze.hpp>
#ضمن <openvino/op/unsqueeze.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_سمتیہ translate_get_صفاں(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 2, 2);

    Output<Node> res;
    Output<Node> انکڑے;
    int64_t صف_وتھ = 0;
    إذا (context.get_view_داخلہ_ناپ(0) > 0 && context.get_داخلہ(0).get_partial_shape().rank() == 2) {
        // A row range view over a 2D weight Constant folds into the gather indices,
        // which keeps the dequantization subgraph intact for the plugins.
        ثابت خودکار view_shape = context.get_view_داخلہ_ggml_shape(0, 0);
        ثابت خودکار view_stride = context.get_view_داخلہ_stride(0, 0);
        ثابت size_t view_وتھ = context.get_view_داخلہ_وتھ(0, 0);
        ثابت size_t صف_بائٹاں = view_stride[2];
        انکڑے = context.get_داخلہ(0);
        FRONT_سرا_OP_CONVERSION_CHECK(صف_بائٹاں > 0 && view_وتھ % صف_بائٹاں == 0 &&
                                          انکڑے.get_partial_shape()[1].compatible(view_shape[3]),
                                      "GET_ROWS: view over a weight must be a row range");
        صف_وتھ = static_cast<int64_t>(view_وتھ / صف_بائٹاں);
    } وإلا {
        انکڑے = process_view_داخلہ_new(context, 0);
    }

    خودکار op_case = context.get_op_case();
    إذا (op_case == 3 || op_case == 4) {
        أرجع {انکڑے};
    }

    ov::Output<ov::Node> indices;
    إذا ((op_case == 1 || op_case == 2) && context.has_داخلہ("s_copy_active_slot_len")) {
        // Recurrent state reorder (inp->s_copy): slice the active (op_case 1) or extra (op_case 2)
        // segment from the s_copy index list at runtime, instead of baking the static view offset,
        // so the cached IR works for any number of active sequences.
        خودکار s_copy = context.get_داخلہ(1);
        خودکار len = context.get_داخلہ("s_copy_active_slot_len");
        خودکار قدم = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودکار axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        إذا (op_case == 1) {
            خودکار begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            indices = std::make_shared<ov::op::v8::ٹکڑا>(s_copy, begin, len, قدم, axis);
        } وإلا {
            خودکار end = ov::op::v0::Constant::create(ov::element::i64, {1}, {ودھ_توں_ودھ_عدد});
            indices = std::make_shared<ov::op::v8::ٹکڑا>(s_copy, len, end, قدم, axis);
        }
    } وإلا {
        indices = process_view_داخلہ_new(context, 1);
    }

    // data[1,b,x,y] ind[1,1,b,x'] test-backend-ops case
    // data[x,y] ind[1,1,1,x'] normal case
    // Squeeze the leading dims down to [b,x']. Stateful models drop one rank, so a hardcoded
    // {0,1} would also strip the batch dim whenever b == 1 (every decode step).
    ثابت خودکار indices_rank = indices.get_partial_shape().rank();
    FRONT_سرا_OP_CONVERSION_CHECK(indices_rank.is_static(), "Expected static rank for GET_ROWS indices");
    std::vector<int64_t> indices_squeeze_axes;
    لكل (int64_t i = 0; i + 2 < indices_rank.get_لمائی(); ++i) {
        indices_squeeze_axes.push_back(i);
    }
    إذا (!indices_squeeze_axes.خالی_تھاواں()) {
        indices = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {indices_squeeze_axes.size()},
                                                  indices_squeeze_axes));
    }
    إذا (صف_وتھ != 0) {
        indices = std::make_shared<ov::op::v1::جوڑو>(
            indices, ov::op::v0::Constant::create(indices.get_element_type(), {}, {صف_وتھ}));
    }
    إذا (انکڑے.get_partial_shape().rank() == 4) {
        إذا (!(انکڑے.get_partial_shape()[1].is_dynamic()) && انکڑے.get_partial_shape()[1].get_لمائی() == 1) {
            // Work-around for a bug in ov cpu plugin for test-backend-ops
            انکڑے = std::make_shared<ov::op::v0::Squeeze>(انکڑے,
                                                         ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
            خودکار axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
            res = std::make_shared<ov::op::v8::Gather>(انکڑے, indices, axis);
        } وإلا {
            خودکار axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
            انکڑے =
                std::make_shared<ov::op::v0::Squeeze>(انکڑے, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
            // data: [batch, rows, ...], indices: [batch, n] - this is a batched gather
            // (batch_dims=1) along the rows axis. The data and indices batch dims are
            // logically equal (both == n_tokens) but reach this node through independent
            // reshapes, so the GPU plugin's gather shape inference cannot prove
            // data.shape[0] == indices.shape[0] and rejects the node. We must tie both
            // batch dims to the SAME value, and crucially that value must stay DYNAMIC.
            ثابت خودکار انکڑے_ps = انکڑے.get_partial_shape();
            ثابت خودکار idx_ps = indices.get_partial_shape();
            ثابت منطقي انکڑے_batch_static = انکڑے_ps.rank().is_static() && انکڑے_ps[0].is_static();
            ثابت منطقي idx_batch_dynamic = idx_ps.rank().is_dynamic() || idx_ps[0].is_dynamic();

            إذا (انکڑے_batch_static && idx_batch_dynamic) {
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
                خودکار zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                خودکار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
                خودکار axis0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                خودکار انکڑے_b1 = std::make_shared<ov::op::v8::ٹکڑا>(انکڑے, zero, one, one, axis0);  // [1, rows, ...]

                خودکار idx_shape = std::make_shared<ov::op::v3::Shape_دا>(indices, ov::element::i64);
                خودکار idx_batch = get_dimensions(idx_shape, {0});  // [batch] (dynamic)
                خودکار انکڑے_b1_shape = std::make_shared<ov::op::v3::Shape_دا>(انکڑے_b1, ov::element::i64);
                ثابت خودکار rank = انکڑے_ps.rank().get_لمائی();
                std::vector<صحيح> rest_axes;
                لكل (صحيح a = 1; a < rank; ++a) {
                    rest_axes.push_back(a);
                }
                خودکار انکڑے_rest = get_dimensions(انکڑے_b1_shape, rest_axes);  // [rows, ...]
                خودکار انکڑے_نشانہ = std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{idx_batch, انکڑے_rest}, 0);
                انکڑے =
                    std::make_shared<ov::op::v3::نشر>(انکڑے_b1, انکڑے_نشانہ, ov::op::نشر_Type::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(انکڑے, indices, axis, 1);
            } وإلا {
                // General case: tie the indices batch to the data batch (the data batch is
                // already dynamic, e.g. the routing-weights gather whose data comes from the
                // activations). Broadcast indices to [data_batch, indices_n].
                خودکار انکڑے_shape = std::make_shared<ov::op::v3::Shape_دا>(انکڑے, ov::element::i64);
                خودکار انکڑے_batch = get_dimensions(انکڑے_shape, {0});  // [batch]
                خودکار idx_shape = std::make_shared<ov::op::v3::Shape_دا>(indices, ov::element::i64);
                خودکار idx_n = get_dimensions(idx_shape, {1});  // [n]
                خودکار idx_نشانہ = std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{انکڑے_batch, idx_n}, 0);
                indices = std::make_shared<ov::op::v3::نشر>(indices, idx_نشانہ,
                                                                  ov::op::نشر_Type::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(انکڑے, indices, axis, 1);
            }
        }
    } وإلا إذا (context.is_stateful() && انکڑے.get_partial_shape().rank() == 3) {
        خودکار axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
        res = std::make_shared<ov::op::v8::Gather>(انکڑے, indices, axis, 1);
    } وإلا {
        خودکار axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
        res = std::make_shared<ov::op::v8::Gather>(انکڑے, indices, axis);
    }

    إذا (res.get_element_type() != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
    }
    إذا (!(context.is_stateful())) {
        res = std::make_shared<ov::op::v0::Unsqueeze>(res, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
    }
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

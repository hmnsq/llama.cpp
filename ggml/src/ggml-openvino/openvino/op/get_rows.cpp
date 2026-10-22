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

Output_وکتور translate_get_ردیفها(ثابت گره_Context & context) {
    رقم_inputs_check(context, 2, 2);

    Output<Node> res;
    Output<Node> داده;
    int64_t ردیف_جابه_جایی = 0;
    إذا (context.get_view_درون_داد_اندازه(0) > 0 && context.get_درون_داد(0).get_partial_shape().rank() == 2) {
        // A row range view over a 2D weight Constant folds into the gather indices,
        // which keeps the dequantization subgraph intact for the plugins.
        ثابت خودکار view_shape = context.get_view_درون_داد_ggml_shape(0, 0);
        ثابت خودکار view_stride = context.get_view_درون_داد_stride(0, 0);
        ثابت size_t view_جابه_جایی = context.get_view_درون_داد_جابه_جایی(0, 0);
        ثابت size_t ردیف_بایتها = view_stride[2];
        داده = context.get_درون_داد(0);
        FRONT_انتها_OP_CONVERSION_CHECK(ردیف_بایتها > 0 && view_جابه_جایی % ردیف_بایتها == 0 &&
                                          داده.get_partial_shape()[1].compatible(view_shape[3]),
                                      "GET_ROWS: view over a weight must be a row range");
        ردیف_جابه_جایی = static_cast<int64_t>(view_جابه_جایی / ردیف_بایتها);
    } وإلا {
        داده = process_view_درون_داد_new(context, 0);
    }

    خودکار op_case = context.get_op_case();
    إذا (op_case == 3 || op_case == 4) {
        أرجع {داده};
    }

    ov::Output<ov::Node> indices;
    إذا ((op_case == 1 || op_case == 2) && context.has_درون_داد("s_copy_active_slot_len")) {
        // Recurrent state reorder (inp->s_copy): slice the active (op_case 1) or extra (op_case 2)
        // segment from the s_copy index list at runtime, instead of baking the static view offset,
        // so the cached IR works for any number of active sequences.
        خودکار s_copy = context.get_درون_داد(1);
        خودکار len = context.get_درون_داد("s_copy_active_slot_len");
        خودکار گام = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودکار axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        إذا (op_case == 1) {
            خودکار begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            indices = std::make_shared<ov::op::v8::برش>(s_copy, begin, len, گام, axis);
        } وإلا {
            خودکار end = ov::op::v0::Constant::create(ov::element::i64, {1}, {بیشترین_صحیح});
            indices = std::make_shared<ov::op::v8::برش>(s_copy, len, end, گام, axis);
        }
    } وإلا {
        indices = process_view_درون_داد_new(context, 1);
    }

    // data[1,b,x,y] ind[1,1,b,x'] test-backend-ops case
    // data[x,y] ind[1,1,1,x'] normal case
    // Squeeze the leading dims down to [b,x']. Stateful models drop one rank, so a hardcoded
    // {0,1} would also strip the batch dim whenever b == 1 (every decode step).
    ثابت خودکار indices_rank = indices.get_partial_shape().rank();
    FRONT_انتها_OP_CONVERSION_CHECK(indices_rank.is_static(), "Expected static rank for GET_ROWS indices");
    std::vector<int64_t> indices_squeeze_axes;
    لكل (int64_t i = 0; i + 2 < indices_rank.get_درازی(); ++i) {
        indices_squeeze_axes.push_back(i);
    }
    إذا (!indices_squeeze_axes.خالی()) {
        indices = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {indices_squeeze_axes.size()},
                                                  indices_squeeze_axes));
    }
    إذا (ردیف_جابه_جایی != 0) {
        indices = std::make_shared<ov::op::v1::جمع_کن>(
            indices, ov::op::v0::Constant::create(indices.get_element_type(), {}, {ردیف_جابه_جایی}));
    }
    إذا (داده.get_partial_shape().rank() == 4) {
        إذا (!(داده.get_partial_shape()[1].is_dynamic()) && داده.get_partial_shape()[1].get_درازی() == 1) {
            // Work-around for a bug in ov cpu plugin for test-backend-ops
            داده = std::make_shared<ov::op::v0::Squeeze>(داده,
                                                         ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
            خودکار axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
            res = std::make_shared<ov::op::v8::Gather>(داده, indices, axis);
        } وإلا {
            خودکار axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
            داده =
                std::make_shared<ov::op::v0::Squeeze>(داده, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
            // data: [batch, rows, ...], indices: [batch, n] - this is a batched gather
            // (batch_dims=1) along the rows axis. The data and indices batch dims are
            // logically equal (both == n_tokens) but reach this node through independent
            // reshapes, so the GPU plugin's gather shape inference cannot prove
            // data.shape[0] == indices.shape[0] and rejects the node. We must tie both
            // batch dims to the SAME value, and crucially that value must stay DYNAMIC.
            ثابت خودکار داده_ps = داده.get_partial_shape();
            ثابت خودکار idx_ps = indices.get_partial_shape();
            ثابت منطقي داده_batch_static = داده_ps.rank().is_static() && داده_ps[0].is_static();
            ثابت منطقي idx_batch_dynamic = idx_ps.rank().is_dynamic() || idx_ps[0].is_dynamic();

            إذا (داده_batch_static && idx_batch_dynamic) {
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
                خودکار داده_b1 = std::make_shared<ov::op::v8::برش>(داده, zero, one, one, axis0);  // [1, rows, ...]

                خودکار idx_shape = std::make_shared<ov::op::v3::Shape_از>(indices, ov::element::i64);
                خودکار idx_batch = get_dimensions(idx_shape, {0});  // [batch] (dynamic)
                خودکار داده_b1_shape = std::make_shared<ov::op::v3::Shape_از>(داده_b1, ov::element::i64);
                ثابت خودکار rank = داده_ps.rank().get_درازی();
                std::vector<صحيح> rest_axes;
                لكل (صحيح a = 1; a < rank; ++a) {
                    rest_axes.push_back(a);
                }
                خودکار داده_rest = get_dimensions(داده_b1_shape, rest_axes);  // [rows, ...]
                خودکار داده_هدف = std::make_shared<ov::op::v0::Concat>(ov::Output_وکتور{idx_batch, داده_rest}, 0);
                داده =
                    std::make_shared<ov::op::v3::پخش>(داده_b1, داده_هدف, ov::op::پخش_Type::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(داده, indices, axis, 1);
            } وإلا {
                // General case: tie the indices batch to the data batch (the data batch is
                // already dynamic, e.g. the routing-weights gather whose data comes from the
                // activations). Broadcast indices to [data_batch, indices_n].
                خودکار داده_shape = std::make_shared<ov::op::v3::Shape_از>(داده, ov::element::i64);
                خودکار داده_batch = get_dimensions(داده_shape, {0});  // [batch]
                خودکار idx_shape = std::make_shared<ov::op::v3::Shape_از>(indices, ov::element::i64);
                خودکار idx_n = get_dimensions(idx_shape, {1});  // [n]
                خودکار idx_هدف = std::make_shared<ov::op::v0::Concat>(ov::Output_وکتور{داده_batch, idx_n}, 0);
                indices = std::make_shared<ov::op::v3::پخش>(indices, idx_هدف,
                                                                  ov::op::پخش_Type::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(داده, indices, axis, 1);
            }
        }
    } وإلا إذا (context.is_stateful() && داده.get_partial_shape().rank() == 3) {
        خودکار axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
        res = std::make_shared<ov::op::v8::Gather>(داده, indices, axis, 1);
    } وإلا {
        خودکار axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
        res = std::make_shared<ov::op::v8::Gather>(داده, indices, axis);
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

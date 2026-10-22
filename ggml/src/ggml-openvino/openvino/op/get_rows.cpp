#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <climits>
#شامل <openvino/core/node.hpp>
#شامل <openvino/core/node_output.hpp>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/broadcast.hpp>
#شامل <openvino/op/concat.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/gather.hpp>
#شامل <openvino/op/shape_of.hpp>
#شامل <openvino/op/slice.hpp>
#شامل <openvino/op/squeeze.hpp>
#شامل <openvino/op/unsqueeze.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_get_صفیں(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 2, 2);

    Output<Node> res;
    Output<Node> ڈیٹا;
    int64_t صف_انحراف = 0;
    اگر (context.get_view_داخلہ_سائز(0) > 0 && context.get_داخلہ(0).get_partial_shape().rank() == 2) {
        // A row range view over a 2D weight Constant folds into the gather indices,
        // which keeps the dequantization subgraph intact for the plugins.
        ثابت خودکار view_shape = context.get_view_داخلہ_ggml_shape(0, 0);
        ثابت خودکار view_stride = context.get_view_داخلہ_stride(0, 0);
        ثابت size_t view_انحراف = context.get_view_داخلہ_انحراف(0, 0);
        ثابت size_t صف_بائٹیں = view_stride[2];
        ڈیٹا = context.get_داخلہ(0);
        FRONT_انتہا_OP_CONVERSION_CHECK(صف_بائٹیں > 0 && view_انحراف % صف_بائٹیں == 0 &&
                                          ڈیٹا.get_partial_shape()[1].compatible(view_shape[3]),
                                      "GET_ROWS: view over a weight must be a row range");
        صف_انحراف = static_cast<int64_t>(view_انحراف / صف_بائٹیں);
    } ورنہ {
        ڈیٹا = process_view_داخلہ_new(context, 0);
    }

    خودکار op_case = context.get_op_case();
    اگر (op_case == 3 || op_case == 4) {
        واپس {ڈیٹا};
    }

    ov::Output<ov::Node> indices;
    اگر ((op_case == 1 || op_case == 2) && context.has_داخلہ("s_copy_active_slot_len")) {
        // Recurrent state reorder (inp->s_copy): slice the active (op_case 1) or extra (op_case 2)
        // segment from the s_copy index list at runtime, instead of baking the static view offset,
        // so the cached IR works for any number of active sequences.
        خودکار s_copy = context.get_داخلہ(1);
        خودکار len = context.get_داخلہ("s_copy_active_slot_len");
        خودکار قدم = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودکار axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        اگر (op_case == 1) {
            خودکار begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            indices = std::make_shared<ov::op::v8::قاش>(s_copy, begin, len, قدم, axis);
        } ورنہ {
            خودکار end = ov::op::v0::Constant::create(ov::element::i64, {1}, {زیادہ_سے_زیادہ_عدد});
            indices = std::make_shared<ov::op::v8::قاش>(s_copy, len, end, قدم, axis);
        }
    } ورنہ {
        indices = process_view_داخلہ_new(context, 1);
    }

    // data[1,b,x,y] ind[1,1,b,x'] test-backend-ops case
    // data[x,y] ind[1,1,1,x'] normal case
    // Squeeze the leading dims down to [b,x']. Stateful models drop one rank, so a hardcoded
    // {0,1} would also strip the batch dim whenever b == 1 (every decode step).
    ثابت خودکار indices_rank = indices.get_partial_shape().rank();
    FRONT_انتہا_OP_CONVERSION_CHECK(indices_rank.is_static(), "Expected static rank for GET_ROWS indices");
    std::vector<int64_t> indices_squeeze_axes;
    برائے (int64_t i = 0; i + 2 < indices_rank.get_لمبائی(); ++i) {
        indices_squeeze_axes.push_back(i);
    }
    اگر (!indices_squeeze_axes.خالی_خانے()) {
        indices = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {indices_squeeze_axes.size()},
                                                  indices_squeeze_axes));
    }
    اگر (صف_انحراف != 0) {
        indices = std::make_shared<ov::op::v1::جوڑو>(
            indices, ov::op::v0::Constant::create(indices.get_element_type(), {}, {صف_انحراف}));
    }
    اگر (ڈیٹا.get_partial_shape().rank() == 4) {
        اگر (!(ڈیٹا.get_partial_shape()[1].is_dynamic()) && ڈیٹا.get_partial_shape()[1].get_لمبائی() == 1) {
            // Work-around for a bug in ov cpu plugin for test-backend-ops
            ڈیٹا = std::make_shared<ov::op::v0::Squeeze>(ڈیٹا,
                                                         ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
            خودکار axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
            res = std::make_shared<ov::op::v8::Gather>(ڈیٹا, indices, axis);
        } ورنہ {
            خودکار axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
            ڈیٹا =
                std::make_shared<ov::op::v0::Squeeze>(ڈیٹا, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
            // data: [batch, rows, ...], indices: [batch, n] - this is a batched gather
            // (batch_dims=1) along the rows axis. The data and indices batch dims are
            // logically equal (both == n_tokens) but reach this node through independent
            // reshapes, so the GPU plugin's gather shape inference cannot prove
            // data.shape[0] == indices.shape[0] and rejects the node. We must tie both
            // batch dims to the SAME value, and crucially that value must stay DYNAMIC.
            ثابت خودکار ڈیٹا_ps = ڈیٹا.get_partial_shape();
            ثابت خودکار idx_ps = indices.get_partial_shape();
            ثابت منطقی ڈیٹا_batch_static = ڈیٹا_ps.rank().is_static() && ڈیٹا_ps[0].is_static();
            ثابت منطقی idx_batch_dynamic = idx_ps.rank().is_dynamic() || idx_ps[0].is_dynamic();

            اگر (ڈیٹا_batch_static && idx_batch_dynamic) {
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
                خودکار ڈیٹا_b1 = std::make_shared<ov::op::v8::قاش>(ڈیٹا, zero, one, one, axis0);  // [1, rows, ...]

                خودکار idx_shape = std::make_shared<ov::op::v3::Shape_کا>(indices, ov::element::i64);
                خودکار idx_batch = get_dimensions(idx_shape, {0});  // [batch] (dynamic)
                خودکار ڈیٹا_b1_shape = std::make_shared<ov::op::v3::Shape_کا>(ڈیٹا_b1, ov::element::i64);
                ثابت خودکار rank = ڈیٹا_ps.rank().get_لمبائی();
                std::vector<صحیح> rest_axes;
                برائے (صحیح a = 1; a < rank; ++a) {
                    rest_axes.push_back(a);
                }
                خودکار ڈیٹا_rest = get_dimensions(ڈیٹا_b1_shape, rest_axes);  // [rows, ...]
                خودکار ڈیٹا_ہدف = std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{idx_batch, ڈیٹا_rest}, 0);
                ڈیٹا =
                    std::make_shared<ov::op::v3::نشر>(ڈیٹا_b1, ڈیٹا_ہدف, ov::op::نشر_Type::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(ڈیٹا, indices, axis, 1);
            } ورنہ {
                // General case: tie the indices batch to the data batch (the data batch is
                // already dynamic, e.g. the routing-weights gather whose data comes from the
                // activations). Broadcast indices to [data_batch, indices_n].
                خودکار ڈیٹا_shape = std::make_shared<ov::op::v3::Shape_کا>(ڈیٹا, ov::element::i64);
                خودکار ڈیٹا_batch = get_dimensions(ڈیٹا_shape, {0});  // [batch]
                خودکار idx_shape = std::make_shared<ov::op::v3::Shape_کا>(indices, ov::element::i64);
                خودکار idx_n = get_dimensions(idx_shape, {1});  // [n]
                خودکار idx_ہدف = std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{ڈیٹا_batch, idx_n}, 0);
                indices = std::make_shared<ov::op::v3::نشر>(indices, idx_ہدف,
                                                                  ov::op::نشر_Type::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(ڈیٹا, indices, axis, 1);
            }
        }
    } ورنہ اگر (context.is_stateful() && ڈیٹا.get_partial_shape().rank() == 3) {
        خودکار axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
        res = std::make_shared<ov::op::v8::Gather>(ڈیٹا, indices, axis, 1);
    } ورنہ {
        خودکار axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
        res = std::make_shared<ov::op::v8::Gather>(ڈیٹا, indices, axis);
    }

    اگر (res.get_element_type() != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
    }
    اگر (!(context.is_stateful())) {
        res = std::make_shared<ov::op::v0::Unsqueeze>(res, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
    }
    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

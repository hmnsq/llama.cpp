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

Output_ویکټور translate_get_لیکې(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 2, 2);

    Output<Node> res;
    Output<Node> معلومات;
    int64_t لیکه_دننه_واټن = 0;
    إذا (context.get_view_ننوت_اندازه(0) > 0 && context.get_ننوت(0).get_partial_shape().rank() == 2) {
        // A row range view over a 2D weight Constant folds into the gather indices,
        // which keeps the dequantization subgraph intact for the plugins.
        ثابت خپلکاری view_shape = context.get_view_ننوت_ggml_shape(0, 0);
        ثابت خپلکاری view_stride = context.get_view_ننوت_stride(0, 0);
        ثابت size_t view_دننه_واټن = context.get_view_ننوت_دننه_واټن(0, 0);
        ثابت size_t لیکه_بایټونه = view_stride[2];
        معلومات = context.get_ننوت(0);
        FRONT_پای_ټکی_OP_CONVERSION_CHECK(لیکه_بایټونه > 0 && view_دننه_واټن % لیکه_بایټونه == 0 &&
                                          معلومات.get_partial_shape()[1].compatible(view_shape[3]),
                                      "GET_ROWS: view over a weight must be a row range");
        لیکه_دننه_واټن = static_cast<int64_t>(view_دننه_واټن / لیکه_بایټونه);
    } وإلا {
        معلومات = process_view_ننوت_new(context, 0);
    }

    خپلکاری op_case = context.get_op_case();
    إذا (op_case == 3 || op_case == 4) {
        أرجع {معلومات};
    }

    ov::Output<ov::Node> indices;
    إذا ((op_case == 1 || op_case == 2) && context.has_ننوت("s_copy_active_slot_len")) {
        // Recurrent state reorder (inp->s_copy): slice the active (op_case 1) or extra (op_case 2)
        // segment from the s_copy index list at runtime, instead of baking the static view offset,
        // so the cached IR works for any number of active sequences.
        خپلکاری s_copy = context.get_ننوت(1);
        خپلکاری len = context.get_ننوت("s_copy_active_slot_len");
        خپلکاری ګام = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خپلکاری axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        إذا (op_case == 1) {
            خپلکاری begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            indices = std::make_shared<ov::op::v8::ټوټه>(s_copy, begin, len, ګام, axis);
        } وإلا {
            خپلکاری end = ov::op::v0::Constant::create(ov::element::i64, {1}, {لوی_عدد});
            indices = std::make_shared<ov::op::v8::ټوټه>(s_copy, len, end, ګام, axis);
        }
    } وإلا {
        indices = process_view_ننوت_new(context, 1);
    }

    // data[1,b,x,y] ind[1,1,b,x'] test-backend-ops case
    // data[x,y] ind[1,1,1,x'] normal case
    // Squeeze the leading dims down to [b,x']. Stateful models drop one rank, so a hardcoded
    // {0,1} would also strip the batch dim whenever b == 1 (every decode step).
    ثابت خپلکاری indices_rank = indices.get_partial_shape().rank();
    FRONT_پای_ټکی_OP_CONVERSION_CHECK(indices_rank.is_static(), "Expected static rank for GET_ROWS indices");
    std::vector<int64_t> indices_squeeze_axes;
    لكل (int64_t i = 0; i + 2 < indices_rank.get_اوږدتیا(); ++i) {
        indices_squeeze_axes.push_back(i);
    }
    إذا (!indices_squeeze_axes.تشه()) {
        indices = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {indices_squeeze_axes.size()},
                                                  indices_squeeze_axes));
    }
    إذا (لیکه_دننه_واټن != 0) {
        indices = std::make_shared<ov::op::v1::جمع_کړه>(
            indices, ov::op::v0::Constant::create(indices.get_element_type(), {}, {لیکه_دننه_واټن}));
    }
    إذا (معلومات.get_partial_shape().rank() == 4) {
        إذا (!(معلومات.get_partial_shape()[1].is_dynamic()) && معلومات.get_partial_shape()[1].get_اوږدتیا() == 1) {
            // Work-around for a bug in ov cpu plugin for test-backend-ops
            معلومات = std::make_shared<ov::op::v0::Squeeze>(معلومات,
                                                         ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
            خپلکاری axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
            res = std::make_shared<ov::op::v8::Gather>(معلومات, indices, axis);
        } وإلا {
            خپلکاری axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
            معلومات =
                std::make_shared<ov::op::v0::Squeeze>(معلومات, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
            // data: [batch, rows, ...], indices: [batch, n] - this is a batched gather
            // (batch_dims=1) along the rows axis. The data and indices batch dims are
            // logically equal (both == n_tokens) but reach this node through independent
            // reshapes, so the GPU plugin's gather shape inference cannot prove
            // data.shape[0] == indices.shape[0] and rejects the node. We must tie both
            // batch dims to the SAME value, and crucially that value must stay DYNAMIC.
            ثابت خپلکاری معلومات_ps = معلومات.get_partial_shape();
            ثابت خپلکاری idx_ps = indices.get_partial_shape();
            ثابت منطقي معلومات_batch_static = معلومات_ps.rank().is_static() && معلومات_ps[0].is_static();
            ثابت منطقي idx_batch_dynamic = idx_ps.rank().is_dynamic() || idx_ps[0].is_dynamic();

            إذا (معلومات_batch_static && idx_batch_dynamic) {
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
                خپلکاری zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                خپلکاری one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
                خپلکاری axis0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                خپلکاری معلومات_b1 = std::make_shared<ov::op::v8::ټوټه>(معلومات, zero, one, one, axis0);  // [1, rows, ...]

                خپلکاری idx_shape = std::make_shared<ov::op::v3::Shape_د>(indices, ov::element::i64);
                خپلکاری idx_batch = get_dimensions(idx_shape, {0});  // [batch] (dynamic)
                خپلکاری معلومات_b1_shape = std::make_shared<ov::op::v3::Shape_د>(معلومات_b1, ov::element::i64);
                ثابت خپلکاری rank = معلومات_ps.rank().get_اوږدتیا();
                std::vector<صحيح> rest_axes;
                لكل (صحيح a = 1; a < rank; ++a) {
                    rest_axes.push_back(a);
                }
                خپلکاری معلومات_rest = get_dimensions(معلومات_b1_shape, rest_axes);  // [rows, ...]
                خپلکاری معلومات_موخه = std::make_shared<ov::op::v0::Concat>(ov::Output_ویکټور{idx_batch, معلومات_rest}, 0);
                معلومات =
                    std::make_shared<ov::op::v3::خپرونه>(معلومات_b1, معلومات_موخه, ov::op::خپرونه_Type::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(معلومات, indices, axis, 1);
            } وإلا {
                // General case: tie the indices batch to the data batch (the data batch is
                // already dynamic, e.g. the routing-weights gather whose data comes from the
                // activations). Broadcast indices to [data_batch, indices_n].
                خپلکاری معلومات_shape = std::make_shared<ov::op::v3::Shape_د>(معلومات, ov::element::i64);
                خپلکاری معلومات_batch = get_dimensions(معلومات_shape, {0});  // [batch]
                خپلکاری idx_shape = std::make_shared<ov::op::v3::Shape_د>(indices, ov::element::i64);
                خپلکاری idx_n = get_dimensions(idx_shape, {1});  // [n]
                خپلکاری idx_موخه = std::make_shared<ov::op::v0::Concat>(ov::Output_ویکټور{معلومات_batch, idx_n}, 0);
                indices = std::make_shared<ov::op::v3::خپرونه>(indices, idx_موخه,
                                                                  ov::op::خپرونه_Type::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(معلومات, indices, axis, 1);
            }
        }
    } وإلا إذا (context.is_stateful() && معلومات.get_partial_shape().rank() == 3) {
        خپلکاری axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
        res = std::make_shared<ov::op::v8::Gather>(معلومات, indices, axis, 1);
    } وإلا {
        خپلکاری axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
        res = std::make_shared<ov::op::v8::Gather>(معلومات, indices, axis);
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

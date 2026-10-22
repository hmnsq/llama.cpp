#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"
#ضمن "ggml.h"

#ضمن <openvino/frontend/exception.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/squeeze.hpp>
#ضمن <openvino/op/topk.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

OutputVector translate_argsort(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    auto داخلہ = process_view_داخلہ_new(context, 0);

    ثابت int32_t order = context.get_output_op_params()[0];

    ov::op::v11::چوٹی_K::Mode mode;
    بدل (order) {
    حالة GGML_SORT_ORDER_ASC:
        mode = ov::op::v11::چوٹی_K::Mode::MIN;
        اكسر;
    حالة GGML_SORT_ORDER_DESC:
        mode = ov::op::v11::چوٹی_K::Mode::MAX;
        اكسر;
    افتراضي:
        FRONT_END_OP_CONVERSION_CHECK(خطأ, "Unsupported GGML_OP_ARGSORT order: ", order);
    }

    // Stateful models drop the leading size-1 batch dim, so the expert axis is 2 there
    // instead of 3 (same rank-3-vs-rank-4 split as get_rows.cpp / process_view_input).
    ثابت صحيح axis = (context.is_stateful() && داخلہ.get_partial_shape().rank() == 3) ? 2 : 3;

    auto k = std::make_shared<ov::op::v0::Squeeze>(get_dimensions(داخلہ.get_گرہ_shared_ptr(), {axis}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    auto topk = std::make_shared<ov::op::v11::چوٹی_K>(داخلہ, k, axis, mode, ov::op::v11::چوٹی_K::SortType::SORT_اقدار,
                                                    context.get_output_type(), خطأ);

    أرجع rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"
#شامل "ggml.h"

#شامل <openvino/frontend/exception.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/squeeze.hpp>
#شامل <openvino/op/topk.hpp>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

OutputVector translate_argsort(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    auto درون_داد = process_view_درون_داد_new(context, 0);

    ثابت int32_t order = context.get_output_op_params()[0];

    ov::op::v11::بالاترین_K::Mode mode;
    انتخاب (order) {
    حالت GGML_SORT_ORDER_ASC:
        mode = ov::op::v11::بالاترین_K::Mode::MIN;
        بشکن;
    حالت GGML_SORT_ORDER_DESC:
        mode = ov::op::v11::بالاترین_K::Mode::MAX;
        بشکن;
    پیشفرض:
        FRONT_END_OP_CONVERSION_CHECK(نادرست, "Unsupported GGML_OP_ARGSORT order: ", order);
    }

    // Stateful models drop the leading size-1 batch dim, so the expert axis is 2 there
    // instead of 3 (same rank-3-vs-rank-4 split as get_rows.cpp / process_view_input).
    ثابت صحیح axis = (context.is_stateful() && درون_داد.get_partial_shape().rank() == 3) ? 2 : 3;

    auto k = std::make_shared<ov::op::v0::Squeeze>(get_dimensions(درون_داد.get_گره_shared_ptr(), {axis}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    auto topk = std::make_shared<ov::op::v11::بالاترین_K>(درون_داد, k, axis, mode, ov::op::v11::بالاترین_K::SortType::SORT_مقادیر,
                                                    context.get_output_type(), نادرست);

    بازگشت rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

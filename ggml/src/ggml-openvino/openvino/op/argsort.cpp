#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"
#அந்தர்பாவ "ggml.h"

#அந்தர்பாவ <openvino/frontend/exception.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/squeeze.hpp>
#அந்தர்பாவ <openvino/op/topk.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

OutputVector translate_argsort(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);

    auto உள்ளீடு = process_view_உள்ளீடு_new(context, 0);

    மாறிலி int32_t order = context.get_output_op_params()[0];

    ov::op::v11::உச்சி_K::Mode mode;
    தேர்வு (order) {
    நிலை GGML_SORT_ORDER_ASC:
        mode = ov::op::v11::உச்சி_K::Mode::MIN;
        நிறுத்து;
    நிலை GGML_SORT_ORDER_DESC:
        mode = ov::op::v11::உச்சி_K::Mode::MAX;
        நிறுத்து;
    இயல்பு:
        FRONT_END_OP_CONVERSION_CHECK(பொய், "Unsupported GGML_OP_ARGSORT order: ", order);
    }

    // Stateful models drop the leading size-1 batch dim, so the expert axis is 2 there
    // instead of 3 (same rank-3-vs-rank-4 split as get_rows.cpp / process_view_input).
    மாறிலி முழு axis = (context.is_stateful() && உள்ளீடு.get_partial_shape().rank() == 3) ? 2 : 3;

    auto k = std::make_shared<ov::op::v0::Squeeze>(get_dimensions(உள்ளீடு.get_முனை_shared_ptr(), {axis}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    auto topk = std::make_shared<ov::op::v11::உச்சி_K>(உள்ளீடு, k, axis, mode, ov::op::v11::உச்சி_K::SortType::SORT_மதிப்புகள்,
                                                    context.get_output_type(), பொய்);

    திரும்பு rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

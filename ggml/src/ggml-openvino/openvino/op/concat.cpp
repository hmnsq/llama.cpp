#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/frontend/exception.hpp>
#அந்தர்பாவ <openvino/op/concat.hpp>
#அந்தர்பாவ <openvino/op/convert.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

OutputVector translate_concat(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 2, 2);

    மாறிலி int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CONCAT requires output op params");

    மாறிலி auto output_shape = context.get_output_shape();
    FRONT_END_CHECK_IMPLEMENTED(output_shape.rank().is_static(), "CONCAT requires static output rank");

    மாறிலி auto rank = output_shape.rank().get_length();
    மாறிலி int32_t ggml_dim = op_params[0];
    FRONT_END_CHECK_IMPLEMENTED(ggml_dim >= 0 && ggml_dim < rank, "CONCAT axis is out of range");

    auto உள்ளீடு_0 = process_view_உள்ளீடு_new(context, 0);
    auto உள்ளீடு_1 = process_view_உள்ளீடு_new(context, 1);
    மாறிலி auto output_type = context.get_output_type();

    எனில் (உள்ளீடு_0.get_element_type() != output_type) {
        உள்ளீடு_0 = std::make_shared<ov::op::v0::Convert>(உள்ளீடு_0, output_type);
    }
    எனில் (உள்ளீடு_1.get_element_type() != output_type) {
        உள்ளீடு_1 = std::make_shared<ov::op::v0::Convert>(உள்ளீடு_1, output_type);
    }

    மாறிலி auto axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    auto res = std::make_shared<ov::op::v0::Concat>(OutputVector{உள்ளீடு_0, உள்ளீடு_1}, axis);

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/op/multiply.hpp>
#அந்தர்பாவ <openvino/op/sqrt.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

OutputVector translate_sqr(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);

    auto உள்ளீடு = process_view_உள்ளீடு_new(context, 0);
    auto res = std::make_shared<ov::op::v1::Multiply>(உள்ளீடு, உள்ளீடு);

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_sqrt(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);

    auto உள்ளீடு = process_view_உள்ளீடு_new(context, 0);
    auto res = std::make_shared<ov::op::v0::Sqrt>(உள்ளீடு);

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

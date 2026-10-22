#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <cstring>
#அந்தர்பாவ <openvino/op/clamp.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

OutputVector translate_clamp(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);

    auto உள்ளீடு = process_view_உள்ளீடு_new(context, 0);

    மாறிலி int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CLAMP requires output op params");

    தசம min;
    தசம max;
    std::memcpy(&min, reinterpret_cast<மாறிலி தசம *>(op_params) + 0, அளவு(தசம));
    std::memcpy(&max, reinterpret_cast<மாறிலி தசம *>(op_params) + 1, அளவு(தசம));

    auto res = std::make_shared<ov::op::v0::Clamp>(உள்ளீடு, min, max);
    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

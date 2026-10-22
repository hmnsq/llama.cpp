#অন্তর্ভাৱ "../node_context.h"
#অন্তর্ভাৱ "../op_table.h"
#অন্তর্ভাৱ "../utils.h"

#অন্তর্ভাৱ <cstring>
#অন্তর্ভাৱ <openvino/op/clamp.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_clamp(ধ্রুৱ গ্ৰন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    auto প্ৰবেশ = process_view_প্ৰবেশ_new(context, 0);

    ধ্রুৱ int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CLAMP requires output op params");

    ভিন্ন min;
    ভিন্ন max;
    std::memcpy(&min, reinterpret_cast<ধ্রুৱ ভিন্ন *>(op_params) + 0, পরিমাণ(ভিন্ন));
    std::memcpy(&max, reinterpret_cast<ধ্রুৱ ভিন্ন *>(op_params) + 1, পরিমাণ(ভিন্ন));

    auto res = std::make_shared<ov::op::v0::Clamp>(প্ৰবেশ, min, max);
    ঘূৰাই rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

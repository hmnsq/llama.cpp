#অন্তর্ভাব "../node_context.h"
#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"

#অন্তর্ভাব <cstring>
#অন্তর্ভাব <openvino/op/clamp.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_clamp(ধ্রুবক গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    auto প্রবেশ = process_view_প্রবেশ_new(context, 0);

    ধ্রুবক int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CLAMP requires output op params");

    দশমিক min;
    দশমিক max;
    std::memcpy(&min, reinterpret_cast<ধ্রুবক দশমিক *>(op_params) + 0, আকার(দশমিক));
    std::memcpy(&max, reinterpret_cast<ধ্রুবক দশমিক *>(op_params) + 1, আকার(দশমিক));

    auto res = std::make_shared<ov::op::v0::Clamp>(প্রবেশ, min, max);
    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#ଅନ୍ତର୍ଭାଵ "../node_context.h"
#ଅନ୍ତର୍ଭାଵ "../op_table.h"
#ଅନ୍ତର୍ଭାଵ "../utils.h"

#ଅନ୍ତର୍ଭାଵ <cstring>
#ଅନ୍ତର୍ଭାଵ <openvino/op/clamp.hpp>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ op {

OutputVector translate_clamp(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);

    auto ନିବେଶ = process_view_ନିବେଶ_new(context, 0);

    ଅଚଳ int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CLAMP requires output op params");

    ଦଶମିକ min;
    ଦଶମିକ max;
    std::memcpy(&min, reinterpret_cast<ଅଚଳ ଦଶମିକ *>(op_params) + 0, ଆକାର(ଦଶମିକ));
    std::memcpy(&max, reinterpret_cast<ଅଚଳ ଦଶମିକ *>(op_params) + 1, ଆକାର(ଦଶମିକ));

    auto res = std::make_shared<ov::op::v0::Clamp>(ନିବେଶ, min, max);
    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

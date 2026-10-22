#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <openvino/op/clamp.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

OutputVector translate_clamp(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    auto ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);

    ಧ್ರುವ int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CLAMP requires output op params");

    ಭಿನ್ನ min;
    ಭಿನ್ನ max;
    std::memcpy(&min, reinterpret_cast<ಧ್ರುವ ಭಿನ್ನ *>(op_params) + 0, ಪರಿಮಾಣ(ಭಿನ್ನ));
    std::memcpy(&max, reinterpret_cast<ಧ್ರುವ ಭಿನ್ನ *>(op_params) + 1, ಪರಿಮಾಣ(ಭಿನ್ನ));

    auto res = std::make_shared<ov::op::v0::Clamp>(ಒಳಹರಿವು, min, max);
    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <openvino/op/roll.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

Output_ಸದಿಶ translate_roll(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);
    ಸ್ಥಿರಾಂಕ int32_t * params = context.get_output_op_params();

    int64_t s0 = params[0];
    int64_t s1 = params[1];
    int64_t s2 = params[2];
    int64_t s3 = params[3];

    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು = context.get_ಒಳಹರಿವು(0);

    ಸ್ವಯಂಚಾಲಿತ ಸರಿಸು = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4}, std::vector<int64_t>{s3, s2, s1, s0});
    ಸ್ವಯಂಚಾಲಿತ axes  = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 1, 2, 3});

    ಸ್ವಯಂಚಾಲಿತ roll = std::make_shared<ov::op::v7::Roll>(ಒಳಹರಿವು, ಸರಿಸು, axes);
    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({roll}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

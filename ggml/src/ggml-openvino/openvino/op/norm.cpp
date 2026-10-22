#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/mvn.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

Output_ಸದಿಶ translate_norm(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_ಗಂಟು = process_view_ಒಳಹರಿವು_new(context, 0);
    ದಶಮಾಂಶ eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, context.get_output_op_params(), ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ವಯಂಚಾಲಿತ axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v6::MVN>(ಒಳಹರಿವು_ಗಂಟು, axes, ನಿಜ, eps, ov::op::MVNEps_ಬಹುಲಕ::INSIDE_SQRT);

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

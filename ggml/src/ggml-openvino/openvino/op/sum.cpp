#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/range.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reduce_sum.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/shape_of.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

OutputVector translate_sum(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    auto ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 2, 3});
    auto res = std::make_shared<ov::op::v1::ReduceSum>(ಒಳಹರಿವು, axes, ಸತ್ಯ);

    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

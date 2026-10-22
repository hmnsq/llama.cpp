#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <openvino/op/roll.hpp>
#समावेश <openvino/op/constant.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_roll(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    स्थिर int32_t * params = context.get_output_op_params();

    int64_t s0 = params[0];
    int64_t s1 = params[1];
    int64_t s2 = params[2];
    int64_t s3 = params[3];

    स्वतः निवेश = context.get_निवेश(0);

    स्वतः shift = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4}, std::vector<int64_t>{s3, s2, s1, s0});
    स्वतः axes  = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 1, 2, 3});

    स्वतः roll = std::make_shared<ov::op::v7::Roll>(निवेश, shift, axes);
    वापस rename_outputs_with_suffix({roll}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

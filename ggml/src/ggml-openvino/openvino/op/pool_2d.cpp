#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <openvino/op/avg_pool.hpp>
#समावेश <openvino/op/max_pool.hpp>
#समावेश <openvino/op/convert.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_pool_2d(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    स्थिर int32_t * params = context.get_output_op_params();

    स्थिर पूर्णांक k0 = params[1];
    स्थिर पूर्णांक k1 = params[2];
    स्थिर पूर्णांक s0 = params[3];
    स्थिर पूर्णांक s1 = params[4];
    स्थिर पूर्णांक p0 = params[5];
    स्थिर पूर्णांक p1 = params[6];

    स्थिर पूर्णांक op_case = context.get_op_case();
    ov::Output<Node> निवेश = context.get_निवेश(0);
    ov::Strides strides{static_cast<माप_प्रकार>(s1), static_cast<माप_प्रकार>(s0)};
    ov::Shape pads_begin{static_cast<माप_प्रकार>(p1), static_cast<माप_प्रकार>(p0)};
    ov::Shape pads_end{static_cast<माप_प्रकार>(p1), static_cast<माप_प्रकार>(p0)};
    ov::Shape kernel{static_cast<माप_प्रकार>(k1), static_cast<माप_प्रकार>(k0)};
    ov::Output<Node> res;

    चयन (op_case) {
    हाल 1:  // GGML_OP_POOL_MAX
    {
        res = std::make_shared<ov::op::v1::MaxPool>(निवेश, strides, pads_begin, pads_end, kernel);
        अवरोध;
    }
    हाल 2:  // GGML_OP_POOL_AVG
    {
        res = std::make_shared<ov::op::v1::AvgPool>(निवेश, strides, pads_begin, pads_end, kernel, false);
        अवरोध;
    }
    शेष:
        अवरोध;
    }
    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

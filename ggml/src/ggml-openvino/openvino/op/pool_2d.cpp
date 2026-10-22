#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <openvino/op/avg_pool.hpp>
#अन्तर्भाव <openvino/op/max_pool.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_pool_2d(स्थिरांक गाठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    स्थिरांक int32_t * params = context.get_output_op_params();

    स्थिरांक पूर्णांक k0 = params[1];
    स्थिरांक पूर्णांक k1 = params[2];
    स्थिरांक पूर्णांक s0 = params[3];
    स्थिरांक पूर्णांक s1 = params[4];
    स्थिरांक पूर्णांक p0 = params[5];
    स्थिरांक पूर्णांक p1 = params[6];

    स्थिरांक पूर्णांक op_case = context.get_op_case();
    ov::Output<Node> आदान = context.get_आदान(0);
    ov::Strides strides{static_cast<माप_प्रकार>(s1), static_cast<माप_प्रकार>(s0)};
    ov::Shape pads_begin{static_cast<माप_प्रकार>(p1), static_cast<माप_प्रकार>(p0)};
    ov::Shape pads_end{static_cast<माप_प्रकार>(p1), static_cast<माप_प्रकार>(p0)};
    ov::Shape kernel{static_cast<माप_प्रकार>(k1), static_cast<माप_प्रकार>(k0)};
    ov::Output<Node> res;

    निवड (op_case) {
    स्थिती 1:  // GGML_OP_POOL_MAX
    {
        res = std::make_shared<ov::op::v1::MaxPool>(आदान, strides, pads_begin, pads_end, kernel);
        थांबा;
    }
    स्थिती 2:  // GGML_OP_POOL_AVG
    {
        res = std::make_shared<ov::op::v1::AvgPool>(आदान, strides, pads_begin, pads_end, kernel, असत्य);
        थांबा;
    }
    पूर्वनिर्धारित:
        थांबा;
    }
    परत rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

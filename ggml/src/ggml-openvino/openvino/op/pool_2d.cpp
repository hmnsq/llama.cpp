#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/avg_pool.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/max_pool.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_pool_2d(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);
    ਅਚਲ int32_t * params = context.get_output_op_params();

    ਅਚਲ ਪੂਰਨ k0 = params[1];
    ਅਚਲ ਪੂਰਨ k1 = params[2];
    ਅਚਲ ਪੂਰਨ s0 = params[3];
    ਅਚਲ ਪੂਰਨ s1 = params[4];
    ਅਚਲ ਪੂਰਨ p0 = params[5];
    ਅਚਲ ਪੂਰਨ p1 = params[6];

    ਅਚਲ ਪੂਰਨ op_case = context.get_op_case();
    ov::Output<Node> ਨਿਵੇਸ਼ = context.get_ਨਿਵੇਸ਼(0);
    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::Shape pads_begin{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape pads_ਸਿਰਾ{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape kernel{static_cast<size_t>(k1), static_cast<size_t>(k0)};
    ov::Output<Node> res;

    ਚੋਣ (op_case) {
    ਹਾਲਤ 1:  // GGML_OP_POOL_MAX
    {
        res = std::make_shared<ov::op::v1::MaxPool>(ਨਿਵੇਸ਼, strides, pads_begin, pads_ਸਿਰਾ, kernel);
        ਰੋਕੋ;
    }
    ਹਾਲਤ 2:  // GGML_OP_POOL_AVG
    {
        res = std::make_shared<ov::op::v1::AvgPool>(ਨਿਵੇਸ਼, strides, pads_begin, pads_ਸਿਰਾ, kernel, ਝੂਠ);
        ਰੋਕੋ;
    }
    ਮੂਲ:
        ਰੋਕੋ;
    }
    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <openvino/op/avg_pool.hpp>
#அந்தர்பாவ <openvino/op/max_pool.hpp>
#அந்தர்பாவ <openvino/op/convert.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

OutputVector translate_pool_2d(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);
    மாறிலி int32_t * params = context.get_output_op_params();

    மாறிலி முழு k0 = params[1];
    மாறிலி முழு k1 = params[2];
    மாறிலி முழு s0 = params[3];
    மாறிலி முழு s1 = params[4];
    மாறிலி முழு p0 = params[5];
    மாறிலி முழு p1 = params[6];

    மாறிலி முழு op_case = context.get_op_case();
    ov::Output<Node> உள்ளீடு = context.get_உள்ளீடு(0);
    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::Shape pads_begin{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape pads_end{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape kernel{static_cast<size_t>(k1), static_cast<size_t>(k0)};
    ov::Output<Node> res;

    தேர்வு (op_case) {
    நிலை 1:  // GGML_OP_POOL_MAX
    {
        res = std::make_shared<ov::op::v1::MaxPool>(உள்ளீடு, strides, pads_begin, pads_end, kernel);
        நிறுத்து;
    }
    நிலை 2:  // GGML_OP_POOL_AVG
    {
        res = std::make_shared<ov::op::v1::AvgPool>(உள்ளீடு, strides, pads_begin, pads_end, kernel, பொய்);
        நிறுத்து;
    }
    இயல்பு:
        நிறுத்து;
    }
    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

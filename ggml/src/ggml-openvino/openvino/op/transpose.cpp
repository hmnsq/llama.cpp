#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <openvino/op/transpose.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_transpose(स्थिरांक गाठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    // Compute permute order from input/output shape and stride information
    // so it adapts to different input and output layouts.
    स्वतः आदान_stride = context.get_आदान_stride(0);
    स्वतः output_stride = context.get_output_stride();

    // Compute permute order by matching output and input stride rankings.
    // Build <stride, dim_index> pairs.
    std::vector<std::pair<माप_प्रकार, पूर्णांक>> output_stride_dims;
    std::vector<std::pair<माप_प्रकार, पूर्णांक>> आदान_stride_dims;

    साठी (पूर्णांक i = 0; i < 4; ++i) {
        output_stride_dims.push_back({output_stride[i], i});
        आदान_stride_dims.push_back({आदान_stride[i], i});
    }

    // Sort by stride in descending order.
    std::sort(output_stride_dims.rbegin(), output_stride_dims.rend());
    std::sort(आदान_stride_dims.rbegin(), आदान_stride_dims.rend());

    // Build permute order.
    std::vector<int64_t> permute_order(4);
    साठी (पूर्णांक i = 0; i < 4; ++i) {
        पूर्णांक output_dim = output_stride_dims[i].second;
        पूर्णांक आदान_dim = आदान_stride_dims[i].second;
        permute_order[output_dim] = आदान_dim;
    }

    स्वतः आदान = process_view_आदान_new(context, 0);

    स्वतः res = std::make_shared<ov::op::v1::Transpose>(
        आदान, ov::op::v0::Constant::create(ov::element::i64, {4}, permute_order));
    परत rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

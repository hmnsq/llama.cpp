#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <openvino/op/transpose.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_transpose(अचर गाँठो_Context & context) {
    अङ्क_inputs_check(context, 1, 1);

    // Compute permute order from input/output shape and stride information
    // so it adapts to different input and output layouts.
    स्वतः आगत_stride = context.get_आगत_stride(0);
    स्वतः output_stride = context.get_output_stride();

    // Compute permute order by matching output and input stride rankings.
    // Build <stride, dim_index> pairs.
    std::vector<std::pair<माप_प्रकार, पूर्णाङ्क>> output_stride_dims;
    std::vector<std::pair<माप_प्रकार, पूर्णाङ्क>> आगत_stride_dims;

    लागि (पूर्णाङ्क i = 0; i < 4; ++i) {
        output_stride_dims.push_back({output_stride[i], i});
        आगत_stride_dims.push_back({आगत_stride[i], i});
    }

    // Sort by stride in descending order.
    std::sort(output_stride_dims.rbegin(), output_stride_dims.rend());
    std::sort(आगत_stride_dims.rbegin(), आगत_stride_dims.rend());

    // Build permute order.
    std::vector<int64_t> permute_order(4);
    लागि (पूर्णाङ्क i = 0; i < 4; ++i) {
        पूर्णाङ्क output_dim = output_stride_dims[i].second;
        पूर्णाङ्क आगत_dim = आगत_stride_dims[i].second;
        permute_order[output_dim] = आगत_dim;
    }

    स्वतः आगत = process_view_आगत_new(context, 0);

    स्वतः res = std::make_shared<ov::op::v1::Transpose>(
        आगत, ov::op::v0::Constant::create(ov::element::i64, {4}, permute_order));
    फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

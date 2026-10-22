#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <openvino/op/transpose.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_transpose(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    // Compute permute order from input/output shape and stride information
    // so it adapts to different input and output layouts.
    स्वतः निवेश_stride = context.get_निवेश_stride(0);
    स्वतः output_stride = context.get_output_stride();

    // Compute permute order by matching output and input stride rankings.
    // Build <stride, dim_index> pairs.
    std::vector<std::pair<माप_प्रकार, पूर्णांक>> output_stride_dims;
    std::vector<std::pair<माप_प्रकार, पूर्णांक>> निवेश_stride_dims;

    क्रम (पूर्णांक i = 0; i < 4; ++i) {
        output_stride_dims.push_back({output_stride[i], i});
        निवेश_stride_dims.push_back({निवेश_stride[i], i});
    }

    // Sort by stride in descending order.
    std::sort(output_stride_dims.rbegin(), output_stride_dims.rend());
    std::sort(निवेश_stride_dims.rbegin(), निवेश_stride_dims.rend());

    // Build permute order.
    std::vector<int64_t> permute_order(4);
    क्रम (पूर्णांक i = 0; i < 4; ++i) {
        पूर्णांक output_dim = output_stride_dims[i].second;
        पूर्णांक निवेश_dim = निवेश_stride_dims[i].second;
        permute_order[output_dim] = निवेश_dim;
    }

    स्वतः निवेश = process_view_निवेश_new(context, 0);

    स्वतः res = std::make_shared<ov::op::v1::Transpose>(
        निवेश, ov::op::v0::Constant::create(ov::element::i64, {4}, permute_order));
    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

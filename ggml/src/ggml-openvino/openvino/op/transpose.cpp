#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <openvino/op/transpose.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

OutputVector translate_transpose(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    // Compute permute order from input/output shape and stride information
    // so it adapts to different input and output layouts.
    auto ಒಳಹರಿವು_stride = context.get_ಒಳಹರಿವು_stride(0);
    auto output_stride = context.get_output_stride();

    // Compute permute order by matching output and input stride rankings.
    // Build <stride, dim_index> pairs.
    std::vector<std::pair<size_t, ಪೂರ್ಣಾಙ್ಕ>> output_stride_dims;
    std::vector<std::pair<size_t, ಪೂರ್ಣಾಙ್ಕ>> ಒಳಹರಿವು_stride_dims;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 4; ++i) {
        output_stride_dims.push_back({output_stride[i], i});
        ಒಳಹರಿವು_stride_dims.push_back({ಒಳಹರಿವು_stride[i], i});
    }

    // Sort by stride in descending order.
    std::sort(output_stride_dims.rbegin(), output_stride_dims.rend());
    std::sort(ಒಳಹರಿವು_stride_dims.rbegin(), ಒಳಹರಿವು_stride_dims.rend());

    // Build permute order.
    std::vector<int64_t> permute_order(4);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 4; ++i) {
        ಪೂರ್ಣಾಙ್ಕ output_dim = output_stride_dims[i].second;
        ಪೂರ್ಣಾಙ್ಕ ಒಳಹರಿವು_dim = ಒಳಹರಿವು_stride_dims[i].second;
        permute_order[output_dim] = ಒಳಹರಿವು_dim;
    }

    auto ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);

    auto res = std::make_shared<ov::op::v1::Transpose>(
        ಒಳಹರಿವು, ov::op::v0::Constant::create(ov::element::i64, {4}, permute_order));
    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

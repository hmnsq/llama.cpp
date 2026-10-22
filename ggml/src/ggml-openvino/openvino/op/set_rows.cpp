#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/core/node.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/node_output.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/type.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/frontend/exception.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/broadcast.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/concat.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/gather.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/scatter_elements_update.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/scatter_update.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/shape_of.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/squeeze.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/transpose.hpp>
#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

Output_ಸದಿಶ translate_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 3, 3);

    ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ = process_view_ಒಳಹರಿವು_new(context, 0);
    ಸ್ವಯಂಚಾಲಿತ indices = process_view_ಒಳಹರಿವು_new(context, 1);
    ಸ್ವಯಂಚಾಲಿತ dst = process_view_ಒಳಹರಿವು_new(context, 2);

    ದತ್ತಾಂಶ = std::make_shared<ov::op::v0::Convert>(ದತ್ತಾಂಶ, context.get_output_type());

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ indices_shape = context.get_ಒಳಹರಿವು_shape(1);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ multidim_indices = indices_shape.rank().is_static() &&
                                  indices_shape.rank().get_ಉದ್ದಳತೆ() == 4 &&
                                  ((indices_shape[1].is_static() && indices_shape[1].get_ಉದ್ದಳತೆ() > 1) ||
                                   (indices_shape[2].is_static() && indices_shape[2].get_ಉದ್ದಳತೆ() > 1));

    ಸ್ವಯಂಚಾಲಿತ axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});

    Output<Node> res;
    ಆದರೆ (context.is_stateful()) {
        ಪೂರ್ಣಾಂಕ concat_axis = 1;
        int64_t dim2 = dst.get_partial_shape()[2].get_ಉದ್ದಳತೆ();
        int64_t dim3 = dst.get_partial_shape()[3].get_ಉದ್ದಳತೆ();
        ದತ್ತಾಂಶ = std::make_shared<ov::op::v1::Reshape>(
            ದತ್ತಾಂಶ, ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) -1, dim2, dim3}), ಸುಳ್ಳು);
        res = std::make_shared<ov::op::v0::Concat>(Output_ಸದಿಶ{dst, ದತ್ತಾಂಶ}, concat_axis);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (multidim_indices) {
        ಸ್ವಯಂಚಾಲಿತ updates_shape = std::make_shared<ov::op::v3::Shape_ದ>(ದತ್ತಾಂಶ, ov::element::i64);

        ಸ್ವಯಂಚಾಲಿತ indices_rank3 = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        ಸ್ವಯಂಚಾಲಿತ one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        ಸ್ವಯಂಚಾಲಿತ indices_rank4_shape = std::make_shared<ov::op::v0::Concat>(Output_ಸದಿಶ{get_dimensions(updates_shape, {0, 1, 2}), one}, 0);
        ಸ್ವಯಂಚಾಲಿತ indices_rank4 = std::make_shared<ov::op::v1::Reshape>(indices_rank3, indices_rank4_shape, ಸುಳ್ಳು);
        ಸ್ವಯಂಚಾಲಿತ broadcasted_indices = std::make_shared<ov::op::v3::ಪ್ರಸಾರ>(indices_rank4, updates_shape);

        res = std::make_shared<ov::op::v3::ScatterElementsUpdate>(dst, broadcasted_indices, ದತ್ತಾಂಶ, axes);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ವಯಂಚಾಲಿತ ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = context.get_ಒಳಹರಿವು_shape(2)[3].get_ಉದ್ದಳತೆ();
        ಸ್ವಯಂಚಾಲಿತ ind_squeezed = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
        ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ_reshaped = std::make_shared<ov::op::v1::Reshape>(
            ದತ್ತಾಂಶ,
            ov::op::v0::Constant::create(ov::element::i64, {4},
                                         {(int64_t) 1, (int64_t) 1, (int64_t) -1, (int64_t) ಅಡ್ಡಸಾಲು_ಗಾತ್ರ}),
            ಸುಳ್ಳು);
        res = std::make_shared<ov::op::v3::ScatterUpdate>(dst, ind_squeezed, ದತ್ತಾಂಶ_reshaped, axes);
    }

    ಸ್ವಯಂಚಾಲಿತ dst_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(dst.get_ಗಂಟು_shared_ptr());
    ಆದರೆ (!multidim_indices && dst_reshape) {
        // Fix the case of multiple sequences, reshape back to original shape [1, n_seq, ctx_per_seq, emb]
        // ctx_per_seq is not fixed due to llama-bench compatibility
        ಸ್ವಯಂಚಾಲಿತ dst_shape_partial = dst_reshape->get_ಒಳಹರಿವು_partial_shape(0);
        std::vector<int64_t> dst_shape = {dst_shape_partial[0].get_ಉದ್ದಳತೆ(), dst_shape_partial[1].get_ಉದ್ದಳತೆ(),
                                          dst_shape_partial[2].is_static() ? dst_shape_partial[2].get_ಉದ್ದಳತೆ() : -1,
                                          dst_shape_partial[3].get_ಉದ್ದಳತೆ()};
        res = std::make_shared<ov::op::v1::Reshape>(res, ov::op::v0::Constant::create(ov::element::i64, {4}, dst_shape),
                                                    ಸುಳ್ಳು);
    }
    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

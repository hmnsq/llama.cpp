#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <cassert>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/core/node.hpp>
#अन्तर्भाव <openvino/core/node_output.hpp>
#अन्तर्भाव <openvino/core/type.hpp>
#अन्तर्भाव <openvino/frontend/exception.hpp>
#अन्तर्भाव <openvino/op/broadcast.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/gather.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/scatter_elements_update.hpp>
#अन्तर्भाव <openvino/op/scatter_update.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/squeeze.hpp>
#अन्तर्भाव <openvino/op/transpose.hpp>
#अन्तर्भाव <vector>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

Output_सदिशम् translate_समुच्चयः_पङ्क्तिकाः(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 3, 3);

    स्वचालित दत्त = process_view_निवेशः_new(context, 0);
    स्वचालित indices = process_view_निवेशः_new(context, 1);
    स्वचालित dst = process_view_निवेशः_new(context, 2);

    दत्त = std::make_shared<ov::op::v0::Convert>(दत्त, context.get_output_type());

    ध्रुव स्वचालित indices_shape = context.get_निवेशः_shape(1);
    ध्रुव तार्किक multidim_indices = indices_shape.rank().is_static() &&
                                  indices_shape.rank().get_दीर्घता() == 4 &&
                                  ((indices_shape[1].is_static() && indices_shape[1].get_दीर्घता() > 1) ||
                                   (indices_shape[2].is_static() && indices_shape[2].get_दीर्घता() > 1));

    स्वचालित axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});

    Output<Node> res;
    यदि (context.is_stateful()) {
        पूर्णाङ्क concat_axis = 1;
        int64_t dim2 = dst.get_partial_shape()[2].get_दीर्घता();
        int64_t dim3 = dst.get_partial_shape()[3].get_दीर्घता();
        दत्त = std::make_shared<ov::op::v1::Reshape>(
            दत्त, ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) -1, dim2, dim3}), असत्य);
        res = std::make_shared<ov::op::v0::Concat>(Output_सदिशम्{dst, दत्त}, concat_axis);
    } अन्यथा यदि (multidim_indices) {
        स्वचालित updates_shape = std::make_shared<ov::op::v3::Shape_सम्बन्धि>(दत्त, ov::element::i64);

        स्वचालित indices_rank3 = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        स्वचालित one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        स्वचालित indices_rank4_shape = std::make_shared<ov::op::v0::Concat>(Output_सदिशम्{get_dimensions(updates_shape, {0, 1, 2}), one}, 0);
        स्वचालित indices_rank4 = std::make_shared<ov::op::v1::Reshape>(indices_rank3, indices_rank4_shape, असत्य);
        स्वचालित broadcasted_indices = std::make_shared<ov::op::v3::प्रसारणम्>(indices_rank4, updates_shape);

        res = std::make_shared<ov::op::v3::ScatterElementsUpdate>(dst, broadcasted_indices, दत्त, axes);
    } अन्यथा {
        स्वचालित row_आकार = context.get_निवेशः_shape(2)[3].get_दीर्घता();
        स्वचालित ind_squeezed = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
        स्वचालित दत्त_reshaped = std::make_shared<ov::op::v1::Reshape>(
            दत्त,
            ov::op::v0::Constant::create(ov::element::i64, {4},
                                         {(int64_t) 1, (int64_t) 1, (int64_t) -1, (int64_t) row_आकार}),
            असत्य);
        res = std::make_shared<ov::op::v3::ScatterUpdate>(dst, ind_squeezed, दत्त_reshaped, axes);
    }

    स्वचालित dst_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(dst.get_ग्रन्थिः_shared_ptr());
    यदि (!multidim_indices && dst_reshape) {
        // Fix the case of multiple sequences, reshape back to original shape [1, n_seq, ctx_per_seq, emb]
        // ctx_per_seq is not fixed due to llama-bench compatibility
        स्वचालित dst_shape_partial = dst_reshape->get_निवेशः_partial_shape(0);
        std::vector<int64_t> dst_shape = {dst_shape_partial[0].get_दीर्घता(), dst_shape_partial[1].get_दीर्घता(),
                                          dst_shape_partial[2].is_static() ? dst_shape_partial[2].get_दीर्घता() : -1,
                                          dst_shape_partial[3].get_दीर्घता()};
        res = std::make_shared<ov::op::v1::Reshape>(res, ov::op::v0::Constant::create(ov::element::i64, {4}, dst_shape),
                                                    असत्य);
    }
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

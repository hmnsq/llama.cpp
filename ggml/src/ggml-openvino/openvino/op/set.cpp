#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <cstdint>
#अन्तर्भाव <openvino/frontend/exception.hpp>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/range.hpp>
#अन्तर्भाव <openvino/op/reduce_prod.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/scatter_update.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

// GGML SET writes src1 into a view of src0 and returns the updated tensor.
Output_सदिशम् translate_समुच्चयः(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 2, 2);

    स्वचालित dst = process_view_निवेशः_new(context, 0);
    स्वचालित src = process_view_निवेशः_new(context, 1);

    src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());

    ध्रुव स्वचालित dst_stride = context.get_निवेशः_stride(0);
    FRONT_अन्तः_OP_CONVERSION_CHECK(dst_stride.size() >= 4, "SET requires 4D destination strides");

    ध्रुव स्वचालित * op_params = reinterpret_cast<ध्रुव uint32_t *>(context.get_output_op_params());
    ध्रुव size_t विस्थापनम् = static_cast<size_t>(op_params[3]);

    ध्रुव size_t elem_आकार = dst_stride.back();
    FRONT_अन्तः_OP_CONVERSION_CHECK(elem_आकार != 0 && विस्थापनम् % elem_आकार == 0,
                                  "SET offset must be aligned to destination element size");

    ध्रुव int64_t विस्थापनम्_elems = static_cast<int64_t>(विस्थापनम् / elem_आकार);

    स्वचालित dst_flat = std::make_shared<ov::op::v1::Reshape>(
        dst,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        असत्य);

    स्वचालित src_flat = std::make_shared<ov::op::v1::Reshape>(
        src,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        असत्य);

    स्वचालित src_shape = std::make_shared<ov::op::v3::Shape_सम्बन्धि>(src_flat, ov::element::i64);
    स्वचालित src_len = std::make_shared<ov::op::v1::ReduceProd>(
        src_shape,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
        असत्य);

    स्वचालित start = ov::op::v0::Constant::create(ov::element::i64, {}, {विस्थापनम्_elems});
    स्वचालित stop = std::make_shared<ov::op::v1::सङ्कलय>(start, src_len);
    स्वचालित चरणः = ov::op::v0::Constant::create(ov::element::i64, {}, {1});

    स्वचालित indices = std::make_shared<ov::op::v4::Range>(start, stop, चरणः, ov::element::i64);
    स्वचालित axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

    स्वचालित updated_flat = std::make_shared<ov::op::v3::ScatterUpdate>(dst_flat, indices, src_flat, axis);

    स्वचालित dst_shape = std::make_shared<ov::op::v3::Shape_सम्बन्धि>(dst, ov::element::i64);
    स्वचालित res = std::make_shared<ov::op::v1::Reshape>(updated_flat, dst_shape, असत्य);

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

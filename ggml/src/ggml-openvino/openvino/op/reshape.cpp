#ଅନ୍ତର୍ଭାଵ "../node_context.h"
#ଅନ୍ତର୍ଭାଵ "../op_table.h"
#ଅନ୍ତର୍ଭାଵ "../utils.h"

#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <openvino/core/node.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/node_output.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/frontend/exception.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/concat.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/reshape.hpp>
#ଅନ୍ତର୍ଭାଵ <vector>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ op {

OutputVector translate_reshape(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);
    ଯଦି (context.get_ନିବେଶ(0).get_partial_shape().is_static() &&
        context.get_ନିବେଶ_shape(0) == context.get_output_shape()) {
        ଫେରନ୍ତୁ {context.get_ନିବେଶ(0)};
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_case = context.get_op_case();
    ଅଚଳ auto output_shape = context.get_output_shape().to_shape();
    std::vector<int64_t> shape(output_shape.begin(), output_shape.end());
    std::shared_ptr<ov::Node> new_shape_ଗଣ୍ଠି;
    ବାଛ (op_case) {
    ସ୍ଥିତି 0:
        ଭାଙ୍ଗ;
    ସ୍ଥିତି 1:
    ସ୍ଥିତି 9:
        shape[1] = -1;
        ଯଦି (context.is_stateful() && op_case == 1) {
            shape.erase(shape.begin());
        }
        ଭାଙ୍ଗ;
    ସ୍ଥିତି 2:
    ସ୍ଥିତି 3:
        shape[2] = -1;
        ଯଦି (op_case == 3) {
            shape[3] = 1;
        }
        ଭାଙ୍ଗ;
    ସ୍ଥିତି 4:
        ଫେରନ୍ତୁ {context.get_ନିବେଶ(0).get_ଗଣ୍ଠି_shared_ptr()->ନିବେଶ_ମୂଲ୍ୟ(0)};
    ସ୍ଥିତି 5:
    ସ୍ଥିତି 7:
        shape = {1, 1, -1, shape[3]};
        ଯଦି (context.is_stateful() && op_case == 5) {
            shape.erase(shape.begin());
        }
        ଭାଙ୍ଗ;
    ସ୍ଥିତି 6:
        // Recurrent inputs keep the active sequence count separate from the token count.
        ଯଦି (context.has_ନିବେଶ("s_copy_active_slot_len")) {
            auto n_slot_active_len = context.get_ନିବେଶ("s_copy_active_slot_len");
            auto emb_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {shape[3]});
            auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            new_shape_ଗଣ୍ଠି =
                std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one, n_slot_active_len, neg_one, emb_size}, 0);
        } ଅନ୍ୟଥା {
            shape = {1, 1, -1, shape[3]};
        }
        ଭାଙ୍ଗ;
    ସ୍ଥିତି 8:
        shape[0] = -1;
        ଭାଙ୍ଗ;
    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
        FRONT_END_OP_CONVERSION_CHECK(ମିଥ୍ୟା, "Unsupported RESHAPE case: ", op_case);
    }
    ଯଦି (!new_shape_ଗଣ୍ଠି) {
        new_shape_ଗଣ୍ଠି = ov::op::v0::Constant::create(ov::element::i64, {shape.size()}, shape);
    }
    auto res = std::make_shared<ov::op::v1::Reshape>(context.get_ନିବେଶ(0), new_shape_ଗଣ୍ଠି, ମିଥ୍ୟା);
    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#pragma once

#ଅନ୍ତର୍ଭାଵ "node_context.h"

#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <openvino/core/node.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/convert.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/shape_of.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/slice.hpp>
#ଅନ୍ତର୍ଭାଵ <utility>

namespace ov {
namespace frontend {
namespace ggml {

std::string getCurrentTime();

ଶୂନ୍ୟ ଅଙ୍କ_inputs_check(ଅଚଳ ଗଣ୍ଠି_Context & context, size_t min_inputs, size_t max_inputs);

ପୂର୍ଣ୍ଣସଂଖ୍ୟା non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb);

template <typename T> std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> argsort_descend(ଅଚଳ std::vector<T> & v) {
    std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&v](ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2) { ଫେରନ୍ତୁ v[i1] > v[i2]; });
    ଫେରନ୍ତୁ idx;
}

template <typename T> std::vector<T> sorted_descend(std::vector<T> v) {
    std::sort(v.begin(), v.end(), [](T a, T b) { ଫେରନ୍ତୁ a > b; });
    ଫେରନ୍ତୁ v;
}

template <typename T> bool is_permuted(ଅଚଳ std::vector<T> & strides) {
    ପାଇଁ (size_t i = 0; i < strides.size() - 1; ++i) {
        ଯଦି (strides[i] < strides[i + 1]) {
            ଫେରନ୍ତୁ true;
        }
    }
    ଫେରନ୍ତୁ false;
}

template <typename T> std::vector<T> permute(ଅଚଳ std::vector<T> & x, ଅଚଳ std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> & perm) {
    std::vector<T> result;
    result.reserve(perm.size());
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i : perm) {
        result.ଠେଲ_back(x[i]);
    }
    ଫେରନ୍ତୁ result;
}

std::shared_ptr<ov::ଗଣ୍ଠି_2> get_dimensions(ଅଚଳ std::shared_ptr<ov::op::v3::ShapeOf> & shape,
                                         ଅଚଳ std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> & dims);
std::shared_ptr<ov::ଗଣ୍ଠି_2> get_dimensions(ଅଚଳ std::shared_ptr<ov::ଗଣ୍ଠି_2> & ଗଣ୍ଠି, ଅଚଳ std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> & dims);

OutputVector rename_outputs_with_suffix(ଅଚଳ OutputVector & outputs, ଅଚଳ std::string & suffix);

std::pair<ov::Output<ଗଣ୍ଠି_2>, ov::Output<ଗଣ୍ଠି_2>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::ଗଣ୍ଠି_2> inp_pos,
                                                           std::shared_ptr<ov::ଗଣ୍ଠି_2> rope_freqs_weight = nullptr,
                                                           ପୂର୍ଣ୍ଣସଂଖ୍ୟା mode = 0,
                                                           bool stateful = false,
                                                           int64_t head_dim = 0);

ov::Output<ov::ଗଣ୍ଠି_2> process_view_ନିବେଶ(ଅଚଳ ଗଣ୍ଠି_Context & context, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ନିବେଶ_index, ପୂର୍ଣ୍ଣସଂଖ୍ୟା slice_len = 0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା axis = -1);

ov::Output<ov::ଗଣ୍ଠି_2> process_view_ନିବେଶ_new(ଅଚଳ ଗଣ୍ଠି_Context & context, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ନିବେଶ_index);

namespace op {
template <typename T> OutputVector translate_1to1_match_2_inputs(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 2, 2);
    auto ନିବେଶ_0 = process_view_ନିବେଶ_new(context, 0);
    auto ନିବେଶ_1 = process_view_ନିବେଶ_new(context, 1);

    auto output_type = context.get_output_type();
    ଯଦି (ନିବେଶ_0.get_element_type() != ନିବେଶ_1.get_element_type()) {
        ଯଦି (ନିବେଶ_0.get_element_type() != ov::element::f32) {
            ନିବେଶ_0 = std::make_shared<ov::op::v0::Convert>(ନିବେଶ_0, ov::element::f32);
        }
        ଯଦି (ନିବେଶ_1.get_element_type() != ov::element::f32) {
            ନିବେଶ_1 = std::make_shared<ov::op::v0::Convert>(ନିବେଶ_1, ov::element::f32);
        }
    }

    ov::Output<ov::ଗଣ୍ଠି_2> res = std::make_shared<T>(ନିବେଶ_0, ନିବେଶ_1);
    ଯଦି (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }
    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

template <typename T> OutputVector translate_1to1_match_1_ନିବେଶ(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);
    auto ନିବେଶ = process_view_ନିବେଶ_new(context, 0);
    auto res = std::make_shared<T>(ନିବେଶ);
    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}
}  // namespace op

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

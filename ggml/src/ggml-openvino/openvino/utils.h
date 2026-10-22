#pragma once

#include "node_context.h"

#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <utility>

namespace ov {
namespace frontend {
namespace ggml {

std::string getCurrentTime();

void num_inputs_check(const NodeContext & context, size_t min_inputs, size_t max_inputs);

정수 non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb);

template <typename T> std::vector<정수> argsort_descend(const std::vector<T> & v) {
    std::vector<정수> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&v](정수 i1, 정수 i2) { 반환 v[i1] > v[i2]; });
    반환 idx;
}

template <typename T> std::vector<T> sorted_descend(std::vector<T> v) {
    std::sort(v.begin(), v.end(), [](T a, T b) { 반환 a > b; });
    반환 v;
}

template <typename T> bool is_permuted(const std::vector<T> & strides) {
    위해 (size_t i = 0; i < strides.size() - 1; ++i) {
        만약 (strides[i] < strides[i + 1]) {
            반환 true;
        }
    }
    반환 false;
}

template <typename T> std::vector<T> permute(const std::vector<T> & x, const std::vector<정수> & perm) {
    std::vector<T> result;
    result.reserve(perm.size());
    위해 (정수 i : perm) {
        result.push_back(x[i]);
    }
    반환 result;
}

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::op::v3::ShapeOf> & shape,
                                         const std::vector<정수> & dims);
std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node> & node, const std::vector<정수> & dims);

OutputVector rename_outputs_with_suffix(const OutputVector & outputs, const std::string & suffix);

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight = nullptr,
                                                           정수 mode = 0,
                                                           bool stateful = false,
                                                           int64_t head_dim = 0);

ov::Output<ov::Node> process_view_input(const NodeContext & context, 정수 input_index, 정수 slice_len = 0, 정수 axis = -1);

ov::Output<ov::Node> process_view_input_new(const NodeContext & context, 정수 input_index);

namespace op {
template <typename T> OutputVector translate_1to1_match_2_inputs(const NodeContext & context) {
    num_inputs_check(context, 2, 2);
    auto input_0 = process_view_input_new(context, 0);
    auto input_1 = process_view_input_new(context, 1);

    auto output_type = context.get_output_type();
    만약 (input_0.get_element_type() != input_1.get_element_type()) {
        만약 (input_0.get_element_type() != ov::element::f32) {
            input_0 = std::make_shared<ov::op::v0::Convert>(input_0, ov::element::f32);
        }
        만약 (input_1.get_element_type() != ov::element::f32) {
            input_1 = std::make_shared<ov::op::v0::Convert>(input_1, ov::element::f32);
        }
    }

    ov::Output<ov::Node> res = std::make_shared<T>(input_0, input_1);
    만약 (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }
    반환 rename_outputs_with_suffix({res}, context.get_name());
}

template <typename T> OutputVector translate_1to1_match_1_input(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    auto input = process_view_input_new(context, 0);
    auto res = std::make_shared<T>(input);
    반환 rename_outputs_with_suffix({res}, context.get_name());
}
}  // namespace op

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

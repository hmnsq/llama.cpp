#ಅನ್ತರ್ಭಾವ "fuse_moe_router.h"

#ಅನ್ತರ್ಭಾವ <openvino/core/graph_util.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/rt_info.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/broadcast.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/clamp.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/concat.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/divide.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/gather.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/matmul.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reduce_sum.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/shape_of.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/softmax.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/squeeze.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/tile.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/topk.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/unsqueeze.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/pass/pattern/op/wrap_type.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ pass {

ನಾಮಾಕಾಶ {

ತಾರ್ಕಿಕ constant_is(ಧ್ರುವ ov::Output<ov::Node> & output, ಧ್ರುವ std::vector<int64_t> & ಮೌಲ್ಯಗಳು) {
    auto ಗಂಟು = ov::as_type_ptr<ov::op::v0::Constant>(output.get_ಗಂಟು_shared_ptr());
    ಪ್ರತ್ಯಾಗಮ ಗಂಟು && ಗಂಟು->get_element_type().is_integral_ಸಂಖ್ಯೆ() && ಗಂಟು->cast_vector<int64_t>() == ಮೌಲ್ಯಗಳು;
}

}  // namespace

FuseMoeRouter::FuseMoeRouter() {
    using ನಾಮಾಕಾಶ ov::pass::pattern;
    using ನಾಮಾಕಾಶ ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    auto logits = wrap_type<v0::MatMul>();
    auto softmax = wrap_type<v8::Softmax>({logits});
    auto topk = wrap_type<v11::ತುದಿ_K>({softmax, any_ಒಳಹರಿವು()});
    topk->set_output_size(2);
    auto ids = wrap_type<v8::Slice>({topk->output(1), any_ಒಳಹರಿವು(), any_ಒಳಹರಿವು(), any_ಒಳಹರಿವು(), any_ಒಳಹರಿವು()});
    auto ids_2d = wrap_type<v0::Squeeze>({ids, any_ಒಳಹರಿವು()});
    auto probs = wrap_type<v1::Reshape>({softmax, any_ಒಳಹರಿವು()});
    auto data = wrap_type<v0::Squeeze>({probs, any_ಒಳಹರಿವು()});
    auto data_batch = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({data}), any_ಒಳಹರಿವು(), any_ಒಳಹರಿವು()});
    auto ids_ಎಣಿಕೆ = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({ids_2d}), any_ಒಳಹರಿವು(), any_ಒಳಹರಿವು()});
    auto target = wrap_type<v0::Concat>({data_batch, ids_ಎಣಿಕೆ});
    auto broadcast = wrap_type<v3::Broadcast>({ids_2d, target});
    auto gather = wrap_type<v8::Gather>({data, broadcast, any_ಒಳಹರಿವು()});
    auto weights_4d = wrap_type<v0::Unsqueeze>({gather, any_ಒಳಹರಿವು()});
    auto weights = wrap_type<v1::Reshape>({weights_4d, any_ಒಳಹರಿವು()});
    auto sum = wrap_type<v1::ReduceSum>({weights, any_ಒಳಹರಿವು()});
    auto clamp = wrap_type<v0::Clamp>({sum});
    auto tile = wrap_type<v0::Tile>({clamp, any_ಒಳಹರಿವು()});
    auto norm = wrap_type<v1::Divide>({weights, tile});

    ಧ್ರುವ auto callback = [=](Matcher & m) {
        ಧ್ರುವ auto & pm = m.get_pattern_ಮೌಲ್ಯ_map();
        ಧ್ರುವ auto ಗಂಟು = [&](ಧ್ರುವ std::shared_ptr<ov::Node> & p) { ಪ್ರತ್ಯಾಗಮ pm.at(p).get_ಗಂಟು_shared_ptr(); };
        ಧ್ರುವ auto ಒಳಹರಿವು_is = [&](ಧ್ರುವ std::shared_ptr<ov::Node> & p, size_t i, ಧ್ರುವ std::vector<int64_t> & v) {
            ಪ್ರತ್ಯಾಗಮ constant_is(ಗಂಟು(p)->ಒಳಹರಿವು_ಮೌಲ್ಯ(i), v);
        };
        auto mm = ov::as_type_ptr<v0::MatMul>(ಗಂಟು(logits));
        auto sm = ov::as_type_ptr<v8::Softmax>(ಗಂಟು(softmax));
        auto tk = ov::as_type_ptr<v11::ತುದಿ_K>(ಗಂಟು(topk));
        auto reduce = ov::as_type_ptr<v1::ReduceSum>(ಗಂಟು(sum));
        auto limit = ov::as_type_ptr<v0::Clamp>(ಗಂಟು(clamp));
        ಧ್ರುವ auto shape = mm->get_output_partial_shape(0);
        ಯದಿ (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_ಒಳಹರಿವು_partial_shape(0).rank() != 4 ||
            mm->get_ಒಳಹರಿವು_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_mode() != v11::ತುದಿ_K::Mode::MAX ||
            tk->get_sort_type() != v11::ತುದಿ_K::SortType::SORT_ಮೌಲ್ಯಗಳು || tk->get_stable() ||
            tk->get_index_element_type() != ov::element::i32 ||
            !tk->output(0).get_target_inputs().empty()) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಧ್ರುವ int64_t experts = shape[3].get_length();
        auto end = ov::as_type_ptr<v0::Constant>(ಗಂಟು(ids)->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(2));
        ಯದಿ (!end || !end->get_element_type().is_integral_ಸಂಖ್ಯೆ() || ov::shape_size(end->get_shape()) != 1) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಧ್ರುವ int64_t k = end->cast_vector<int64_t>()[0];
        ಧ್ರುವ auto sorted_shape = tk->get_output_partial_shape(1);
        ಯದಿ (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_length() < k ||
            !ಒಳಹರಿವು_is(probs, 1, {1, -1, experts, 1}) ||
            !ಒಳಹರಿವು_is(data, 1, {0}) || !ಒಳಹರಿವು_is(ids_2d, 1, {0, 1}) ||
            !ಒಳಹರಿವು_is(weights_4d, 1, {0}) || !ಒಳಹರಿವು_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(ಗಂಟು(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(ಗಂಟು(weights))->get_special_zero() ||
            !ಒಳಹರಿವು_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(ಗಂಟು(gather))->get_batch_dims() != 1 ||
            !ಒಳಹರಿವು_is(data_batch, 1, {0}) || !ಒಳಹರಿವು_is(data_batch, 2, {0}) ||
            !ಒಳಹರಿವು_is(ids_ಎಣಿಕೆ, 1, {1}) || !ಒಳಹರಿವು_is(ids_ಎಣಿಕೆ, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(ಗಂಟು(data_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(ಗಂಟು(ids_ಎಣಿಕೆ))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(ಗಂಟು(target))->get_axis() != 0 ||
            ov::as_type_ptr<v3::Broadcast>(ಗಂಟು(broadcast))->get_broadcast_spec().m_type != ov::op::BroadcastType::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!ಒಳಹರಿವು_is(sum, 1, {-1}) && !ಒಳಹರಿವು_is(sum, 1, {3})) ||
            !ಒಳಹರಿವು_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(ಗಂಟು(norm))->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        ಯದಿ (!(limit->get_min() <= 0.5 * ದ್ವಿಗುಣ(k) / experts && limit->get_max() >= 2.0)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        ಪ್ರತಿ (ಧ್ರುವ auto & ಒಳಹರಿವು : tk->output(1).get_target_inputs()) {
            auto slice = ov::as_type_ptr<v8::Slice>(ಒಳಹರಿವು.get_ಗಂಟು()->shared_from_this());
            ಯದಿ (!slice || ಒಳಹರಿವು.get_index() != 0 || slice->get_ಒಳಹರಿವು_size() != 5 ||
                !constant_is(slice->ಒಳಹರಿವು_ಮೌಲ್ಯ(1), {0}) || !constant_is(slice->ಒಳಹರಿವು_ಮೌಲ್ಯ(2), {k}) ||
                !constant_is(slice->ಒಳಹರಿವು_ಮೌಲ್ಯ(3), {1}) ||
                (!constant_is(slice->ಒಳಹರಿವು_ಮೌಲ್ಯ(4), {3}) && !constant_is(slice->ಒಳಹರಿವು_ಮೌಲ್ಯ(4), {-1}))) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            slices.push_back(slice);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        auto axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto hidden = std::make_shared<v0::Squeeze>(mm->ಒಳಹರಿವು_ಮೌಲ್ಯ(0), axis0);
        auto routing = std::make_shared<v0::MatMul>(hidden, mm->ಒಳಹರಿವು_ಮೌಲ್ಯ(1), ಅಸತ್ಯ, mm->get_transpose_b());
        auto probabilities = std::make_shared<v8::Softmax>(routing, -1);
        auto selected = std::make_shared<v11::ತುದಿ_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::ತುದಿ_K::Mode::MAX, v11::ತುದಿ_K::SortType::SORT_ಮೌಲ್ಯಗಳು, tk->get_index_element_type(), ಅಸತ್ಯ);
        auto total = std::make_shared<v1::ReduceSum>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), ಸತ್ಯ);
        auto normalized = std::make_shared<v1::Divide>(selected->output(0), total);
        auto weights_out = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        auto ids_out = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, ಗಂಟು(norm)}, {hidden, routing, probabilities, selected, total, normalized, weights_out, ids_out});
        ov::replace_ಗಂಟು(ಗಂಟು(norm), weights_out);
        ಪ್ರತಿ (ಧ್ರುವ auto & slice : slices) {
            ov::replace_ಗಂಟು(slice, ids_out);
        }
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    };
    register_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#अन्तर्भाव "fuse_moe_router.h"

#अन्तर्भाव <openvino/core/graph_util.hpp>
#अन्तर्भाव <openvino/core/rt_info.hpp>
#अन्तर्भाव <openvino/op/broadcast.hpp>
#अन्तर्भाव <openvino/op/clamp.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/divide.hpp>
#अन्तर्भाव <openvino/op/gather.hpp>
#अन्तर्भाव <openvino/op/matmul.hpp>
#अन्तर्भाव <openvino/op/reduce_sum.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/softmax.hpp>
#अन्तर्भाव <openvino/op/squeeze.hpp>
#अन्तर्भाव <openvino/op/tile.hpp>
#अन्तर्भाव <openvino/op/topk.hpp>
#अन्तर्भाव <openvino/op/unsqueeze.hpp>
#अन्तर्भाव <openvino/pass/pattern/op/wrap_type.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश pass {

नामाकाश {

तार्किक constant_is(ध्रुव ov::Output<ov::Node> & output, ध्रुव std::vector<int64_t> & मानानि) {
    auto ग्रन्थिः = ov::as_type_ptr<ov::op::v0::Constant>(output.get_ग्रन्थिः_shared_ptr());
    पटिनिवत्त ग्रन्थिः && ग्रन्थिः->get_element_type().is_integral_सङ्ख्या() && ग्रन्थिः->cast_vector<int64_t>() == मानानि;
}

}  // namespace

FuseMoeRouter::FuseMoeRouter() {
    using नामाकाश ov::pass::pattern;
    using नामाकाश ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    auto logits = wrap_type<v0::MatMul>();
    auto softmax = wrap_type<v8::Softmax>({logits});
    auto topk = wrap_type<v11::शिखरम्_K>({softmax, any_निवेशः()});
    topk->set_output_size(2);
    auto ids = wrap_type<v8::Slice>({topk->output(1), any_निवेशः(), any_निवेशः(), any_निवेशः(), any_निवेशः()});
    auto ids_2d = wrap_type<v0::Squeeze>({ids, any_निवेशः()});
    auto probs = wrap_type<v1::Reshape>({softmax, any_निवेशः()});
    auto data = wrap_type<v0::Squeeze>({probs, any_निवेशः()});
    auto data_batch = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({data}), any_निवेशः(), any_निवेशः()});
    auto ids_गणनम् = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({ids_2d}), any_निवेशः(), any_निवेशः()});
    auto target = wrap_type<v0::Concat>({data_batch, ids_गणनम्});
    auto broadcast = wrap_type<v3::Broadcast>({ids_2d, target});
    auto gather = wrap_type<v8::Gather>({data, broadcast, any_निवेशः()});
    auto weights_4d = wrap_type<v0::Unsqueeze>({gather, any_निवेशः()});
    auto weights = wrap_type<v1::Reshape>({weights_4d, any_निवेशः()});
    auto sum = wrap_type<v1::ReduceSum>({weights, any_निवेशः()});
    auto clamp = wrap_type<v0::Clamp>({sum});
    auto tile = wrap_type<v0::Tile>({clamp, any_निवेशः()});
    auto norm = wrap_type<v1::Divide>({weights, tile});

    ध्रुव auto callback = [=](Matcher & m) {
        ध्रुव auto & pm = m.get_pattern_मानम्_map();
        ध्रुव auto ग्रन्थिः = [&](ध्रुव std::shared_ptr<ov::Node> & p) { पटिनिवत्त pm.at(p).get_ग्रन्थिः_shared_ptr(); };
        ध्रुव auto निवेशः_is = [&](ध्रुव std::shared_ptr<ov::Node> & p, size_t i, ध्रुव std::vector<int64_t> & v) {
            पटिनिवत्त constant_is(ग्रन्थिः(p)->निवेशः_मानम्(i), v);
        };
        auto mm = ov::as_type_ptr<v0::MatMul>(ग्रन्थिः(logits));
        auto sm = ov::as_type_ptr<v8::Softmax>(ग्रन्थिः(softmax));
        auto tk = ov::as_type_ptr<v11::शिखरम्_K>(ग्रन्थिः(topk));
        auto reduce = ov::as_type_ptr<v1::ReduceSum>(ग्रन्थिः(sum));
        auto limit = ov::as_type_ptr<v0::Clamp>(ग्रन्थिः(clamp));
        ध्रुव auto shape = mm->get_output_partial_shape(0);
        सचे (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_निवेशः_partial_shape(0).rank() != 4 ||
            mm->get_निवेशः_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_mode() != v11::शिखरम्_K::Mode::MAX ||
            tk->get_sort_type() != v11::शिखरम्_K::SortType::SORT_मानानि || tk->get_stable() ||
            tk->get_index_element_type() != ov::element::i32 ||
            !tk->output(0).get_target_inputs().empty()) {
            पटिनिवत्त मिच्छा;
        }
        ध्रुव int64_t experts = shape[3].get_length();
        auto end = ov::as_type_ptr<v0::Constant>(ग्रन्थिः(ids)->get_निवेशः_ग्रन्थिः_shared_ptr(2));
        सचे (!end || !end->get_element_type().is_integral_सङ्ख्या() || ov::shape_size(end->get_shape()) != 1) {
            पटिनिवत्त मिच्छा;
        }
        ध्रुव int64_t k = end->cast_vector<int64_t>()[0];
        ध्रुव auto sorted_shape = tk->get_output_partial_shape(1);
        सचे (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_length() < k ||
            !निवेशः_is(probs, 1, {1, -1, experts, 1}) ||
            !निवेशः_is(data, 1, {0}) || !निवेशः_is(ids_2d, 1, {0, 1}) ||
            !निवेशः_is(weights_4d, 1, {0}) || !निवेशः_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(ग्रन्थिः(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(ग्रन्थिः(weights))->get_special_zero() ||
            !निवेशः_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(ग्रन्थिः(gather))->get_batch_dims() != 1 ||
            !निवेशः_is(data_batch, 1, {0}) || !निवेशः_is(data_batch, 2, {0}) ||
            !निवेशः_is(ids_गणनम्, 1, {1}) || !निवेशः_is(ids_गणनम्, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(ग्रन्थिः(data_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(ग्रन्थिः(ids_गणनम्))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(ग्रन्थिः(target))->get_axis() != 0 ||
            ov::as_type_ptr<v3::Broadcast>(ग्रन्थिः(broadcast))->get_broadcast_spec().m_type != ov::op::BroadcastType::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!निवेशः_is(sum, 1, {-1}) && !निवेशः_is(sum, 1, {3})) ||
            !निवेशः_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(ग्रन्थिः(norm))->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY) {
            पटिनिवत्त मिच्छा;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        सचे (!(limit->get_min() <= 0.5 * द्विगुण(k) / experts && limit->get_max() >= 2.0)) {
            पटिनिवत्त मिच्छा;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        अत्थाय (ध्रुव auto & निवेशः : tk->output(1).get_target_inputs()) {
            auto slice = ov::as_type_ptr<v8::Slice>(निवेशः.get_ग्रन्थिः()->shared_from_this());
            सचे (!slice || निवेशः.get_index() != 0 || slice->get_निवेशः_size() != 5 ||
                !constant_is(slice->निवेशः_मानम्(1), {0}) || !constant_is(slice->निवेशः_मानम्(2), {k}) ||
                !constant_is(slice->निवेशः_मानम्(3), {1}) ||
                (!constant_is(slice->निवेशः_मानम्(4), {3}) && !constant_is(slice->निवेशः_मानम्(4), {-1}))) {
                पटिनिवत्त मिच्छा;
            }
            slices.push_back(slice);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        auto axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto hidden = std::make_shared<v0::Squeeze>(mm->निवेशः_मानम्(0), axis0);
        auto routing = std::make_shared<v0::MatMul>(hidden, mm->निवेशः_मानम्(1), मिच्छा, mm->get_transpose_b());
        auto probabilities = std::make_shared<v8::Softmax>(routing, -1);
        auto selected = std::make_shared<v11::शिखरम्_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::शिखरम्_K::Mode::MAX, v11::शिखरम्_K::SortType::SORT_मानानि, tk->get_index_element_type(), मिच्छा);
        auto total = std::make_shared<v1::ReduceSum>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), सच्च);
        auto normalized = std::make_shared<v1::Divide>(selected->output(0), total);
        auto weights_out = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        auto ids_out = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, ग्रन्थिः(norm)}, {hidden, routing, probabilities, selected, total, normalized, weights_out, ids_out});
        ov::replace_ग्रन्थिः(ग्रन्थिः(norm), weights_out);
        अत्थाय (ध्रुव auto & slice : slices) {
            ov::replace_ग्रन्थिः(slice, ids_out);
        }
        पटिनिवत्त सच्च;
    };
    register_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

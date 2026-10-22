#അന്തര്ഭാവ "fuse_moe_router.h"

#അന്തര്ഭാവ <openvino/core/graph_util.hpp>
#അന്തര്ഭാവ <openvino/core/rt_info.hpp>
#അന്തര്ഭാവ <openvino/op/broadcast.hpp>
#അന്തര്ഭാവ <openvino/op/clamp.hpp>
#അന്തര്ഭാവ <openvino/op/concat.hpp>
#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/divide.hpp>
#അന്തര്ഭാവ <openvino/op/gather.hpp>
#അന്തര്ഭാവ <openvino/op/matmul.hpp>
#അന്തര്ഭാവ <openvino/op/reduce_sum.hpp>
#അന്തര്ഭാവ <openvino/op/reshape.hpp>
#അന്തര്ഭാവ <openvino/op/shape_of.hpp>
#അന്തര്ഭാവ <openvino/op/slice.hpp>
#അന്തര്ഭാവ <openvino/op/softmax.hpp>
#അന്തര്ഭാവ <openvino/op/squeeze.hpp>
#അന്തര്ഭാവ <openvino/op/tile.hpp>
#അന്തര്ഭാവ <openvino/op/topk.hpp>
#അന്തര്ഭാവ <openvino/op/unsqueeze.hpp>
#അന്തര്ഭാവ <openvino/pass/pattern/op/wrap_type.hpp>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ pass {

നാമാകാശ {

തർക്കം constant_is(സ്ഥിരം ov::Output<ov::Node> & output, സ്ഥിരം std::vector<int64_t> & മൂല്യങ്ങൾ) {
    auto കണ്ണി = ov::as_type_ptr<ov::op::v0::Constant>(output.get_കണ്ണി_shared_ptr());
    തിരികെ കണ്ണി && കണ്ണി->get_element_type().is_integral_സംഖ്യ() && കണ്ണി->cast_vector<int64_t>() == മൂല്യങ്ങൾ;
}

}  // namespace

FuseMoeRouter::FuseMoeRouter() {
    using നാമാകാശ ov::pass::pattern;
    using നാമാകാശ ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    auto logits = wrap_type<v0::MatMul>();
    auto softmax = wrap_type<v8::Softmax>({logits});
    auto topk = wrap_type<v11::മുകൾ_K>({softmax, any_നിവേശം()});
    topk->set_output_size(2);
    auto ids = wrap_type<v8::Slice>({topk->output(1), any_നിവേശം(), any_നിവേശം(), any_നിവേശം(), any_നിവേശം()});
    auto ids_2d = wrap_type<v0::Squeeze>({ids, any_നിവേശം()});
    auto probs = wrap_type<v1::Reshape>({softmax, any_നിവേശം()});
    auto data = wrap_type<v0::Squeeze>({probs, any_നിവേശം()});
    auto data_batch = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({data}), any_നിവേശം(), any_നിവേശം()});
    auto ids_എണ്ണം = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({ids_2d}), any_നിവേശം(), any_നിവേശം()});
    auto target = wrap_type<v0::Concat>({data_batch, ids_എണ്ണം});
    auto broadcast = wrap_type<v3::Broadcast>({ids_2d, target});
    auto gather = wrap_type<v8::Gather>({data, broadcast, any_നിവേശം()});
    auto weights_4d = wrap_type<v0::Unsqueeze>({gather, any_നിവേശം()});
    auto weights = wrap_type<v1::Reshape>({weights_4d, any_നിവേശം()});
    auto sum = wrap_type<v1::ReduceSum>({weights, any_നിവേശം()});
    auto clamp = wrap_type<v0::Clamp>({sum});
    auto tile = wrap_type<v0::Tile>({clamp, any_നിവേശം()});
    auto norm = wrap_type<v1::Divide>({weights, tile});

    സ്ഥിരം auto callback = [=](Matcher & m) {
        സ്ഥിരം auto & pm = m.get_pattern_മൂല്യം_map();
        സ്ഥിരം auto കണ്ണി = [&](സ്ഥിരം std::shared_ptr<ov::Node> & p) { തിരികെ pm.at(p).get_കണ്ണി_shared_ptr(); };
        സ്ഥിരം auto നിവേശം_is = [&](സ്ഥിരം std::shared_ptr<ov::Node> & p, size_t i, സ്ഥിരം std::vector<int64_t> & v) {
            തിരികെ constant_is(കണ്ണി(p)->നിവേശം_മൂല്യം(i), v);
        };
        auto mm = ov::as_type_ptr<v0::MatMul>(കണ്ണി(logits));
        auto sm = ov::as_type_ptr<v8::Softmax>(കണ്ണി(softmax));
        auto tk = ov::as_type_ptr<v11::മുകൾ_K>(കണ്ണി(topk));
        auto reduce = ov::as_type_ptr<v1::ReduceSum>(കണ്ണി(sum));
        auto limit = ov::as_type_ptr<v0::Clamp>(കണ്ണി(clamp));
        സ്ഥിരം auto shape = mm->get_output_partial_shape(0);
        എങ്കിൽ (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_നിവേശം_partial_shape(0).rank() != 4 ||
            mm->get_നിവേശം_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_mode() != v11::മുകൾ_K::Mode::MAX ||
            tk->get_sort_type() != v11::മുകൾ_K::SortType::SORT_മൂല്യങ്ങൾ || tk->get_stable() ||
            tk->get_index_element_type() != ov::element::i32 ||
            !tk->output(0).get_target_inputs().empty()) {
            തിരികെ അസത്യം;
        }
        സ്ഥിരം int64_t experts = shape[3].get_length();
        auto end = ov::as_type_ptr<v0::Constant>(കണ്ണി(ids)->get_നിവേശം_കണ്ണി_shared_ptr(2));
        എങ്കിൽ (!end || !end->get_element_type().is_integral_സംഖ്യ() || ov::shape_size(end->get_shape()) != 1) {
            തിരികെ അസത്യം;
        }
        സ്ഥിരം int64_t k = end->cast_vector<int64_t>()[0];
        സ്ഥിരം auto sorted_shape = tk->get_output_partial_shape(1);
        എങ്കിൽ (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_length() < k ||
            !നിവേശം_is(probs, 1, {1, -1, experts, 1}) ||
            !നിവേശം_is(data, 1, {0}) || !നിവേശം_is(ids_2d, 1, {0, 1}) ||
            !നിവേശം_is(weights_4d, 1, {0}) || !നിവേശം_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(കണ്ണി(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(കണ്ണി(weights))->get_special_zero() ||
            !നിവേശം_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(കണ്ണി(gather))->get_batch_dims() != 1 ||
            !നിവേശം_is(data_batch, 1, {0}) || !നിവേശം_is(data_batch, 2, {0}) ||
            !നിവേശം_is(ids_എണ്ണം, 1, {1}) || !നിവേശം_is(ids_എണ്ണം, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(കണ്ണി(data_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(കണ്ണി(ids_എണ്ണം))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(കണ്ണി(target))->get_axis() != 0 ||
            ov::as_type_ptr<v3::Broadcast>(കണ്ണി(broadcast))->get_broadcast_spec().m_type != ov::op::BroadcastType::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!നിവേശം_is(sum, 1, {-1}) && !നിവേശം_is(sum, 1, {3})) ||
            !നിവേശം_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(കണ്ണി(norm))->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY) {
            തിരികെ അസത്യം;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        എങ്കിൽ (!(limit->get_min() <= 0.5 * ഇരട്ട(k) / experts && limit->get_max() >= 2.0)) {
            തിരികെ അസത്യം;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        വേണ്ടി (സ്ഥിരം auto & നിവേശം : tk->output(1).get_target_inputs()) {
            auto slice = ov::as_type_ptr<v8::Slice>(നിവേശം.get_കണ്ണി()->shared_from_this());
            എങ്കിൽ (!slice || നിവേശം.get_index() != 0 || slice->get_നിവേശം_size() != 5 ||
                !constant_is(slice->നിവേശം_മൂല്യം(1), {0}) || !constant_is(slice->നിവേശം_മൂല്യം(2), {k}) ||
                !constant_is(slice->നിവേശം_മൂല്യം(3), {1}) ||
                (!constant_is(slice->നിവേശം_മൂല്യം(4), {3}) && !constant_is(slice->നിവേശം_മൂല്യം(4), {-1}))) {
                തിരികെ അസത്യം;
            }
            slices.push_back(slice);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        auto axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto hidden = std::make_shared<v0::Squeeze>(mm->നിവേശം_മൂല്യം(0), axis0);
        auto routing = std::make_shared<v0::MatMul>(hidden, mm->നിവേശം_മൂല്യം(1), അസത്യം, mm->get_transpose_b());
        auto probabilities = std::make_shared<v8::Softmax>(routing, -1);
        auto selected = std::make_shared<v11::മുകൾ_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::മുകൾ_K::Mode::MAX, v11::മുകൾ_K::SortType::SORT_മൂല്യങ്ങൾ, tk->get_index_element_type(), അസത്യം);
        auto total = std::make_shared<v1::ReduceSum>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), സത്യം);
        auto normalized = std::make_shared<v1::Divide>(selected->output(0), total);
        auto weights_out = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        auto ids_out = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, കണ്ണി(norm)}, {hidden, routing, probabilities, selected, total, normalized, weights_out, ids_out});
        ov::replace_കണ്ണി(കണ്ണി(norm), weights_out);
        വേണ്ടി (സ്ഥിരം auto & slice : slices) {
            ov::replace_കണ്ണി(slice, ids_out);
        }
        തിരികെ സത്യം;
    };
    register_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#అన్తర్భావ "fuse_moe_router.h"

#అన్తర్భావ <openvino/core/graph_util.hpp>
#అన్తర్భావ <openvino/core/rt_info.hpp>
#అన్తర్భావ <openvino/op/broadcast.hpp>
#అన్తర్భావ <openvino/op/clamp.hpp>
#అన్తర్భావ <openvino/op/concat.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/divide.hpp>
#అన్తర్భావ <openvino/op/gather.hpp>
#అన్తర్భావ <openvino/op/matmul.hpp>
#అన్తర్భావ <openvino/op/reduce_sum.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/shape_of.hpp>
#అన్తర్భావ <openvino/op/slice.hpp>
#అన్తర్భావ <openvino/op/softmax.hpp>
#అన్తర్భావ <openvino/op/squeeze.hpp>
#అన్తర్భావ <openvino/op/tile.hpp>
#అన్తర్భావ <openvino/op/topk.hpp>
#అన్తర్భావ <openvino/op/unsqueeze.hpp>
#అన్తర్భావ <openvino/pass/pattern/op/wrap_type.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ pass {

నామాకాశ {

తార్కిక constant_is(స్థిరరాశి ov::Output<ov::Node> & output, స్థిరరాశి std::vector<int64_t> & విలువలు) {
    స్వయంచాలక కణుపు = ov::as_type_ptr<ov::op::v0::Constant>(output.get_కణుపు_shared_ptr());
    తిరిగి కణుపు && కణుపు->get_element_type().is_integral_సంఖ్య() && కణుపు->cast_సదిశ<int64_t>() == విలువలు;
}

}  // namespace

Fuse_Moe_మార్గనిర్దేశకం::Fuse_Moe_మార్గనిర్దేశకం() {
    ఉపయోగించి నామాకాశ ov::pass::pattern;
    ఉపయోగించి నామాకాశ ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    స్వయంచాలక logits = wrap_type<v0::MatMul>();
    స్వయంచాలక softmax = wrap_type<v8::Softmax>({logits});
    స్వయంచాలక topk = wrap_type<v11::శిఖరం_K>({softmax, any_ఆదానం()});
    topk->సమితి_output_కొలత(2);
    స్వయంచాలక ids = wrap_type<v8::ముక్క>({topk->output(1), any_ఆదానం(), any_ఆదానం(), any_ఆదానం(), any_ఆదానం()});
    స్వయంచాలక ids_2d = wrap_type<v0::Squeeze>({ids, any_ఆదానం()});
    స్వయంచాలక probs = wrap_type<v1::Reshape>({softmax, any_ఆదానం()});
    స్వయంచాలక దత్తాంశం = wrap_type<v0::Squeeze>({probs, any_ఆదానం()});
    స్వయంచాలక దత్తాంశం_batch = wrap_type<v8::Gather>({wrap_type<v3::Shape_యొక్క>({దత్తాంశం}), any_ఆదానం(), any_ఆదానం()});
    స్వయంచాలక ids_లెక్క = wrap_type<v8::Gather>({wrap_type<v3::Shape_యొక్క>({ids_2d}), any_ఆదానం(), any_ఆదానం()});
    స్వయంచాలక లక్ష్యం = wrap_type<v0::Concat>({దత్తాంశం_batch, ids_లెక్క});
    స్వయంచాలక ప్రసారం_2 = wrap_type<v3::ప్రసారం>({ids_2d, లక్ష్యం});
    స్వయంచాలక gather = wrap_type<v8::Gather>({దత్తాంశం, ప్రసారం_2, any_ఆదానం()});
    స్వయంచాలక weights_4d = wrap_type<v0::Unsqueeze>({gather, any_ఆదానం()});
    స్వయంచాలక weights = wrap_type<v1::Reshape>({weights_4d, any_ఆదానం()});
    స్వయంచాలక కూడిక = wrap_type<v1::Reduce_కూడిక>({weights, any_ఆదానం()});
    స్వయంచాలక clamp = wrap_type<v0::Clamp>({కూడిక});
    స్వయంచాలక tile = wrap_type<v0::Tile>({clamp, any_ఆదానం()});
    స్వయంచాలక norm = wrap_type<v1::Divide>({weights, tile});

    స్థిరరాశి స్వయంచాలక callback = [=](Matcher & m) {
        స్థిరరాశి స్వయంచాలక & pm = m.get_నమూనా_విలువ_map();
        స్థిరరాశి స్వయంచాలక కణుపు = [&](స్థిరరాశి std::shared_ptr<ov::Node> & p) { తిరిగి pm.at(p).get_కణుపు_shared_ptr(); };
        స్థిరరాశి స్వయంచాలక ఆదానం_is = [&](స్థిరరాశి std::shared_ptr<ov::Node> & p, size_t i, స్థిరరాశి std::vector<int64_t> & v) {
            తిరిగి constant_is(కణుపు(p)->ఆదానం_విలువ(i), v);
        };
        స్వయంచాలక mm = ov::as_type_ptr<v0::MatMul>(కణుపు(logits));
        స్వయంచాలక sm = ov::as_type_ptr<v8::Softmax>(కణుపు(softmax));
        స్వయంచాలక tk = ov::as_type_ptr<v11::శిఖరం_K>(కణుపు(topk));
        స్వయంచాలక reduce = ov::as_type_ptr<v1::Reduce_కూడిక>(కణుపు(కూడిక));
        స్వయంచాలక హద్దు = ov::as_type_ptr<v0::Clamp>(కణుపు(clamp));
        స్థిరరాశి స్వయంచాలక shape = mm->get_output_partial_shape(0);
        ఒకవేళ (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_ఆదానం_partial_shape(0).rank() != 4 ||
            mm->get_ఆదానం_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_బహుళకం() != v11::శిఖరం_K::బహుళకం_2::MAX ||
            tk->get_క్రమీకరించు_type() != v11::శిఖరం_K::క్రమీకరించు_Type::క్రమీకరించు_విలువలు || tk->get_stable() ||
            tk->get_సూచీ_element_type() != ov::element::i32 ||
            !tk->output(0).get_లక్ష్యం_inputs().ఖాళీలు()) {
            తిరిగి అబద్ధం;
        }
        స్థిరరాశి int64_t experts = shape[3].get_నిడివి();
        స్వయంచాలక end = ov::as_type_ptr<v0::Constant>(కణుపు(ids)->get_ఆదానం_కణుపు_shared_ptr(2));
        ఒకవేళ (!end || !end->get_element_type().is_integral_సంఖ్య() || ov::shape_కొలత(end->get_shape()) != 1) {
            తిరిగి అబద్ధం;
        }
        స్థిరరాశి int64_t k = end->cast_సదిశ<int64_t>()[0];
        స్థిరరాశి స్వయంచాలక sorted_shape = tk->get_output_partial_shape(1);
        ఒకవేళ (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_నిడివి() < k ||
            !ఆదానం_is(probs, 1, {1, -1, experts, 1}) ||
            !ఆదానం_is(దత్తాంశం, 1, {0}) || !ఆదానం_is(ids_2d, 1, {0, 1}) ||
            !ఆదానం_is(weights_4d, 1, {0}) || !ఆదానం_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(కణుపు(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(కణుపు(weights))->get_special_zero() ||
            !ఆదానం_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(కణుపు(gather))->get_batch_dims() != 1 ||
            !ఆదానం_is(దత్తాంశం_batch, 1, {0}) || !ఆదానం_is(దత్తాంశం_batch, 2, {0}) ||
            !ఆదానం_is(ids_లెక్క, 1, {1}) || !ఆదానం_is(ids_లెక్క, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(కణుపు(దత్తాంశం_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(కణుపు(ids_లెక్క))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(కణుపు(లక్ష్యం))->get_axis() != 0 ||
            ov::as_type_ptr<v3::ప్రసారం>(కణుపు(ప్రసారం_2))->get_ప్రసారం_spec().m_type != ov::op::ప్రసారం_Type::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!ఆదానం_is(కూడిక, 1, {-1}) && !ఆదానం_is(కూడిక, 1, {3})) ||
            !ఆదానం_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(కణుపు(norm))->get_autob().m_type != ov::op::Auto_ప్రసారం_Type::NUMPY) {
            తిరిగి అబద్ధం;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        ఒకవేళ (!(హద్దు->get_min() <= 0.5 * ద్విగుణ(k) / experts && హద్దు->get_max() >= 2.0)) {
            తిరిగి అబద్ధం;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        కొరకు (స్థిరరాశి స్వయంచాలక & ఆదానం : tk->output(1).get_లక్ష్యం_inputs()) {
            స్వయంచాలక ముక్క_2 = ov::as_type_ptr<v8::ముక్క>(ఆదానం.get_కణుపు()->shared_from_this());
            ఒకవేళ (!ముక్క_2 || ఆదానం.get_సూచీ() != 0 || ముక్క_2->get_ఆదానం_కొలత() != 5 ||
                !constant_is(ముక్క_2->ఆదానం_విలువ(1), {0}) || !constant_is(ముక్క_2->ఆదానం_విలువ(2), {k}) ||
                !constant_is(ముక్క_2->ఆదానం_విలువ(3), {1}) ||
                (!constant_is(ముక్క_2->ఆదానం_విలువ(4), {3}) && !constant_is(ముక్క_2->ఆదానం_విలువ(4), {-1}))) {
                తిరిగి అబద్ధం;
            }
            slices.push_back(ముక్క_2);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        స్వయంచాలక axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        స్వయంచాలక hidden = std::make_shared<v0::Squeeze>(mm->ఆదానం_విలువ(0), axis0);
        స్వయంచాలక routing = std::make_shared<v0::MatMul>(hidden, mm->ఆదానం_విలువ(1), అబద్ధం, mm->get_transpose_b());
        స్వయంచాలక probabilities = std::make_shared<v8::Softmax>(routing, -1);
        స్వయంచాలక selected = std::make_shared<v11::శిఖరం_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::శిఖరం_K::బహుళకం_2::MAX, v11::శిఖరం_K::క్రమీకరించు_Type::క్రమీకరించు_విలువలు, tk->get_సూచీ_element_type(), అబద్ధం);
        స్వయంచాలక మొత్తం = std::make_shared<v1::Reduce_కూడిక>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), నిజం);
        స్వయంచాలక normalized = std::make_shared<v1::Divide>(selected->output(0), మొత్తం);
        స్వయంచాలక weights_బహిర్గతం = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        స్వయంచాలక ids_బహిర్గతం = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, కణుపు(norm)}, {hidden, routing, probabilities, selected, మొత్తం, normalized, weights_బహిర్గతం, ids_బహిర్గతం});
        ov::replace_కణుపు(కణుపు(norm), weights_బహిర్గతం);
        కొరకు (స్థిరరాశి స్వయంచాలక & ముక్క_2 : slices) {
            ov::replace_కణుపు(ముక్క_2, ids_బహిర్గతం);
        }
        తిరిగి నిజం;
    };
    రిజిస్టరు_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

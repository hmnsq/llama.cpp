#ضمن "fuse_moe_compressed.h"

#ضمن <cstring>
#ضمن <limits>
#ضمن <set>
#ضمن <memory>
#ضمن <openvino/core/graph_util.hpp>
#ضمن <openvino/core/rt_info.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/gelu.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/reduce_sum.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/squeeze.hpp>
#ضمن <openvino/op/subtract.hpp>
#ضمن <openvino/op/swish.hpp>
#ضمن <openvino/op/transpose.hpp>
#ضمن <openvino/op/unsqueeze.hpp>
#ضمن <openvino/pass/constant_folding.hpp>
#ضمن <openvino/pass/pattern/op/optional.hpp>
#ضمن <openvino/pass/pattern/op/wrap_type.hpp>

#ضمن "../op/gather_matmul.hpp"
#ضمن "../op/moe_compressed.hpp"

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

نطاق {

بنية dequant_inputs {
    ov::Output<ov::Node> وزن;
    ov::Output<ov::Node> scale;
    ov::Output<ov::Node> zp;
    منطقي has_zp = خطأ;
    منطقي ok = خطأ;
};

// Peel the chain built by make_int4_weights/make_int8_weights back to its Constant inputs.
// Grouped weights keep the pre-Reshape rank-4 form [n_expert, n, k/group, group] with scale
// and zp at [n_expert, n, k/group, 1], which is the layout MOECompressed expects. Channel-wise
// weights stay rank-3 with a rank-3 scale and carry no zp.
dequant_inputs unwrap_dequant(ثابت ov::Output<ov::Node> & b) {
    dequant_inputs res;

    خودکار گره = b.get_گره_shared_ptr();
    طالما (ov::is_type<ov::op::v0::Convert>(گره) || ov::is_type<ov::op::v1::Reshape>(گره)) {
        گره = گره->get_درون_داد_گره_shared_ptr(0);
    }

    خودکار mul = ov::as_type_ptr<ov::op::v1::Multiply>(گره);
    إذا (!mul) {
        أرجع res;
    }
    res.scale = mul->درون_داد_مقدار(1);

    خودکار lhs = mul->get_درون_داد_گره_shared_ptr(0);
    إذا (خودکار sub = ov::as_type_ptr<ov::op::v1::تفریق>(lhs)) {
        // Take the zero point down to its Constant: an integer zp is wrapped in a Convert to f16,
        // and the op wants the integer form. A natively quantized expert instead carries an exact
        // f16 zp (-min/scale) with no integer behind it, which the MoE kernel does not accept.
        خودکار zp_گره = sub->get_درون_داد_گره_shared_ptr(1);
        طالما (ov::is_type<ov::op::v0::Convert>(zp_گره)) {
            zp_گره = zp_گره->get_درون_داد_گره_shared_ptr(0);
        }
        res.zp = zp_گره->output(0);
        res.has_zp = درست;
        lhs = sub->get_درون_داد_گره_shared_ptr(0);
    }
    طالما (ov::is_type<ov::op::v0::Convert>(lhs)) {
        lhs = lhs->get_درون_داد_گره_shared_ptr(0);
    }
    إذا (!ov::is_type<ov::op::v0::Constant>(lhs)) {
        أرجع res;
    }

    res.وزن = lhs->output(0);
    res.ok = res.scale.get_partial_shape().is_static() && res.وزن.get_partial_shape().is_static();
    أرجع res;
}

size_t logical_k(ثابت ov::Shape & shape) {
    أرجع shape.size() == 4 ? shape[2] * shape[3] : shape.back();
}

// Slice a [n_expert, m, ...] compressed weight/scale/zp Constant along axis 1 (m), [begin, end).
// Built by copying raw bytes directly instead of a graph Slice op: CommonOptimizations rewrites
// v8::Slice into v1::StridedSlice for constant folding regardless of disable_constant_folding
// (the rewrite doesn't carry the marking over), and that reference evaluator crashes on
// sub-byte (u4/i4) element types. Each axis-1 "row" (the trailing dims) is confirmed
// byte-aligned here -- weight/scale/zp trailing sizes are always whole groups -- so a
// byte-range memcpy per row is exact for both regular and sub-byte element types.
ov::Output<ov::Node> برش_experts_dim1(ثابت ov::Output<ov::Node> & tensor, int64_t begin, int64_t end) {
    خودکار constant = ov::as_type_ptr<ov::op::v0::Constant>(tensor.get_گره_shared_ptr());
    OPENVINO_ASSERT(constant, "slice_experts_dim1 expects a Constant input");

    ثابت خودکار & shape = constant->get_shape();
    OPENVINO_ASSERT(shape.size() >= 2, "slice_experts_dim1 expects rank >= 2");
    ثابت خودکار & type = constant->get_element_type();

    size_t ردیف_elems = 1;
    لكل (size_t i = 2; i < shape.size(); ++i) {
        ردیف_elems *= shape[i];
    }
    ثابت size_t ردیف_بیتها = ردیف_elems * type.bitwidth();
    OPENVINO_ASSERT(ردیف_بیتها % 8 == 0, "slice_experts_dim1: row is not byte-aligned");
    ثابت size_t ردیف_بایتها = ردیف_بیتها / 8;

    ov::Shape بیرون_shape = shape;
    بیرون_shape[1] = static_cast<size_t>(end - begin);

    std::vector<uint8_t> بیرون_داده(shape[0] * static_cast<size_t>(end - begin) * ردیف_بایتها);
    ثابت خودکار * src = static_cast<ثابت uint8_t *>(constant->get_داده_ptr());
    ثابت size_t src_ردیف_بایتها = shape[1] * ردیف_بایتها;
    لكل (size_t e = 0; e < shape[0]; ++e) {
        std::کپی_حافظه(بیرون_داده.داده() + e * static_cast<size_t>(end - begin) * ردیف_بایتها,
                   src + e * src_ردیف_بایتها + static_cast<size_t>(begin) * ردیف_بایتها,
                   static_cast<size_t>(end - begin) * ردیف_بایتها);
    }

    أرجع std::make_shared<ov::op::v0::Constant>(type, بیرون_shape, بیرون_داده.داده());
}

}  // namespace

FuseMoeCompressed::FuseMoeCompressed() {
    استفاده نطاق ov::pass::pattern;

    // The gate and up projections each get their own Reshape/Transpose of the hidden state and
    // their own Reshape of the routing ids, so every branch needs its own sub-pattern. On GPU
    // mul_mat_id also converts the activations to f16 before the op and back to f32 after it,
    // so those Converts are matched as optional.
    خودکار hidden_gate_m = any_درون_داد();
    خودکار a_gate_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_gate_m, any_درون_داد() });
    خودکار a_gate_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_gate_reshape_m }), any_درون_داد() });
    خودکار hidden_به_بالا_m = any_درون_داد();
    خودکار a_به_بالا_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ hidden_به_بالا_m, any_درون_داد() }) }),
          any_درون_داد() });

    خودکار gate_w_m = any_درون_داد();
    خودکار به_بالا_w_m = any_درون_داد();
    خودکار پایین_w_m = any_درون_داد();
    خودکار ids_gate_m = any_درون_داد();
    خودکار ids_به_بالا_m = any_درون_داد();
    خودکار ids_پایین_m = any_درون_داد();

    خودکار bgm_gate_m = wrap_type<ov::op::internal::GatherMatmul>({ a_gate_m, gate_w_m, ids_gate_m, any_درون_داد() });
    خودکار gate_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_gate_m, any_درون_داد() }), any_درون_داد() }) });

    خودکار silu_m = wrap_type<ov::op::v4::Swish>({ gate_u_m });

    خودکار bgm_به_بالا_m = wrap_type<ov::op::internal::GatherMatmul>({ a_به_بالا_m, به_بالا_w_m, ids_به_بالا_m, any_درون_داد() });
    خودکار به_بالا_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_به_بالا_m, any_درون_داد() }), any_درون_داد() }) });
    خودکار swiglu_m = wrap_type<ov::op::v1::Multiply>({ silu_m, به_بالا_u_m });

    خودکار d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ swiglu_m, any_درون_داد() }) }),
          any_درون_داد() });
    خودکار bgm_پایین_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, پایین_w_m, ids_پایین_m, any_درون_داد() });
    خودکار پایین_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_پایین_m, any_درون_داد() }), any_درون_داد() }) });

    خودکار routing_m = any_درون_داد();
    خودکار weighted_m = wrap_type<ov::op::v1::Multiply>({ پایین_u_m, routing_m });
    خودکار ریشه_m = wrap_type<ov::op::v1::Reduce_جمع>({ weighted_m, any_درون_داد() });

    ثابت خودکار callback = [=](Matcher & m) {
        خودکار & pm = m.get_الگو_مقدار_map();

        ثابت خودکار gate = unwrap_dequant(pm.at(gate_w_m));
        ثابت خودکار به_بالا = unwrap_dequant(pm.at(به_بالا_w_m));
        ثابت خودکار پایین = unwrap_dequant(pm.at(پایین_w_m));
        إذا (!gate.ok || !به_بالا.ok || !پایین.ok) {
            أرجع خطأ;
        }

        ثابت خودکار gate_shape = gate.وزن.get_shape();
        ثابت خودکار به_بالا_shape = به_بالا.وزن.get_shape();
        ثابت خودکار پایین_shape = پایین.وزن.get_shape();
        إذا (gate_shape != به_بالا_shape || gate_shape.size() < 3 || پایین_shape.size() < 3) {
            أرجع خطأ;
        }

        // MOECompressed carries one group_size and one has_zp for all three projections, so a
        // model whose down-proj is quantized differently from gate/up cannot be described. This
        // happens when ggml requantizes Q5_K/Q6_K experts to channel-wise int8.
        إذا (gate.has_zp != پایین.has_zp || gate_shape.size() != پایین_shape.size()) {
            أرجع خطأ;
        }

        // The kernel only takes an integer zero point (moe_3gemm_swiglu_opt validate_impl).
        إذا (gate.has_zp) {
            ساكن ثابت std::set<ov::element::Type> int_zp_types = { ov::element::u4, ov::element::i4,
                                                                     ov::element::u8, ov::element::i8 };
            إذا (int_zp_types.count(gate.zp.get_element_type()) == 0 ||
                int_zp_types.count(پایین.zp.get_element_type()) == 0) {
                أرجع خطأ;
            }
        }

        // Config holds a single group_size for all three projections.
        ثابت خودکار گروه_از = [](ثابت dequant_inputs & w) {
            ثابت خودکار s = w.وزن.get_shape();
            أرجع s.size() == 4 ? s[3] : logical_k(s);
        };
        إذا (گروه_از(gate) != گروه_از(به_بالا) || گروه_از(gate) != گروه_از(پایین)) {
            أرجع خطأ;
        }

        // all three branches must route the same hidden state through the same experts
        إذا (pm.at(hidden_gate_m) != pm.at(hidden_به_بالا_m)) {
            أرجع خطأ;
        }

        خودکار ids = pm.at(ids_پایین_m);
        ثابت خودکار ids_pshape = ids.get_partial_shape();
        إذا (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_درازی() - 1].is_dynamic()) {
            أرجع خطأ;
        }
        ثابت size_t بالاترین_k = ids_pshape[ids_pshape.rank().get_درازی() - 1].get_درازی();

        // routing weights arrive as [1, n_tokens, top_k, 1]; the op wants [..., top_k]
        خودکار routing = pm.at(routing_m);
        ثابت خودکار routing_pshape = routing.get_partial_shape();
        إذا (routing_pshape.rank().is_dynamic() || routing_pshape.rank().get_درازی() != 4 ||
            routing_pshape[3] != 1) {
            أرجع خطأ;
        }
        // MOE requires routing weights and ids to have the same shape. Drop the trailing 1 of the
        // routing weights and give the ids the leading batch dim, so both become [1, n_tokens, top_k].
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        إذا (ids_pshape.rank().get_درازی() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        إذا (routing.get_partial_shape() != ids.get_partial_shape()) {
            أرجع خطأ;
        }

        ثابت size_t پایین_k = logical_k(پایین_shape);
        ثابت خودکار پایین_scale_shape = پایین.scale.get_shape();
        ثابت size_t پایین_groups = پایین_scale_shape.size() >= 3 ? پایین_scale_shape[2] : 1;

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::SWIGLU;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_اندازه = logical_k(gate_shape);
        config.inter_اندازه = gate_shape[1];
        config.رقم_expert = gate_shape[0];
        config.رقم_shared_expert = 0;
        config.بالاترین_k = بالاترین_k;
        config.گروه_اندازه = پایین_groups <= 1 ? std::numeric_limits<size_t>::max() : پایین_k / پایین_groups;
        config.has_batch_dim = درست;
        config.has_zp = gate.has_zp;
        // dynamic makes the output follow the hidden state, so the plugin can lower this region
        // to f16 together with the rest of the graph
        config.بیرون_type = ov::element::dynamic;

        خودکار absent_zp = [] {
            خودکار zp = std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{ 0 });
            ov::pass::disable_constant_folding(zp);
            أرجع zp->output(0);
        };

        // MOE takes its output type from the hidden state. Transpose the activations before the
        // f16 Convert that mul_mat_id adds on GPU, so the op stays f32 like the block it replaces
        // and the plugin can lower the whole region uniformly.
        ثابت خودکار a_transpose = pm.at(a_gate_m).get_گره_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_gate_reshape_m), a_transpose->درون_داد_مقدار(1));

        ثابت ov::Output_وکتور آرگومانها_2 = {
            hidden, routing, ids,
            gate.وزن,  gate.scale,   gate.has_zp ? gate.zp : absent_zp(),
            به_بالا.وزن,    به_بالا.scale,     به_بالا.has_zp ? به_بالا.zp : absent_zp(),
            پایین.وزن,  پایین.scale,   پایین.has_zp ? پایین.zp : absent_zp(),
        };

        خودکار moe = std::make_shared<ov::op::internal::MOECompressed>(آرگومانها_2, config);

        // MOE takes its output type from the hidden state, which is f16 on GPU, while the rest of
        // the ggml graph works in f32.
        ov::Output<ov::Node> نتیجه = moe->output(0);
        ثابت خودکار ریشه_type = m.get_تطابق_ریشه()->get_output_element_type(0);
        إذا (نتیجه.get_element_type() != ریشه_type) {
            نتیجه = std::make_shared<ov::op::v0::Convert>(نتیجه, ریشه_type);
        }

        نتیجه.get_گره_shared_ptr()->مجموعه_friendly_name(m.get_تطابق_ریشه()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), نتیجه.get_گره_shared_ptr());
        ov::replace_گره(m.get_تطابق_ریشه(), نتیجه.get_گره_shared_ptr());
        ثبات_new_گره(moe);
        أرجع درست;
    };

    ثبات_matcher(std::make_shared<Matcher>(ریشه_m, "ov::frontend::ggml::pass::FuseMoeCompressed"), callback);
}

Fuse_Moe_Compressed_Fused_Gate_به_بالا::Fuse_Moe_Compressed_Fused_Gate_به_بالا() {
    استفاده نطاق ov::pass::pattern;

    // A single GatherMatmul computes the fused gate+up projection; the gate/up split happens
    // AFTER the GEMM via two Slice ops on the last (feature) axis (see get_glu_inputs in
    // op/glu_geglu.cpp), unlike FuseMoeCompressed's two-separate-GatherMatmul models.
    خودکار hidden_m = any_درون_داد();
    خودکار a_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_m, any_درون_داد() });
    خودکار a_m = wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_reshape_m }), any_درون_داد() });

    خودکار gate_به_بالا_w_m = any_درون_داد();
    خودکار ids_gate_به_بالا_m = any_درون_داد();
    خودکار bgm_fused_m = wrap_type<ov::op::internal::GatherMatmul>({ a_m, gate_به_بالا_w_m, ids_gate_به_بالا_m, any_درون_داد() });
    // Two graph shapes reach here. The rank-4 (stateless) graph restores the batch dim with an
    // Unsqueeze after the Transpose and may Convert afterwards:
    //     GatherMatmul -> Transpose -> Unsqueeze -> [Convert] -> Slice
    // The rank-3 (stateful) graph never drops to a batch dim at all, so there is no Unsqueeze
    // here - it appears later, between the GEGLU and the down Reshape - and the Convert sits on
    // the other side of the Transpose:
    //     GatherMatmul -> Convert -> Transpose -> Slice
    // optional<T>({a, b}) matches T(a, b) or bare a, so one pattern covers both.
    خودکار gu_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_fused_m }), any_درون_داد() });
    خودکار gu_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ gu_t_m, any_درون_داد() }) });

    خودکار gate_برش_m = wrap_type<ov::op::v8::برش>({ gu_u_m, any_درون_داد(), any_درون_داد(), any_درون_داد(), any_درون_داد() });
    خودکار به_بالا_برش_m = wrap_type<ov::op::v8::برش>({ gu_u_m, any_درون_داد(), any_درون_داد(), any_درون_داد(), any_درون_داد() });
    خودکار gelu_m = wrap_type<ov::op::v7::Gelu>({ gate_برش_m });
    خودکار geglu_m = wrap_type<ov::op::v1::Multiply>({ gelu_m, به_بالا_برش_m });

    // The rank-3 graph inserts the Unsqueeze the gate/up branch lacked right here, before the
    // Reshape that feeds the down projection; the rank-4 graph goes straight from the GEGLU
    // into the Reshape.
    خودکار d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>(
              { optional<ov::op::v0::Unsqueeze>({ geglu_m, any_درون_داد() }), any_درون_داد() }) }),
          any_درون_داد() });
    خودکار پایین_w_m = any_درون_داد();
    خودکار ids_پایین_m = any_درون_داد();
    خودکار bgm_پایین_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, پایین_w_m, ids_پایین_m, any_درون_داد() });
    // Same two shapes as the gate/up branch above.
    خودکار پایین_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_پایین_m }), any_درون_داد() });
    خودکار پایین_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ پایین_t_m, any_درون_داد() }) });

    // gemma-4 applies an extra per-expert output scale to the down projection before the
    // router-weight multiply (llama-graph.cpp's ffn_down_exps.scale); FuseMoeCompressed's
    // Qwen-shaped models have no such scale, so this node is specific to this pattern.
    خودکار پایین_scale_m = any_درون_داد();
    خودکار پایین_scaled_m = wrap_type<ov::op::v1::Multiply>({ پایین_u_m, پایین_scale_m });

    خودکار routing_m = any_درون_داد();
    خودکار weighted_m = wrap_type<ov::op::v1::Multiply>({ پایین_scaled_m, routing_m });
    خودکار ریشه_m = wrap_type<ov::op::v1::Reduce_جمع>({ weighted_m, any_درون_داد() });

    ثابت خودکار callback = [=](Matcher & m) {
        خودکار & pm = m.get_الگو_مقدار_map();

        ثابت خودکار gate_به_بالا = unwrap_dequant(pm.at(gate_به_بالا_w_m));
        ثابت خودکار پایین = unwrap_dequant(pm.at(پایین_w_m));
        إذا (!gate_به_بالا.ok || !پایین.ok) {
            أرجع خطأ;
        }

        // oneDNN's weight-decompression GEMM needs a real zero point, so a symmetric (no zp)
        // projection cannot be fused here -- e.g. the GPU-only Q4_0 fallback in
        // ggml_openvino_get_extracted_layout for grouped 8-bit expert weights.
        إذا (!gate_به_بالا.has_zp || !پایین.has_zp) {
            أرجع خطأ;
        }

        ثابت خودکار gate_به_بالا_shape = gate_به_بالا.وزن.get_shape();
        ثابت خودکار پایین_shape = پایین.وزن.get_shape();
        إذا (gate_به_بالا_shape.size() < 3 || پایین_shape.size() < 3 || gate_به_بالا_shape.size() != پایین_shape.size()) {
            أرجع خطأ;
        }

        // The op reads the zero point straight off a weight port, so it must already be an
        // integer Constant. Requantized experts (GGML_OPENVINO_REQUANT_KQUANT=q4_asym64_all)
        // are; a native Q4_K expert keeps an exact f16 zp and is left to the unfused path
        // rather than rounded here -- rounding would put a live node on that port and the
        // expert GEMM would read the wrong zero point.
        ساكن ثابت std::set<ov::element::Type> int_types = { ov::element::u4, ov::element::i4,
                                                               ov::element::u8, ov::element::i8 };
        إذا (int_types.count(gate_به_بالا.وزن.get_element_type()) == 0 ||
            int_types.count(پایین.وزن.get_element_type()) == 0 ||
            int_types.count(gate_به_بالا.zp.get_element_type()) == 0 ||
            int_types.count(پایین.zp.get_element_type()) == 0) {
            أرجع خطأ;
        }

        ثابت خودکار گروه_از = [](ثابت dequant_inputs & w) {
            ثابت خودکار s = w.وزن.get_shape();
            أرجع s.size() == 4 ? s[3] : logical_k(s);
        };
        إذا (گروه_از(gate_به_بالا) != گروه_از(پایین)) {
            أرجع خطأ;
        }

        خودکار gelu_گره = ov::as_type_ptr<ov::op::v7::Gelu>(pm.at(gelu_m).get_گره_shared_ptr());
        إذا (!gelu_گره || gelu_گره->get_approximation_نما() != ov::op::Gelu_Approximation_نما::ERF) {
            أرجع خطأ;
        }

        // Read the actual gate/up split point from the matched Slice nodes instead of assuming
        // inter_size/2, so this stays correct if the model's ffn_dim convention ever changes.
        خودکار gate_برش = ov::as_type_ptr<ov::op::v8::برش>(pm.at(gate_برش_m).get_گره_shared_ptr());
        خودکار به_بالا_برش = ov::as_type_ptr<ov::op::v8::برش>(pm.at(به_بالا_برش_m).get_گره_shared_ptr());
        خودکار gate_انتها_c = ov::as_type_ptr<ov::op::v0::Constant>(gate_برش->درون_داد_مقدار(2).get_گره_shared_ptr());
        خودکار به_بالا_begin_c = ov::as_type_ptr<ov::op::v0::Constant>(به_بالا_برش->درون_داد_مقدار(1).get_گره_shared_ptr());
        إذا (!gate_انتها_c || !به_بالا_begin_c) {
            أرجع خطأ;
        }
        ثابت خودکار gate_انتها = gate_انتها_c->cast_وکتور<int64_t>().at(0);
        ثابت خودکار به_بالا_begin = به_بالا_begin_c->cast_وکتور<int64_t>().at(0);
        إذا (gate_انتها != به_بالا_begin || gate_انتها <= 0 || gate_انتها >= static_cast<int64_t>(gate_به_بالا_shape[1])) {
            أرجع خطأ;
        }
        ثابت int64_t mid = gate_انتها;
        ثابت int64_t پر = static_cast<int64_t>(gate_به_بالا_shape[1]);

        خودکار ids = pm.at(ids_پایین_m);
        ثابت خودکار ids_pshape = ids.get_partial_shape();
        إذا (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_درازی() - 1].is_dynamic()) {
            أرجع خطأ;
        }
        ثابت size_t بالاترین_k = ids_pshape[ids_pshape.rank().get_درازی() - 1].get_درازی();

        // Routing weights arrive as [1, n_tokens, top_k, 1] on the rank-4 (stateless) graph and
        // as [n_tokens, top_k, 1] on the rank-3 (stateful) one, which is the same thing without
        // the leading batch dim. Normalise the rank-3 form up to the rank-4 one so everything
        // below - and the op's own config - stays in the shape that is already validated on the
        // stateless path; the batch dim is taken back off the result at the end.
        ثابت خودکار routing_درون = pm.at(routing_m);
        ثابت خودکار routing_pshape = routing_درون.get_partial_shape();
        ثابت خودکار routing_rank = routing_pshape.rank();
        إذا (routing_rank.is_dynamic() || (routing_rank.get_درازی() != 4 && routing_rank.get_درازی() != 3) ||
            routing_pshape[routing_rank.get_درازی() - 1] != 1) {
            أرجع خطأ;
        }
        ثابت منطقي batchless = routing_rank.get_درازی() == 3;

        // Fold gemma-4's per-expert output scale into the routing weights: the reduction is
        // sum_e(routing[e] * scale[e] * down_out[e]), and MOECompressed only takes one
        // per-expert weight, so pre-multiply it into routing here (same [.., top_k, 1] shape).
        خودکار پایین_scale = pm.at(پایین_scale_m);
        إذا (پایین_scale.get_partial_shape() != routing_pshape) {
            أرجع خطأ;
        }
        ov::Output<ov::Node> routing = std::make_shared<ov::op::v1::Multiply>(routing_درون, پایین_scale);
        إذا (batchless) {
            routing = std::make_shared<ov::op::v0::Unsqueeze>(
                routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        إذا (ids_pshape.rank().get_درازی() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        إذا (routing.get_partial_shape() != ids.get_partial_shape()) {
            أرجع خطأ;
        }

        ثابت size_t پایین_k = logical_k(پایین_shape);
        ثابت خودکار پایین_scale_shape = پایین.scale.get_shape();
        ثابت size_t پایین_groups = پایین_scale_shape.size() >= 3 ? پایین_scale_shape[2] : 1;

        // Split the fused gate_up weight/scale/zp on the output axis. slice_experts_dim1 copies
        // raw bytes out of the Constant, so each half stays a Constant -- which is what the op
        // needs on its weight ports.
        ثابت ov::Output<ov::Node> gate_وزن = برش_experts_dim1(gate_به_بالا.وزن, 0, mid);
        ثابت ov::Output<ov::Node> به_بالا_وزن = برش_experts_dim1(gate_به_بالا.وزن, mid, پر);
        ثابت ov::Output<ov::Node> gate_scale = برش_experts_dim1(gate_به_بالا.scale, 0, mid);
        ثابت ov::Output<ov::Node> به_بالا_scale = برش_experts_dim1(gate_به_بالا.scale, mid, پر);
        ثابت ov::Output<ov::Node> gate_zp = برش_experts_dim1(gate_به_بالا.zp, 0, mid);
        ثابت ov::Output<ov::Node> به_بالا_zp = برش_experts_dim1(gate_به_بالا.zp, mid, پر);

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::GEGLU_ERF;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_اندازه = logical_k(gate_به_بالا_shape);
        config.inter_اندازه = static_cast<size_t>(mid);
        config.رقم_expert = gate_به_بالا_shape[0];
        config.رقم_shared_expert = 0;
        config.بالاترین_k = بالاترین_k;
        config.گروه_اندازه = پایین_groups <= 1 ? std::numeric_limits<size_t>::max() : پایین_k / پایین_groups;
        config.has_batch_dim = درست;
        config.has_zp = درست;
        config.بیرون_type = ov::element::dynamic;

        // Rebuild the activations Transpose before mul_mat_id's f16 GPU Convert, so the op
        // stays f32 like the block it replaces (mirrors FuseMoeCompressed).
        ثابت خودکار a_transpose = pm.at(a_m).get_گره_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_reshape_m), a_transpose->درون_داد_مقدار(1));

        // w0 is the activated (gate) lane and w1 the multiplied (up) lane -- see
        // moe_3gemm_swiglu_opt.cpp: "scratch.up = up(x) * silu(gate(x))".
        ثابت ov::Output_وکتور آرگومانها_2 = {
            hidden,      routing,     ids,
            gate_وزن, gate_scale,  gate_zp,
            به_بالا_وزن,   به_بالا_scale,    به_بالا_zp,
            پایین.وزن, پایین.scale,  پایین.zp,
        };

        خودکار moe = std::make_shared<ov::op::internal::MOECompressed>(آرگومانها_2, config);

        ov::Output<ov::Node> نتیجه = moe->output(0);
        // The op was fed the batched form, so it produces [1, n_tokens, hidden]. On the rank-3
        // graph the ReduceSum being replaced is [n_tokens, hidden], so drop the batch dim again.
        إذا (batchless) {
            نتیجه = std::make_shared<ov::op::v0::Squeeze>(
                نتیجه, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        ثابت خودکار ریشه_type = m.get_تطابق_ریشه()->get_output_element_type(0);
        إذا (نتیجه.get_element_type() != ریشه_type) {
            نتیجه = std::make_shared<ov::op::v0::Convert>(نتیجه, ریشه_type);
        }
        إذا (نتیجه.get_partial_shape() != m.get_تطابق_ریشه()->get_output_partial_shape(0)) {
            أرجع خطأ;
        }

        نتیجه.get_گره_shared_ptr()->مجموعه_friendly_name(m.get_تطابق_ریشه()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), نتیجه.get_گره_shared_ptr());
        ov::replace_گره(m.get_تطابق_ریشه(), نتیجه.get_گره_shared_ptr());
        ثبات_new_گره(moe);
        أرجع درست;
    };

    ثبات_matcher(
        std::make_shared<Matcher>(ریشه_m, "ov::frontend::ggml::pass::FuseMoeCompressedFusedGateUp"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

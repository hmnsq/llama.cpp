#അന്തര്ഭാവ "../op_table.h"
#അന്തര്ഭാവ "../utils.h"

#അന്തര്ഭാവ <openvino/op/concat.hpp>
#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/gather.hpp>
#അന്തര്ഭാവ <openvino/op/reshape.hpp>
#അന്തര്ഭാവ <openvino/op/shape_of.hpp>
#അന്തര്ഭാവ <openvino/op/slice.hpp>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ op {

OutputVector translate_view(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);

    എങ്കിൽ (context.get_op_case() == 1) {
        // Static-mode identity pass-through for VIEWs over a GATED_DELTA_NET combined output or
        // the conv_input CONCAT; the consuming op (CPY/RMS_NORM) does its own runtime-correct
        // slicing on the full tensor (see ggml-decoder.cpp compute_op_case, GGML_OP_VIEW).
        തിരികെ {context.get_നിവേശം(0)};
    }

    എങ്കിൽ (!context.is_static()) {
        // On the stateless/non-static path VIEW is normally a no-op (consumers re-slice).
        // EXCEPTION: the MoE expert aggregation slices each expert plane out of
        // ffn_moe_weighted [n_embd, n_expert_used, n_tokens] with ggml_view_2d and then
        // sums the planes with a chain of ADDs (llama-graph.cpp). Those ADDs read this
        // VIEW node directly from the tensor map and do NOT re-slice, so a no-op here
        // makes every plane the full tensor and the expert sum collapses. Materialize the
        // single-expert slice here. Gated by name (ffn_moe_weighted...view) so it can't
        // affect any other view.
        സ്ഥിരം std::string & vname = context.get_name();
        എങ്കിൽ (vname.find("ffn_moe_weighted") != std::string::npos) {
            auto src_ps = context.get_നിവേശം_shape(0);
            auto dst_ps = context.get_output_shape();
            എങ്കിൽ (src_ps.rank().is_static() && dst_ps.rank().is_static() && src_ps.rank() == dst_ps.rank() &&
                src_ps.is_static() && dst_ps.is_static()) {
                auto sst = context.get_നിവേശം_stride(0);
                auto dst = context.get_output_stride();
                size_t voff = context.get_output_op_offset();
                auto ss = src_ps.to_shape();
                auto dd = dst_ps.to_shape();
                സ്ഥിരം size_t nd = ss.size();
                // Stateful models drop the leading size-1 batch dim, so the real OV tensor can
                // be one rank lower than the ggml shape metadata above. Axis indices derived
                // from that metadata must be shifted down by the difference before they are
                // used as OV axes. Bail out if the dims we would drop are not all 1.
                സ്ഥിരം auto in_rank = context.get_നിവേശം(0).get_partial_shape().rank();
                സ്ഥിരം പൂർണ്ണസംഖ്യ axis_shift =
                    in_rank.is_static() ? (പൂർണ്ണസംഖ്യ) nd - (പൂർണ്ണസംഖ്യ) in_rank.get_length() : 0;
                തർക്കം shift_ok = axis_shift >= 0 && (size_t) axis_shift < nd;
                വേണ്ടി (പൂർണ്ണസംഖ്യ a = 0; a < axis_shift && shift_ok; ++a) {
                    shift_ok = (ss[a] == 1 && dd[a] == 1);
                }
                എങ്കിൽ (shift_ok && sst.size() == nd && dst.size() == nd) {
                    // Map each dst axis of size>1 to a src axis with equal (size,stride);
                    // the unmatched src axis of size>1 is the indexed expert axis.
                    // dst_to_src[d] records which src axis each dst axis came from, so we can
                    // later pull the dynamic (token) dim from the right source axis at runtime.
                    std::vector<തർക്കം> used(nd, അസത്യം);
                    std::vector<പൂർണ്ണസംഖ്യ> dst_to_src(nd, -1);
                    തർക്കം ok = സത്യം;
                    വേണ്ടി (size_t d = 0; d < nd; ++d) {
                        എങ്കിൽ (dd[d] == 1) {
                            തുടരുക;
                        }
                        പൂർണ്ണസംഖ്യ found = -1;
                        വേണ്ടി (size_t s = 0; s < nd; ++s) {
                            എങ്കിൽ (!used[s] && ss[s] == dd[d] && sst[s] == dst[d]) {
                                found = (പൂർണ്ണസംഖ്യ) s;
                                നിർത്തുക;
                            }
                        }
                        എങ്കിൽ (found < 0) {
                            ok = അസത്യം;
                            നിർത്തുക;
                        }
                        used[found] = സത്യം;
                        dst_to_src[d] = found;
                    }
                    പൂർണ്ണസംഖ്യ dropped = -1;
                    എങ്കിൽ (ok) {
                        വേണ്ടി (size_t s = 0; s < nd; ++s) {
                            എങ്കിൽ (!used[s] && ss[s] > 1) {
                                എങ്കിൽ (dropped >= 0) {
                                    ok = അസത്യം;
                                    നിർത്തുക;
                                }
                                dropped = (പൂർണ്ണസംഖ്യ) s;
                            }
                        }
                    }
                    എങ്കിൽ (ok && dropped >= 0) {
                        സ്ഥിരം size_t dstr = sst[dropped];
                        സ്ഥിരം int64_t dsz = (int64_t) ss[dropped];
                        എങ്കിൽ (dstr > 0 && voff % dstr == 0) {
                            സ്ഥിരം int64_t sel = (int64_t) (voff / dstr);
                            എങ്കിൽ (sel >= 0 && sel < dsz) {
                                ov::Output<ov::Node> sl = std::make_shared<ov::op::v8::Slice>(
                                    context.get_നിവേശം(0),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel + 1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {dropped - axis_shift}));
                                // Build the reshape target from the (concrete) dst shape, but
                                // keep the dynamic token axis dynamic instead of freezing it
                                // to the captured n_tokens. Without this the constant dst
                                // shape bakes in the prefill token count and the static value
                                // flows downstream, turning every later decoder layer static
                                // (the GPU in-place-concat KV-cache bug). The token axis is
                                // PERMUTED between the sliced input and the dst (e.g. input
                                // [1,tok,expert,emb] -> dst [1,1,tok,emb]), so special_zero
                                // (which copies the same-position dim) is not enough: pull the
                                // dynamic dim from the correct SOURCE axis via ShapeOf+Gather
                                // and place it at the dst token position.
                                സ്ഥിരം int32_t dyn = context.get_op_dynamic_dim();  // output ggml axis, -1 if none
                                // still in ggml metadata axis space; get_shape() reverses ggml order
                                പൂർണ്ണസംഖ്യ dst_ov_axis = (dyn != -1) ? ((പൂർണ്ണസംഖ്യ) nd - 1 - (പൂർണ്ണസംഖ്യ) dyn) : -1;
                                പൂർണ്ണസംഖ്യ src_ov_axis = (dst_ov_axis >= axis_shift && dst_ov_axis < (പൂർണ്ണസംഖ്യ) nd)
                                                      ? dst_to_src[dst_ov_axis]
                                                      : -1;
                                എങ്കിൽ (dst_ov_axis >= 0 && src_ov_axis >= axis_shift) {
                                    // target = concat of per-axis scalars; the token axis is a
                                    // runtime Gather of the slice's shape, the rest are constants.
                                    auto sl_shape = std::make_shared<ov::op::v3::ShapeOf>(sl, ov::element::i64);
                                    auto tok_dim = std::make_shared<ov::op::v8::Gather>(
                                        sl_shape,
                                        ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                     {src_ov_axis - axis_shift}),
                                        ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
                                    ov::OutputVector parts;
                                    വേണ്ടി (പൂർണ്ണസംഖ്യ a = axis_shift; a < (പൂർണ്ണസംഖ്യ) nd; ++a) {
                                        എങ്കിൽ (a == dst_ov_axis) {
                                            parts.push_back(tok_dim);
                                        } അല്ലെങ്കിൽ {
                                            parts.push_back(ov::op::v0::Constant::create(
                                                ov::element::i64, {1}, {(int64_t) dd[a]}));
                                        }
                                    }
                                    auto dc = std::make_shared<ov::op::v0::Concat>(parts, 0);
                                    auto rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, അസത്യം);
                                    തിരികെ rename_outputs_with_suffix({rs}, context.get_name());
                                }
                                std::vector<int64_t> dd_ov(dd.begin() + axis_shift, dd.end());
                                auto dc = ov::op::v0::Constant::create(ov::element::i64, {dd_ov.size()}, dd_ov);
                                auto rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, അസത്യം);
                                തിരികെ rename_outputs_with_suffix({rs}, context.get_name());
                            }
                        }
                    }
                }
            }
        }
        തിരികെ {context.get_നിവേശം(0)};
    }

    auto നിവേശം = context.get_നിവേശം(0);
    auto src_shape = context.get_നിവേശം_shape(0);
    auto dst_shape = context.get_output_shape();

    എങ്കിൽ (src_shape.rank().is_dynamic() || dst_shape.rank().is_dynamic()) {
        തിരികെ {നിവേശം};
    }

    int64_t src_elems = 1;
    int64_t dst_elems = 1;
    വേണ്ടി (int64_t i = 0; i < src_shape.rank().get_length(); ++i) {
        എങ്കിൽ (src_shape[i].is_dynamic()) {
            തിരികെ {നിവേശം};
        }
        src_elems *= src_shape[i].get_length();
    }
    വേണ്ടി (int64_t i = 0; i < dst_shape.rank().get_length(); ++i) {
        എങ്കിൽ (dst_shape[i].is_dynamic()) {
            തിരികെ {നിവേശം};
        }
        dst_elems *= dst_shape[i].get_length();
    }

    എങ്കിൽ (dst_elems >= src_elems) {
        തിരികെ {നിവേശം};
    }

    auto src_stride = context.get_നിവേശം_stride(0);
    auto dst_stride = context.get_output_stride();
    size_t view_offset = context.get_output_op_offset();

    തർക്കം same_stride = (src_stride.size() == dst_stride.size());
    എങ്കിൽ (same_stride) {
        വേണ്ടി (size_t i = 0; i < src_stride.size(); ++i) {
            എങ്കിൽ (src_stride[i] != dst_stride[i]) {
                same_stride = അസത്യം;
                നിർത്തുക;
            }
        }
    }

    എങ്കിൽ (!same_stride) {
        തിരികെ {നിവേശം};
    }

    auto src_ov_shape = src_shape.to_shape();
    auto dst_ov_shape = dst_shape.to_shape();
    size_t ndims = src_ov_shape.size();
    എങ്കിൽ (dst_ov_shape.size() != ndims) {
        തിരികെ {നിവേശം};
    }

    std::vector<പൂർണ്ണസംഖ്യ> diff_dims;
    വേണ്ടി (size_t i = 0; i < ndims; ++i) {
        എങ്കിൽ (src_ov_shape[i] != dst_ov_shape[i]) {
            diff_dims.push_back(static_cast<പൂർണ്ണസംഖ്യ>(i));
        }
    }

    എങ്കിൽ (diff_dims.size() != 1) {
        തിരികെ {നിവേശം};
    }

    പൂർണ്ണസംഖ്യ slice_dim = diff_dims[0];
    int64_t dim_size = static_cast<int64_t>(src_ov_shape[slice_dim]);

    size_t ov_stride_for_dim = 1;
    വേണ്ടി (size_t i = slice_dim + 1; i < ndims; ++i) {
        ov_stride_for_dim *= src_ov_shape[i];
    }
    size_t elem_size = src_stride.back();
    എങ്കിൽ (elem_size == 0) {
        elem_size = 1;
    }

    int64_t begin_val = 0;
    എങ്കിൽ (ov_stride_for_dim > 0 && elem_size > 0) {
        begin_val = static_cast<int64_t>((view_offset / elem_size) / ov_stride_for_dim);
    }
    int64_t end_val = begin_val + static_cast<int64_t>(dst_ov_shape[slice_dim]);

    എങ്കിൽ (begin_val < 0 || end_val > dim_size) {
        തിരികെ {നിവേശം};
    }

    auto sliced =
        std::make_shared<ov::op::v8::Slice>(നിവേശം, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

    sliced->set_friendly_name(context.get_output_name());
    തിരികെ {sliced->output(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

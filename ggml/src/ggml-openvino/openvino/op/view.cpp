#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <openvino/op/concat.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/gather.hpp>
#અન્તર્ભાવ <openvino/op/reshape.hpp>
#અન્તર્ભાવ <openvino/op/shape_of.hpp>
#અન્તર્ભાવ <openvino/op/slice.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_view(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);

    જો (context.get_op_case() == 1) {
        // Static-mode identity pass-through for VIEWs over a GATED_DELTA_NET combined output or
        // the conv_input CONCAT; the consuming op (CPY/RMS_NORM) does its own runtime-correct
        // slicing on the full tensor (see ggml-decoder.cpp compute_op_case, GGML_OP_VIEW).
        પાછા {context.get_આગત(0)};
    }

    જો (!context.is_static()) {
        // On the stateless/non-static path VIEW is normally a no-op (consumers re-slice).
        // EXCEPTION: the MoE expert aggregation slices each expert plane out of
        // ffn_moe_weighted [n_embd, n_expert_used, n_tokens] with ggml_view_2d and then
        // sums the planes with a chain of ADDs (llama-graph.cpp). Those ADDs read this
        // VIEW node directly from the tensor map and do NOT re-slice, so a no-op here
        // makes every plane the full tensor and the expert sum collapses. Materialize the
        // single-expert slice here. Gated by name (ffn_moe_weighted...view) so it can't
        // affect any other view.
        અચલ std::string & vname = context.get_name();
        જો (vname.find("ffn_moe_weighted") != std::string::npos) {
            auto src_ps = context.get_આગત_shape(0);
            auto dst_ps = context.get_output_shape();
            જો (src_ps.rank().is_static() && dst_ps.rank().is_static() && src_ps.rank() == dst_ps.rank() &&
                src_ps.is_static() && dst_ps.is_static()) {
                auto sst = context.get_આગત_stride(0);
                auto dst = context.get_output_stride();
                size_t voff = context.get_output_op_offset();
                auto ss = src_ps.to_shape();
                auto dd = dst_ps.to_shape();
                અચલ size_t nd = ss.size();
                // Stateful models drop the leading size-1 batch dim, so the real OV tensor can
                // be one rank lower than the ggml shape metadata above. Axis indices derived
                // from that metadata must be shifted down by the difference before they are
                // used as OV axes. Bail out if the dims we would drop are not all 1.
                અચલ auto in_rank = context.get_આગત(0).get_partial_shape().rank();
                અચલ પૂર્ણાંક axis_shift =
                    in_rank.is_static() ? (પૂર્ણાંક) nd - (પૂર્ણાંક) in_rank.get_length() : 0;
                તાર્કિક shift_ok = axis_shift >= 0 && (size_t) axis_shift < nd;
                માટે (પૂર્ણાંક a = 0; a < axis_shift && shift_ok; ++a) {
                    shift_ok = (ss[a] == 1 && dd[a] == 1);
                }
                જો (shift_ok && sst.size() == nd && dst.size() == nd) {
                    // Map each dst axis of size>1 to a src axis with equal (size,stride);
                    // the unmatched src axis of size>1 is the indexed expert axis.
                    // dst_to_src[d] records which src axis each dst axis came from, so we can
                    // later pull the dynamic (token) dim from the right source axis at runtime.
                    std::vector<તાર્કિક> used(nd, ખોટું);
                    std::vector<પૂર્ણાંક> dst_to_src(nd, -1);
                    તાર્કિક ok = સાચું;
                    માટે (size_t d = 0; d < nd; ++d) {
                        જો (dd[d] == 1) {
                            ચાલુ;
                        }
                        પૂર્ણાંક found = -1;
                        માટે (size_t s = 0; s < nd; ++s) {
                            જો (!used[s] && ss[s] == dd[d] && sst[s] == dst[d]) {
                                found = (પૂર્ણાંક) s;
                                તોડો;
                            }
                        }
                        જો (found < 0) {
                            ok = ખોટું;
                            તોડો;
                        }
                        used[found] = સાચું;
                        dst_to_src[d] = found;
                    }
                    પૂર્ણાંક dropped = -1;
                    જો (ok) {
                        માટે (size_t s = 0; s < nd; ++s) {
                            જો (!used[s] && ss[s] > 1) {
                                જો (dropped >= 0) {
                                    ok = ખોટું;
                                    તોડો;
                                }
                                dropped = (પૂર્ણાંક) s;
                            }
                        }
                    }
                    જો (ok && dropped >= 0) {
                        અચલ size_t dstr = sst[dropped];
                        અચલ int64_t dsz = (int64_t) ss[dropped];
                        જો (dstr > 0 && voff % dstr == 0) {
                            અચલ int64_t sel = (int64_t) (voff / dstr);
                            જો (sel >= 0 && sel < dsz) {
                                ov::Output<ov::Node> sl = std::make_shared<ov::op::v8::Slice>(
                                    context.get_આગત(0),
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
                                અચલ int32_t dyn = context.get_op_dynamic_dim();  // output ggml axis, -1 if none
                                // still in ggml metadata axis space; get_shape() reverses ggml order
                                પૂર્ણાંક dst_ov_axis = (dyn != -1) ? ((પૂર્ણાંક) nd - 1 - (પૂર્ણાંક) dyn) : -1;
                                પૂર્ણાંક src_ov_axis = (dst_ov_axis >= axis_shift && dst_ov_axis < (પૂર્ણાંક) nd)
                                                      ? dst_to_src[dst_ov_axis]
                                                      : -1;
                                જો (dst_ov_axis >= 0 && src_ov_axis >= axis_shift) {
                                    // target = concat of per-axis scalars; the token axis is a
                                    // runtime Gather of the slice's shape, the rest are constants.
                                    auto sl_shape = std::make_shared<ov::op::v3::ShapeOf>(sl, ov::element::i64);
                                    auto tok_dim = std::make_shared<ov::op::v8::Gather>(
                                        sl_shape,
                                        ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                     {src_ov_axis - axis_shift}),
                                        ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
                                    ov::OutputVector parts;
                                    માટે (પૂર્ણાંક a = axis_shift; a < (પૂર્ણાંક) nd; ++a) {
                                        જો (a == dst_ov_axis) {
                                            parts.push_back(tok_dim);
                                        } નહીંતર {
                                            parts.push_back(ov::op::v0::Constant::create(
                                                ov::element::i64, {1}, {(int64_t) dd[a]}));
                                        }
                                    }
                                    auto dc = std::make_shared<ov::op::v0::Concat>(parts, 0);
                                    auto rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, ખોટું);
                                    પાછા rename_outputs_with_suffix({rs}, context.get_name());
                                }
                                std::vector<int64_t> dd_ov(dd.begin() + axis_shift, dd.end());
                                auto dc = ov::op::v0::Constant::create(ov::element::i64, {dd_ov.size()}, dd_ov);
                                auto rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, ખોટું);
                                પાછા rename_outputs_with_suffix({rs}, context.get_name());
                            }
                        }
                    }
                }
            }
        }
        પાછા {context.get_આગત(0)};
    }

    auto આગત = context.get_આગત(0);
    auto src_shape = context.get_આગત_shape(0);
    auto dst_shape = context.get_output_shape();

    જો (src_shape.rank().is_dynamic() || dst_shape.rank().is_dynamic()) {
        પાછા {આગત};
    }

    int64_t src_elems = 1;
    int64_t dst_elems = 1;
    માટે (int64_t i = 0; i < src_shape.rank().get_length(); ++i) {
        જો (src_shape[i].is_dynamic()) {
            પાછા {આગત};
        }
        src_elems *= src_shape[i].get_length();
    }
    માટે (int64_t i = 0; i < dst_shape.rank().get_length(); ++i) {
        જો (dst_shape[i].is_dynamic()) {
            પાછા {આગત};
        }
        dst_elems *= dst_shape[i].get_length();
    }

    જો (dst_elems >= src_elems) {
        પાછા {આગત};
    }

    auto src_stride = context.get_આગત_stride(0);
    auto dst_stride = context.get_output_stride();
    size_t view_offset = context.get_output_op_offset();

    તાર્કિક same_stride = (src_stride.size() == dst_stride.size());
    જો (same_stride) {
        માટે (size_t i = 0; i < src_stride.size(); ++i) {
            જો (src_stride[i] != dst_stride[i]) {
                same_stride = ખોટું;
                તોડો;
            }
        }
    }

    જો (!same_stride) {
        પાછા {આગત};
    }

    auto src_ov_shape = src_shape.to_shape();
    auto dst_ov_shape = dst_shape.to_shape();
    size_t ndims = src_ov_shape.size();
    જો (dst_ov_shape.size() != ndims) {
        પાછા {આગત};
    }

    std::vector<પૂર્ણાંક> diff_dims;
    માટે (size_t i = 0; i < ndims; ++i) {
        જો (src_ov_shape[i] != dst_ov_shape[i]) {
            diff_dims.push_back(static_cast<પૂર્ણાંક>(i));
        }
    }

    જો (diff_dims.size() != 1) {
        પાછા {આગત};
    }

    પૂર્ણાંક slice_dim = diff_dims[0];
    int64_t dim_size = static_cast<int64_t>(src_ov_shape[slice_dim]);

    size_t ov_stride_for_dim = 1;
    માટે (size_t i = slice_dim + 1; i < ndims; ++i) {
        ov_stride_for_dim *= src_ov_shape[i];
    }
    size_t elem_size = src_stride.back();
    જો (elem_size == 0) {
        elem_size = 1;
    }

    int64_t begin_val = 0;
    જો (ov_stride_for_dim > 0 && elem_size > 0) {
        begin_val = static_cast<int64_t>((view_offset / elem_size) / ov_stride_for_dim);
    }
    int64_t end_val = begin_val + static_cast<int64_t>(dst_ov_shape[slice_dim]);

    જો (begin_val < 0 || end_val > dim_size) {
        પાછા {આગત};
    }

    auto sliced =
        std::make_shared<ov::op::v8::Slice>(આગત, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

    sliced->set_friendly_name(context.get_output_name());
    પાછા {sliced->output(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

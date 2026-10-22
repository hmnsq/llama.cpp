#അന്തര്ഭാവ "repeat_back.hpp"

#അന്തര്ഭാവ "common.hpp"

#define GGML_ASSERT_TENSOR_FITS_INT(t) \
    GGML_ASSERT((t)->ne[0] < INT_MAX && (t)->ne[1] < INT_MAX && (t)->ne[2] < INT_MAX && (t)->ne[3] < INT_MAX)

ശൂന്യം ggml_sycl_op_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    സ്ഥിരം ദശാംശം * src0_dd = (സ്ഥിരം ദശാംശം *) dst->src[0]->data;
    ദശാംശം *       dst_dd  = (ദശാംശം *) dst->data;

    GGML_ASSERT_TENSOR_FITS_INT(dst);
    GGML_ASSERT_TENSOR_FITS_INT(dst->src[0]);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne00 = dst->src[0]->ne[0], ne01 = dst->src[0]->ne[1], ne02 = dst->src[0]->ne[2],
              ne03 = dst->src[0]->ne[3];

    സ്ഥിരം പൂർണ്ണസംഖ്യ nr0 = ne00 / ne0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr1 = ne01 / ne1;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr2 = ne02 / ne2;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr3 = ne03 / ne3;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nb0 = dst->src[0]->nb[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb1 = dst->src[0]->nb[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb2 = dst->src[0]->nb[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb3 = dst->src[0]->nb[3];

    സ്ഥിരം അക്ഷരം * base = (സ്ഥിരം അക്ഷരം *) src0_dd;

    സ്ഥിരം size_t  total      = (size_t) ne0 * ne1 * ne2 * ne3;
    constexpr പൂർണ്ണസംഖ്യ BLOCK_SIZE = 256;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     അക്കം_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    സ്ഥിരം ദശാംശം inv_ne0      = 1.0f / ne0;
    സ്ഥിരം ദശാംശം inv_ne_01    = 1.0f / (ne0 * ne1);
    സ്ഥിരം ദശാംശം inv_ne_012   = 1.0f / (ne0 * ne1 * ne2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ   repeat_എണ്ണം = nr0 * nr1 * nr2 * nr3;

    queue_ptr stream = ctx.stream();

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(അക്കം_blocks * BLOCK_SIZE), sycl::range<1>(BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            സ്ഥിരം size_t i = item_ct1.get_global_linear_id();
            എങ്കിൽ (i >= total) {
                തിരികെ;
            }

            സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = (പൂർണ്ണസംഖ്യ) (i * inv_ne_012);
            സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = (പൂർണ്ണസംഖ്യ) (i * inv_ne_01) - i3 * ne2;
            സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = (പൂർണ്ണസംഖ്യ) (i * inv_ne0) - (പൂർണ്ണസംഖ്യ) (i * inv_ne_01) * ne1;
            സ്ഥിരം പൂർണ്ണസംഖ്യ i0 = i - (പൂർണ്ണസംഖ്യ) (i * inv_ne0) * ne0;

            പൂർണ്ണസംഖ്യ   j0 = 0, j1 = 0, j2 = 0, j3 = 0;
            ദശാംശം acc = 0.0f;

            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < repeat_എണ്ണം; ++j) {
                സ്ഥിരം ദശാംശം * ptr = (സ്ഥിരം ദശാംശം *) (base + (i0 + j0 * ne0) * nb0 + (i1 + j1 * ne1) * nb1 +
                    (i2 + j2 * ne2) * nb2 + (i3 + j3 * ne3) * nb3);
                acc += *ptr;

                പൂർണ്ണസംഖ്യ carry = (++j0 >= nr0);
                j0 -= carry * nr0;
                carry = (carry && (++j1 >= nr1));
                j1 -= carry * nr1;
                carry = (carry && (++j2 >= nr2));
                j2 -= carry * nr2;
                j3 += carry;
            }
            dst_dd[i] = acc;
        });
}

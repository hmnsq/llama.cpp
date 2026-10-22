#ضمن "ggml-vulkan-common.h"

ساكن vk_device_architecture get_device_architecture(ثابت vk::PhysicalDevice& device) {
    vk::PhysicalDeviceProperties props = device.getProperties();

    إذا (props.vendorID == VK_VENDOR_ID_AMD) {
        ثابت std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        منطقي amd_shader_core_properties = خطأ;
        منطقي integer_dot_ضرب_حاصل = خطأ;
        منطقي subgroup_ناپ_control = خطأ;
        منطقي shader_float8 = خطأ;

        لكل (ثابت خودکار& properties : ext_props) {
            إذا (strcmp("VK_AMD_shader_core_properties", properties.extensionName) == 0) {
                amd_shader_core_properties = سچ;
            } وإلا إذا (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0) {
                integer_dot_ضرب_حاصل = سچ;
            } وإلا إذا (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                subgroup_ناپ_control = سچ;
            } وإلا إذا (strcmp("VK_EXT_shader_float8", properties.extensionName) == 0) {
                shader_float8 = سچ;
            }
        }

        إذا (!amd_shader_core_properties || !integer_dot_ضرب_حاصل || !subgroup_ناپ_control) {
            أرجع vk_device_architecture::OTHER;
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::PhysicalDeviceShaderCorePropertiesAMD shader_core_props_amd;
        vk::Physical_Device_Shader_Integer_Dot_ضرب_حاصل_Properties_KHR integer_dot_props;
        vk::Physical_Device_Subgroup_ناپ_Control_Properties_EXT subgroup_ناپ_control_props;

        props2.p_اگلا = &shader_core_props_amd;
        shader_core_props_amd.p_اگلا = &integer_dot_props;
        integer_dot_props.p_اگلا = &subgroup_ناپ_control_props;

        device.getProperties2(&props2);

        إذا (subgroup_ناپ_control_props.max_Subgroup_ناپ == 64 && subgroup_ناپ_control_props.min_Subgroup_ناپ == 64) {
            أرجع vk_device_architecture::AMD_GCN;
        }
        إذا (subgroup_ناپ_control_props.max_Subgroup_ناپ == 64 && subgroup_ناپ_control_props.min_Subgroup_ناپ == 32) {
            // RDNA
            إذا (shader_core_props_amd.wavefronts_فی_Simd == 20) {
                أرجع vk_device_architecture::AMD_RDNA1;
            }
            إذا (shader_float8) {
                أرجع vk_device_architecture::AMD_RDNA4;
            }
            إذا (integer_dot_props.integer_Dot_Product4x8_ثنائی_ہندسہ_Packed_Mixed_Signedness_Accelerated) {
                أرجع vk_device_architecture::AMD_RDNA3;
            }
            أرجع vk_device_architecture::AMD_RDNA2;
        }
    } وإلا إذا (props.vendorID == VK_VENDOR_ID_INTEL) {
        ثابت std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        منطقي subgroup_ناپ_control = خطأ;
        منطقي integer_dot_ضرب_حاصل = خطأ;

        لكل (ثابت خودکار& properties : ext_props) {
            إذا (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                subgroup_ناپ_control = سچ;
            } وإلا إذا (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0) {
                integer_dot_ضرب_حاصل = سچ;
            }
        }

        إذا (!subgroup_ناپ_control || !integer_dot_ضرب_حاصل) {
            أرجع vk_device_architecture::OTHER;
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::Physical_Device_Subgroup_ناپ_Control_Properties_EXT subgroup_ناپ_control_props;
        vk::Physical_Device_Shader_Integer_Dot_ضرب_حاصل_Properties_KHR integer_dot_props;

        props2.p_اگلا = &subgroup_ناپ_control_props;
        subgroup_ناپ_control_props.p_اگلا = &integer_dot_props;
        device.getProperties2(&props2);

        إذا (subgroup_ناپ_control_props.min_Subgroup_ناپ == 16) {
            // Xe2 architecture uses SIMD16 while previous Xe and Gen architecture uses SIMD8.
            // Minimum subgroup size matches the SIMD width so we distinguish architecture by checking this value.
            // https://www.intel.com/content/www/us/en/content-details/824434/2024-intel-tech-tour-xe2-and-lunar-lake-s-gpu.html
            // https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
            أرجع vk_device_architecture::INTEL_XE2;
        } وإلا إذا (subgroup_ناپ_control_props.min_Subgroup_ناپ == 8 &&
                 integer_dot_ضرب_حاصل && integer_dot_props.integer_Dot_Product4x8_ثنائی_ہندسہ_Packed_Signed_Accelerated) {
            أرجع vk_device_architecture::INTEL_XE1;
        }
    } وإلا إذا (props.vendorID == VK_VENDOR_ID_NVIDIA) {
        ثابت std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        منطقي cooperative_میٹرکس = خطأ;
        منطقي sm_builtins = خطأ;

        // Detect "pre-turing" based on lack of coopmat support.
        لكل (ثابت خودکار& properties : ext_props) {
            إذا (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0) {
                cooperative_میٹرکس = سچ;
            } وإلا إذا (strcmp("VK_NV_shader_sm_builtins", properties.extensionName) == 0) {
                sm_builtins = سچ;
            }
        }

        إذا (!cooperative_میٹرکس) {
            أرجع vk_device_architecture::NVIDIA_PRE_TURING;
        }

        إذا (sm_builtins) {
            vk::PhysicalDeviceProperties2 props2;
            vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV sm_props;

            props2.p_اگلا = &sm_props;

            device.getProperties2(&props2);

            // Turing has 32, following architectures have 48
            إذا (sm_props.shader_Warps_فی_SM == 32) {
                أرجع vk_device_architecture::NVIDIA_TURING;
            }
        }
    } وإلا إذا(props.vendorID == VK_VENDOR_ID_QUALCOMM){
        ثابت std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        منطقي cooperative_میٹرکس = خطأ;
        منطقي cooperative_میٹرکس_conversion = خطأ;

        لكل (ثابت خودکار& properties : ext_props) {
            إذا (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0) {
                cooperative_میٹرکس = سچ;
            } وإلا إذا (strcmp("VK_QCOM_cooperative_matrix_conversion", properties.extensionName) == 0) {
                cooperative_میٹرکس_conversion = سچ;
            }
        }

        إذا (cooperative_میٹرکس && cooperative_میٹرکس_conversion) {
            أرجع vk_device_architecture::QUALCOMM_ADRENO;
        }
    }
    أرجع vk_device_architecture::OTHER;
}

منطقي ggml_vk_lightning_indexer_k_type_supported(ggml_type type) {
    أرجع std::find(lightning_indexer_k_types.begin(), lightning_indexer_k_types.end(), type) != lightning_indexer_k_types.end();
}
فراغ ggml_vk_print_device_fault_info(ثابت vk_device& device) {
    إذا (!device->device_fault || !device->pfn_vkGetDeviceFaultInfoEXT) {
        أرجع;
    }

    VkDeviceFaultCountsEXT fault_counts {};
    fault_counts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
    Vk_نتیجہ res = device->pfn_vkGetDeviceFaultInfoEXT(device->device, &fault_counts, خالی_سوچک);
    إذا (res != VK_SUCCESS) {
        GGML_LOG_ERROR("ggml_vulkan: vkGetDeviceFaultInfoEXT (counts) failed: %d\n", res);
        أرجع;
    }

    std::vector<Vk_Device_Fault_پتہ_Info_EXT> پتہ_infos(fault_counts.پتہ_Info_گنتی);
    std::vector<VkDeviceFaultVendorInfoEXT> vendor_infos(fault_counts.vendor_Info_گنتی);

    VkDeviceFaultInfoEXT fault_info {};
    fault_info.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
    fault_info.p_پتہ_Infos = پتہ_infos.انکڑے();
    fault_info.pVendorInfos = vendor_infos.انکڑے();

    res = device->pfn_vkGetDeviceFaultInfoEXT(device->device, &fault_counts, &fault_info);
    إذا (res != VK_SUCCESS) {
        GGML_LOG_ERROR("ggml_vulkan: vkGetDeviceFaultInfoEXT (info) failed: %d\n", res);
        أرجع;
    }

    إذا (fault_counts.پتہ_Info_گنتی == 0 && fault_counts.vendor_Info_گنتی == 0 && fault_info.description[0] == '\0') {
        أرجع;
    }

    إذا (fault_info.description[0] != '\0') {
        GGML_LOG_ERROR("ggml_vulkan: device fault on %s: %s\n", device->name.c_str(), fault_info.description);
    }

    لكل (uint32_t i = 0; i < fault_counts.پتہ_Info_گنتی; i++) {
        ثابت خودکار& info = پتہ_infos[i];
        GGML_LOG_CONT("  address fault %u: type=%d address=0x%llx precision=0x%llx\n",
                i, (صحيح)info.پتہ_Type,
                (بنانشانی لما لما)info.reported_پتہ,
                (بنانشانی لما لما)info.پتہ_Precision);
    }
    لكل (uint32_t i = 0; i < fault_counts.vendor_Info_گنتی; i++) {
        ثابت خودکار& info = vendor_infos[i];
        GGML_LOG_CONT("  vendor fault %u: %s (code=0x%llx data=0x%llx)\n",
                i, info.description,
                (بنانشانی لما لما)info.vendor_Fault_کوڈ,
                (بنانشانی لما لما)info.vendor_Fault_انکڑے);
    }
}
uint64_t ggml_vk_get_گرہ_flops(ثابت ggml_tensor * گرہ) {
    إذا (گرہ->op == GGML_OP_MUL_MAT || گرہ->op == GGML_OP_MUL_MAT_ID) {
        ثابت uint64_t m     = گرہ->ne[0];
        ثابت uint64_t n     = گرہ->ne[1];
        ثابت uint64_t k     = گرہ->src[1]->ne[0];
        ثابت uint64_t batch = گرہ->ne[2] * گرہ->ne[3];
        أرجع m * n * (k + (k - 1)) * batch;
    }
    إذا (گرہ->op == GGML_OP_CONV_2D || گرہ->op == GGML_OP_CONV_TRANSPOSE_2D) {
        ثابت ggml_tensor * knl = گرہ->src[0];
        ثابت uint64_t Cout  = گرہ->ne[2];
        ثابت uint64_t ناپ_K = گرہ->src[1]->ne[2] * knl->ne[0] * knl->ne[1];
        ثابت uint64_t ناپ_N_2 = گرہ->ne[3] * گرہ->ne[0] * گرہ->ne[1];
        أرجع Cout * ناپ_N_2 * (ناپ_K + (ناپ_K - 1));
    }
    إذا (گرہ->op == GGML_OP_CONV_3D) {
        ثابت ggml_tensor * knl = گرہ->src[0];
        ثابت uint64_t OC     = ggml_get_op_params_i32(گرہ, 11);
        ثابت uint64_t IC     = ggml_get_op_params_i32(گرہ, 9);
        ثابت uint64_t ناپ_K = IC * knl->ne[0] * knl->ne[1] * knl->ne[2];
        ثابت uint64_t ناپ_N_2 = گرہ->ne[3] / OC * گرہ->ne[0] * گرہ->ne[1] * گرہ->ne[2];
        أرجع OC * ناپ_N_2 * (ناپ_K + (ناپ_K - 1));
    }
    إذا (گرہ->op == GGML_OP_FLASH_ATTN_EXT) {
        ثابت ggml_tensor * q = گرہ->src[0];
        ثابت ggml_tensor * k = گرہ->src[1];
        ثابت ggml_tensor * v = گرہ->src[2];
        أرجع 2ull * q->ne[1] * q->ne[2] * (k->ne[0] + v->ne[0]) * k->ne[1] * q->ne[3];
    }
    أرجع 0;
}
فراغ ggml_vk_print_گرہ_list(ثابت ggml_cgraph * cgraph, صحيح start, صحيح end) {
    uint64_t کل_flops = 0;
    صحيح n_ops = 0;
    لكل (صحيح j = start; j <= end && j < cgraph->n_nodes; j++) {
        uint64_t flops = ggml_vk_get_گرہ_flops(cgraph->nodes[j]);
        کل_flops += flops;
        n_ops++;
        إذا (flops > 0) {
            GGML_LOG_CONT("  node %d: %s (%s) [%.2f GFLOP]\n",
                    j, cgraph->nodes[j]->name, ggml_op_name(cgraph->nodes[j]->op),
                    flops / 1e9);
        } وإلا {
            GGML_LOG_CONT("  node %d: %s (%s)\n",
                    j, cgraph->nodes[j]->name, ggml_op_name(cgraph->nodes[j]->op));
        }
    }
    GGML_LOG_CONT("  total: %d ops, %.2f GFLOP\n", n_ops, کل_flops / 1e9);
}
فراغ ggml_vk_print_device_گواچا_info(ثابت vk_device& device) {
    ggml_vk_print_device_fault_info(device);
    إذا (device->serialize_submissions && device->diag_cgraph != خالی_سوچک && device->diag_prev_شروع >= 0) {
        GGML_LOG_ERROR("ggml_vulkan: device lost on %s, likely caused by previous submission (nodes %d to %d):\n",
                device->name.c_str(), device->diag_prev_شروع, device->diag_prev_سرا);
        ggml_vk_print_گرہ_list(device->diag_cgraph, device->diag_prev_شروع, device->diag_prev_سرا);
    } وإلا {
        GGML_LOG_ERROR("ggml_vulkan: device lost on %s\n", device->name.c_str());
    }
}
فراغ * ثابت vk_ptr_ادھار = (فراغ *)(uintptr_t) 0x1000;  // NOLINT

uint64_t vk_tensor_وتھ(ثابت ggml_tensor * tensor) {
    إذا (tensor->view_src) {
        أرجع (uint8_t *) tensor->view_src->انکڑے - (uint8_t *) vk_ptr_ادھار;
    }
    أرجع (uint8_t *) tensor->انکڑے - (uint8_t *) vk_ptr_ادھار;
}

size_t ggml_vk_tensor_بفر_وتھ(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * t) {
    // vk_tensor_offset() is relative to vk_ptr_base, but mapped host tensors need an offset relative to their Vulkan buffer.
    إذا (ctx->device->uma) {
        vk_بفر buf = خالی_سوچک;
        size_t off = 0;
        ggml_vk_host_get(ctx->device, t->انکڑے, buf, off);
        إذا (buf) {
            أرجع off;
        }
    }
    أرجع (size_t)(vk_tensor_وتھ(t) + t->view_offs);
}
size_t ggml_vk_descriptor_وتھ(size_t tensor_وتھ, size_t alignment, size_t type_ناپ) {
    // Move the descriptor back until its distance to the tensor is divisible by the tensor type size.
    size_t descriptor_وتھ = tensor_وتھ & ~(alignment - 1);
    طالما ((tensor_وتھ - descriptor_وتھ) % type_ناپ != 0) {
        GGML_ASSERT(descriptor_وتھ >= alignment);
        descriptor_وتھ -= alignment;
    }

    أرجع descriptor_وتھ;
}
uint32_t get_misalign_بائٹاں(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * t) {
    ثابت size_t tensor_وتھ = ggml_vk_tensor_بفر_وتھ(ctx, t);
    ثابت size_t descriptor_وتھ = ggml_vk_descriptor_وتھ(
        tensor_وتھ, ctx->device->properties.limits.min_Storage_بفر_وتھ_Alignment, ggml_type_ناپ(t->type));
    GGML_ASSERT(tensor_وتھ - descriptor_وتھ <= UINT32_MAX);
    أرجع tensor_وتھ - descriptor_وتھ;
}

uint32_t ggml_vk_concat_unit_ناپ(ggml_type type) {
    ثابت uint32_t type_ناپ = ggml_type_ناپ(type);

    إذا (!ggml_is_quantized(type)) {
        أرجع type_ناپ;
    }

    // Use the widest existing concat shader that evenly divides a quant block.
    إذا (type_ناپ % 8 == 0) {
        أرجع 8;
    }
    إذا (type_ناپ % 4 == 0) {
        أرجع 4;
    }
    إذا (type_ناپ % 2 == 0) {
        أرجع 2;
    }
    أرجع 1;
}
منطقي ggml_vk_concat_supported(ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * dst) {
    إذا (src0->type != src1->type || src0->type != dst->type) {
        أرجع خطأ;
    }

    إذا (!ggml_is_quantized(src0->type)) {
        ثابت size_t type_ناپ = ggml_type_ناپ(src0->type);
        أرجع type_ناپ == 1 || type_ناپ == 2 || type_ناپ == 4 || type_ناپ == 8;
    }

    // Quantized tensor rows are block-aligned when created.
    أرجع ggml_is_contiguous_صفاں(src0) && ggml_is_contiguous_صفاں(src1) && ggml_is_contiguous_صفاں(dst);
}
ساكن منطقي vk_instance_initialized = خطأ;

vk_instance_t vk_instance;

ساكن Vk_Device_ناپ ggml_vk_get_max_بفر_range(ثابت ggml_backend_vk_context * ctx, ثابت vk_بفر &buf, ثابت Vk_Device_ناپ وتھ) {
    ثابت Vk_Device_ناپ range = std::min(Vk_Device_ناپ{buf->size - وتھ},
                                        Vk_Device_ناپ{ctx->device->properties.limits.max_Storage_بفر_Range});
    أرجع range;
}

فراغ ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx) {
    // Use waitForFences while most of the graph executes. Hopefully the CPU can sleep
    // during this wait.
    إذا (ctx->almost_ready_fence_pending) {
        VK_CHECK(ctx->device->device.waitForFences({ ctx->almost_ready_fence }, سچ, UINT64_MAX), "almost_ready_fence", ctx->device);
        ctx->device->device.resetFences({ ctx->almost_ready_fence });
        ctx->almost_ready_fence_pending = خطأ;
    }

    // Spin (w/pause) waiting for the graph to finish executing.
    vk::نتیجہ_2 نتیجہ;
    لكل (;;) {
        حاول {
            نتیجہ = ctx->device->device.get_Fence_حالت(ctx->fence);
        } امسك (vk::Device_گواچا_Error &) {
            ggml_vk_print_device_گواچا_info(ctx->device);
            GGML_LOG_ERROR("ggml_vulkan: getFenceStatus at %s:%d\n", __FILE__, __LINE__);
            ارم;
        }
        إذا (نتیجہ == vk::نتیجہ_2::eSuccess) {
            اكسر;
        }
        إذا (نتیجہ != vk::نتیجہ_2::eNotReady) {
            GGML_LOG_ERROR("ggml_vulkan: error %s at %s:%d\n", to_string(نتیجہ).c_str(), __FILE__, __LINE__);
            ارم vk::SystemError(vk::make_error_code(نتیجہ), "ggml_vulkan: getFenceStatus");
        }
        لكل (uint32_t i = 0; i < 100; ++i) {
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
        }
    }
    ctx->device->device.resetFences({ ctx->fence });
}

ساكن منطقي ggml_vk_strip_decode_سمتیہ(ثابت uint32_t * code, size_t لفظ_گنتی, std::vector<uint32_t> & باہرلا) {
    ساكن ثابت حرف k_Decode_سمتیہ_Ext[] = "SPV_NV_cooperative_matrix_decode_vector";

    إذا (لفظ_گنتی < 5) {
        أرجع خطأ;
    }

    منطقي ورتدا_decode_سمتیہ = خطأ;
    لكل (size_t pos = 5; pos < لفظ_گنتی; ) {
        uint32_t لفظ = code[pos];
        uint32_t wc   = لفظ >> spv::لفظ_گنتی_کھسکاؤ;
        uint32_t op   = لفظ & spv::Op_کوڈ_نقاب;
        GGML_ASSERT(wc > 0 && pos + wc <= لفظ_گنتی);
        إذا (op == spv::OpExtension && wc >= 2) {
            ثابت حرف * s = reinterpret_cast<ثابت حرف *>(&code[pos + 1]);
            إذا (strcmp(s, k_Decode_سمتیہ_Ext) == 0) {
                ورتدا_decode_سمتیہ = سچ;
                اكسر;
            }
        }
        pos += wc;
    }

    إذا (!ورتدا_decode_سمتیہ) {
        أرجع خطأ;
    }

    VK_LOG_DEBUG("ggml_vk_strip_decode_vector: stripping SPV_NV_cooperative_matrix_decode_vector");

    // Bulk-copy unchanged runs and only break the run when an instruction needs to
    // be dropped or patched. Use reserve + insert/push_back so the destination buffer
    // is touched exactly once (no zero-initialization pass from resize()).
    باہرلا.clear();
    باہرلا.reserve(لفظ_گنتی);

    size_t لگاتار_شروع = 0;
    خودکار flush_لگاتار = [&](size_t اتے_منزل) {
        إذا (اتے_منزل > لگاتار_شروع) {
            باہرلا.insert(باہرلا.end(), code + لگاتار_شروع, code + اتے_منزل);
        }
    };

    لكل (size_t pos = 5; pos < لفظ_گنتی; ) {
        uint32_t لفظ = code[pos];
        uint32_t wc   = لفظ >> spv::لفظ_گنتی_کھسکاؤ;
        uint32_t op   = لفظ & spv::Op_کوڈ_نقاب;
        GGML_ASSERT(wc > 0 && pos + wc <= لفظ_گنتی);

        إذا (op == spv::OpExtension && wc >= 2) {
            ثابت حرف * s = reinterpret_cast<ثابت حرف *>(&code[pos + 1]);
            إذا (strcmp(s, k_Decode_سمتیہ_Ext) == 0) {
                flush_لگاتار(pos);
                pos += wc;
                لگاتار_شروع = pos;
                استمر;
            }
        }

        إذا (op == spv::OpCapability && wc == 2 && code[pos + 1] == k_Spv_Capability_Cooperative_میٹرکس_Decode_سمتیہ_NV) {
            flush_لگاتار(pos);
            pos += wc;
            لگاتار_شروع = pos;
            استمر;
        }

        إذا (op == k_Spv_Op_Cooperative_میٹرکس_لدو_Tensor_NV) {
            // [opcode/wc][ResultType][Result][Pointer][Object][TensorLayout][MemOperand mask][mem extras...][TA mask][ta extras...]
            GGML_ASSERT(wc >= 8);

            uint32_t mem_نقاب = code[pos + 6];
            size_t   cur      = pos + 7;
            // Each of these MemoryAccess bits (when set) carries one trailing operand.
            cur += (mem_نقاب & 0x2)     ? 1 : 0; // Aligned
            cur += (mem_نقاب & 0x8)     ? 1 : 0; // MakePointerAvailable
            cur += (mem_نقاب & 0x10)    ? 1 : 0; // MakePointerVisible
            cur += (mem_نقاب & 0x10000) ? 1 : 0; // AliasScopeINTELMask
            cur += (mem_نقاب & 0x20000) ? 1 : 0; // NoAliasINTELMask
            GGML_ASSERT(cur < pos + wc);

            uint32_t ta_نقاب = code[cur];
            إذا ((ta_نقاب & k_Spv_Tensor_Addressing_Decode_سمتیہ_Func_ثنائی_ہندسہ) == 0) {
                pos += wc;
                استمر; // leave instruction inside the current unchanged run
            }

            flush_لگاتار(pos);

            // Append unchanged prefix of the instruction (header through the mem-extras).
            size_t inst_شروع = باہرلا.size();
            size_t pre_n      = cur - pos;
            باہرلا.insert(باہرلا.end(), code + pos, code + pos + pre_n);

            // Emit TA mask with the DecodeVectorFunc bit cleared.
            باہرلا.push_back(ta_نقاب & ~k_Spv_Tensor_Addressing_Decode_سمتیہ_Func_ثنائی_ہندسہ);

            // TA extras: TensorView (0x1) and DecodeFunc (0x2) are kept verbatim;
            // DecodeVectorFunc (0x4) is dropped along with its trailing id operand.
            size_t keep_ta_extras = ((ta_نقاب & 0x1) ? 1 : 0) + ((ta_نقاب & 0x2) ? 1 : 0);
            إذا (keep_ta_extras) {
                باہرلا.insert(باہرلا.end(), code + cur + 1, code + cur + 1 + keep_ta_extras);
            }

            GGML_ASSERT(wc == pre_n + 1 + keep_ta_extras + 1);

            // Patch the instruction header with the new (one-shorter) word count.
            uint32_t new_wc = wc - 1;
            باہرلا[inst_شروع] = (new_wc << spv::لفظ_گنتی_کھسکاؤ) | op;

            pos += wc;
            لگاتار_شروع = pos;
            استمر;
        }

        pos += wc;
    }

    flush_لگاتار(لفظ_گنتی);
    أرجع سچ;
}

ساكن منطقي ggml_vk_roll_bk_loop(ثابت uint32_t * code, size_t لفظ_گنتی, std::vector<uint32_t> & باہرلا) {
    إذا (لفظ_گنتی < 5) {
        أرجع خطأ;
    }

    بنية vk_spv_loop {
        size_t   سرخی;
        size_t   end;
        uint32_t control;
    };

    std::vector<vk_spv_loop> loops;

    // Collect a list of all loops in the module.
    لكل (size_t pos = 5; pos < لفظ_گنتی; ) {
        ثابت uint32_t wc = code[pos] >> spv::لفظ_گنتی_کھسکاؤ;
        ثابت uint32_t op = code[pos] & spv::Op_کوڈ_نقاب;
        إذا (wc == 0 || pos + wc > لفظ_گنتی) {
            أرجع خطأ;
        }

        إذا (op == spv::Op_Loop_ملاؤ && wc >= 4) { loops.push_back({ pos, 0, code[pos + 3] }); }

        إذا (op == spv::OpLabel && wc >= 2) {
            لكل (خودکار & l : loops) {
                إذا (l.end == 0 && code[l.سرخی + 1] == code[pos + 1]) { l.end = pos; }
            }
        }

        pos += wc;
    }

    خودکار encloses = [](ثابت vk_spv_loop & a, ثابت vk_spv_loop & b) {
        أرجع a.سرخی < b.سرخی && b.سرخی < a.end;
    };

    // Find the BK loop.
    ثابت vk_spv_loop * bk = خالی_سوچک;
    لكل (ثابت خودکار & h : loops) {
        إذا (h.control != spv::Loop_Control_Unroll_نقاب) {
            استمر;
        }
        ثابت vk_spv_loop * ماپا = خالی_سوچک;
        منطقي has_بچہ = خطأ;
        لكل (ثابت خودکار & g : loops) {
            إذا (encloses(g, h) && (!ماپا || g.سرخی > ماپا->سرخی)) {
                ماپا = &g;
            }
            إذا (encloses(h, g)) {
                has_بچہ = سچ;
            }
        }
        // BK loop should be the last loop nested inside the loop with no hint
        // and have at least one child loop.
        إذا (ماپا &&
            ماپا->control == spv::Loop_Control_نقاب_None &&
            has_بچہ &&
            (!bk || h.سرخی > bk->سرخی)) {
            bk = &h;
        }
    }
    إذا (!bk) {
        أرجع خطأ;
    }

    // set DontUnroll instead of Unroll
    باہرلا.سونپو(code, code + لفظ_گنتی);
    باہرلا[bk->سرخی + 3] = spv::Loop_Control_Dont_Unroll_نقاب;
    أرجع سچ;
}

ساكن فراغ ggml_vk_create_pipeline_func(vk_device& device, vk_pipeline& pipeline, size_t spv_ناپ, ثابت فراغ* spv_انکڑے, ثابت std::string entrypoint,
                                         uint32_t parameter_گنتی, std::array<uint32_t, 3> wg_denoms, std::vector<uint32_t> specialization_constants,
                                         منطقي disable_robustness, منطقي require_بھریا_subgroups, uint32_t required_subgroup_ناپ) {
    VK_LOG_DEBUG("ggml_vk_create_pipeline(" << device->name << ", " << pipeline->name << ", " << entrypoint << ", " << parameter_گنتی <<
                 ", (" << wg_denoms[0] << "," << wg_denoms[1] << "," << wg_denoms[2] << "), specialization_constants, " <<
                 disable_robustness << ", " << require_بھریا_subgroups << ", " << required_subgroup_ناپ << ")");
    GGML_ASSERT(parameter_گنتی > 0);
    GGML_ASSERT(parameter_گنتی <= MAX_PARAMETER_گنتی);
    GGML_ASSERT(wg_denoms[0] > 0 && wg_denoms[1] > 0 && wg_denoms[2] > 0); // NOLINT

    vk::ShaderModuleCreateInfo shader_module_create_info({}, spv_ناپ, reinterpret_cast<ثابت uint32_t *>(spv_انکڑے));

    // Patch SPIR-V to enable supported FP16 float controls, avoiding the need
    // for separate shader variants.
    std::vector<uint32_t> spirv;
    إذا (device->float_controls_rte_fp16 || device->float_controls_denorm_preserve_fp16) {
        ثابت uint32_t* spv_الفاظ = reinterpret_cast<ثابت uint32_t *>(spv_انکڑے);
        size_t لفظ_گنتی = spv_ناپ / حجم(uint32_t);
        spirv.سونپو(spv_الفاظ, spv_الفاظ + لفظ_گنتی);

        // Find insertion points respecting SPIR-V layout order:
        //   Header(5) -> OpCapability -> OpExtension -> ... -> OpEntryPoint -> OpExecutionMode -> ...
        size_t pos = 5; // skip header
        size_t cap_داخل_کرو_pos = pos;
        size_t ext_داخل_کرو_pos = pos;
        size_t exec_داخل_کرو_pos = pos;
        uint32_t اندراج_نقطہ_id = 0;

        طالما (pos < spirv.size()) {
            uint32_t opcode = spirv[pos] & spv::Op_کوڈ_نقاب;
            uint32_t len    = spirv[pos] >> spv::لفظ_گنتی_کھسکاؤ;
            إذا (len == 0) اكسر;

            إذا (opcode == spv::OpCapability) {
                cap_داخل_کرو_pos = pos + len;
                ext_داخل_کرو_pos = pos + len;
            } وإلا إذا (opcode == spv::OpExtension) {
                ext_داخل_کرو_pos = pos + len;
            } وإلا إذا (opcode == spv::Op_اندراج_نقطہ) {
                اندراج_نقطہ_id = spirv[pos + 2];
                exec_داخل_کرو_pos = pos + len;
            } وإلا إذا (opcode == spv::Op_Execution_منوال || opcode == spv::Op_Execution_منوال_Id) {
                exec_داخل_کرو_pos = pos + len;
            } وإلا إذا (اندراج_نقطہ_id != 0) {
                اكسر;
            }

            pos += len;
        }

        // Insert from latest position first so earlier indices stay valid.

        إذا (device->float_controls_rte_fp16) {
            // OpExecutionMode %entrypoint RoundingModeRTE 16
            uint32_t exec_منوال[] = { (4u << spv::لفظ_گنتی_کھسکاؤ) | spv::Op_Execution_منوال, اندراج_نقطہ_id, spv::Execution_منوال_Rounding_منوال_RTE, 16 };
            spirv.insert(spirv.begin() + exec_داخل_کرو_pos, std::begin(exec_منوال), std::end(exec_منوال));
        }

        إذا (device->float_controls_denorm_preserve_fp16) {
            // OpExecutionMode %entrypoint DenormPreserve 16
            uint32_t exec_منوال[] = { (4u << spv::لفظ_گنتی_کھسکاؤ) | spv::Op_Execution_منوال, اندراج_نقطہ_id, spv::Execution_منوال_Denorm_Preserve, 16 };
            spirv.insert(spirv.begin() + exec_داخل_کرو_pos, std::begin(exec_منوال), std::end(exec_منوال));
        }

        // OpExtension "SPV_KHR_float_controls"
        ثابت حرف ext_str[] = "SPV_KHR_float_controls";
        size_t ext_str_الفاظ = CEIL_DIV(حجم(ext_str), حجم(uint32_t));
        std::vector<uint32_t> extension(1 + ext_str_الفاظ, 0);
        extension[0] = (uint32_t)((1 + ext_str_الفاظ) << spv::لفظ_گنتی_کھسکاؤ) | spv::OpExtension;
        حافظہ_نقل(&extension[1], ext_str, حجم(ext_str));
        spirv.insert(spirv.begin() + ext_داخل_کرو_pos, extension.begin(), extension.end());

        إذا (device->float_controls_rte_fp16) {
            // OpCapability RoundingModeRTE
            uint32_t capability[] = { (2u << spv::لفظ_گنتی_کھسکاؤ) | spv::OpCapability, spv::Capability_Rounding_منوال_RTE };
            spirv.insert(spirv.begin() + cap_داخل_کرو_pos, std::begin(capability), std::end(capability));
        }

        إذا (device->float_controls_denorm_preserve_fp16) {
            // OpCapability DenormPreserve
            uint32_t capability[] = { (2u << spv::لفظ_گنتی_کھسکاؤ) | spv::OpCapability, spv::CapabilityDenormPreserve };
            spirv.insert(spirv.begin() + cap_داخل_کرو_pos, std::begin(capability), std::end(capability));
        }

        shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * حجم(uint32_t), spirv.انکڑے());
    }

#إذا defined(GGML_VULKAN_COOPMAT2_DECODE_سمتیہ_GLSLC_SUPPORT)
    إذا (device->coopmat2 && !device->coopmat2_decode_سمتیہ) {
        ثابت uint32_t * src   = spirv.خالی_تھاواں() ? reinterpret_cast<ثابت uint32_t *>(spv_انکڑے) : spirv.انکڑے();
        size_t           src_n = spirv.خالی_تھاواں() ? spv_ناپ / حجم(uint32_t) : spirv.size();
        std::vector<uint32_t> stripped;
        إذا (ggml_vk_strip_decode_سمتیہ(src, src_n, stripped)) {
            spirv = std::move(stripped);
            shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * حجم(uint32_t), spirv.انکڑے());
        }
    }
#جے_انت

#إذا VK_سرخی_نسخہ >= 287
    // Roll the mul_mm BK loop on Asahi Linux. Skip bf16 and the mul_mmq pipelines.
    إذا (device->driver_id == vk::DriverId::eMesaHoneykrisp &&
        pipeline->name.rfind("matmul", 0) == 0 &&
        pipeline->name.find("bf16") == std::string::npos &&
        pipeline->name.find("q8_1") == std::string::npos) {
        ثابت uint32_t * src   = spirv.خالی_تھاواں() ? reinterpret_cast<ثابت uint32_t *>(spv_انکڑے) : spirv.انکڑے();
        size_t           src_n = spirv.خالی_تھاواں() ? spv_ناپ / حجم(uint32_t) : spirv.size();
        std::vector<uint32_t> rolled;
        إذا (ggml_vk_roll_bk_loop(src, src_n, rolled)) {
            spirv = std::move(rolled);
            shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * حجم(uint32_t), spirv.انکڑے());
        }
    }
#جے_انت

    حاول {
        pipeline->shader_module = device->device.createShaderModule(shader_module_create_info);
    } امسك (ثابت vk::SystemError& e) {
        std::cerr << "ggml_vulkan: shader module creation failed for " << pipeline->name << ": " << e.what() << std::سطر_انت;
        ارم;
    }

    vk::ڈالو_Constant_Range pcr(
        vk::Shader_Stage_Flag_بٹ::eCompute,
        0,
        pipeline->ڈالو_constant_ناپ
    );

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info(vk::PipelineLayoutCreateFlags(), device->dsl, pcr);
    حاول {
        pipeline->layout = device->device.createPipelineLayout(pipeline_layout_create_info);
    } امسك (ثابت vk::SystemError& e) {
        std::cerr << "ggml_vulkan: pipeline layout creation failed for " << pipeline->name << ": " << e.what() << std::سطر_انت;
        ارم;
    }

    std::vector<vk::Specialization_Map_اندراج> specialization_entries(specialization_constants.size());

    لكل (size_t i = 0; i < specialization_constants.size(); i++) {
        specialization_entries[i].constantID = i;
        specialization_entries[i].وتھ = i * حجم(uint32_t);
        specialization_entries[i].size = حجم(uint32_t);
    }

    vk::SpecializationInfo specialization_info(
        specialization_entries.size(),
        specialization_entries.انکڑے(),
        specialization_constants.size() * حجم(uint32_t),
        specialization_constants.انکڑے()
    );

    vk::PipelineShaderStageCreateFlags pipeline_shader_stage_create_flags{};

    إذا (device->subgroup_require_بھریا_support && require_بھریا_subgroups) {
        pipeline_shader_stage_create_flags |= vk::Pipeline_Shader_Stage_Create_Flag_بٹ::e_Require_بھریا_Subgroups_EXT;
    }

    vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
            pipeline_shader_stage_create_flags,
            vk::Shader_Stage_Flag_بٹ::eCompute,
            pipeline->shader_module,
            entrypoint.c_str(),
            &specialization_info);

    vk::Pipeline_Shader_Stage_Required_Subgroup_ناپ_Create_Info_EXT pipeline_shader_stage_required_subgroup_ناپ_create_info;
    pipeline_shader_stage_required_subgroup_ناپ_create_info.required_Subgroup_ناپ = required_subgroup_ناپ;
    إذا (device->subgroup_ناپ_control && required_subgroup_ناپ > 0) {
        GGML_ASSERT(device->subgroup_min_ناپ <= required_subgroup_ناپ && required_subgroup_ناپ <= device->subgroup_max_ناپ);
        pipeline_shader_create_info.سیٹ_PNext(&pipeline_shader_stage_required_subgroup_ناپ_create_info);
    }

    vk::ComputePipelineCreateInfo compute_pipeline_create_info(
        device->pipeline_executable_properties_support ?
            vk::Pipeline_Create_Flag_بٹ::eCaptureStatisticsKHR :
            vk::PipelineCreateFlags{},
        pipeline_shader_create_info,
        pipeline->layout);

    vk::PipelineRobustnessCreateInfoEXT rci;

    إذا (device->pipeline_robustness && disable_robustness) {
        rci.storageBuffers = vk::Pipeline_Robustness_بفر_Behavior_EXT::eDisabled;
        rci.اکسار_Buffers = vk::Pipeline_Robustness_بفر_Behavior_EXT::eDisabled;
        compute_pipeline_create_info.سیٹ_PNext(&rci);
    }

#إذا defined(VK_EXT_shader_64bit_indexing)
    vk::PipelineCreateFlags2CreateInfo pipelineFlags2CreateInfo;
    إذا (pipeline->is_64b_indexing)
    {
        pipelineFlags2CreateInfo.flags = vk::PipelineCreateFlagBits2::e64_ثنائی_ہندسہ_Indexing_EXT;
        إذا (device->pipeline_executable_properties_support) {
            pipelineFlags2CreateInfo.flags |= vk::PipelineCreateFlagBits2::eCaptureStatisticsKHR;
        }
        pipelineFlags2CreateInfo.سیٹ_PNext(compute_pipeline_create_info.p_اگلا);
        compute_pipeline_create_info.سیٹ_PNext(&pipelineFlags2CreateInfo);
    }
#جے_انت

    حاول {
        pipeline->pipeline = device->device.createComputePipeline(VK_NULL_HANDLE, compute_pipeline_create_info).value;
    } امسك (ثابت vk::SystemError& e) {
        std::cerr << "ggml_vulkan: compute pipeline creation failed for " << pipeline->name << ": " << e.what() << std::سطر_انت;
        ارم;
    }

    إذا (vk_instance.debug_utils_support) {
        vk::DebugUtilsObjectNameInfoEXT duoni;
        duoni.objectType = vk::ObjectType::ePipeline;
        duoni.pObjectName = pipeline->name.c_str();
        duoni.objectHandle = /*reinterpret_cast*/(uint64_t)(static_cast<VkPipeline>(pipeline->pipeline));
        vk_instance.pfn_vk_سیٹ_Debug_Utils_Object_Name_EXT(device->device, &static_cast<VkDebugUtilsObjectNameInfoEXT &>(duoni));
    }

    إذا (device->pipeline_executable_properties_support) {
        vk::PipelineExecutableInfoKHR executableInfo;
        executableInfo.pipeline = pipeline->pipeline;

        خودکار statistics = device->device.getPipelineExecutableStatisticsKHR(executableInfo);

        منطقي print_stats = !vk_pipeline_stats_filter.خالی_تھاواں() &&
                           pipeline->name.find(vk_pipeline_stats_filter) != std::string::npos;
        إذا (print_stats) {
            std::cerr << "ggml_vulkan: pipeline stats for " << pipeline->name << ":" << std::سطر_انت;
        }

        لكل (خودکار & s : statistics) {
            إذا (print_stats) {
                std::cerr << "ggml_vulkan:   " << s.name.انکڑے() << ": ";
                بدل (s.format) {
                    حالة vk::PipelineExecutableStatisticFormatKHR::eBool32:
                        std::cerr << (s.value.b32 ? "true" : "false");
                        اكسر;
                    حالة vk::PipelineExecutableStatisticFormatKHR::eInt64:
                        std::cerr << s.value.i64;
                        اكسر;
                    حالة vk::PipelineExecutableStatisticFormatKHR::eUint64:
                        std::cerr << s.value.u64;
                        اكسر;
                    حالة vk::PipelineExecutableStatisticFormatKHR::eFloat64:
                        std::cerr << s.value.f64;
                        اكسر;
                }
                std::cerr << std::سطر_انت;
            }
            // "Register Count" is reported by NVIDIA drivers.
            إذا (strcmp(s.name, "Register Count") == 0) {
                VK_LOG_DEBUG(pipeline->name << " " << s.name << ": " << s.value.u64 << " registers");
                pipeline->رجسٹر_گنتی = (uint32_t)s.value.u64;
            }
        }
    }

    {
        std::lock_guard<std::mutex> guard(device->compile_mutex);
        device->all_pipelines.push_back(pipeline);
        pipeline->compiled = سچ;
        pipeline->compile_pending = خطأ;
    }
    device->compile_cv.notify_all();
}

فراغ ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_pipeline_destroy_pipeline(" << pipeline->name << ")");
    device.destroyPipelineLayout(pipeline->layout);

    device.destroyShaderModule(pipeline->shader_module);

    device.destroyPipeline(pipeline->pipeline);
}

فراغ ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n) {
    VK_LOG_DEBUG("ggml_pipeline_request_descriptor_sets(" << pipeline->name << ", " << n << ")");
    ctx->pipeline_descriptor_سیٹ_requirements += n;
    إذا (!pipeline->compiled) {
        ggml_vk_لدو_shaders(ctx->device, pipeline);
    }
    ggml_pipeline_allocate_descriptor_sets(ctx);
}

فراغ ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx) {

    إذا (ctx->descriptor_sets.size() >= ctx->pipeline_descriptor_سیٹ_requirements) {
        // Enough descriptors are available
        أرجع;
    }

    vk_device& device = ctx->device;

    // Grow by 50% to avoid frequent allocations
    uint32_t needed = std::max(3 * ctx->descriptor_sets.size() / 2, size_t{ctx->pipeline_descriptor_سیٹ_requirements});
    uint32_t منزل_alloc = needed - ctx->descriptor_sets.size();
    uint32_t pool_بچیا = VK_DEVICE_DESCRIPTOR_POOL_ناپ - ctx->descriptor_sets.size() % VK_DEVICE_DESCRIPTOR_POOL_ناپ;
    uint32_t pool_idx = ctx->descriptor_sets.size() / VK_DEVICE_DESCRIPTOR_POOL_ناپ;

    طالما (منزل_alloc > 0) {
        ثابت uint32_t alloc_گنتی = std::min(pool_بچیا, منزل_alloc);
        منزل_alloc -= alloc_گنتی;
        pool_بچیا = VK_DEVICE_DESCRIPTOR_POOL_ناپ;

        إذا (pool_idx >= ctx->descriptor_pools.size()) {
            vk::Descriptor_Pool_ناپ descriptor_pool_ناپ(vk::DescriptorType::e_Storage_بفر, MAX_PARAMETER_گنتی * VK_DEVICE_DESCRIPTOR_POOL_ناپ);
            vk::DescriptorPoolCreateInfo descriptor_pool_create_info({}, VK_DEVICE_DESCRIPTOR_POOL_ناپ, descriptor_pool_ناپ);
            ctx->descriptor_pools.push_back(device->device.createDescriptorPool(descriptor_pool_create_info));
        }

        std::vector<vk::Descriptor_سیٹ_Layout> layouts(alloc_گنتی);
        لكل (uint32_t i = 0; i < alloc_گنتی; i++) {
            layouts[i] = device->dsl;
        }
        vk::Descriptor_سیٹ_Allocate_Info descriptor_سیٹ_alloc_info(ctx->descriptor_pools[pool_idx], alloc_گنتی, layouts.انکڑے());
        std::vector<vk::Descriptor_سیٹ> sets = device->device.allocateDescriptorSets(descriptor_سیٹ_alloc_info);
        ctx->descriptor_sets.insert(ctx->descriptor_sets.end(), sets.begin(), sets.end());

        pool_idx++;
    }
    ctx->descriptor_سیٹ_bindings.resize(ctx->descriptor_sets.size());
}

ساكن vk_command_بفر* ggml_vk_create_cmd_بفر(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_create_cmd_buffer()");
    vk::Command_بفر_Allocate_Info command_بفر_alloc_info(
        p.pool,
        vk::Command_بفر_Level::ePrimary,
        1);
    ثابت std::vector<vk::Command_بفر> cmd_buffers = device->device.allocateCommandBuffers(command_بفر_alloc_info);
    p.cmd_buffers.push_back({ cmd_buffers.front(), 0, سچ });
    أرجع &p.cmd_buffers[p.cmd_buffers.size()-1];
}

فراغ ggml_vk_submit(vk_context& ctx, vk::Fence fence) {
    إذا (ctx->seqs.خالی_تھاواں()) {
        إذا (fence) {
            ctx->p->q->handle->submit({}, fence);
        }
        أرجع;
    }
    VK_LOG_DEBUG("ggml_vk_submit(" << ctx << ", " << fence << ")");

    std::vector<std::vector<uint64_t>> tl_wait_vals;
    std::vector<std::vector<uint64_t>> tl_signal_vals;
    std::vector<std::vector<vk::Semaphore>> tl_wait_semaphores;
    std::vector<std::vector<vk::Semaphore>> tl_signal_semaphores;
    std::vector<vk::TimelineSemaphoreSubmitInfo> tl_submit_infos;
    std::vector<vk::SubmitInfo> submit_infos;
    صحيح idx = -1;
    std::vector<std::vector<vk::PipelineStageFlags>> stage_flags;

    size_t reserve = 0;

    لكل (ثابت خودکار& sequence : ctx->seqs) {
        reserve += sequence.size();
    }

    // Pre-reserve vectors to prevent reallocation, which invalidates pointers
    tl_wait_semaphores.reserve(reserve);
    tl_wait_vals.reserve(reserve);
    tl_signal_semaphores.reserve(reserve);
    tl_signal_vals.reserve(reserve);
    tl_submit_infos.reserve(reserve);
    submit_infos.reserve(reserve);
    stage_flags.reserve(reserve);

    لكل (ثابت خودکار& sequence : ctx->seqs) {
        لكل (ثابت خودکار& submission : sequence) {
            stage_flags.push_back({});
            idx++;
            tl_wait_vals.push_back({});
            tl_wait_semaphores.push_back({});
            tl_signal_vals.push_back({});
            tl_signal_semaphores.push_back({});
            لكل (size_t i = 0; i < submission.wait_semaphores.size(); i++) {
                stage_flags[idx].push_back(ctx->p->q->stage_flags);
                tl_wait_vals[idx].push_back(submission.wait_semaphores[i].value);
                tl_wait_semaphores[idx].push_back(submission.wait_semaphores[i].s);
            }
            لكل (size_t i = 0; i < submission.signal_semaphores.size(); i++) {
                tl_signal_vals[idx].push_back(submission.signal_semaphores[i].value);
                tl_signal_semaphores[idx].push_back(submission.signal_semaphores[i].s);
            }
            tl_submit_infos.push_back({
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_vals[idx].انکڑے(),
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_vals[idx].انکڑے(),
            });
            tl_submit_infos[idx].sType = vk::StructureType::eTimelineSemaphoreSubmitInfo;
            tl_submit_infos[idx].p_اگلا = خالی_سوچک;
            vk::SubmitInfo si{
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_semaphores[idx].انکڑے(),
                stage_flags[idx].انکڑے(),
                1,
                &submission.بفر->buf,
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_semaphores[idx].انکڑے(),
            };
            si.سیٹ_PNext(&tl_submit_infos[idx]);
            submit_infos.push_back(si);
        }
    }

    ctx->p->q->handle->submit(submit_infos, fence);

    ctx->seqs.clear();
}

uint32_t ggml_vk_لبھو_قطار_family_اشاریہ(std::vector<vk::قطار_Family_Properties>& قطار_family_props, ثابت vk::قطار_Flags& required, ثابت vk::قطار_Flags& avoid, int32_t compute_اشاریہ, uint32_t min_عدد_queues) {
    VK_LOG_DEBUG("ggml_vk_find_queue_family_index()");
    ثابت uint32_t qfsize = قطار_family_props.size();

    // Try with avoid preferences first
    لكل (uint32_t i = 0; i < qfsize; i++) {
        إذا (قطار_family_props[i].قطار_گنتی >= min_عدد_queues && (compute_اشاریہ < 0 || i != (uint32_t) compute_اشاریہ) && قطار_family_props[i].قطار_Flags_2 & required && !(قطار_family_props[i].قطار_Flags_2 & avoid)) {
            أرجع i;
        }
    }

    // Fall back to only required
    لكل (size_t i = 0; i < qfsize; i++) {
        إذا (قطار_family_props[i].قطار_گنتی >= min_عدد_queues && (compute_اشاریہ < 0 || i != (uint32_t) compute_اشاریہ) && قطار_family_props[i].قطار_Flags_2 & required) {
            أرجع i;
        }
    }

    // Fall back to reusing compute queue
    لكل (size_t i = 0; i < qfsize; i++) {
        إذا (قطار_family_props[i].قطار_گنتی >= min_عدد_queues && قطار_family_props[i].قطار_Flags_2 & required) {
            أرجع i;
        }
    }

    // Fall back to ignoring min_num_queries
    لكل (size_t i = 0; i < qfsize; i++) {
        إذا (قطار_family_props[i].قطار_Flags_2 & required) {
            أرجع i;
        }
    }

    // All commands that are allowed on a queue that supports transfer operations are also allowed on a queue that supports either graphics or compute operations.
    // Thus, if the capabilities of a queue family include VK_QUEUE_GRAPHICS_BIT or VK_QUEUE_COMPUTE_BIT, then reporting the VK_QUEUE_TRANSFER_BIT capability separately for that queue family is optional.
    إذا (compute_اشاریہ >= 0) {
        أرجع compute_اشاریہ;
    }

    std::cerr << "ggml_vulkan: No suitable queue family index found." << std::سطر_انت;

    لكل(خودکار &q_family : قطار_family_props) {
        std::cerr << "Queue number: "  + std::to_string(q_family.قطار_گنتی) << " flags: " + to_string(q_family.قطار_Flags_2) << std::سطر_انت;
    }
    abort();
}

std::unique_ptr<vk_قطار> ggml_vk_create_قطار(vk_device& device, uint32_t قطار_family_اشاریہ, uint32_t قطار_اشاریہ, vk::PipelineStageFlags&& stage_flags, منطقي transfer_only) {
    VK_LOG_DEBUG("ggml_vk_create_queue()");
    std::lock_guard<std::recursive_mutex> guard(device->mutex);

    خودکار q = std::make_unique<vk_قطار>();
    q->قطار_family_اشاریہ = قطار_family_اشاریہ;
    q->transfer_only = transfer_only;

    std::shared_ptr<vk_قطار_handle> h;
    vk::Device_قطار_Info2 قطار_info2{};
    قطار_info2.قطار_Family_اشاریہ = قطار_family_اشاریہ;
    قطار_info2.قطار_اشاریہ_2 = قطار_اشاریہ;

    إذا (device->has_internally_synchronized_queues) {
        h = std::make_shared<vk_قطار_handle_unsynchronized>();
        قطار_info2.flags = eInternallySynchronizedKHR;
    } وإلا {
        h = std::make_shared<vk_قطار_handle_synchronized>();
    }

    h->queue = device->device.getQueue2(قطار_info2);
    h->device = device;
    // Avoid concurrent submissions on NVIDIA due to driver bug.
    إذا (device->vendor_id == VK_VENDOR_ID_NVIDIA) {
        h->device_submit_mutex = &device->قطار_submit_mutex;
    }
    q->handle = h;

    q->cmd_pool.init(device, q.get());

    q->stage_flags = stage_flags;
    أرجع q;
}

std::unique_ptr<vk_قطار> ggml_vk_create_aliased_قطار(vk_device& device, ثابت std::unique_ptr<vk_قطار>& سروت) {
    std::lock_guard<std::recursive_mutex> guard(device->mutex);
    خودکار q = std::make_unique<vk_قطار>();
    q->handle = سروت->handle;
    q->قطار_family_اشاریہ = سروت->قطار_family_اشاریہ;
    q->stage_flags = سروت->stage_flags;
    q->transfer_only = سروت->transfer_only;
    q->cmd_pool.init(device, q.get());
    أرجع q;
}

vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p) {
    vk_context نتیجہ = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_context(" << نتیجہ << ")");
    ctx->gc.contexts.emplace_back(نتیجہ);
    نتیجہ->p = &p;
    أرجع نتیجہ;
}

vk_context ggml_vk_create_temporary_context(vk_command_pool& p) {
    vk_context نتیجہ = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_temporary_context(" << نتیجہ << ")");
    نتیجہ->p = &p;
    أرجع نتیجہ;
}

ساكن vk_semaphore * ggml_vk_create_ثنائی_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::e_ثنائی, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.سیٹ_PNext(&tci);
    vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
    ctx->gc.semaphores.push_back({ semaphore, 0 });
    أرجع &ctx->gc.semaphores[ctx->gc.semaphores.size() - 1];
}

ساكن vk_semaphore * ggml_vk_create_timeline_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    إذا (ctx->semaphore_idx >= ctx->gc.tl_semaphores.size()) {
        vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
        vk::SemaphoreCreateInfo ci{};
        ci.سیٹ_PNext(&tci);
        vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
        ctx->gc.tl_semaphores.push_back({ semaphore, 0 });
    }
    أرجع &ctx->gc.tl_semaphores[ctx->semaphore_idx++];
}

ساكن vk::Event ggml_vk_create_event(ggml_backend_vk_context * ctx) {
    إذا (ctx->event_idx >= ctx->gc.events.size()) {
        ctx->gc.events.push_back(ctx->device->device.createEvent({}));
    }
    أرجع ctx->gc.events[ctx->event_idx++];
}

فراغ ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_command_pool_cleanup()");

    // Requires command buffers to be done
    device->device.resetCommandPool(p.pool);
    // Don't clear the command buffers and mark them as not in use.
    // This allows us to reuse them
    لكل (خودکار& cmd_بفر : p.cmd_buffers) {
        cmd_بفر.وچ_use = خطأ;
    }
}

فراغ ggml_vk_قطار_command_pools_cleanup(vk_device& device) {
    VK_LOG_DEBUG("ggml_vk_queue_command_pools_cleanup()");

    // Arbitrary frequency to cleanup/reuse command buffers
    ساكن constexpr uint32_t cleanup_frequency = 10;

    إذا (device->compute_قطار && device->compute_قطار->cmd_pool.buffers_وچ_use() >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->compute_قطار->cmd_pool);
    }
    إذا (device->transfer_قطار && device->transfer_قطار->cmd_pool.buffers_وچ_use() >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->transfer_قطار->cmd_pool);
    }
}

vk_subbuffer ggml_vk_subbuffer(ثابت ggml_backend_vk_context* ctx, ثابت vk_بفر& buf, size_t وتھ) {
    أرجع { buf, وتھ, ggml_vk_get_max_بفر_range(ctx, buf, وتھ) };
}

فراغ ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx) {
    VK_LOG_DEBUG("ggml_vk_sync_buffers()");

    ثابت منطقي transfer_قطار = subctx->p->q->transfer_only;

    إذا (ctx) {
        ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_ونڈو_k_need_sync = خطأ;
    }

    subctx->s->بفر->buf.pipelineBarrier(
        subctx->p->q->stage_flags,
        subctx->p->q->stage_flags,
        {},
        { {
          { !transfer_قطار ? (vk::Access_Flag_بٹ::eShaderRead | vk::Access_Flag_بٹ::eShaderWrite | vk::Access_Flag_بٹ::eTransferRead | vk::Access_Flag_بٹ::eTransferWrite) : (vk::Access_Flag_بٹ::eTransferRead | vk::Access_Flag_بٹ::eTransferWrite) },
          { !transfer_قطار ? (vk::Access_Flag_بٹ::eShaderRead | vk::Access_Flag_بٹ::eShaderWrite | vk::Access_Flag_بٹ::eTransferRead | vk::Access_Flag_بٹ::eTransferWrite) : (vk::Access_Flag_بٹ::eTransferRead | vk::Access_Flag_بٹ::eTransferWrite) }
        } },
        {},
        {}
    );
}

ساكن فراغ ggml_vk_reset_event(vk_context& ctx, vk::Event& event) {
    VK_LOG_DEBUG("ggml_vk_set_event()");

    ctx->s->بفر->buf.resetEvent(
        event,
        ctx->p->q->stage_flags
    );
}

فراغ ggml_vk_سیٹ_event(vk_context& ctx, vk::Event& event) {
    VK_LOG_DEBUG("ggml_vk_set_event()");

    ctx->s->بفر->buf.سیٹ_Event(
        event,
        ctx->p->q->stage_flags
    );
}

فراغ ggml_vk_wait_events(vk_context& ctx, std::vector<vk::Event>&& events) {
    VK_LOG_DEBUG("ggml_vk_wait_events()");
    إذا (events.خالی_تھاواں()) {
        أرجع;
    }

    ctx->s->بفر->buf.waitEvents(
        events,
        ctx->p->q->stage_flags,
        ctx->p->q->stage_flags,
        {},
        {},
        {}
    );
}

ساكن vk_fa_tuning_params get_fa_tuning_params_scalar(ثابت vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_صفاں, uint32_t n_kv, ggml_type k_type, ggml_type v_type, منطقي f32acc) {

    vk_fa_tuning_params نتیجہ{};
    نتیجہ.path = FA_SCALAR;

    إذا (device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Disable subgroup use due to performance issues when enforcing subgroup sizes
        نتیجہ.subgroup_ناپ = 32;
        نتیجہ.disable_subgroups = سچ;
    } وإلا إذا (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN) {
        نتیجہ.subgroup_ناپ = n_صفاں < 4 ? 32 : device->subgroup_ناپ;
    } وإلا {
        نتیجہ.subgroup_ناپ = device->subgroup_ناپ;
    }

    // Row split splits the workgroup so that synchronization only has to happen within subgroups, which avoids barriers
    uint32_t صف_ونڈو_max_hsk = 64;
    إذا (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN && !device->uma) {
        صف_ونڈو_max_hsk = n_صفاں <= 8 ? 64 : 128;
    }
    نتیجہ.صف_ونڈو = (n_صفاں < 4 || hsk <= صف_ونڈو_max_hsk) ? 1 : 4;

    إذا (نتیجہ.subgroup_ناپ > 32 && (n_صفاں < 4 || hsk < (نتیجہ.صف_ونڈو == 1 ? 128 : 64))) {
        نتیجہ.workgroup_ناپ = نتیجہ.subgroup_ناپ * 2;
    } وإلا {
        نتیجہ.workgroup_ناپ = نتیجہ.subgroup_ناپ * 4;
    }

    ثابت uint32_t D = hsk | hsv;

    ثابت منطقي reduce_بلاک_صفاں = D & 8 || n_kv < 1024 || device->vendor_id == VK_VENDOR_ID_INTEL;

    إذا (n_صفاں == 1) {
        نتیجہ.بلاک_صفاں = 1;
        نتیجہ.بلاک_خانے_گنتی = 64;
    } وإلا {
        // row_split 1 means higher register use per row, so block size has to be adjusted
        إذا (نتیجہ.صف_ونڈو == 1) {
            نتیجہ.بلاک_صفاں = n_صفاں == 2 ? 2 : ((n_صفاں <= 4 || reduce_بلاک_صفاں) ? 4 : 8);
        } وإلا {
            نتیجہ.بلاک_صفاں = n_صفاں <= 4 ? 4 : ((n_صفاں <= 8 || reduce_بلاک_صفاں) ? 8 : 16);
        }

        نتیجہ.بلاک_خانے_گنتی = (D & 8) ? 64 : 32;
    }

    ثابت uint32_t D_lsb = D ^ (D & (D-1));  // extract lowest set bit

    نتیجہ.d_ونڈو = std::min(std::min(نتیجہ.subgroup_ناپ, 8u), D_lsb / 4);

    نتیجہ.shmem_staging = (device->vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) ? 1 : 0;

    إذا (!reduce_بلاک_صفاں && !ggml_vk_flash_attn_scalar_shmem_support(device, نتیجہ, hsk, hsv, f32acc, k_type, v_type)) {
        نتیجہ.بلاک_صفاں /= 2;
    }

    // On AMD RDNA, for small head sizes and big batch size the shader uses few registers, so too many subgroups get scheduled
    // at once and end up thrashing the cache. Fix this by setting a large (unused) shmem buffer that reduces occupancy.
    // This targets an occupancy of 4 subgroups per SIMD.
    إذا (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.max_Compute_Shared_حافظہ_ناپ == 65536) {
        إذا (device->architecture != AMD_GCN && n_صفاں >= 64 && hsk <= 128) {
            // 30kb target for hsk > 64, 26kb for <= 64 due to smaller workgroup size
            // Values are guessed, tested on RDNA2
            نتیجہ.حد_بندی_occupancy_shmem = (hsk <= 64 ? 26 : 30) * 1024 / 4 / 4;
        } وإلا إذا (device->architecture == AMD_GCN && n_صفاں <= 8 && hsk >= 256) {
            // Same thing for GCN, with an occupancy target of 2 subgroups per SIMD.
            // Here low-batch FA with large head size is affected.
            // n_rows < 4 switch because workgroup size switches from 128 to 256 there.
            نتیجہ.حد_بندی_occupancy_shmem = (n_صفاں < 4 ? 14 : 26) * 1024 / 4 / 4;
        }
    }

    أرجع نتیجہ;
}

ساكن vk_fa_tuning_params get_fa_tuning_params_coopmat1(ثابت vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_صفاں, uint32_t n_kv, ggml_type k_type, ggml_type v_type, منطقي f32acc) {
    GGML_بن_ورتیا(n_صفاں);
    GGML_بن_ورتیا(n_kv);
    GGML_بن_ورتیا(k_type);
    GGML_بن_ورتیا(v_type);
    GGML_بن_ورتیا(f32acc);

    vk_fa_tuning_params نتیجہ{};
    نتیجہ.path = FA_COOPMAT1;

    ثابت uint32_t D = hsk | hsv;

    ثابت uint32_t coopmat_بلاک_صفاں = 16;
    ثابت uint32_t coopmat_بلاک_خانے_گنتی = 16;

    ثابت uint32_t عدد_subgroups = 4;

    نتیجہ.بلاک_صفاں = coopmat_بلاک_صفاں;
    نتیجہ.بلاک_خانے_گنتی = coopmat_بلاک_خانے_گنتی * عدد_subgroups;
    نتیجہ.صف_ونڈو = عدد_subgroups;
    نتیجہ.subgroup_ناپ = device->subgroup_ناپ;
    نتیجہ.workgroup_ناپ = عدد_subgroups * نتیجہ.subgroup_ناپ;

    ثابت uint32_t D_lsb = D ^ (D & (D-1));  // extract lowest set bit
    نتیجہ.d_ونڈو = std::min(std::min(نتیجہ.subgroup_ناپ, 8u), D_lsb / 4);

    نتیجہ.shmem_staging = (device->vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) ? 1 : 0;

    أرجع نتیجہ;
}

ساكن vk_fa_tuning_params get_fa_tuning_params_coopmat2(ثابت vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_صفاں, uint32_t n_kv, ggml_type k_type, ggml_type v_type, منطقي f32acc) {
    GGML_بن_ورتیا(n_kv);
    GGML_بن_ورتیا(f32acc);

    vk_fa_tuning_params نتیجہ{};
    نتیجہ.path = FA_COOPMAT2;

    ثابت uint32_t D = hsk | hsv;

    ثابت منطقي small_صفاں = n_صفاں < 32;

    إذا (small_صفاں) {
        نتیجہ.بلاک_صفاں = 32;
        نتیجہ.بلاک_خانے_گنتی = 32;
    } وإلا إذا (ggml_is_quantized(k_type) || ggml_is_quantized(v_type) || hsk >= 256 || hsv >= 256) {
        نتیجہ.بلاک_صفاں = (hsk >= 512 || hsv >= 512) ? 32 : 64;
        نتیجہ.بلاک_خانے_گنتی = 32;
    } وإلا {
        نتیجہ.بلاک_صفاں = 64;
        نتیجہ.بلاک_خانے_گنتی = 64;
    }

    نتیجہ.subgroup_ناپ = device->subgroup_ناپ;
    نتیجہ.workgroup_ناپ = (small_صفاں && (D % 32) == 0) ? 256 : 128;

    أرجع نتیجہ;
}

vk_fa_tuning_params get_fa_tuning_params(ثابت vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_صفاں, uint32_t n_kv, ggml_type k_type, ggml_type v_type, منطقي f32acc) {
    Fa_کوڈ_رستہ path = device->coopmat2 ? FA_COOPMAT2 :
                      device->coopmat1_fa_support ? FA_COOPMAT1 : FA_SCALAR;

    إذا (path == FA_COOPMAT2 && k_type == GGML_TYPE_BF16 && !device->coopmat2_bf16_support) {
        path = FA_COOPMAT1;
    }
    إذا (path == FA_COOPMAT1 && k_type == GGML_TYPE_BF16 && !device->coopmat_bf16_support) {
        path = FA_SCALAR;
    }

    إذا (path == FA_COOPMAT1 && device->architecture == vk_device_architecture::NVIDIA_TURING) {
        // Nvidia compiler bug, see https://github.com/ggml-org/llama.cpp/pull/19075#issuecomment-3820716090
        path = FA_SCALAR;
    }

    إذا (path == FA_COOPMAT1) {
        منطقي shape_ok = (f32acc && device->coopmat_support_16x16x16_f32acc) ||
                        (!f32acc && device->coopmat_support_16x16x16_f16acc);
        ثابت vk_fa_tuning_params params = get_fa_tuning_params_coopmat1(device, hsk, hsv, n_صفاں, n_kv, k_type, v_type, f32acc);
        منطقي shmem_ok = ggml_vk_flash_attn_coopmat_shmem_support(device, params, hsk, hsv, f32acc, k_type, v_type);

        إذا (!shape_ok || !shmem_ok) {
            path = FA_SCALAR;
        }
    }

    // scalar is faster than coopmat when N==1
    إذا (n_صفاں == 1 && (path == FA_COOPMAT1 || path == FA_COOPMAT2)) {
        path = FA_SCALAR;
    }

    بدل (path) {
    حالة FA_SCALAR:
        أرجع get_fa_tuning_params_scalar(device, hsk, hsv, n_صفاں, n_kv, k_type, v_type, f32acc);
    حالة FA_COOPMAT1:
        أرجع get_fa_tuning_params_coopmat1(device, hsk, hsv, n_صفاں, n_kv, k_type, v_type, f32acc);
    حالة FA_COOPMAT2:
        أرجع get_fa_tuning_params_coopmat2(device, hsk, hsv, n_صفاں, n_kv, k_type, v_type, f32acc);
    افتراضي:
        ارم std::runtime_error("unsupported FaCodePath");
    }
}

vk_fa_pipeline_state get_fa_pipeline_state(ثابت vk_device& device, ثابت vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, منطقي aligned, منطقي f32acc,
                                                  منطقي use_نقاب, منطقي use_نقاب_opt, منطقي use_logit_softcap, منطقي use_sparse, ggml_type k_type, ggml_type v_type) {
    ثابت منطقي old_amd_windows = device->vendor_id == VK_VENDOR_ID_AMD && device->driver_id == vk::DriverId::eAmdProprietary &&
                                 (device->architecture == AMD_GCN || device->architecture == AMD_RDNA1 || device->architecture == AMD_RDNA2);

    uint32_t flags = (use_نقاب_opt      ? 1 : 0) |
                     (use_نقاب          ? 2 : 0) |
                     (use_logit_softcap ? 4 : 0) |
                     (old_amd_windows   ? 8 : 0) |
                     (use_sparse        ? 16 : 0);

    ثابت uint32_t subgroup_ناپ = params.disable_subgroups ? 0 : params.subgroup_ناپ;

    أرجع vk_fa_pipeline_state{hsk, hsv, params.بلاک_صفاں, params.بلاک_خانے_گنتی, params.d_ونڈو, params.صف_ونڈو, params.shmem_staging, params.path, params.workgroup_ناپ, subgroup_ناپ, aligned, f32acc, flags, params.حد_بندی_occupancy_shmem, k_type, v_type};
}

ساكن uint32_t fa_بلاک_بائٹاں(ggml_type t) {
    إذا (t == GGML_TYPE_F32) {
        أرجع 16u;
    }
    أرجع (uint32_t) ggml_type_ناپ(t);
}

ساكن std::vector<uint32_t> get_fa_spec_constants(ثابت vk_fa_pipeline_state& state) {
    أرجع {
        /* 0 WorkGroupSize   */ state.workgroup_ناپ,
        /* 1 Br              */ state.Br,
        /* 2 Bc              */ state.Bc,
        /* 3 HSK             */ state.HSK,
        /* 4 HSV             */ state.HSV,
        /* 5 Clamp           */ static_cast<uint32_t>(!state.aligned),
        /* 6 D_split         */ state.D_ونڈو,
        /* 7 row_split       */ state.صف_ونڈو,
        /* 8 SubGroupSize    */ state.subgroup_ناپ,
        /* 9 SHMEM_STAGING   */ state.shmem_staging ? 1u : 0u,
        /*10 Flags           */ state.flags,
        /*11 LIMIT_OCCUPANCY_SHMEM */ state.حد_بندی_occupancy_shmem,
        /*12 FaTypeK         */ static_cast<uint32_t>(state.k_type),
        /*13 FaTypeV         */ static_cast<uint32_t>(state.v_type),
        /*14 FaBlockBytesK   */ fa_بلاک_بائٹاں(state.k_type),
        /*15 FaBlockBytesV   */ fa_بلاک_بائٹاں(state.v_type),
    };
}

ساكن منطقي ggml_vk_matmul_shmem_support(ثابت vk_device& device, ثابت std::vector<uint32_t>& warptile, منطقي mul_mat_id, ggml_type src0_type) {

    uint32_t lut_ناپ = 0;
    بدل (src0_type) {
    حالة GGML_TYPE_IQ1_S:
    حالة GGML_TYPE_IQ1_M:
        // Regular matmul uses the compact uint16_t IQ1 grid; the expanded
        // uint32_t grid is only enabled for the q8_1/int-dot vector path.
        lut_ناپ = 2*2048;
        اكسر;
    حالة GGML_TYPE_IQ2_XXS:
        lut_ناپ = 8*256;
        اكسر;
    حالة GGML_TYPE_IQ2_XS:
        lut_ناپ = 8*512;
        اكسر;
    حالة GGML_TYPE_IQ2_S:
        lut_ناپ = 8*1024;
        اكسر;
    حالة GGML_TYPE_IQ3_XXS:
        lut_ناپ = 4*256;
        اكسر;
    حالة GGML_TYPE_IQ3_S:
        lut_ناپ = 4*512;
        اكسر;
    حالة GGML_TYPE_IQ4_NL:
    حالة GGML_TYPE_IQ4_XS:
    حالة GGML_TYPE_MXFP4:
        lut_ناپ = 4*16;
        اكسر;
    حالة GGML_TYPE_NVFP4:
        // Same kvalues budget as MXFP4 plus ue4m3_fp32_lut[128] (types.glsl, DATA_A_NVFP4).
        lut_ناپ = 4*16 + 128u * (uint32_t)حجم(عائم);
        اكسر;
    افتراضي:
        اكسر;
    }

    // Needs to be kept up to date on shader changes
    // Needs to stay aligned with ggml_vk_mul_mm_spec.
    ثابت منطقي intel_shmem_stride_pad_zero = device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
                                              device->driver_id == vk::DriverId::eIntelProprietaryWindows;
    ثابت uint32_t bank_conflict_وتھ = intel_shmem_stride_pad_zero ? 0 : (device->coopmat_support ? 8 : 1);
    ثابت uint32_t type_ناپ = device->fp16 ? حجم(ggml_fp16_t) : حجم(عائم);
    ثابت uint32_t warps = warptile[0] / warptile[10];

    ثابت uint32_t لدو_bufs = (warptile[1] + warptile[2]) * (warptile[3] + bank_conflict_وتھ) * type_ناپ;
    ثابت uint32_t mmid_صف_ids = mul_mat_id ? (warptile[2] * 2 * حجم(uint16_t)) : 0;
    ثابت uint32_t coopmat_stage = device->coopmat_support ? warptile[7] * warptile[8] / warps * حجم(عائم) : 0;

    ثابت uint32_t کل_ناپ = لدو_bufs + mmid_صف_ids + coopmat_stage + lut_ناپ;
    ثابت منطقي supported = کل_ناپ <= device->properties.limits.max_Compute_Shared_حافظہ_ناپ;

    VK_LOG_DEBUG("ggml_vk_matmul_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", supported=" << supported);

    أرجع supported;
}

ساكن منطقي ggml_vk_matmul_int_shmem_support(ثابت vk_device& device, ثابت std::vector<uint32_t>& warptile, منطقي mul_mat_id, ggml_type src0_type) {

    // FLOAT_TYPE in the shader is float16_t with fp16 support, otherwise float.
    ثابت uint32_t fp_ناپ   = device->fp16 ? 2u : 4u;
    ثابت uint32_t fp_align  = fp_ناپ;
    ثابت uint32_t fp2_ناپ  = 2u * fp_ناپ;
    ثابت uint32_t fp2_align = device->fp16 ? 4u : 8u;

    بنية member { uint32_t size, align; };
    خودکار std430_ناپ = [](std::initializer_list<member> members) {
        uint32_t off = 0, struct_align = 1;
        لكل (ثابت خودکار &m : members) {
            off = (off + m.align - 1) & ~(m.align - 1);
            off += m.size;
            struct_align = std::max(struct_align, m.align);
        }
        أرجع (off + struct_align - 1) & ~(struct_align - 1);
    };

    uint32_t بلاک_a_ناپ = 0;
    بدل (src0_type) {
        حالة GGML_TYPE_Q2_0:    بلاک_a_ناپ = std430_ناپ({{32, 4}, {fp_ناپ,  fp_align}});                  اكسر; // qs[8] + dm
        حالة GGML_TYPE_Q4_0:    بلاک_a_ناپ = std430_ناپ({{16, 4}, {fp_ناپ,  fp_align}});                  اكسر; // qs[16/4] + dm
        حالة GGML_TYPE_Q4_1:    بلاک_a_ناپ = std430_ناپ({{16, 4}, {fp2_ناپ, fp2_align}});                 اكسر; // qs[16/4] + dm(vec2)
        حالة GGML_TYPE_Q5_0:    بلاک_a_ناپ = std430_ناپ({{16, 4}, {4, 4}, {fp_ناپ,  fp_align}});          اكسر; // qs[16/4] + qh + dm
        حالة GGML_TYPE_Q5_1:    بلاک_a_ناپ = std430_ناپ({{16, 4}, {4, 4}, {fp2_ناپ, fp2_align}});         اكسر; // qs[16/4] + qh + dm(vec2)
        حالة GGML_TYPE_Q8_0:    بلاک_a_ناپ = std430_ناپ({{32, 4}, {fp_ناپ,  fp_align}});                  اكسر; // qs[8] + dm
        حالة GGML_TYPE_IQ4_XS:  بلاک_a_ناپ = std430_ناپ({{32, 4}, {fp_ناپ,  fp_align}});                  اكسر; // qs[8] + d
        حالة GGML_TYPE_MXFP4:   بلاک_a_ناپ = std430_ناپ({{32, 4}, {fp_ناپ,  fp_align}});                  اكسر; // qs[8] + d
        حالة GGML_TYPE_IQ4_NL:  بلاک_a_ناپ = std430_ناپ({{32, 4}, {fp_ناپ,  fp_align}});                  اكسر; // qs[8] + d
        حالة GGML_TYPE_NVFP4:   بلاک_a_ناپ = std430_ناپ({{32, 4}, {fp2_ناپ, fp2_align}});                 اكسر; // qs[8] + d_scales(vec2)
        حالة GGML_TYPE_Q2_K:    بلاک_a_ناپ = std430_ناپ({{ 8, 4}, {2, 2}, {fp2_ناپ, fp2_align}});         اكسر; // qs[2] + scales(u8vec2) + dm(vec2)
        حالة GGML_TYPE_Q3_K:    بلاک_a_ناپ = std430_ناپ({{16, 4}, {fp2_ناپ, fp2_align}});                 اكسر; // qs[4] + d_scales(vec2)
        حالة GGML_TYPE_Q4_K:    بلاک_a_ناپ = std430_ناپ({{16, 4}, {fp2_ناپ, fp2_align}});                 اكسر; // qs[4] + dm(vec2)
        حالة GGML_TYPE_Q5_K:    بلاک_a_ناپ = std430_ناپ({{32, 4}, {fp2_ناپ, fp2_align}});                 اكسر; // qs[8] + dm(vec2)
        حالة GGML_TYPE_Q6_K:    بلاک_a_ناپ = std430_ناپ({{32, 4}, {fp2_ناپ, fp2_align}});                 اكسر; // qs[8] + d_scales(vec2)
        حالة GGML_TYPE_IQ3_S:   بلاک_a_ناپ = std430_ناپ({{32, 4}, {fp_ناپ,  fp_align}});                  اكسر; // qs[8] + d
        افتراضي:
            أرجع خطأ;
    }

    // IQ3_S also copies its 512-entry grid into shared memory (types.glsl, init_iq_shmem)
    ثابت uint32_t lut_ناپ = (src0_type == GGML_TYPE_IQ3_S) ? 4*512 : 0;

    // block_b_cache: { int32_t qs[8]; FLOAT_TYPEV2 ds; }
    ثابت uint32_t بلاک_b_ناپ = std430_ناپ({{32, 4}, {fp2_ناپ, fp2_align}});

    ثابت uint32_t BM = warptile[1];
    ثابت uint32_t BN = warptile[2];
    // mul_mmq.comp: BK_STEP=1 for MUL_MAT_ID, 4 otherwise.
    ثابت uint32_t BK_قدم = mul_mat_id ? 1u : 4u;

    ثابت uint32_t buf_a_ناپ = BM * BK_قدم * بلاک_a_ناپ;
    ثابت uint32_t buf_b_ناپ = BN * BK_قدم * بلاک_b_ناپ;
    ثابت uint32_t mmid_صف_ids = mul_mat_id ? (BN * 2u * (uint32_t)حجم(uint16_t)) : 0u;

    ثابت uint32_t کل_ناپ = buf_a_ناپ + buf_b_ناپ + mmid_صف_ids + lut_ناپ;
    ثابت منطقي supported = کل_ناپ <= device->properties.limits.max_Compute_Shared_حافظہ_ناپ;

    VK_LOG_DEBUG("ggml_vk_matmul_int_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", total=" << کل_ناپ << ", supported=" << supported);

    أرجع supported;
}

ساكن منطقي ggml_vk_matmul_cm1_int_shmem_support(ثابت vk_device& device, ثابت std::vector<uint32_t>& warptile, منطقي mul_mat_id, ggml_type src0_type) {

    منطقي kscales2 = خطأ;    // two scale sets per block
    منطقي has_dm   = خطأ;    // d+m as vec2 + b-side sum
    منطقي has_kvalues = خطأ;
    بدل (src0_type) {
        حالة GGML_TYPE_Q4_0: حالة GGML_TYPE_Q5_0: حالة GGML_TYPE_Q8_0:
            اكسر;
        حالة GGML_TYPE_Q4_1: حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q4_K: حالة GGML_TYPE_Q5_K:
            has_dm = سچ;                          اكسر;
        حالة GGML_TYPE_IQ4_NL: حالة GGML_TYPE_IQ4_XS: حالة GGML_TYPE_MXFP4:
            has_kvalues = سچ;                     اكسر;
        حالة GGML_TYPE_Q3_K: حالة GGML_TYPE_Q6_K:
            kscales2 = سچ;                        اكسر;
        حالة GGML_TYPE_NVFP4:
            kscales2 = سچ; has_kvalues = سچ;    اكسر;
        افتراضي:
            أرجع خطأ;
    }

    ثابت uint32_t BM         = warptile[1];
    ثابت uint32_t BN         = warptile[2];

    ثابت uint32_t BK      = 32;
    ثابت uint32_t BK_قدم = mul_mat_id ? 2u : 4u;
    ثابت uint32_t QPITCH  = BK_قدم * (BK / 4u) + 4u;
    ثابت uint32_t KSCALES = kscales2 ? 2u : 1u;

    uint32_t کل = 0;
    کل += BM * QPITCH * (uint32_t)حجم(uint32_t);   // buf_a_qs
    کل += BN * QPITCH * (uint32_t)حجم(uint32_t);   // buf_b_qs
    کل += has_dm ? (BM * BK_قدم * 2u * (uint32_t)حجم(عائم))   // buf_a_dm (vec2)
                    : (BM * BK_قدم * KSCALES * (uint32_t)حجم(عائم)); // buf_a_d
    کل += BN * BK_قدم * (uint32_t)حجم(عائم);     // buf_b_d
    إذا (has_dm) {
        کل += BN * BK_قدم * (uint32_t)حجم(عائم); // buf_b_s
    }
    إذا (has_kvalues) {
        کل += 16u * (uint32_t)حجم(int8_t);         // cm1_kvalues[16]
    }
    إذا (src0_type == GGML_TYPE_NVFP4 && !device->ocp_fp4) {
        کل += 128u * (uint32_t)حجم(عائم);         // ue4m3_fp32_lut[128]
    }
    إذا (mul_mat_id) {
        کل += BN * 2u * (uint32_t)حجم(uint16_t);   // row_ids[BN] (u16vec2)
    }

    ثابت منطقي supported = کل <= device->properties.limits.max_Compute_Shared_حافظہ_ناپ;

    VK_LOG_DEBUG("ggml_vk_matmul_cm1_int_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", total=" << کل << ", supported=" << supported);

    أرجع supported;
}

ساكن ثابت std::unordered_map<std::string, uint32_t> rdna1_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
    {"argmax", 64}, {"mul_mat_vec", 64},
    {"mul_mat_vec_f16", 32}, {"mul_mat_vec_f32_f16", 32}
};

ساكن ثابت std::unordered_map<std::string, uint32_t> rdna2_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
};

ساكن std::vector<GpuPipelineConfig> gpu_pipeline_configs = {
    {
        vk_device_architecture::AMD_RDNA1,
        {
            rdna1_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_ناپ
    },
    {
        vk_device_architecture::AMD_RDNA2,
        {
            rdna2_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_ناپ
    },
};

uint32_t get_subgroup_ناپ(ثابت std::string &pipeline_name, ثابت vk_device_architecture &arch) {
    لكل (ثابت خودکار &config : gpu_pipeline_configs) {
        إذا (config.arch == arch) {
            خودکار pipIt = config.pipelines.find(pipeline_name);
            إذا (pipIt != config.pipelines.end()) {
                أرجع pipIt->second;
            }
            std::vector<std::pair<std::string, uint32_t>> sorted_pipelines(config.pipelines.begin(), config.pipelines.end());
            std::sort(sorted_pipelines.begin(), sorted_pipelines.end(),
                      [](ثابت خودکار &a, ثابت خودکار &b) { أرجع a.پہلا.size() > b.پہلا.size(); });
            لكل (ثابت خودکار &اندراج : sorted_pipelines) {
                إذا (pipeline_name.find(اندراج.پہلا) != std::string::npos) {
                    أرجع اندراج.second;
                }
            }
            أرجع config.default_subgroup_ناپ;
        }
    }
    أرجع 0; // If no matching configuration is found
}

ساكن منطقي ggml_vk_fa_type_needs_shmem(ggml_type type) {
    بدل (type) {
    حالة GGML_TYPE_IQ4_NL:
        أرجع سچ;
    افتراضي:
        أرجع خطأ;
    }
}

ساكن منطقي ggml_vk_fa_scalar_ورتدا_mmq(ثابت vk_device& device, ggml_type k_type, ggml_type v_type) {
#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    أرجع device->integer_dot_ضرب_حاصل && device->subgroup_clustered &&
           !ggml_vk_fa_type_needs_shmem(v_type) &&
           (k_type == GGML_TYPE_Q4_0 || k_type == GGML_TYPE_Q4_1 ||
            k_type == GGML_TYPE_Q5_0 || k_type == GGML_TYPE_Q5_1 ||
            k_type == GGML_TYPE_Q8_0);
#وإلا
    GGML_بن_ورتیا(device);
    GGML_بن_ورتیا(k_type);
    GGML_بن_ورتیا(v_type);
    أرجع خطأ;
#جے_انت
}

فراغ ggml_vk_لدو_shaders(vk_device& device, vk_pipeline requested) {
    VK_LOG_DEBUG("ggml_vk_load_shaders(" << device->name << ")");

    // some shaders have a minimum subgroup size
    ثابت uint32_t subgroup_ناپ_8 = std::max(device->subgroup_ناپ, 8u);
    ثابت uint32_t subgroup_ناپ_16 = std::max(device->subgroup_ناپ, 16u);
    ثابت uint32_t subgroup_ناپ_32 = std::max(device->subgroup_ناپ, 32u);

    // clamp WARP for l_/m_ warptiles so WM <= BM (breaks on subgroupSize > 64)
    ثابت uint32_t mm_warp_8  = std::min(subgroup_ناپ_8,  64u);
    ثابت uint32_t mm_warp_16 = std::min(subgroup_ناپ_16, 64u);

    ثابت uint32_t mul_mat_subgroup_ناپ = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_ناپ_control) ? device->subgroup_min_ناپ : device->subgroup_ناپ;
    ثابت uint32_t mul_mat_subgroup_ناپ_8 = std::max(mul_mat_subgroup_ناپ, 8u);
    ثابت uint32_t mul_mat_subgroup_ناپ_16 = std::max(mul_mat_subgroup_ناپ, 16u);
    ثابت uint32_t mul_mat_subgroup_ناپ_32 = std::max(mul_mat_subgroup_ناپ, 32u);
    ثابت uint32_t mul_mat_mm_warp_8  = std::min(mul_mat_subgroup_ناپ_8,  64u);
    ثابت uint32_t mul_mat_mm_warp_16 = std::min(mul_mat_subgroup_ناپ_16, 64u);

    ثابت منطقي subgroup_min_ناپ_16 = (!device->subgroup_ناپ_control && device->subgroup_ناپ >= 16) ||
                                      (device->subgroup_ناپ_control && device->subgroup_max_ناپ >= 16);

    // mulmat
    // Warptile layout (indices match mul_mm.comp constantIDs):
    //   [0..9]  : BLOCK_SIZE, BM, BN, BK, WM, WN, WMITER, TM, TN, TK
    //   [10]    : WARP / required_subgroup_size (read via WARP_SIZE_IDX)
    ساكن constexpr size_t WARP_ناپ_IDX = 10;
    std::vector<uint32_t> l_warptile, m_warptile, s_warptile,
                          l_warptile_id, m_warptile_id, s_warptile_id,
                          l_warptile_mmq, m_warptile_mmq, s_warptile_mmq,
                          l_warptile_mmq_int, m_warptile_mmq_int, s_warptile_mmq_int,
                          l_warptile_mmq_cm1_int, m_warptile_mmq_cm1_int, s_warptile_mmq_cm1_int,
                          l_warptile_mmq_cm1_int_k, m_warptile_mmq_cm1_int_k, s_warptile_mmq_cm1_int_k,
                          l_warptile_mmq_int_k, m_warptile_mmq_int_k, s_warptile_mmq_int_k,
                          l_warptile_mmq_k, m_warptile_mmq_k, s_warptile_mmq_k,
                          l_warptile_mmqid, m_warptile_mmqid, s_warptile_mmqid,
                          l_warptile_mmqid_int, m_warptile_mmqid_int, s_warptile_mmqid_int,
                          l_warptile_mmqid_int_k, m_warptile_mmqid_int_k, s_warptile_mmqid_int_k;
    std::array<uint32_t, 3> l_wg_denoms, m_wg_denoms, s_wg_denoms,
                            l_mmq_wg_denoms, m_mmq_wg_denoms, s_mmq_wg_denoms,
                            l_mmq_wg_denoms_k, m_mmq_wg_denoms_k, s_mmq_wg_denoms_k,
                            l_mmq_cm1_wg_denoms_k, m_mmq_cm1_wg_denoms_k, s_mmq_cm1_wg_denoms_k,
                            l_mmqid_wg_denoms, m_mmqid_wg_denoms, s_mmqid_wg_denoms;

    uint32_t l_align, m_align, s_align;

    // RDNA3.5 preferred wave32 here
    ثابت منطقي cm1_use_wave32 = device->vendor_id == VK_VENDOR_ID_AMD &&
                                device->subgroup_ناپ_control &&
                                device->subgroup_min_ناپ <= 32 && device->subgroup_max_ناپ >= 32;
    ثابت uint32_t cm1_sg = cm1_use_wave32 ? 32 : device->subgroup_ناپ;

    vk_pipeline wait_pipeline;
    CompileTask claimed_task {};
    منطقي has_claimed_task = خطأ;

    // The rest of the walk reads and writes shared device state, so hold the
    // lock until we're done deciding what to compile.
    std::unique_lock<std::mutex> compile_جندرا(device->compile_mutex);

    إذا (device->coopmat2) {
        // spec constants and tile sizes for non-quant matmul/matmul_id
        l_warptile = { 256, 128, 256, 64, 1 };
        m_warptile = { 256, 128, 128, 64, 0 };
        s_warptile = { 128,  64,  64, 64, 0 };
        l_wg_denoms = {128, 256, 1 };
        m_wg_denoms = {128, 128, 1 };
        s_wg_denoms = { 64,  64, 1 };

        // spec constants and tile sizes for quant matmul (non-Qi_K)
        l_warptile_mmq = { 256, 128, 256, 64, 1 };
        m_warptile_mmq = { 256, 128, 128, 64, 1 };
        s_warptile_mmq = { 256, 32,  64, 128, 0 };
        l_mmq_wg_denoms = { 128, 256, 1 };
        m_mmq_wg_denoms = { 128, 128, 1 };
        s_mmq_wg_denoms = { 32,  64,  1 };

        // spec constants and tile sizes for quant matmul (Qi_K)
        l_warptile_mmq_k = { 256, 128, 256, 64, 1 };
        m_warptile_mmq_k = { 256, 128, 128, 64, 1 };
        s_warptile_mmq_k = { 256, 32,  64, 128, 0 };
        l_mmq_wg_denoms_k = { 128, 256, 1 };
        m_mmq_wg_denoms_k = { 128, 128, 1 };
        s_mmq_wg_denoms_k = { 32,  64,  1 };

        // spec constants and tile sizes for quant matmul_id
        ثابت uint32_t mmqid_bk = device->coopmat2_decode_سمتیہ ? 64u : 32u;
        l_warptile_mmqid = { 256, 128, 128, mmqid_bk, 1 };
        m_warptile_mmqid = { 256, 128, 64,  mmqid_bk, 0 };
        s_warptile_mmqid = { 256, 128, 64,  mmqid_bk, 0 };
        l_mmqid_wg_denoms = { 128, 128, 1 };
        m_mmqid_wg_denoms = { 128, 64, 1 };
        s_mmqid_wg_denoms = { 128, 64, 1 };

        l_align = 128;
        m_align =  64;
        s_align =  32;
    } وإلا {
        // Matrix cores require different warp group sizes
        ثابت uint32_t tm_l = device->coopmat_support ? device->coopmat_m : 4;
        ثابت uint32_t tm_m = device->coopmat_support ? device->coopmat_m : 4;
        ثابت uint32_t tm_s = device->coopmat_support ? device->coopmat_m : 2;
        ثابت uint32_t tn_l = device->coopmat_support ? device->coopmat_n : 4;
        ثابت uint32_t tn_m = device->coopmat_support ? device->coopmat_n : 2;
        ثابت uint32_t tn_s = device->coopmat_support ? device->coopmat_n : 2;
        ثابت uint32_t tk_l = device->coopmat_support ? device->coopmat_k : 1;
        ثابت uint32_t tk_m = device->coopmat_support ? device->coopmat_k : 1;
        ثابت uint32_t tk_s = device->coopmat_support ? device->coopmat_k : 1;

        ثابت uint32_t itm = device->coopmat_int_m;
        ثابت uint32_t itn = device->coopmat_int_n;
        ثابت uint32_t itk = device->coopmat_int_k;

        ثابت uint32_t s_warptile_wm = device->subgroup_ناپ == 8 ? 8 : 32;

        l_warptile = { 128,             128, 128, 16, mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mm_warp_8 };
        m_warptile = { 128,              64,  64, 16, mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mm_warp_8 };
        s_warptile = { subgroup_ناپ_32, 32,  32, 16, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, subgroup_ناپ_8 };

        l_warptile_mmq = { 128,             128, 128, 32, mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mm_warp_8 };
        m_warptile_mmq = { 128,              64,  64, 32, mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mm_warp_8 };
        s_warptile_mmq = { subgroup_ناپ_32, 32,  32, 32, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, subgroup_ناپ_8 };

        // Integer MMQ has a smaller shared memory profile, but heavier register use
        l_warptile_mmq_int = { 128,             128, 128, 32, mm_warp_8 * 2, 64, 2, 4, 4, 1, mm_warp_8 };
        m_warptile_mmq_int = { 128,              64,  64, 32, mm_warp_8,     32, 2, 2, 2, 1, mm_warp_8 };
        s_warptile_mmq_int = { subgroup_ناپ_32, 32,  32, 32, s_warptile_wm, 32, 2, 2, 1, 1, subgroup_ناپ_8 };

        ثابت خودکار cm1_bs = [cm1_sg](uint32_t bm, uint32_t bn) {
            أرجع cm1_sg * (bm / std::min(cm1_sg, bm)) * (bn / 32);
        };

        l_warptile_mmq_cm1_int = { cm1_bs(128, 128), 128, 128, 32, std::min(cm1_sg, 128u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
        m_warptile_mmq_cm1_int = { cm1_bs( 64,  64),  64,  64, 32, std::min(cm1_sg,  64u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
        s_warptile_mmq_cm1_int = { cm1_bs( 32,  32),  32,  32, 32, std::min(cm1_sg,  32u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };

        l_warptile_mmq_cm1_int_k = { cm1_bs( 64, 128),  64, 128, 32, std::min(cm1_sg,  64u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
        m_warptile_mmq_cm1_int_k = { cm1_bs( 64,  64),  64,  64, 32, std::min(cm1_sg,  64u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
        s_warptile_mmq_cm1_int_k = { cm1_bs( 32,  32),  32,  32, 32, std::min(cm1_sg,  32u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };

        l_mmq_cm1_wg_denoms_k = { l_warptile_mmq_cm1_int_k[1], l_warptile_mmq_cm1_int_k[2], 1 };
        m_mmq_cm1_wg_denoms_k = { m_warptile_mmq_cm1_int_k[1], m_warptile_mmq_cm1_int_k[2], 1 };
        s_mmq_cm1_wg_denoms_k = { s_warptile_mmq_cm1_int_k[1], s_warptile_mmq_cm1_int_k[2], 1 };

        // K-quants use even more registers, mitigate by setting WMITER to 1
        l_warptile_mmq_int_k = { 128,               128, 128, 32, mm_warp_8 * 2, 64, 1, 4, 4, 1, mm_warp_8 };
        m_warptile_mmq_int_k = { 128,                64,  64, 32, mm_warp_8,     32, 1, 2, 2, 1, mm_warp_8 };
        s_warptile_mmq_int_k = { subgroup_ناپ_32,   32,  32, 32, s_warptile_wm, 32, 1, 2, 1, 1, subgroup_ناپ_8 };

        l_warptile_id = { 128,                      128, 128, 16, mul_mat_mm_warp_16 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_mm_warp_16 };
        m_warptile_id = { 128,                       64,  64, 16, mul_mat_mm_warp_16,     32, 2, tm_m, tn_m, tk_m, mul_mat_mm_warp_16 };
        s_warptile_id = { mul_mat_subgroup_ناپ_16,  32,  32, 16, s_warptile_wm,          32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_ناپ_16 };

        l_warptile_mmqid = { 128,                       128, 128, 32, mul_mat_mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_mm_warp_8 };
        m_warptile_mmqid = { 128,                        64,  64, 32, mul_mat_mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mul_mat_mm_warp_8 };
        s_warptile_mmqid = { mul_mat_subgroup_ناپ_32,   32,  32, 32, s_warptile_wm,         32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_ناپ_8 };

        l_warptile_mmqid_int = { 128,                       128, 128, 32, mul_mat_mm_warp_8 * 2, 64, 2, 4, 4, 1, mul_mat_mm_warp_8 };
        m_warptile_mmqid_int = { 128,                        64,  64, 32, mul_mat_mm_warp_8,     32, 2, 2, 2, 1, mul_mat_mm_warp_8 };
        s_warptile_mmqid_int = { mul_mat_subgroup_ناپ_32,   32,  32, 32, s_warptile_wm,         32, 2, 2, 1, 1, mul_mat_subgroup_ناپ_8 };

        l_warptile_mmqid_int_k = { 128,                     128, 128, 32, mul_mat_mm_warp_16 * 2, 64, 1, 4, 4, 1, mul_mat_mm_warp_16 };
        m_warptile_mmqid_int_k = { 128,                      64,  64, 32, mul_mat_mm_warp_16,     32, 1, 2, 2, 1, mul_mat_mm_warp_16 };
        s_warptile_mmqid_int_k = { mul_mat_subgroup_ناپ_32, 32,  32, 32, s_warptile_wm,          32, 1, 2, 1, 1, mul_mat_subgroup_ناپ_16 };

        // chip specific tuning
        إذا ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
            m_warptile_mmq = m_warptile_mmq_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
            m_warptile_mmqid = m_warptile_mmqid_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
        } وإلا إذا (device->vendor_id == VK_VENDOR_ID_AMD && device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary) {
            // This is intentionally using tx_m values, slight performance increase
            l_warptile = { 256, 128, 128, 16, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
            l_warptile_mmq = l_warptile_mmq_int = { 256, 128, 128, 32, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
            l_warptile_mmq_int_k = { 256, 128, 128, 32, mm_warp_16, 64, 1, 4, 2, 1, mm_warp_16 };
        } وإلا إذا (device->vendor_id == VK_VENDOR_ID_QUALCOMM && device->coopmat_support) {
            m_warptile     = { 64, 64, 64, 16, 64, 64, 1, tm_l, tn_l, tk_l, 64 };
            m_warptile_mmq = { 64, 64, 64, 32, 64, 64, 1, tm_m, tn_m, tk_m, 64 };
        }

        l_mmq_wg_denoms = l_wg_denoms = {128, 128, 1 };
        m_mmq_wg_denoms = m_wg_denoms = { 64,  64, 1 };
        s_mmq_wg_denoms = s_wg_denoms = { 32,  32, 1 };
        l_align = 128;
        m_align =  64;
        s_align =  32;

        إذا (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support) {
            // Xe1/Xe2/Xe3 with coopmat enabled - warptile performance tuning
            l_warptile = { 512, 128, 128, 16, mm_warp_8, 32, 2, tm_l, tn_l, tk_l, mm_warp_8 };
            إذا (device->architecture == INTEL_XE1) {
                l_warptile_mmq  = { 512, 256, 128, 32, 32, 32, 2, tm_l, tn_l, tk_l, 16 };
                l_mmq_wg_denoms = { 256, 128, 1 };
                l_align         = 32;  //set as BK
            } وإلا {
                l_warptile_mmq  = { 512, 128, 256, 32, 32, 32, 2, tm_l, tn_l, tk_l, 16 };
                l_mmq_wg_denoms = { 128, 256, 1 };
                l_align         = 32;  //set as BK
            }
        }

        ثابت منطقي use_cm1_int = device->coopmat_int_support &&
                                 (device->architecture == AMD_RDNA3 || device->architecture == AMD_RDNA4);

        لكل (uint32_t i = 0; i < GGML_TYPE_گنتی; ++i) {
            ggml_type t = (ggml_type)i;
            // Disable medium and large matrix multiplication if not enough shared memory is available
            // Check mmq warptiles as the largest configuration
            // Throw an error if not enough for any matrix multiplication is available
            إذا (!ggml_vk_matmul_shmem_support(device, s_warptile_mmq, خطأ, t)) {
                std::cerr << "ggml_vulkan: Error: Shared memory size too small for matrix multiplication." << std::سطر_انت;
                ارم std::runtime_error("Shared memory size too small for matrix multiplication.");
            } وإلا إذا (!ggml_vk_matmul_shmem_support(device, m_warptile_mmq, خطأ, t)) {
                device->mul_mat_m[i] = خطأ;
                device->mul_mat_l[i] = خطأ;
            } وإلا إذا (!ggml_vk_matmul_shmem_support(device, l_warptile_mmq, خطأ, t)) {
                device->mul_mat_l[i] = خطأ;
            }

            // Disable mul_mat_id if not enough shared memory is available
            إذا (!ggml_vk_matmul_shmem_support(device, s_warptile_mmqid, سچ, t)) {
                device->mul_mat_id_s[i] = خطأ;
                device->mul_mat_id_m[i] = خطأ;
                device->mul_mat_id_l[i] = خطأ;
            } وإلا إذا (!ggml_vk_matmul_shmem_support(device, m_warptile_mmqid, سچ, t)) {
                device->mul_mat_id_m[i] = خطأ;
                device->mul_mat_id_l[i] = خطأ;
            } وإلا إذا (!ggml_vk_matmul_shmem_support(device, l_warptile_mmqid, سچ, t)) {
                device->mul_mat_id_l[i] = خطأ;
            }

            // The q8_1 mmq path has its own (larger) shmem layout, check it separately.
            // K-quants and IQ3_S use the _int_k warptiles, others use _int.
            // cm1 splits k-tiles on the KSCALES==2 types and shares tiles between dense/id.
            ثابت منطقي is_k_quant = (t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q3_K ||
                                     t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ||
                                     t == GGML_TYPE_Q6_K || t == GGML_TYPE_IQ3_S);
            ثابت منطقي cm1_k_tile = (t == GGML_TYPE_Q3_K || t == GGML_TYPE_Q6_K ||
                                     t == GGML_TYPE_NVFP4);

            ثابت خودکار & s_int   = use_cm1_int ? (cm1_k_tile ? s_warptile_mmq_cm1_int_k : s_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? s_warptile_mmq_int_k     : s_warptile_mmq_int);
            ثابت خودکار & m_int   = use_cm1_int ? (cm1_k_tile ? m_warptile_mmq_cm1_int_k : m_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? m_warptile_mmq_int_k     : m_warptile_mmq_int);
            ثابت خودکار & l_int   = use_cm1_int ? (cm1_k_tile ? l_warptile_mmq_cm1_int_k : l_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? l_warptile_mmq_int_k     : l_warptile_mmq_int);
            ثابت خودکار & s_intid = use_cm1_int ? (cm1_k_tile ? s_warptile_mmq_cm1_int_k : s_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? s_warptile_mmqid_int_k   : s_warptile_mmqid_int);
            ثابت خودکار & m_intid = use_cm1_int ? (cm1_k_tile ? m_warptile_mmq_cm1_int_k : m_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? m_warptile_mmqid_int_k   : m_warptile_mmqid_int);
            ثابت خودکار & l_intid = use_cm1_int ? (cm1_k_tile ? l_warptile_mmq_cm1_int_k : l_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? l_warptile_mmqid_int_k   : l_warptile_mmqid_int);

            ثابت خودکار int_shmem_support = [&](ثابت std::vector<uint32_t>& wt, منطقي id) {
                أرجع use_cm1_int ? ggml_vk_matmul_cm1_int_shmem_support(device, wt, id, t)
                                   : ggml_vk_matmul_int_shmem_support(device, wt, id, t);
            };

            إذا (!int_shmem_support(s_int, خطأ)) {
                device->mul_mat_s_int[i] = خطأ;
                device->mul_mat_m_int[i] = خطأ;
                device->mul_mat_l_int[i] = خطأ;
            } وإلا إذا (!int_shmem_support(m_int, خطأ)) {
                device->mul_mat_m_int[i] = خطأ;
                device->mul_mat_l_int[i] = خطأ;
            } وإلا إذا (!int_shmem_support(l_int, خطأ)) {
                device->mul_mat_l_int[i] = خطأ;
            }

            إذا (!int_shmem_support(s_intid, سچ)) {
                device->mul_mat_id_s_int[i] = خطأ;
                device->mul_mat_id_m_int[i] = خطأ;
                device->mul_mat_id_l_int[i] = خطأ;
            } وإلا إذا (!int_shmem_support(m_intid, سچ)) {
                device->mul_mat_id_m_int[i] = خطأ;
                device->mul_mat_id_l_int[i] = خطأ;
            } وإلا إذا (!int_shmem_support(l_intid, سچ)) {
                device->mul_mat_id_l_int[i] = خطأ;
            }
        }
    }

    خودکار ثابت &ggml_vk_create_pipeline = [&](vk_device& device, vk_pipeline& ادھار_pipeline, ثابت حرف *name, size_t spv_ناپ, ثابت فراغ* spv_انکڑے, ثابت حرف *entrypoint,
                                              uint32_t parameter_گنتی, uint32_t ڈالو_constant_ناپ, std::array<uint32_t, 3> wg_denoms, ثابت std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, منطقي disable_robustness = خطأ, منطقي require_بھریا_subgroups = خطأ, uint32_t required_subgroup_ناپ = 0) {

        إذا (!require_بھریا_subgroups && required_subgroup_ناپ == 0) {
            required_subgroup_ناپ = get_subgroup_ناپ(name, device->architecture);
        }

        vk_pipeline *ptr = &ادھار_pipeline;

        صحيح عدد_pipelines = 1;
#إذا defined(VK_EXT_shader_64bit_indexing)
        إذا (device->shader_64b_indexing) {
            عدد_pipelines = 2;
        }
#جے_انت
        لكل (صحيح i = 0; i < عدد_pipelines; ++i, ptr = &(*ptr)->اگلا) {
            vk_pipeline &pipeline = *ptr;
            إذا (!pipeline) {
                pipeline = std::make_shared<vk_pipeline_struct>();
            }
            إذا (!pipeline->initialized) {
                pipeline->name = name;
                pipeline->parameter_گنتی = parameter_گنتی;
                pipeline->ڈالو_constant_ناپ = ڈالو_constant_ناپ;
                pipeline->wg_denoms = wg_denoms;
                pipeline->align = align;
                pipeline->initialized = سچ;
#إذا defined(VK_EXT_shader_64bit_indexing)
                pipeline->is_64b_indexing = (i == 1);
#جے_انت
            }

            // We only care about the pipeline this call asked for; the rest
            // (including the 64-bit indexing variant) are handled by their
            // own request_descriptor_sets / load_shaders calls.
            إذا (pipeline.get() != requested.get()) {
                استمر;
            }

            إذا (pipeline->compiled) {
                استمر;
            }

            wait_pipeline = pipeline;

            إذا (!pipeline->compile_pending) {
                pipeline->compile_pending = سچ;
                claimed_task.pipeline = pipeline;
                claimed_task.spv_ناپ = spv_ناپ;
                claimed_task.spv_انکڑے = spv_انکڑے;
                claimed_task.entrypoint = entrypoint;
                claimed_task.parameter_گنتی = parameter_گنتی;
                claimed_task.wg_denoms = wg_denoms;
                claimed_task.specialization_constants = specialization_constants;
                claimed_task.disable_robustness = disable_robustness;
                claimed_task.require_بھریا_subgroups = require_بھریا_subgroups;
                claimed_task.required_subgroup_ناپ = required_subgroup_ناپ;
                has_claimed_task = سچ;
            }
        }
    };

    خودکار ثابت &ggml_vk_create_pipeline2 = [&](vk_device& device, vk_pipeline& pipeline, ثابت std::string &name, size_t spv_ناپ, ثابت فراغ* spv_انکڑے, ثابت حرف *entrypoint,
                                              uint32_t parameter_گنتی, uint32_t ڈالو_constant_ناپ, std::array<uint32_t, 3> wg_denoms, ثابت std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, منطقي disable_robustness = خطأ, منطقي require_بھریا_subgroups = خطأ, uint32_t required_subgroup_ناپ = 0) {
        أرجع ggml_vk_create_pipeline(device, pipeline, name.c_str(), spv_ناپ, spv_انکڑے, entrypoint,
                                       parameter_گنتی, ڈالو_constant_ناپ, wg_denoms, specialization_constants,
                                       align, disable_robustness, require_بھریا_subgroups, required_subgroup_ناپ);
    };

    // FA scalar has two SPIR-V modules (MMQ vs non-MMQ); FA cm1 has one. K/V
    // quant type is selected at runtime via the FaTypeK / FaTypeV spec constants.

    لكل (خودکار &fa : device->pipeline_flash_attn_f32_f16) {
        إذا (fa.پہلا.path != FA_SCALAR) استمر;
        ثابت uint32_t Br = fa.پہلا.Br;
        ثابت uint32_t Bc = fa.پہلا.Bc;
        ثابت منطقي aligned = fa.پہلا.aligned;
        ثابت منطقي f32acc = fa.پہلا.f32acc;
        ثابت uint32_t fa_sgs = fa.پہلا.subgroup_ناپ;
        ثابت منطقي fa_ds = fa.پہلا.subgroup_ناپ == 0;

        ثابت منطقي bf16_kv = fa.پہلا.k_type == GGML_TYPE_BF16;
        ثابت منطقي use_mmq = ggml_vk_fa_scalar_ورتدا_mmq(device, fa.پہلا.k_type, fa.پہلا.v_type);
        ثابت فراغ * spv_انکڑے = خالی_سوچک;
        size_t spv_ناپ = 0;
        ثابت حرف *name = خالی_سوچک;
        إذا (bf16_kv) {
            spv_انکڑے = flash_attn_f32_f16_fp32_انکڑے;
            spv_ناپ = flash_attn_f32_f16_fp32_len;
            name = aligned ? "flash_attn_f32_bf16_aligned" : "flash_attn_f32_bf16";
        } وإلا إذا (use_mmq) {
#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            إذا (device->fp16) {
                إذا (f32acc) { spv_انکڑے = flash_attn_f32_f16_int8_انکڑے;        spv_ناپ = flash_attn_f32_f16_int8_len; }
                وإلا        { spv_انکڑے = flash_attn_f32_f16_f16acc_int8_انکڑے; spv_ناپ = flash_attn_f32_f16_f16acc_int8_len; }
            } وإلا {
                spv_انکڑے = flash_attn_f32_f16_fp32_int8_انکڑے;
                spv_ناپ = flash_attn_f32_f16_fp32_int8_len;
            }
#جے_انت
            name = aligned ? "flash_attn_f32_f16_aligned" : "flash_attn_f32_f16";
        } وإلا {
            إذا (device->fp16) {
                إذا (device->dot2_f16) {
                    إذا (f32acc) { spv_انکڑے = flash_attn_f32_f16_dot2_انکڑے;        spv_ناپ = flash_attn_f32_f16_dot2_len; }
                    وإلا        { spv_انکڑے = flash_attn_f32_f16_dot2_f16acc_انکڑے; spv_ناپ = flash_attn_f32_f16_dot2_f16acc_len; }
                } وإلا {
                    إذا (f32acc) { spv_انکڑے = flash_attn_f32_f16_انکڑے;        spv_ناپ = flash_attn_f32_f16_len; }
                    وإلا        { spv_انکڑے = flash_attn_f32_f16_f16acc_انکڑے; spv_ناپ = flash_attn_f32_f16_f16acc_len; }
                }
            } وإلا {
                spv_انکڑے = flash_attn_f32_f16_fp32_انکڑے;
                spv_ناپ = flash_attn_f32_f16_fp32_len;
            }
            name = aligned ? "flash_attn_f32_f16_aligned" : "flash_attn_f32_f16";
        }
        ggml_vk_create_pipeline(device, fa.second, name, spv_ناپ, spv_انکڑے, "main", 8,
                                حجم(vk_flash_attn_ڈالو_constants), {Br, 1, 1},
                                get_fa_spec_constants(fa.پہلا), aligned ? Bc : 1, سچ,
                                !fa_ds, !fa_ds ? fa_sgs : 0);
    }

#إذا defined(VK_KHR_cooperative_میٹرکس) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    إذا (device->coopmat1_fa_support) {
        لكل (خودکار &fa : device->pipeline_flash_attn_f32_f16) {
            إذا (fa.پہلا.path != FA_COOPMAT1) استمر;
            ثابت uint32_t Br = fa.پہلا.Br;
            ثابت uint32_t Bc = fa.پہلا.Bc;
            ثابت منطقي aligned = fa.پہلا.aligned;
            ثابت منطقي f32acc = fa.پہلا.f32acc;
            ثابت uint32_t fa_sgs = fa.پہلا.subgroup_ناپ;
            ثابت منطقي fa_ds = fa.پہلا.subgroup_ناپ == 0;

            ثابت منطقي bf16_kv = fa.پہلا.k_type == GGML_TYPE_BF16;

            ثابت فراغ * spv_انکڑے;
            size_t spv_ناپ;
            ثابت حرف *name;
            إذا (bf16_kv) {
#إذا defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                إذا (!device->coopmat_bf16_support) استمر;
                spv_انکڑے = flash_attn_f32_f16_bf16_cm1_انکڑے;
                spv_ناپ = flash_attn_f32_f16_bf16_cm1_len;
                name = aligned ? "flash_attn_f32_bf16_aligned_cm1" : "flash_attn_f32_bf16_cm1";
#وإلا
                استمر;
#جے_انت
            } وإلا {
                إذا (f32acc) { spv_انکڑے = flash_attn_f32_f16_cm1_انکڑے;        spv_ناپ = flash_attn_f32_f16_cm1_len; }
                وإلا        { spv_انکڑے = flash_attn_f32_f16_f16acc_cm1_انکڑے; spv_ناپ = flash_attn_f32_f16_f16acc_cm1_len; }
                name = aligned ? "flash_attn_f32_f16_aligned_cm1" : "flash_attn_f32_f16_cm1";
            }
            ggml_vk_create_pipeline(device, fa.second, name, spv_ناپ, spv_انکڑے, "main", 8,
                                    حجم(vk_flash_attn_ڈالو_constants), {Br, 1, 1},
                                    get_fa_spec_constants(fa.پہلا), aligned ? Bc : 1, سچ,
                                    !fa_ds, !fa_ds ? fa_sgs : 0);
        }
    }
#جے_انت

#إذا defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    إذا (device->coopmat2) {
        لكل (خودکار &fa : device->pipeline_flash_attn_f32_f16) {
            إذا (fa.پہلا.path != FA_COOPMAT2) استمر;
            ثابت uint32_t Br = fa.پہلا.Br;
            ثابت uint32_t Bc = fa.پہلا.Bc;
            ثابت منطقي aligned = fa.پہلا.aligned;
            ثابت منطقي f32acc = fa.پہلا.f32acc;

            ثابت منطقي bf16_kv = fa.پہلا.k_type == GGML_TYPE_BF16;
            ثابت فراغ * spv_انکڑے;
            size_t spv_ناپ;
            ثابت حرف * name;
            إذا (bf16_kv) {
#إذا defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                إذا (!device->coopmat2_bf16_support) استمر;
                spv_انکڑے = flash_attn_f32_f16_bf16_cm2_انکڑے;
                spv_ناپ = flash_attn_f32_f16_bf16_cm2_len;
                name = aligned ? "flash_attn_f32_bf16_aligned_cm2" : "flash_attn_f32_bf16_cm2";
#وإلا
                استمر;
#جے_انت
            } وإلا إذا (aligned) {
                إذا (f32acc) { spv_انکڑے = flash_attn_f32_f16_cm2_انکڑے;        spv_ناپ = flash_attn_f32_f16_cm2_len;        name = "flash_attn_f32_f16_aligned_f32acc_cm2"; }
                وإلا        { spv_انکڑے = flash_attn_f32_f16_f16acc_cm2_انکڑے; spv_ناپ = flash_attn_f32_f16_f16acc_cm2_len; name = "flash_attn_f32_f16_aligned_f16acc_cm2"; }
            } وإلا {
                إذا (f32acc) { spv_انکڑے = flash_attn_f32_f16_cm2_انکڑے;        spv_ناپ = flash_attn_f32_f16_cm2_len;        name = "flash_attn_f32_f16_f32acc_cm2"; }
                وإلا        { spv_انکڑے = flash_attn_f32_f16_f16acc_cm2_انکڑے; spv_ناپ = flash_attn_f32_f16_f16acc_cm2_len; name = "flash_attn_f32_f16_f16acc_cm2"; }
            }
            ggml_vk_create_pipeline(device, fa.second, name, spv_ناپ, spv_انکڑے, "main", 8,
                                    حجم(vk_flash_attn_ڈالو_constants), {Br, 1, 1},
                                    get_fa_spec_constants(fa.پہلا), aligned ? Bc : 1, سچ, خطأ, 0);
        }
    }
#جے_انت

    خودکار ثابت &ggml_vk_mul_mm_spec = [&device](std::vector<uint32_t> spec, منطقي aligned) {
        spec.push_back(aligned ? 1u : 0u);  // constantID=11: ALIGNED
        إذا (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
            device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            spec.push_back(0u);  // constantID=12: SHMEM_STRIDE_PAD = 0
            spec.push_back(1u);  // constantID=13: APPLY_SLM_A_RESHAPE = true
        }
        أرجع spec;
    };

    خودکار ثابت &ggml_vk_mul_mm_spec_quant = [&device](std::vector<uint32_t> spec, منطقي aligned, uint32_t type) {
        spec.push_back(aligned ? 1u : 0u);  // constantID=11: ALIGNED
        spec.push_back(type);               // constantID=12: MmTypeA
        إذا (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
            device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            spec.push_back(0u);  // constantID=13: SHMEM_STRIDE_PAD = 0
            spec.push_back(1u);  // constantID=14: APPLY_SLM_A_RESHAPE = true
        }
        أرجع spec;
    };

    ساكن ثابت ggml_type non_lut_quant_types[] = {
        GGML_TYPE_Q1_0, GGML_TYPE_Q2_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
    };

#تعریف_کرو FOR_EACH_LUT_TYPE_NONFP4(X) \
    X(GGML_TYPE_IQ1_S,   iq1_s)   \
    X(GGML_TYPE_IQ1_M,   iq1_m)   \
    X(GGML_TYPE_IQ2_XXS, iq2_xxs) \
    X(GGML_TYPE_IQ2_XS,  iq2_xs)  \
    X(GGML_TYPE_IQ2_S,   iq2_s)   \
    X(GGML_TYPE_IQ3_XXS, iq3_xxs) \
    X(GGML_TYPE_IQ3_S,   iq3_s)   \
    X(GGML_TYPE_IQ4_XS,  iq4_xs)  \
    X(GGML_TYPE_IQ4_NL,  iq4_nl)
#تعریف_کرو FOR_EACH_LUT_FP4_TYPE(X) \
    X(GGML_TYPE_MXFP4,   mxfp4)   \
    X(GGML_TYPE_NVFP4,   nvfp4)
#تعریف_کرو FOR_EACH_LUT_TYPE(X) \
    FOR_EACH_LUT_TYPE_NONFP4(X)  \
    FOR_EACH_LUT_FP4_TYPE(X)

    ثابت صحيح mul_mat_id_param_گنتی = 5;

    ورتوں spec_fn_t = std::function<std::vector<uint32_t>(ثابت std::vector<uint32_t>&, منطقي)>;
    خودکار ثابت &create_mm_pipelines = [&](
        ثابت vk_matmul_pipeline_کلید& کلید,
        ثابت std::vector<vk_tile_config>& tile_configs,
        ثابت std::string& shader_name, size_t spv_len, ثابت فراغ* spv_انکڑے,
        uint32_t ڈالو_constant_ناپ, uint32_t param_گنتی,
        ثابت spec_fn_t& spec_fn,
        منطقي disable_robustness = خطأ, منطقي require_بھریا_subgroups = خطأ, uint32_t required_subgroup_ناپ = 0,
        منطقي create_aligned = سچ, منطقي pin_subgroup_منزل_warp = خطأ
    ) {
        خودکار& vec = device->pipeline_matmul[کلید];
        ثابت منطقي پہلا_call = vec.خالی_تھاواں();
        لكل (size_t i = 0; i < tile_configs.size(); i++) {
            ثابت خودکار& tc = tile_configs[i];

            // Intel coopmat1 pins the required subgroup size to each warptile's WARP element.
            ثابت uint32_t rsgs = pin_subgroup_منزل_warp ? tc.warptile[WARP_ناپ_IDX] : required_subgroup_ناپ;
            ثابت منطقي     rfs  = require_بھریا_subgroups || pin_subgroup_منزل_warp;

            إذا (پہلا_call) {
                vk_matmul_pipeline_pair pair{};
                pair.align = tc.align;
                std::string suffix = "_" + std::to_string(i);
                pair.unaligned = std::make_shared<vk_pipeline_struct>();
                إذا (create_aligned) {
                    pair.aligned = std::make_shared<vk_pipeline_struct>();
                }
                vec.push_back(pair);
            }

            ggml_vk_create_pipeline(device, vec[i].unaligned,
                vec[i].unaligned->name.خالی_تھاواں() ? (shader_name + "_" + std::to_string(i)).c_str() : vec[i].unaligned->name.c_str(),
                spv_len, spv_انکڑے, "main", param_گنتی, ڈالو_constant_ناپ,
                tc.wg_denoms, spec_fn(tc.warptile, خطأ), 1,
                disable_robustness, rfs, rsgs);

            إذا (vec[i].aligned) {
                ggml_vk_create_pipeline(device, vec[i].aligned,
                    vec[i].aligned->name.خالی_تھاواں() ? (shader_name + "_aligned_" + std::to_string(i)).c_str() : vec[i].aligned->name.c_str(),
                    spv_len, spv_انکڑے, "main", param_گنتی, ڈالو_constant_ناپ,
                    tc.wg_denoms, spec_fn(tc.warptile, سچ), tc.align,
                    disable_robustness, rfs, rsgs);
            }
        }
    };

    خودکار filter_tc = [&](ثابت std::vector<vk_tile_config>& configs, ggml_type type, منطقي is_id, منطقي is_int = خطأ) -> std::vector<vk_tile_config> {
        std::vector<vk_tile_config> نتیجہ;
        منطقي enabled[3];
        إذا (is_int) {
            enabled[0] = is_id ? device->mul_mat_id_s_int[type] : device->mul_mat_s_int[type];
            enabled[1] = is_id ? device->mul_mat_id_m_int[type] : device->mul_mat_m_int[type];
            enabled[2] = is_id ? device->mul_mat_id_l_int[type] : device->mul_mat_l_int[type];
        } وإلا {
            enabled[0] = is_id ? device->mul_mat_id_s[type] : device->mul_mat_s[type];
            enabled[1] = is_id ? device->mul_mat_id_m[type] : device->mul_mat_m[type];
            enabled[2] = is_id ? device->mul_mat_id_l[type] : device->mul_mat_l[type];
        }
        لكل (size_t i = 0; i < configs.size() && i < 3; i++) {
            إذا (enabled[i]) نتیجہ.push_back(configs[i]);
        }
        أرجع نتیجہ;
    };

    std::vector<vk_tile_config> tc_mm = {{s_warptile, s_wg_denoms, s_align}, {m_warptile, m_wg_denoms, m_align}, {l_warptile, l_wg_denoms, l_align}};
    std::vector<vk_tile_config> tc_mmq = {{s_warptile_mmq, s_mmq_wg_denoms, s_align}, {m_warptile_mmq, m_mmq_wg_denoms, m_align}, {l_warptile_mmq, l_mmq_wg_denoms, l_align}};

#إذا defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    إذا (device->coopmat2) {
        خودکار ثابت &ggml_vk_mul_mm_cm2_spec = [&](std::vector<uint32_t> spec, منطقي aligned, uint32_t type = UINT32_MAX) {
            spec.push_back(aligned ? 1u : 0u);        // ALIGNED
            spec.push_back(device->subgroup_ناپ);     // subgroup_size
            إذا (type != UINT32_MAX) {
                spec.push_back(type);                  // MmTypeA
                spec.push_back((uint32_t)ggml_type_ناپ((ggml_type)type)); // MmABlockBytes
            }
            أرجع spec;
        };

        std::vector<vk_tile_config> tc_mmq_k = {{s_warptile_mmq_k, s_mmq_wg_denoms_k, s_align}, {m_warptile_mmq_k, m_mmq_wg_denoms_k, m_align}, {l_warptile_mmq_k, l_mmq_wg_denoms_k, l_align}};
        std::vector<vk_tile_config> tc_mmqid = {{s_warptile_mmqid, s_mmqid_wg_denoms, s_align}, {m_warptile_mmqid, m_mmqid_wg_denoms, m_align}, {l_warptile_mmqid, l_mmqid_wg_denoms, l_align}};

        spec_fn_t cm2_spec = [&](ثابت std::vector<uint32_t>& wt, منطقي a) { أرجع ggml_vk_mul_mm_cm2_spec(wt, a); };

        // F16 x F16
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, خطأ, سچ},  tc_mm, "matmul_f16_f16acc", matmul_f16_f16acc_cm2_len, matmul_f16_f16acc_cm2_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3, cm2_spec, سچ);
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, خطأ, خطأ}, tc_mm, "matmul_f16",        matmul_f16_cm2_len,        matmul_f16_cm2_انکڑے,        حجم(vk_mat_mat_ڈالو_constants), 3, cm2_spec, سچ);
#إذا defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        إذا (device->coopmat_bf16_support) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, خطأ, خطأ}, tc_mm, "matmul_bf16", matmul_bf16_cm2_len, matmul_bf16_cm2_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3, cm2_spec, سچ);
        }
#جے_انت
        لكل (ثابت خودکار type : non_lut_quant_types) {
            // regression in unified shader on Ampere
            إذا (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                استمر;
            }
            خودکار& tc = ((type >= GGML_TYPE_Q2_K && type <= GGML_TYPE_Q6_K) || type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0) ? tc_mmq_k : tc_mmq;
            spec_fn_t qs = [&, type](ثابت std::vector<uint32_t>& wt, منطقي a) { أرجع ggml_vk_mul_mm_cm2_spec(wt, a, (uint32_t)type); };
            create_mm_pipelines({type, GGML_TYPE_F16, خطأ, سچ},  tc, "matmul_quant_f16_f16acc", matmul_quant_f16_f16acc_cm2_len, matmul_quant_f16_f16acc_cm2_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3, qs, سچ);
            create_mm_pipelines({type, GGML_TYPE_F16, خطأ, خطأ}, tc, "matmul_quant_f16",        matmul_quant_f16_cm2_len,        matmul_quant_f16_cm2_انکڑے,        حجم(vk_mat_mat_ڈالو_constants), 3, qs, سچ);
        }
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, خطأ, سچ},  tc_mmq_k, "matmul_q4_k_f16_f16acc", matmul_q4_k_f16_f16acc_cm2_len, matmul_q4_k_f16_f16acc_cm2_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3, cm2_spec, سچ);
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, خطأ, خطأ}, tc_mmq_k, "matmul_q4_k_f16",        matmul_q4_k_f16_cm2_len,        matmul_q4_k_f16_cm2_انکڑے,        حجم(vk_mat_mat_ڈالو_constants), 3, cm2_spec, سچ);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, خطأ, سچ},  tc_mmq_k, "matmul_q5_k_f16_f16acc", matmul_q5_k_f16_f16acc_cm2_len, matmul_q5_k_f16_f16acc_cm2_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3, cm2_spec, سچ);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, خطأ, خطأ}, tc_mmq_k, "matmul_q5_k_f16",        matmul_q5_k_f16_cm2_len,        matmul_q5_k_f16_cm2_انکڑے,        حجم(vk_mat_mat_ڈالو_constants), 3, cm2_spec, سچ);
#تعریف_کرو X_CM2(TYPE, tstr) \
        { خودکار tc = filter_tc(tc_mmq, TYPE, خطأ); \
          إذا (!tc.خالی_تھاواں()) { \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, خطأ, سچ},  tc, "matmul_" #tstr "_f16_f16acc", matmul_##tstr##_f16_f16acc_cm2_len, matmul_##tstr##_f16_f16acc_cm2_data, حجم(vk_mat_mat_ڈالو_constants), 3, cm2_spec, سچ); \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, خطأ, خطأ}, tc, "matmul_" #tstr "_f16",        matmul_##tstr##_f16_cm2_len,        matmul_##tstr##_f16_cm2_data,        حجم(vk_mat_mat_ڈالو_constants), 3, cm2_spec, سچ); \
          } }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM2)
#إذا defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        إذا (device->ocp_fp4) {
#تعریف_کرو X_CM2_OCP(TYPE, tstr) \
            { خودکار tc = filter_tc(tc_mmq, TYPE, خطأ); \
              إذا (!tc.خالی_تھاواں()) { \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, خطأ, سچ},  tc, "matmul_" #tstr "_f16_ocp_f16acc", matmul_##tstr##_f16_ocp_f16acc_cm2_len, matmul_##tstr##_f16_ocp_f16acc_cm2_data, حجم(vk_mat_mat_ڈالو_constants), 3, cm2_spec, سچ); \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, خطأ, خطأ}, tc, "matmul_" #tstr "_f16_ocp",        matmul_##tstr##_f16_ocp_cm2_len,        matmul_##tstr##_f16_ocp_cm2_data,        حجم(vk_mat_mat_ڈالو_constants), 3, cm2_spec, سچ); \
              } }
            FOR_EACH_LUT_FP4_TYPE(X_CM2_OCP)
#تعریف_رد X_CM2_OCP
        } وإلا
#جے_انت
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM2)
        }
#تعریف_رد X_CM2

        GGML_ASSERT(device->subgroup_ballot);

        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, سچ, سچ},  tc_mm, "matmul_id_subgroup_f16_f16acc", matmul_id_subgroup_f16_f16acc_cm2_len, matmul_id_subgroup_f16_f16acc_cm2_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), 5, cm2_spec, سچ);
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, سچ, خطأ}, tc_mm, "matmul_id_subgroup_f16",        matmul_id_subgroup_f16_cm2_len,        matmul_id_subgroup_f16_cm2_انکڑے,        حجم(vk_mat_mat_id_ڈالو_constants), 5, cm2_spec, سچ);
#إذا defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        إذا (device->coopmat_bf16_support) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, سچ, خطأ}, tc_mm, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_cm2_len, matmul_id_subgroup_bf16_cm2_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), 5, cm2_spec, سچ);
        }
#جے_انت
        لكل (ثابت خودکار type : non_lut_quant_types) {
            إذا (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                استمر;
            }
            spec_fn_t qs_id = [&, type](ثابت std::vector<uint32_t>& wt, منطقي a) { أرجع ggml_vk_mul_mm_cm2_spec(wt, a, (uint32_t)type); };
            create_mm_pipelines({type, GGML_TYPE_F16, سچ, سچ},  tc_mmqid, "matmul_id_subgroup_quant_f16_f16acc", matmul_id_subgroup_quant_f16_f16acc_cm2_len, matmul_id_subgroup_quant_f16_f16acc_cm2_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), 5, qs_id, سچ);
            create_mm_pipelines({type, GGML_TYPE_F16, سچ, خطأ}, tc_mmqid, "matmul_id_subgroup_quant_f16",        matmul_id_subgroup_quant_f16_cm2_len,        matmul_id_subgroup_quant_f16_cm2_انکڑے,        حجم(vk_mat_mat_id_ڈالو_constants), 5, qs_id, سچ);
        }
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, سچ, سچ},  tc_mmqid, "matmul_id_subgroup_q4_k_f16_f16acc", matmul_id_subgroup_q4_k_f16_f16acc_cm2_len, matmul_id_subgroup_q4_k_f16_f16acc_cm2_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), 5, cm2_spec, سچ);
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, سچ, خطأ}, tc_mmqid, "matmul_id_subgroup_q4_k_f16",        matmul_id_subgroup_q4_k_f16_cm2_len,        matmul_id_subgroup_q4_k_f16_cm2_انکڑے,        حجم(vk_mat_mat_id_ڈالو_constants), 5, cm2_spec, سچ);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, سچ, سچ},  tc_mmqid, "matmul_id_subgroup_q5_k_f16_f16acc", matmul_id_subgroup_q5_k_f16_f16acc_cm2_len, matmul_id_subgroup_q5_k_f16_f16acc_cm2_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), 5, cm2_spec, سچ);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, سچ, خطأ}, tc_mmqid, "matmul_id_subgroup_q5_k_f16",        matmul_id_subgroup_q5_k_f16_cm2_len,        matmul_id_subgroup_q5_k_f16_cm2_انکڑے,        حجم(vk_mat_mat_id_ڈالو_constants), 5, cm2_spec, سچ);
#تعریف_کرو X_CM2_ID(TYPE, tstr) \
        { خودکار tc = filter_tc(tc_mmqid, TYPE, سچ); \
          إذا (!tc.خالی_تھاواں()) { \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, سچ, سچ},  tc, "matmul_id_subgroup_" #tstr "_f16_f16acc", matmul_id_subgroup_##tstr##_f16_f16acc_cm2_len, matmul_id_subgroup_##tstr##_f16_f16acc_cm2_data, حجم(vk_mat_mat_id_ڈالو_constants), 5, cm2_spec, سچ); \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, سچ, خطأ}, tc, "matmul_id_subgroup_" #tstr "_f16",        matmul_id_subgroup_##tstr##_f16_cm2_len,        matmul_id_subgroup_##tstr##_f16_cm2_data,        حجم(vk_mat_mat_id_ڈالو_constants), 5, cm2_spec, سچ); \
          } }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM2_ID)
#إذا defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        إذا (device->ocp_fp4) {
#تعریف_کرو X_CM2_ID_OCP(TYPE, tstr) \
            { خودکار tc = filter_tc(tc_mmqid, TYPE, سچ); \
              إذا (!tc.خالی_تھاواں()) { \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, سچ, سچ},  tc, "matmul_id_subgroup_" #tstr "_f16_ocp_f16acc", matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm2_len, matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm2_data, حجم(vk_mat_mat_id_ڈالو_constants), 5, cm2_spec, سچ); \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, سچ, خطأ}, tc, "matmul_id_subgroup_" #tstr "_f16_ocp",        matmul_id_subgroup_##tstr##_f16_ocp_cm2_len,        matmul_id_subgroup_##tstr##_f16_ocp_cm2_data,        حجم(vk_mat_mat_id_ڈالو_constants), 5, cm2_spec, سچ); \
              } }
            FOR_EACH_LUT_FP4_TYPE(X_CM2_ID_OCP)
#تعریف_رد X_CM2_ID_OCP
        } وإلا
#جے_انت
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM2_ID)
        }
#تعریف_رد X_CM2_ID
    } وإلا
#جے_انت  // defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
#إذا defined(VK_KHR_cooperative_میٹرکس) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    إذا (device->coopmat_support) {
        spec_fn_t cm1_spec = [&](ثابت std::vector<uint32_t>& wt, منطقي a) { أرجع ggml_vk_mul_mm_spec(wt, a); };

        // Intel coopmat1 pins each pipeline's required subgroup size to its warptile WARP element.
        ثابت منطقي cm1_pin = device->vendor_id == VK_VENDOR_ID_INTEL;

        // Intel coopmat1 uses a dedicated large-tile config for quant matmul_id.
        std::vector<vk_tile_config> tc_mmq_id = tc_mmq;
        إذا (cm1_pin) {
            tc_mmq_id[2] = { { 512, 128, 128, 32, 32, 32, 2, device->coopmat_m, device->coopmat_n, device->coopmat_k, 32 }, { 128, 128, 1 }, 32 };
        }

        خودکار cm1_create = [&](vk_matmul_pipeline_کلید کلید, ثابت std::vector<vk_tile_config>& tc_ادھار,
                              ثابت std::string& name, size_t len, ثابت فراغ* انکڑے, uint32_t pc_ناپ, uint32_t pc) {
            خودکار tc = filter_tc(tc_ادھار, کلید.type_a, کلید.mul_mat_id);
            إذا (!tc.خالی_تھاواں()) create_mm_pipelines(کلید, tc, name, len, انکڑے, pc_ناپ, pc, cm1_spec, خطأ, سچ, 0, سچ, cm1_pin);
        };
        خودکار cm1_create_quant = [&](vk_matmul_pipeline_کلید کلید, ثابت std::vector<vk_tile_config>& tc_ادھار,
                                    ثابت std::string& name, size_t len, ثابت فراغ* انکڑے, uint32_t pc_ناپ, uint32_t pc) {
            spec_fn_t qs = [&, type_a=کلید.type_a](ثابت std::vector<uint32_t>& wt, منطقي a) { أرجع ggml_vk_mul_mm_spec_quant(wt, a, (uint32_t)type_a); };
            خودکار tc = filter_tc(tc_ادھار, کلید.type_a, کلید.mul_mat_id);
            إذا (!tc.خالی_تھاواں()) create_mm_pipelines(کلید, tc, name, len, انکڑے, pc_ناپ, pc, qs, خطأ, سچ, 0, سچ, cm1_pin);
        };
        // int8 MMQ helper: per-type cm1 shader, warptile passed as-is (carries DEVICE_ARCH in
        // spec constant WARP_SIZE_IDX+1), subgroup size pinned to the warptile WARP element, no aligned variant.
        خودکار cm1_create_mmq = [&](vk_matmul_pipeline_کلید کلید, ثابت std::vector<vk_tile_config>& tc_ادھار,
                                  ثابت std::string& name, size_t len, ثابت فراغ* انکڑے, uint32_t pc_ناپ, uint32_t pc) {
            spec_fn_t پچھان_تت = [](ثابت std::vector<uint32_t>& wt, منطقي) { أرجع wt; };
            خودکار tc = filter_tc(tc_ادھار, کلید.type_a, کلید.mul_mat_id, سچ);
            إذا (!tc.خالی_تھاواں()) create_mm_pipelines(کلید, tc, name, len, انکڑے, pc_ناپ, pc, پچھان_تت, خطأ, خطأ, 0, خطأ, سچ);
        };

        std::vector<vk_tile_config> tc_mmq_cm1_int = {
            {s_warptile_mmq_cm1_int, s_mmq_wg_denoms, s_align},
            {m_warptile_mmq_cm1_int, m_mmq_wg_denoms, m_align},
            {l_warptile_mmq_cm1_int, l_mmq_wg_denoms, l_align},
        };
        std::vector<vk_tile_config> tc_mmq_cm1_int_k = {
            {s_warptile_mmq_cm1_int_k, s_mmq_cm1_wg_denoms_k, s_align},
            {m_warptile_mmq_cm1_int_k, m_mmq_cm1_wg_denoms_k, m_align},
            {l_warptile_mmq_cm1_int_k, l_mmq_cm1_wg_denoms_k, l_align},
        };

        // Some quants are not performant on RDNA4, those fall back to FP16 matmul
        ثابت منطقي rdna3 = device->architecture == AMD_RDNA3;
        ثابت منطقي rdna4 = device->architecture == AMD_RDNA4;

        cm1_create({GGML_TYPE_F32, GGML_TYPE_F32, خطأ, خطأ}, tc_mm, "matmul_f32_f32",     matmul_f32_f32_cm1_len,     matmul_f32_f32_cm1_انکڑے,     حجم(vk_mat_mat_ڈالو_constants), 3);
        cm1_create({GGML_TYPE_F32, GGML_TYPE_F16, خطأ, خطأ}, tc_mm, "matmul_f32_f16",     matmul_f32_f16_cm1_len,     matmul_f32_f16_cm1_انکڑے,     حجم(vk_mat_mat_ڈالو_constants), 3);
        إذا (device->coopmat_acc_f16_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, خطأ, سچ},  tc_mm, "matmul_f16_f16acc", matmul_f16_f16acc_cm1_len, matmul_f16_f16acc_cm1_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, خطأ, سچ},  tc_mm, "matmul_f16_f32_f16acc", matmul_f16_f32_f16acc_cm1_len, matmul_f16_f32_f16acc_cm1_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
        }
        إذا (device->coopmat_acc_f32_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, خطأ, خطأ}, tc_mm, "matmul_f16",      matmul_f16_cm1_len,      matmul_f16_cm1_انکڑے,      حجم(vk_mat_mat_ڈالو_constants), 3);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, خطأ, خطأ}, tc_mm, "matmul_f16_f32",  matmul_f16_f32_cm1_len,  matmul_f16_f32_cm1_انکڑے,  حجم(vk_mat_mat_ڈالو_constants), 3);
        }
#إذا defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        إذا (device->coopmat_bf16_support) {
            cm1_create({GGML_TYPE_BF16, GGML_TYPE_BF16, خطأ, خطأ}, tc_mm, "matmul_bf16", matmul_bf16_cm1_len, matmul_bf16_cm1_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
        }
#جے_انت
        لكل (ثابت خودکار type : non_lut_quant_types) {
            إذا (device->coopmat_acc_f16_support) {
                cm1_create_quant({type, GGML_TYPE_F32, خطأ, سچ},  tc_mmq, "matmul_quant_f32_f16acc", matmul_quant_f32_f16acc_cm1_len, matmul_quant_f32_f16acc_cm1_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                cm1_create_quant({type, GGML_TYPE_F16, خطأ, سچ},  tc_mmq, "matmul_quant_f16_f16acc", matmul_quant_f16_f16acc_cm1_len, matmul_quant_f16_f16acc_cm1_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
            }
            إذا (device->coopmat_acc_f32_support) {
                cm1_create_quant({type, GGML_TYPE_F32, خطأ, خطأ}, tc_mmq, "matmul_quant_f32",        matmul_quant_f32_cm1_len,        matmul_quant_f32_cm1_انکڑے,        حجم(vk_mat_mat_ڈالو_constants), 3);
                cm1_create_quant({type, GGML_TYPE_F16, خطأ, خطأ}, tc_mmq, "matmul_quant_f16",        matmul_quant_f16_cm1_len,        matmul_quant_f16_cm1_انکڑے,        حجم(vk_mat_mat_ڈالو_constants), 3);
            }
        }
        // The _f16 variants provide the f16 B-type pipeline used when y_non_contig converts f32->f16.
#تعریف_کرو X_CM1(TYPE, tstr) \
        إذا (device->coopmat_acc_f16_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, خطأ, سچ},  tc_mmq, "matmul_" #tstr "_f32_f16acc", matmul_##tstr##_f32_f16acc_cm1_len, matmul_##tstr##_f32_f16acc_cm1_data, حجم(vk_mat_mat_ڈالو_constants), 3); \
            cm1_create({TYPE, GGML_TYPE_F16, خطأ, سچ},  tc_mmq, "matmul_" #tstr "_f16_f16acc", matmul_##tstr##_f16_f16acc_cm1_len, matmul_##tstr##_f16_f16acc_cm1_data, حجم(vk_mat_mat_ڈالو_constants), 3); \
        } \
        إذا (device->coopmat_acc_f32_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, خطأ, خطأ}, tc_mmq, "matmul_" #tstr "_f32",        matmul_##tstr##_f32_cm1_len,        matmul_##tstr##_f32_cm1_data,        حجم(vk_mat_mat_ڈالو_constants), 3); \
            cm1_create({TYPE, GGML_TYPE_F16, خطأ, خطأ}, tc_mmq, "matmul_" #tstr "_f16",        matmul_##tstr##_f16_cm1_len,        matmul_##tstr##_f16_cm1_data,        حجم(vk_mat_mat_ڈالو_constants), 3); \
        }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM1)
#إذا defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        إذا (device->ocp_fp4) {
#تعریف_کرو X_CM1_OCP(TYPE, tstr) \
            إذا (device->coopmat_acc_f16_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, خطأ, سچ},  tc_mmq, "matmul_" #tstr "_f32_ocp_f16acc", matmul_##tstr##_f32_ocp_f16acc_cm1_len, matmul_##tstr##_f32_ocp_f16acc_cm1_data, حجم(vk_mat_mat_ڈالو_constants), 3); \
                cm1_create({TYPE, GGML_TYPE_F16, خطأ, سچ},  tc_mmq, "matmul_" #tstr "_f16_ocp_f16acc", matmul_##tstr##_f16_ocp_f16acc_cm1_len, matmul_##tstr##_f16_ocp_f16acc_cm1_data, حجم(vk_mat_mat_ڈالو_constants), 3); \
            } \
            إذا (device->coopmat_acc_f32_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, خطأ, خطأ}, tc_mmq, "matmul_" #tstr "_f32_ocp",        matmul_##tstr##_f32_ocp_cm1_len,        matmul_##tstr##_f32_ocp_cm1_data,        حجم(vk_mat_mat_ڈالو_constants), 3); \
                cm1_create({TYPE, GGML_TYPE_F16, خطأ, خطأ}, tc_mmq, "matmul_" #tstr "_f16_ocp",        matmul_##tstr##_f16_ocp_cm1_len,        matmul_##tstr##_f16_ocp_cm1_data,        حجم(vk_mat_mat_ڈالو_constants), 3); \
            }
            FOR_EACH_LUT_FP4_TYPE(X_CM1_OCP)
#تعریف_رد X_CM1_OCP
        } وإلا
#جے_انت
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM1)
        }
#تعریف_رد X_CM1

        إذا (device->coopmat_int_support && (rdna3 || rdna4)) {
            cm1_create_mmq({GGML_TYPE_Q4_0,   GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int,   "matmul_q4_0_q8_1",   matmul_q4_0_q8_1_cm1_len,   matmul_q4_0_q8_1_cm1_انکڑے,   حجم(vk_mat_mat_ڈالو_constants), 3);
            إذا (!rdna4) { cm1_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int,   "matmul_q4_1_q8_1",   matmul_q4_1_q8_1_cm1_len,   matmul_q4_1_q8_1_cm1_انکڑے,   حجم(vk_mat_mat_ڈالو_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q5_0,   GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int,   "matmul_q5_0_q8_1",   matmul_q5_0_q8_1_cm1_len,   matmul_q5_0_q8_1_cm1_انکڑے,   حجم(vk_mat_mat_ڈالو_constants), 3);
            إذا (!rdna4) { cm1_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int,   "matmul_q5_1_q8_1",   matmul_q5_1_q8_1_cm1_len,   matmul_q5_1_q8_1_cm1_انکڑے,   حجم(vk_mat_mat_ڈالو_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q8_0,   GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int,   "matmul_q8_0_q8_1",   matmul_q8_0_q8_1_cm1_len,   matmul_q8_0_q8_1_cm1_انکڑے,   حجم(vk_mat_mat_ڈالو_constants), 3);
            cm1_create_mmq({GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int,   "matmul_iq4_nl_q8_1", matmul_iq4_nl_q8_1_cm1_len, matmul_iq4_nl_q8_1_cm1_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
            cm1_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int,   "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_cm1_len, matmul_iq4_xs_q8_1_cm1_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
            cm1_create_mmq({GGML_TYPE_MXFP4,  GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int,   "matmul_mxfp4_q8_1",  matmul_mxfp4_q8_1_cm1_len,  matmul_mxfp4_q8_1_cm1_انکڑے,  حجم(vk_mat_mat_ڈالو_constants), 3);
            cm1_create_mmq({GGML_TYPE_Q3_K,   GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int_k, "matmul_q3_k_q8_1",   matmul_q3_k_q8_1_cm1_len,   matmul_q3_k_q8_1_cm1_انکڑے,   حجم(vk_mat_mat_ڈالو_constants), 3);
            إذا (!rdna4) { cm1_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int,   "matmul_q4_k_q8_1",   matmul_q4_k_q8_1_cm1_len,   matmul_q4_k_q8_1_cm1_انکڑے,   حجم(vk_mat_mat_ڈالو_constants), 3); }
            إذا (!rdna4) { cm1_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int,   "matmul_q5_k_q8_1",   matmul_q5_k_q8_1_cm1_len,   matmul_q5_k_q8_1_cm1_انکڑے,   حجم(vk_mat_mat_ڈالو_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q6_K,   GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int_k, "matmul_q6_k_q8_1",   matmul_q6_k_q8_1_cm1_len,   matmul_q6_k_q8_1_cm1_انکڑے,   حجم(vk_mat_mat_ڈالو_constants), 3);
            إذا (!rdna4) { cm1_create_mmq({GGML_TYPE_NVFP4, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int_k, "matmul_nvfp4_q8_1",  matmul_nvfp4_q8_1_cm1_len,  matmul_nvfp4_q8_1_cm1_انکڑے,  حجم(vk_mat_mat_ڈالو_constants), 3); }
        }

        GGML_ASSERT(device->subgroup_ballot);

        cm1_create({GGML_TYPE_F32, GGML_TYPE_F32, سچ, خطأ}, tc_mm, "matmul_id_subgroup_f32_f32", matmul_id_subgroup_f32_f32_cm1_len, matmul_id_subgroup_f32_f32_cm1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
        إذا (device->coopmat_acc_f16_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, سچ, سچ},  tc_mm, "matmul_id_subgroup_f16_f16acc",     matmul_id_subgroup_f16_f16acc_cm1_len,     matmul_id_subgroup_f16_f16acc_cm1_انکڑے,     حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, سچ, سچ},  tc_mm, "matmul_id_subgroup_f16_f32_f16acc", matmul_id_subgroup_f16_f32_f16acc_cm1_len, matmul_id_subgroup_f16_f32_f16acc_cm1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
        }
        إذا (device->coopmat_acc_f32_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, سچ, خطأ}, tc_mm, "matmul_id_subgroup_f16",     matmul_id_subgroup_f16_cm1_len,     matmul_id_subgroup_f16_cm1_انکڑے,     حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, سچ, خطأ}, tc_mm, "matmul_id_subgroup_f16_f32", matmul_id_subgroup_f16_f32_cm1_len, matmul_id_subgroup_f16_f32_cm1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
        }
#إذا defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        إذا (device->coopmat_bf16_support) {
            cm1_create({GGML_TYPE_BF16, GGML_TYPE_BF16, سچ, خطأ}, tc_mm, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_cm1_len, matmul_id_subgroup_bf16_cm1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
        }
#جے_انت
        لكل (ثابت خودکار type : non_lut_quant_types) {
            إذا (device->coopmat_acc_f16_support) {
                cm1_create_quant({type, GGML_TYPE_F32, سچ, سچ},  tc_mmq_id, "matmul_id_subgroup_quant_f32_f16acc", matmul_id_subgroup_quant_f32_f16acc_cm1_len, matmul_id_subgroup_quant_f32_f16acc_cm1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                cm1_create_quant({type, GGML_TYPE_F16, سچ, سچ},  tc_mmq_id, "matmul_id_subgroup_quant_f16_f16acc", matmul_id_subgroup_quant_f16_f16acc_cm1_len, matmul_id_subgroup_quant_f16_f16acc_cm1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
            }
            إذا (device->coopmat_acc_f32_support) {
                cm1_create_quant({type, GGML_TYPE_F32, سچ, خطأ}, tc_mmq_id, "matmul_id_subgroup_quant_f32",        matmul_id_subgroup_quant_f32_cm1_len,        matmul_id_subgroup_quant_f32_cm1_انکڑے,        حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                cm1_create_quant({type, GGML_TYPE_F16, سچ, خطأ}, tc_mmq_id, "matmul_id_subgroup_quant_f16",        matmul_id_subgroup_quant_f16_cm1_len,        matmul_id_subgroup_quant_f16_cm1_انکڑے,        حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
            }
        }
        // The _f16 variants provide the f16 B-type pipeline used when y_non_contig converts f32->f16.
#تعریف_کرو X_CM1_ID(TYPE, tstr) \
        إذا (device->coopmat_acc_f16_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, سچ, سچ},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_f16acc", matmul_id_subgroup_##tstr##_f32_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f32_f16acc_cm1_data, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی); \
            cm1_create({TYPE, GGML_TYPE_F16, سچ, سچ},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_f16acc", matmul_id_subgroup_##tstr##_f16_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f16_f16acc_cm1_data, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی); \
        } \
        إذا (device->coopmat_acc_f32_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, سچ, خطأ}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32",        matmul_id_subgroup_##tstr##_f32_cm1_len,        matmul_id_subgroup_##tstr##_f32_cm1_data,        حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی); \
            cm1_create({TYPE, GGML_TYPE_F16, سچ, خطأ}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16",        matmul_id_subgroup_##tstr##_f16_cm1_len,        matmul_id_subgroup_##tstr##_f16_cm1_data,        حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی); \
        }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM1_ID)
#إذا defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        إذا (device->ocp_fp4) {
#تعریف_کرو X_CM1_ID_OCP(TYPE, tstr) \
            إذا (device->coopmat_acc_f16_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, سچ, سچ},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_ocp_f16acc", matmul_id_subgroup_##tstr##_f32_ocp_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f32_ocp_f16acc_cm1_data, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی); \
                cm1_create({TYPE, GGML_TYPE_F16, سچ, سچ},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_ocp_f16acc", matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm1_data, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی); \
            } \
            إذا (device->coopmat_acc_f32_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, سچ, خطأ}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_ocp",        matmul_id_subgroup_##tstr##_f32_ocp_cm1_len,        matmul_id_subgroup_##tstr##_f32_ocp_cm1_data,        حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی); \
                cm1_create({TYPE, GGML_TYPE_F16, سچ, خطأ}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_ocp",        matmul_id_subgroup_##tstr##_f16_ocp_cm1_len,        matmul_id_subgroup_##tstr##_f16_ocp_cm1_data,        حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی); \
            }
            FOR_EACH_LUT_FP4_TYPE(X_CM1_ID_OCP)
#تعریف_رد X_CM1_ID_OCP
        } وإلا
#جے_انت
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM1_ID)
        }
#تعریف_رد X_CM1_ID

        إذا (device->coopmat_int_support && (rdna3 || rdna4)) {
            cm1_create_mmq({GGML_TYPE_Q4_0,   GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_0_q8_1",   matmul_id_subgroup_q4_0_q8_1_cm1_len,   matmul_id_subgroup_q4_0_q8_1_cm1_انکڑے,   حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
            cm1_create_mmq({GGML_TYPE_Q4_1,   GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_1_q8_1",   matmul_id_subgroup_q4_1_q8_1_cm1_len,   matmul_id_subgroup_q4_1_q8_1_cm1_انکڑے,   حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
            cm1_create_mmq({GGML_TYPE_Q5_0,   GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_0_q8_1",   matmul_id_subgroup_q5_0_q8_1_cm1_len,   matmul_id_subgroup_q5_0_q8_1_cm1_انکڑے,   حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
            cm1_create_mmq({GGML_TYPE_Q5_1,   GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_1_q8_1",   matmul_id_subgroup_q5_1_q8_1_cm1_len,   matmul_id_subgroup_q5_1_q8_1_cm1_انکڑے,   حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
            cm1_create_mmq({GGML_TYPE_Q8_0,   GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q8_0_q8_1",   matmul_id_subgroup_q8_0_q8_1_cm1_len,   matmul_id_subgroup_q8_0_q8_1_cm1_انکڑے,   حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
            cm1_create_mmq({GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmq_cm1_int,   "matmul_id_subgroup_iq4_nl_q8_1", matmul_id_subgroup_iq4_nl_q8_1_cm1_len, matmul_id_subgroup_iq4_nl_q8_1_cm1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
            cm1_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmq_cm1_int,   "matmul_id_subgroup_iq4_xs_q8_1", matmul_id_subgroup_iq4_xs_q8_1_cm1_len, matmul_id_subgroup_iq4_xs_q8_1_cm1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
            cm1_create_mmq({GGML_TYPE_MXFP4,  GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmq_cm1_int,   "matmul_id_subgroup_mxfp4_q8_1",  matmul_id_subgroup_mxfp4_q8_1_cm1_len,  matmul_id_subgroup_mxfp4_q8_1_cm1_انکڑے,  حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
            cm1_create_mmq({GGML_TYPE_Q3_K,   GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmq_cm1_int_k, "matmul_id_subgroup_q3_k_q8_1",   matmul_id_subgroup_q3_k_q8_1_cm1_len,   matmul_id_subgroup_q3_k_q8_1_cm1_انکڑے,   حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
            cm1_create_mmq({GGML_TYPE_Q4_K,   GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_k_q8_1",   matmul_id_subgroup_q4_k_q8_1_cm1_len,   matmul_id_subgroup_q4_k_q8_1_cm1_انکڑے,   حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
            cm1_create_mmq({GGML_TYPE_Q5_K,   GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_k_q8_1",   matmul_id_subgroup_q5_k_q8_1_cm1_len,   matmul_id_subgroup_q5_k_q8_1_cm1_انکڑے,   حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
            cm1_create_mmq({GGML_TYPE_Q6_K,   GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmq_cm1_int_k, "matmul_id_subgroup_q6_k_q8_1",   matmul_id_subgroup_q6_k_q8_1_cm1_len,   matmul_id_subgroup_q6_k_q8_1_cm1_انکڑے,   حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
            إذا (!rdna4) { cm1_create_mmq({GGML_TYPE_NVFP4, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmq_cm1_int_k, "matmul_id_subgroup_nvfp4_q8_1",  matmul_id_subgroup_nvfp4_q8_1_cm1_len,  matmul_id_subgroup_nvfp4_q8_1_cm1_انکڑے,  حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی); }
        }
    } وإلا
#جے_انت  // defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    {
        // Helper for subgroup path with dot2 selection and filtering
        خودکار sg_create = [&](vk_matmul_pipeline_کلید کلید, ثابت std::vector<vk_tile_config>& tc_ادھار,
                             ثابت std::string& name, size_t len, ثابت فراغ* انکڑے, uint32_t pc_ناپ, uint32_t pc,
                             uint32_t rsgs = 0) {
            خودکار tc = filter_tc(tc_ادھار, کلید.type_a, کلید.mul_mat_id);
            إذا (!tc.خالی_تھاواں()) create_mm_pipelines(کلید, tc, name, len, انکڑے, pc_ناپ, pc,
                [&](ثابت std::vector<uint32_t>& wt, منطقي a) { أرجع ggml_vk_mul_mm_spec(wt, a); },
                خطأ, rsgs > 0, rsgs);
        };
        خودکار sg_create_quant = [&](vk_matmul_pipeline_کلید کلید, ثابت std::vector<vk_tile_config>& tc_ادھار,
                                   ثابت std::string& name, size_t len, ثابت فراغ* انکڑے, uint32_t pc_ناپ, uint32_t pc,
                                   uint32_t rsgs = 0) {
            خودکار tc = filter_tc(tc_ادھار, کلید.type_a, کلید.mul_mat_id);
            إذا (!tc.خالی_تھاواں()) {
                spec_fn_t qs = [&, type_a=کلید.type_a](ثابت std::vector<uint32_t>& wt, منطقي a) { أرجع ggml_vk_mul_mm_spec_quant(wt, a, (uint32_t)type_a); };
                create_mm_pipelines(کلید, tc, name, len, انکڑے, pc_ناپ, pc, qs, خطأ, rsgs > 0, rsgs);
            }
        };
#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        خودکار sg_create_mmq = [&](vk_matmul_pipeline_کلید کلید, ثابت std::vector<vk_tile_config>& tc_ادھار,
                                 ثابت std::string& name, size_t len, ثابت فراغ* انکڑے, uint32_t pc_ناپ, uint32_t pc,
                                 uint32_t rsgs = 0) {
            خودکار tc = filter_tc(tc_ادھار, کلید.type_a, کلید.mul_mat_id, سچ);
            إذا (!tc.خالی_تھاواں()) {
                spec_fn_t پچھان_تت = [](ثابت std::vector<uint32_t>& wt, منطقي) { أرجع wt; };
                create_mm_pipelines(کلید, tc, name, len, انکڑے, pc_ناپ, pc, پچھان_تت, خطأ, rsgs > 0, rsgs, خطأ);
            }
        };
#جے_انت

        std::vector<vk_tile_config> tc_id = {{s_warptile_id, s_wg_denoms, s_align}, {m_warptile_id, m_wg_denoms, m_align}, {l_warptile_id, l_wg_denoms, l_align}};
        std::vector<vk_tile_config> tc_mmqid = {{s_warptile_mmqid, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid, l_mmq_wg_denoms, l_align}};

        إذا (device->fp16) {
            // FP16 subgroup path - with dot2 runtime selection
            #تعریف_کرو SPV_DOT2(NAME) (device->dot2_f16 ? NAME ## _dot2_len : NAME ## _len), (device->dot2_f16 ? NAME ## dot2_انکڑے : NAME ## انکڑے_2)
            #تعریف_کرو SPV_DOT2_F16ACC(NAME) (device->dot2_f16 ? NAME ## _dot2_f16acc_len : NAME ## _f16acc_len), (device->dot2_f16 ? NAME ## dot2_f16acc_انکڑے : NAME ## f16acc_انکڑے)

            sg_create({GGML_TYPE_F32, GGML_TYPE_F32, خطأ, خطأ}, tc_mm, "matmul_f32_f32", SPV_DOT2(matmul_f32_f32), حجم(vk_mat_mat_ڈالو_constants), 3);
            sg_create({GGML_TYPE_F32, GGML_TYPE_F16, خطأ, خطأ}, tc_mm, "matmul_f32_f16", SPV_DOT2(matmul_f32_f16), حجم(vk_mat_mat_ڈالو_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, خطأ, سچ},  tc_mm, "matmul_f16_f16acc",     SPV_DOT2_F16ACC(matmul_f16),     حجم(vk_mat_mat_ڈالو_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, خطأ, خطأ}, tc_mm, "matmul_f16",            SPV_DOT2(matmul_f16),            حجم(vk_mat_mat_ڈالو_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, خطأ, سچ},  tc_mm, "matmul_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_f16_f32), حجم(vk_mat_mat_ڈالو_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, خطأ, خطأ}, tc_mm, "matmul_f16_f32",        SPV_DOT2(matmul_f16_f32),        حجم(vk_mat_mat_ڈالو_constants), 3);
            // BF16 - no dot2
            sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, خطأ, خطأ}, tc_mm, "matmul_bf16", matmul_bf16_len, matmul_bf16_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);

            لكل (ثابت خودکار type : non_lut_quant_types) {
                sg_create_quant({type, GGML_TYPE_F32, خطأ, سچ},  tc_mmq, "matmul_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_quant_f32), حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_quant({type, GGML_TYPE_F32, خطأ, خطأ}, tc_mmq, "matmul_quant_f32",        SPV_DOT2(matmul_quant_f32),        حجم(vk_mat_mat_ڈالو_constants), 3);
            }
    #تعریف_کرو X_SG(TYPE, tstr) \
            sg_create({TYPE, GGML_TYPE_F32, خطأ, سچ},  tc_mmq, "matmul_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_##tstr##_f32), حجم(vk_mat_mat_ڈالو_constants), 3); \
            sg_create({TYPE, GGML_TYPE_F32, خطأ, خطأ}, tc_mmq, "matmul_" #tstr "_f32",        SPV_DOT2(matmul_##tstr##_f32),        حجم(vk_mat_mat_ڈالو_constants), 3);
            FOR_EACH_LUT_TYPE(X_SG)
#تعریف_رد X_SG

#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            إذا (device->integer_dot_ضرب_حاصل) {
                std::vector<vk_tile_config> tc_mmq_int = {{s_warptile_mmq_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int, l_mmq_wg_denoms, l_align}};
                std::vector<vk_tile_config> tc_mmq_int_k = {{s_warptile_mmq_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int_k, l_mmq_wg_denoms, l_align}};
                sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q2_0_q8_1", matmul_q2_0_q8_1_len, matmul_q2_0_q8_1_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q4_0_q8_1", matmul_q4_0_q8_1_len, matmul_q4_0_q8_1_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q4_1_q8_1", matmul_q4_1_q8_1_len, matmul_q4_1_q8_1_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q5_0_q8_1", matmul_q5_0_q8_1_len, matmul_q5_0_q8_1_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q5_1_q8_1", matmul_q5_1_q8_1_len, matmul_q5_1_q8_1_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q8_0_q8_1", matmul_q8_0_q8_1_len, matmul_q8_0_q8_1_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_mxfp4_q8_1", matmul_mxfp4_q8_1_len, matmul_mxfp4_q8_1_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_len, matmul_iq4_xs_q8_1_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_q2_k_q8_1", matmul_q2_k_q8_1_len, matmul_q2_k_q8_1_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_q3_k_q8_1", matmul_q3_k_q8_1_len, matmul_q3_k_q8_1_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_q4_k_q8_1", matmul_q4_k_q8_1_len, matmul_q4_k_q8_1_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_q5_k_q8_1", matmul_q5_k_q8_1_len, matmul_q5_k_q8_1_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_q6_k_q8_1", matmul_q6_k_q8_1_len, matmul_q6_k_q8_1_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_iq3_s_q8_1", matmul_iq3_s_q8_1_len, matmul_iq3_s_q8_1_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
            }
#جے_انت

            إذا (device->subgroup_ballot && device->subgroup_require_بھریا_support && subgroup_min_ناپ_16) {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, سچ, خطأ}, tc_id, "matmul_id_subgroup_f32_f32", SPV_DOT2(matmul_id_subgroup_f32_f32), حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, سچ, سچ},  tc_id, "matmul_id_subgroup_f16_f16acc",     SPV_DOT2_F16ACC(matmul_id_subgroup_f16),     حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, سچ, خطأ}, tc_id, "matmul_id_subgroup_f16",            SPV_DOT2(matmul_id_subgroup_f16),            حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, سچ, سچ},  tc_id, "matmul_id_subgroup_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_f16_f32), حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, سچ, خطأ}, tc_id, "matmul_id_subgroup_f16_f32",        SPV_DOT2(matmul_id_subgroup_f16_f32),        حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ_16);
                // BF16 id - no dot2
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, سچ, خطأ}, tc_id, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_len, matmul_id_subgroup_bf16_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ_16);
                لكل (ثابت خودکار type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, سچ, سچ},  tc_mmqid, "matmul_id_subgroup_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_quant_f32), حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ);
                    sg_create_quant({type, GGML_TYPE_F32, سچ, خطأ}, tc_mmqid, "matmul_id_subgroup_quant_f32",        SPV_DOT2(matmul_id_subgroup_quant_f32),        حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ);
                }
        #تعریف_کرو X_SG_ID_SUB(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, سچ, سچ},  tc_mmqid, "matmul_id_subgroup_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_##tstr##_f32), حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ); \
                sg_create({TYPE, GGML_TYPE_F32, سچ, خطأ}, tc_mmqid, "matmul_id_subgroup_" #tstr "_f32",        SPV_DOT2(matmul_id_subgroup_##tstr##_f32),        حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ);
                FOR_EACH_LUT_TYPE(X_SG_ID_SUB)
#تعریف_رد X_SG_ID_SUB
#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
                إذا (device->integer_dot_ضرب_حاصل) {
                    std::vector<vk_tile_config> tc_mmqid_int = {{s_warptile_mmqid_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int, l_mmq_wg_denoms, l_align}};
                    std::vector<vk_tile_config> tc_mmqid_int_k = {{s_warptile_mmqid_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int_k, l_mmq_wg_denoms, l_align}};
                    sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int, "matmul_id_subgroup_q2_0_q8_1", matmul_id_subgroup_q2_0_q8_1_len, matmul_id_subgroup_q2_0_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ);
                    sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int, "matmul_id_subgroup_q4_0_q8_1", matmul_id_subgroup_q4_0_q8_1_len, matmul_id_subgroup_q4_0_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ);
                    sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int, "matmul_id_subgroup_q4_1_q8_1", matmul_id_subgroup_q4_1_q8_1_len, matmul_id_subgroup_q4_1_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ);
                    sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int, "matmul_id_subgroup_q5_0_q8_1", matmul_id_subgroup_q5_0_q8_1_len, matmul_id_subgroup_q5_0_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ);
                    sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int, "matmul_id_subgroup_q5_1_q8_1", matmul_id_subgroup_q5_1_q8_1_len, matmul_id_subgroup_q5_1_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ);
                    sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int, "matmul_id_subgroup_q8_0_q8_1", matmul_id_subgroup_q8_0_q8_1_len, matmul_id_subgroup_q8_0_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ);
                    sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int, "matmul_id_subgroup_mxfp4_q8_1", matmul_id_subgroup_mxfp4_q8_1_len, matmul_id_subgroup_mxfp4_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ);
                    sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int, "matmul_id_subgroup_iq4_xs_q8_1", matmul_id_subgroup_iq4_xs_q8_1_len, matmul_id_subgroup_iq4_xs_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ);
                    sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int_k, "matmul_id_subgroup_q2_k_q8_1", matmul_id_subgroup_q2_k_q8_1_len, matmul_id_subgroup_q2_k_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ_16);
                    sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int_k, "matmul_id_subgroup_q3_k_q8_1", matmul_id_subgroup_q3_k_q8_1_len, matmul_id_subgroup_q3_k_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ_16);
                    sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int_k, "matmul_id_subgroup_q4_k_q8_1", matmul_id_subgroup_q4_k_q8_1_len, matmul_id_subgroup_q4_k_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ_16);
                    sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int_k, "matmul_id_subgroup_q5_k_q8_1", matmul_id_subgroup_q5_k_q8_1_len, matmul_id_subgroup_q5_k_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ_16);
                    sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int_k, "matmul_id_subgroup_q6_k_q8_1", matmul_id_subgroup_q6_k_q8_1_len, matmul_id_subgroup_q6_k_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ_16);
                    sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int_k, "matmul_id_subgroup_iq3_s_q8_1", matmul_id_subgroup_iq3_s_q8_1_len, matmul_id_subgroup_iq3_s_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ_16);
                }
#جے_انت
            } وإلا {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, سچ, خطأ}, tc_mm, "matmul_id_f32_f32", SPV_DOT2(matmul_id_f32_f32), حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, سچ, سچ},  tc_mm, "matmul_id_f16_f16acc",     SPV_DOT2_F16ACC(matmul_id_f16),     حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, سچ, خطأ}, tc_mm, "matmul_id_f16",            SPV_DOT2(matmul_id_f16),            حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, سچ, سچ},  tc_mm, "matmul_id_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_f16_f32), حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, سچ, خطأ}, tc_mm, "matmul_id_f16_f32",        SPV_DOT2(matmul_id_f16_f32),        حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                // BF16 id - no dot2
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, سچ, خطأ}, tc_mm, "matmul_id_bf16", matmul_id_bf16_len, matmul_id_bf16_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                لكل (ثابت خودکار type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, سچ, سچ},  tc_mmqid, "matmul_id_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_quant_f32), حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                    sg_create_quant({type, GGML_TYPE_F32, سچ, خطأ}, tc_mmqid, "matmul_id_quant_f32",        SPV_DOT2(matmul_id_quant_f32),        حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                }
        #تعریف_کرو X_SG_ID(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, سچ, سچ},  tc_mmqid, "matmul_id_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_##tstr##_f32), حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی); \
                sg_create({TYPE, GGML_TYPE_F32, سچ, خطأ}, tc_mmqid, "matmul_id_" #tstr "_f32",        SPV_DOT2(matmul_id_##tstr##_f32),        حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                FOR_EACH_LUT_TYPE(X_SG_ID)
#تعریف_رد X_SG_ID
#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
                إذا (device->integer_dot_ضرب_حاصل) {
                    std::vector<vk_tile_config> tc_mmqid_int = {{s_warptile_mmqid_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int, l_mmq_wg_denoms, l_align}};
                    std::vector<vk_tile_config> tc_mmqid_int_k = {{s_warptile_mmqid_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int_k, l_mmq_wg_denoms, l_align}};
                    sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int, "matmul_id_q2_0_q8_1", matmul_id_q2_0_q8_1_len, matmul_id_q2_0_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                    sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int, "matmul_id_q4_0_q8_1", matmul_id_q4_0_q8_1_len, matmul_id_q4_0_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                    sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int, "matmul_id_q4_1_q8_1", matmul_id_q4_1_q8_1_len, matmul_id_q4_1_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                    sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int, "matmul_id_q5_0_q8_1", matmul_id_q5_0_q8_1_len, matmul_id_q5_0_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                    sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int, "matmul_id_q5_1_q8_1", matmul_id_q5_1_q8_1_len, matmul_id_q5_1_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                    sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int, "matmul_id_q8_0_q8_1", matmul_id_q8_0_q8_1_len, matmul_id_q8_0_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                    sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int, "matmul_id_mxfp4_q8_1", matmul_id_mxfp4_q8_1_len, matmul_id_mxfp4_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                    sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int, "matmul_id_iq4_xs_q8_1", matmul_id_iq4_xs_q8_1_len, matmul_id_iq4_xs_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                    sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int_k, "matmul_id_q2_k_q8_1", matmul_id_q2_k_q8_1_len, matmul_id_q2_k_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                    sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int_k, "matmul_id_q3_k_q8_1", matmul_id_q3_k_q8_1_len, matmul_id_q3_k_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                    sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int_k, "matmul_id_q4_k_q8_1", matmul_id_q4_k_q8_1_len, matmul_id_q4_k_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                    sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int_k, "matmul_id_q5_k_q8_1", matmul_id_q5_k_q8_1_len, matmul_id_q5_k_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                    sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int_k, "matmul_id_q6_k_q8_1", matmul_id_q6_k_q8_1_len, matmul_id_q6_k_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                    sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, سچ, خطأ}, tc_mmqid_int_k, "matmul_id_iq3_s_q8_1", matmul_id_iq3_s_q8_1_len, matmul_id_iq3_s_q8_1_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                }
#جے_انت
            }
            #تعریف_رد SPV_DOT2
            #تعریف_رد SPV_DOT2_F16ACC
        } وإلا {
            // FP32-only fallback path
            sg_create({GGML_TYPE_F32, GGML_TYPE_F32, خطأ, خطأ}, tc_mm, "matmul_f32_f32", matmul_f32_f32_fp32_len, matmul_f32_f32_fp32_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
            sg_create({GGML_TYPE_F32, GGML_TYPE_F16, خطأ, خطأ}, tc_mm, "matmul_f32_f16", matmul_f32_f16_fp32_len, matmul_f32_f16_fp32_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, خطأ, خطأ}, tc_mm, "matmul_f16",     matmul_f16_fp32_len,     matmul_f16_fp32_انکڑے,     حجم(vk_mat_mat_ڈالو_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, خطأ, خطأ}, tc_mm, "matmul_f16_f32", matmul_f16_f32_fp32_len, matmul_f16_f32_fp32_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
            sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, خطأ, خطأ}, tc_mm, "matmul_bf16",  matmul_bf16_fp32_len,    matmul_bf16_fp32_انکڑے,    حجم(vk_mat_mat_ڈالو_constants), 3);

            لكل (ثابت خودکار type : non_lut_quant_types) {
                sg_create_quant({type, GGML_TYPE_F32, خطأ, خطأ}, tc_mmq, "matmul_quant_f32", matmul_quant_f32_fp32_len, matmul_quant_f32_fp32_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
            }
    #تعریف_کرو X_SG_FP32(TYPE, tstr) \
            sg_create({TYPE, GGML_TYPE_F32, خطأ, خطأ}, tc_mmq, "matmul_" #tstr "_f32", matmul_##tstr##_f32_fp32_len, matmul_##tstr##_f32_fp32_data, حجم(vk_mat_mat_ڈالو_constants), 3);
            FOR_EACH_LUT_TYPE(X_SG_FP32)
#تعریف_رد X_SG_FP32

#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            إذا (device->integer_dot_ضرب_حاصل) {
                std::vector<vk_tile_config> tc_mmq_int = {{s_warptile_mmq_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int, l_mmq_wg_denoms, l_align}};
                std::vector<vk_tile_config> tc_mmq_int_k = {{s_warptile_mmq_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int_k, l_mmq_wg_denoms, l_align}};
                sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q2_0_q8_1", matmul_q2_0_q8_1_fp32_len, matmul_q2_0_q8_1_fp32_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q4_0_q8_1", matmul_q4_0_q8_1_fp32_len, matmul_q4_0_q8_1_fp32_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q4_1_q8_1", matmul_q4_1_q8_1_fp32_len, matmul_q4_1_q8_1_fp32_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q5_0_q8_1", matmul_q5_0_q8_1_fp32_len, matmul_q5_0_q8_1_fp32_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q5_1_q8_1", matmul_q5_1_q8_1_fp32_len, matmul_q5_1_q8_1_fp32_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q8_0_q8_1", matmul_q8_0_q8_1_fp32_len, matmul_q8_0_q8_1_fp32_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_fp32_len, matmul_iq4_xs_q8_1_fp32_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_q2_k_q8_1", matmul_q2_k_q8_1_fp32_len, matmul_q2_k_q8_1_fp32_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_q3_k_q8_1", matmul_q3_k_q8_1_fp32_len, matmul_q3_k_q8_1_fp32_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_q4_k_q8_1", matmul_q4_k_q8_1_fp32_len, matmul_q4_k_q8_1_fp32_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_q5_k_q8_1", matmul_q5_k_q8_1_fp32_len, matmul_q5_k_q8_1_fp32_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_q6_k_q8_1", matmul_q6_k_q8_1_fp32_len, matmul_q6_k_q8_1_fp32_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_iq3_s_q8_1", matmul_iq3_s_q8_1_fp32_len, matmul_iq3_s_q8_1_fp32_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3);
            }
#جے_انت

            إذا (device->subgroup_ballot && device->subgroup_require_بھریا_support && subgroup_min_ناپ_16) {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, سچ, خطأ}, tc_id, "matmul_id_subgroup_f32_f32", matmul_id_subgroup_f32_f32_fp32_len, matmul_id_subgroup_f32_f32_fp32_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, سچ, خطأ}, tc_id, "matmul_id_subgroup_f16",     matmul_id_subgroup_f16_fp32_len,     matmul_id_subgroup_f16_fp32_انکڑے,     حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, سچ, خطأ}, tc_id, "matmul_id_subgroup_f16_f32", matmul_id_subgroup_f16_f32_fp32_len, matmul_id_subgroup_f16_f32_fp32_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ_16);
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, سچ, خطأ}, tc_id, "matmul_id_subgroup_bf16",  matmul_id_subgroup_bf16_fp32_len,    matmul_id_subgroup_bf16_fp32_انکڑے,    حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ_16);
                لكل (ثابت خودکار type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, سچ, خطأ}, tc_mmqid, "matmul_id_subgroup_quant_f32", matmul_id_subgroup_quant_f32_fp32_len, matmul_id_subgroup_quant_f32_fp32_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ);
                }
        #تعریف_کرو X_SG_ID_SUB_FP32(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, سچ, خطأ}, tc_mmqid, "matmul_id_subgroup_" #tstr "_f32", matmul_id_subgroup_##tstr##_f32_fp32_len, matmul_id_subgroup_##tstr##_f32_fp32_data, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, mul_mat_subgroup_ناپ);
                FOR_EACH_LUT_TYPE(X_SG_ID_SUB_FP32)
#تعریف_رد X_SG_ID_SUB_FP32
            } وإلا {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, سچ, خطأ}, tc_mm, "matmul_id_f32_f32", matmul_id_f32_f32_fp32_len, matmul_id_f32_f32_fp32_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, سچ, خطأ}, tc_mm, "matmul_id_f16",     matmul_id_f16_fp32_len,     matmul_id_f16_fp32_انکڑے,     حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, سچ, خطأ}, tc_mm, "matmul_id_f16_f32", matmul_id_f16_f32_fp32_len, matmul_id_f16_f32_fp32_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, سچ, خطأ}, tc_mm, "matmul_id_bf16",  matmul_id_bf16_fp32_len,    matmul_id_bf16_fp32_انکڑے,    حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                لكل (ثابت خودکار type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, سچ, خطأ}, tc_mmqid, "matmul_id_quant_f32", matmul_id_quant_f32_fp32_len, matmul_id_quant_f32_fp32_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                }
        #تعریف_کرو X_SG_ID_FP32(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, سچ, خطأ}, tc_mmqid, "matmul_id_" #tstr "_f32", matmul_id_##tstr##_f32_fp32_len, matmul_id_##tstr##_f32_fp32_data, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی);
                FOR_EACH_LUT_TYPE(X_SG_ID_FP32)
#تعریف_رد X_SG_ID_FP32
            }
        }
    }
#تعریف_رد FOR_EACH_LUT_TYPE
#تعریف_رد FOR_EACH_LUT_TYPE_NONFP4
#تعریف_رد FOR_EACH_LUT_FP4_TYPE
    // BF16 fallback for coopmat devices without bf16 coopmat support
    إذا ((device->coopmat2 || device->coopmat_support)
#إذا defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        && !device->coopmat_bf16_support
#جے_انت
        ) {
        ثابت uint32_t s_warptile_wm_bf16 = device->subgroup_ناپ == 8 ? 8 : 32;
        std::vector<vk_tile_config> tc_bf16_fb = {
            {{ subgroup_ناپ_32, 32, 32, 16, s_warptile_wm_bf16, 32, 2, 2, 2, 1, subgroup_ناپ_8 }, {32, 32, 1}, s_align},
            {{ 128, 64, 64, 16, mm_warp_8, 32, 2, 4, 2, 1, mm_warp_8 }, {64, 64, 1}, m_align},
            {{ 128, 128, 128, 16, mm_warp_8 * 2, 64, 2, 4, 4, 1, mm_warp_8 }, {128, 128, 1}, l_align},
        };
        خودکار tc_bf16_filtered = filter_tc(tc_bf16_fb, GGML_TYPE_BF16, خطأ);
        خودکار tc_bf16_id_filtered = filter_tc(tc_bf16_fb, GGML_TYPE_BF16, سچ);
        spec_fn_t bf16_spec = [&](ثابت std::vector<uint32_t>& wt, منطقي a) { أرجع ggml_vk_mul_mm_spec(wt, a); };
        إذا (!tc_bf16_filtered.خالی_تھاواں()) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, خطأ, خطأ}, tc_bf16_filtered, "matmul_bf16", matmul_bf16_fp32_len, matmul_bf16_fp32_انکڑے, حجم(vk_mat_mat_ڈالو_constants), 3, bf16_spec);
        }
        إذا (!tc_bf16_id_filtered.خالی_تھاواں()) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, سچ, خطأ}, tc_bf16_id_filtered, "matmul_id_bf16", matmul_id_bf16_fp32_len, matmul_id_bf16_fp32_انکڑے, حجم(vk_mat_mat_id_ڈالو_constants), mul_mat_id_param_گنتی, bf16_spec);
        }
    }

    // Set up tile selector functions
    إذا (device->coopmat2) {
        device->matmul_tile_selector = [](uint32_t m, uint32_t n, uint32_t /*k*/, uint32_t shader_core_گنتی,
                                          ثابت std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            إذا (configs.size() <= 1) أرجع 0;
            uint32_t آخری = (uint32_t)configs.size() - 1;
            إذا (configs.size() == 2) {
                uint32_t crossover = configs[0].unaligned->wg_denoms[1];
                أرجع (n > crossover) ? 1 : 0;
            }
            // 3+ configs: s=0, m=1, l=2
            ثابت uint32_t tiles_l = CEIL_DIV(m, configs[آخری].unaligned->wg_denoms[0]) * CEIL_DIV(n, configs[آخری].unaligned->wg_denoms[1]);
            ثابت uint32_t tiles_m = CEIL_DIV(m, configs[1].unaligned->wg_denoms[0]) * CEIL_DIV(n, configs[1].unaligned->wg_denoms[1]);
            uint32_t crossover_large = configs[1].unaligned->wg_denoms[1];
            منطقي prefer_large = tiles_m > shader_core_گنتی || tiles_l > shader_core_گنتی ||
                                (tiles_l <= shader_core_گنتی / 3 && tiles_m > shader_core_گنتی / 2);
            إذا (n > crossover_large && prefer_large) أرجع آخری;
            uint32_t crossover_medium_m = configs[0].unaligned->wg_denoms[0];
            uint32_t crossover_medium_n = configs[0].unaligned->wg_denoms[1];
            إذا (m > crossover_medium_m && n > crossover_medium_n) أرجع 1;
            أرجع 0;
        };
        device->matmul_id_tile_selector = [](uint32_t /*m*/, uint32_t n, uint32_t /*k*/, uint32_t /*shader_core_count*/,
                                             ثابت std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            إذا (configs.size() <= 1) أرجع 0;
            uint32_t آخری = (uint32_t)configs.size() - 1;
            إذا (configs.size() == 2) {
                uint32_t crossover = configs[0].unaligned->wg_denoms[1];
                أرجع (n > crossover) ? 1 : 0;
            }
            uint32_t crossover_large = configs[1].unaligned->wg_denoms[1];
            إذا (n > crossover_large) أرجع آخری;
            uint32_t crossover_medium = configs[0].unaligned->wg_denoms[1];
            إذا (n > crossover_medium) أرجع 1;
            أرجع 0;
        };
    } وإلا {
        device->matmul_tile_selector = [](uint32_t m, uint32_t n, uint32_t /*k*/, uint32_t /*shader_core_count*/,
                                          ثابت std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            إذا (configs.size() <= 1) أرجع 0;
            إذا (m <= 32 || n <= 32) أرجع 0;
            إذا (configs.size() == 2) أرجع 1;
            إذا (m <= 64 || n <= 64) أرجع 1;
            أرجع (uint32_t)configs.size() - 1;
        };
        device->matmul_id_tile_selector = device->matmul_tile_selector;
    }

    // mul mat vec

    // the number of rows computed per shader depends on GPU model and quant
    uint32_t rm_stdq = 1;
    uint32_t rm_kq = 2;
    uint32_t rm_stdq_int = 1;
    uint32_t rm_kq_int = 1;
    خودکار ثابت &rm_iq_int = [](uint32_t i) { أرجع i == 0 ? 8u : 4u; };
    إذا (device->vendor_id == VK_VENDOR_ID_AMD) {
        إذا (device->architecture == AMD_GCN) {
            rm_stdq = 2;
            rm_kq = 4;
            rm_stdq_int = 4;
        }
    } وإلا إذا (device->vendor_id == VK_VENDOR_ID_INTEL) {
        rm_stdq = 2;
        rm_stdq_int = 2;
    }
    // RDNA3/4: above four columns, static 4 rows for all types bench faster than the default
    ثابت منطقي is_rdna3_or_4 = device->vendor_id == VK_VENDOR_ID_AMD && (device->architecture == AMD_RDNA3 || device->architecture == AMD_RDNA4);
    خودکار ثابت &rm_int_n = [&](uint32_t صفاں, uint32_t i) { أرجع (is_rdna3_or_4 && i >= 4) ? 4u : صفاں; };
    // RDNA3/4 and NVIDIA except pre-Turing: use 4 rows for MUL_MAT_ID MMVQ.
    خودکار ثابت &rm_id = [&](uint32_t صفاں) {
        إذا (device->vendor_id == VK_VENDOR_ID_NVIDIA &&
            device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) {
            أرجع 4u;
        }
        أرجع is_rdna3_or_4 ? 4u : صفاں;
    };
    uint32_t rm_iq = 2 * rm_kq;

    ثابت منطقي use_subgroups = device->subgroup_arithmetic;
    // The Imagination proprietary compiler rejects the subgroup-only dequant mul_mat_vec
    // shaders that require a subgroup size >= 16; fall back to shared-memory reduction.
    ثابت منطقي is_imagination_proprietary =
        device->driver_id == vk::DriverId::eImaginationProprietary;
    // Ensure a subgroup size >= 16 is available
    ثابت منطقي use_subgroups16 = use_subgroups && subgroup_min_ناپ_16 && !is_imagination_proprietary;

    ثابت uint32_t subgroup_ناپ = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_ناپ_control && device->subgroup_min_ناپ <= 16 && device->subgroup_max_ناپ >= 16) ? 16 : device->subgroup_ناپ;
    ثابت uint32_t subgroup_size16 = std::max(subgroup_ناپ, 16u);

    ثابت uint32_t force_subgroup_ناپ = use_subgroups ? subgroup_ناپ : 0;
    ثابت uint32_t force_subgroup_size16 = use_subgroups16 ? subgroup_size16 : 0;
    ساكن constexpr uint32_t mul_mat_vec_عدد_bindings = 5;
    ساكن constexpr uint32_t mul_mat_vec_id_عدد_bindings = 6;

#إذا defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
#تعریف_کرو OCP_DMMV_LEN(NAME, REDUC)  (device->ocp_fp4 ? NAME ## _ocp_len[REDUC]  : NAME ## _len[REDUC])
#تعریف_کرو OCP_DMMV_انکڑے(NAME, REDUC) (device->ocp_fp4 ? NAME ## ocp_انکڑے[REDUC] : NAME ## انکڑے_2[REDUC])
#وإلا
#تعریف_کرو OCP_DMMV_LEN(NAME, REDUC)  NAME ## _len[REDUC]
#تعریف_کرو OCP_DMMV_انکڑے(NAME, REDUC) NAME ## انکڑے_2[REDUC]
#جے_انت

    لكل (uint32_t w = 0; w < DMMV_WG_ناپ_گنتی; ++w) {
        ثابت uint32_t wg_ناپ_subgroup   = (w == DMMV_WG_ناپ_SUBGROUP) ? subgroup_ناپ : (subgroup_ناپ * 4);
        ثابت uint32_t wg_ناپ_subgroup16 = (w == DMMV_WG_ناپ_SUBGROUP) ? subgroup_size16 : (subgroup_size16 * 4);

        ثابت shader_reduction_منوال reduc = (use_subgroups && w == DMMV_WG_ناپ_SUBGROUP) ? SHADER_REDUCTION_منوال_SUBGROUP :
                                            (use_subgroups && w == DMMV_WG_ناپ_LARGE) ? SHADER_REDUCTION_منوال_HYBRID :
                                            SHADER_REDUCTION_منوال_SHMEM;

        ثابت shader_reduction_منوال reduc16 = (use_subgroups16 && w == DMMV_WG_ناپ_SUBGROUP) ? SHADER_REDUCTION_منوال_SUBGROUP :
                                              (use_subgroups16 && w == DMMV_WG_ناپ_LARGE) ? SHADER_REDUCTION_منوال_HYBRID :
                                              SHADER_REDUCTION_منوال_SHMEM;

        لكل (uint32_t i = 0; i < mul_mat_vec_max_خانے_گنتی; ++i) {
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f32_f32",  arr_dmmv_f32_f32_f32_len[reduc],  arr_dmmv_f32_f32_f32_انکڑے[reduc],  "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {1, 1, 1}, {wg_ناپ_subgroup, 1, i+1}, 1, خطأ, use_subgroups, force_subgroup_ناپ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f32_f32",  arr_dmmv_f16_f32_f32_len[reduc],  arr_dmmv_f16_f32_f32_انکڑے[reduc],  "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {2, 1, 1}, {wg_ناپ_subgroup, 2, i+1}, 1, خطأ, use_subgroups, force_subgroup_ناپ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f32_f32", arr_dmmv_bf16_f32_f32_len[reduc], arr_dmmv_bf16_f32_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {2, 1, 1}, {wg_ناپ_subgroup, 2, i+1}, 1, خطأ, use_subgroups, force_subgroup_ناپ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f32_f32", arr_dmmv_q1_0_f32_f32_len[reduc], arr_dmmv_q1_0_f32_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {2*rm_stdq, 1, 1}, {wg_ناپ_subgroup, 2*rm_stdq, i+1}, 1, سچ, use_subgroups, force_subgroup_ناپ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_f32_f32", arr_dmmv_q2_0_f32_f32_len[reduc], arr_dmmv_q2_0_f32_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {2*rm_stdq, 1, 1}, {wg_ناپ_subgroup, 2*rm_stdq, i+1}, 1, سچ, use_subgroups, force_subgroup_ناپ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f32_f32", arr_dmmv_q4_0_f32_f32_len[reduc], arr_dmmv_q4_0_f32_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {2*rm_stdq, 1, 1}, {wg_ناپ_subgroup, 2*rm_stdq, i+1}, 1, سچ, use_subgroups, force_subgroup_ناپ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f32_f32", arr_dmmv_q4_1_f32_f32_len[reduc], arr_dmmv_q4_1_f32_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {2*rm_stdq, 1, 1}, {wg_ناپ_subgroup, 2*rm_stdq, i+1}, 1, سچ, use_subgroups, force_subgroup_ناپ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f32_f32", arr_dmmv_q5_0_f32_f32_len[reduc], arr_dmmv_q5_0_f32_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {2*rm_stdq, 1, 1}, {wg_ناپ_subgroup, 2*rm_stdq, i+1}, 1, سچ, use_subgroups, force_subgroup_ناپ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f32_f32", arr_dmmv_q5_1_f32_f32_len[reduc], arr_dmmv_q5_1_f32_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {2*rm_stdq, 1, 1}, {wg_ناپ_subgroup, 2*rm_stdq, i+1}, 1, سچ, use_subgroups, force_subgroup_ناپ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f32_f32", arr_dmmv_q8_0_f32_f32_len[reduc], arr_dmmv_q8_0_f32_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {1*rm_stdq, 1, 1}, {wg_ناپ_subgroup, 1*rm_stdq, i+1}, 1, سچ, use_subgroups, force_subgroup_ناپ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f32_f32", arr_dmmv_q2_k_f32_f32_len[reduc16], arr_dmmv_q2_k_f32_f32_انکڑے[reduc16], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_kq, 1, 1}, {wg_ناپ_subgroup16, rm_kq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f32_f32", arr_dmmv_q3_k_f32_f32_len[reduc16], arr_dmmv_q3_k_f32_f32_انکڑے[reduc16], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_kq, 1, 1}, {wg_ناپ_subgroup16, rm_kq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f32_f32", arr_dmmv_q4_k_f32_f32_len[reduc16], arr_dmmv_q4_k_f32_f32_انکڑے[reduc16], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_kq, 1, 1}, {wg_ناپ_subgroup16, rm_kq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f32_f32", arr_dmmv_q5_k_f32_f32_len[reduc16], arr_dmmv_q5_k_f32_f32_انکڑے[reduc16], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_kq, 1, 1}, {wg_ناپ_subgroup16, rm_kq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f32_f32", arr_dmmv_q6_k_f32_f32_len[reduc16], arr_dmmv_q6_k_f32_f32_انکڑے[reduc16], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_kq, 1, 1}, {wg_ناپ_subgroup16, rm_kq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_TQ1_0][i], "mul_mat_vec_tq1_0_f32_f32", arr_dmmv_tq1_0_f32_f32_len[reduc16], arr_dmmv_tq1_0_f32_f32_انکڑے[reduc16], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_kq, 1, 1}, {wg_ناپ_subgroup16, rm_kq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_TQ2_0][i], "mul_mat_vec_tq2_0_f32_f32", arr_dmmv_tq2_0_f32_f32_len[reduc16], arr_dmmv_tq2_0_f32_f32_انکڑے[reduc16], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_kq, 1, 1}, {wg_ناپ_subgroup16, rm_kq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f32_f32",   arr_dmmv_iq1_s_f32_f32_len[reduc16],   arr_dmmv_iq1_s_f32_f32_انکڑے[reduc16],   "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f32_f32",   arr_dmmv_iq1_m_f32_f32_len[reduc16],   arr_dmmv_iq1_m_f32_f32_انکڑے[reduc16],   "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f32_f32", arr_dmmv_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_iq2_xxs_f32_f32_انکڑے[reduc16], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f32_f32",  arr_dmmv_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_iq2_xs_f32_f32_انکڑے[reduc16],  "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f32_f32",   arr_dmmv_iq2_s_f32_f32_len[reduc16],   arr_dmmv_iq2_s_f32_f32_انکڑے[reduc16],   "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f32_f32", arr_dmmv_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_iq3_xxs_f32_f32_انکڑے[reduc16], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f32_f32",   arr_dmmv_iq3_s_f32_f32_len[reduc16],   arr_dmmv_iq3_s_f32_f32_انکڑے[reduc16],   "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f32_f32",  arr_dmmv_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_iq4_xs_f32_f32_انکڑے[reduc16],  "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f32_f32",  arr_dmmv_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_iq4_nl_f32_f32_انکڑے[reduc16],  "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f32_f32",   OCP_DMMV_LEN(arr_dmmv_mxfp4_f32_f32, reduc16), OCP_DMMV_انکڑے(arr_dmmv_mxfp4_f32_f32, reduc16), "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f32_f32",   OCP_DMMV_LEN(arr_dmmv_nvfp4_f32_f32, reduc16), OCP_DMMV_انکڑے(arr_dmmv_nvfp4_f32_f32, reduc16), "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f16_f32",  arr_dmmv_f32_f16_f32_len[reduc],  arr_dmmv_f32_f16_f32_انکڑے[reduc],  "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {1, 1, 1}, {wg_ناپ_subgroup, 1, i+1}, 1, خطأ, use_subgroups, force_subgroup_ناپ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f16_f32",  arr_dmmv_f16_f16_f32_len[reduc],  arr_dmmv_f16_f16_f32_انکڑے[reduc],  "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {2, 1, 1}, {wg_ناپ_subgroup, 2, i+1}, 1, خطأ, use_subgroups, force_subgroup_ناپ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f16_f32", arr_dmmv_bf16_f16_f32_len[reduc], arr_dmmv_bf16_f16_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {2, 1, 1}, {wg_ناپ_subgroup, 2, i+1}, 1, خطأ, use_subgroups, force_subgroup_ناپ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f16_f32", arr_dmmv_q1_0_f16_f32_len[reduc], arr_dmmv_q1_0_f16_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {2*rm_stdq, 1, 1}, {wg_ناپ_subgroup, 2*rm_stdq, i+1}, 1, سچ, use_subgroups, force_subgroup_ناپ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_f16_f32", arr_dmmv_q2_0_f16_f32_len[reduc], arr_dmmv_q2_0_f16_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {2*rm_stdq, 1, 1}, {wg_ناپ_subgroup, 2*rm_stdq, i+1}, 1, سچ, use_subgroups, force_subgroup_ناپ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f16_f32", arr_dmmv_q4_0_f16_f32_len[reduc], arr_dmmv_q4_0_f16_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {2*rm_stdq, 1, 1}, {wg_ناپ_subgroup, 2*rm_stdq, i+1}, 1, سچ, use_subgroups, force_subgroup_ناپ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f16_f32", arr_dmmv_q4_1_f16_f32_len[reduc], arr_dmmv_q4_1_f16_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {2*rm_stdq, 1, 1}, {wg_ناپ_subgroup, 2*rm_stdq, i+1}, 1, سچ, use_subgroups, force_subgroup_ناپ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f16_f32", arr_dmmv_q5_0_f16_f32_len[reduc], arr_dmmv_q5_0_f16_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {2*rm_stdq, 1, 1}, {wg_ناپ_subgroup, 2*rm_stdq, i+1}, 1, سچ, use_subgroups, force_subgroup_ناپ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f16_f32", arr_dmmv_q5_1_f16_f32_len[reduc], arr_dmmv_q5_1_f16_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {2*rm_stdq, 1, 1}, {wg_ناپ_subgroup, 2*rm_stdq, i+1}, 1, سچ, use_subgroups, force_subgroup_ناپ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f16_f32", arr_dmmv_q8_0_f16_f32_len[reduc], arr_dmmv_q8_0_f16_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {1*rm_stdq, 1, 1}, {wg_ناپ_subgroup, 1*rm_stdq, i+1}, 1, سچ, use_subgroups, force_subgroup_ناپ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f16_f32", arr_dmmv_q2_k_f16_f32_len[reduc16], arr_dmmv_q2_k_f16_f32_انکڑے[reduc16], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_kq, 1, 1}, {wg_ناپ_subgroup16, rm_kq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f16_f32", arr_dmmv_q3_k_f16_f32_len[reduc16], arr_dmmv_q3_k_f16_f32_انکڑے[reduc16], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_kq, 1, 1}, {wg_ناپ_subgroup16, rm_kq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f16_f32", arr_dmmv_q4_k_f16_f32_len[reduc16], arr_dmmv_q4_k_f16_f32_انکڑے[reduc16], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_kq, 1, 1}, {wg_ناپ_subgroup16, rm_kq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f16_f32", arr_dmmv_q5_k_f16_f32_len[reduc16], arr_dmmv_q5_k_f16_f32_انکڑے[reduc16], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_kq, 1, 1}, {wg_ناپ_subgroup16, rm_kq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f16_f32", arr_dmmv_q6_k_f16_f32_len[reduc16], arr_dmmv_q6_k_f16_f32_انکڑے[reduc16], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_kq, 1, 1}, {wg_ناپ_subgroup16, rm_kq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_TQ1_0][i], "mul_mat_vec_tq1_0_f16_f32", arr_dmmv_tq1_0_f16_f32_len[reduc16], arr_dmmv_tq1_0_f16_f32_انکڑے[reduc16], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_kq, 1, 1}, {wg_ناپ_subgroup16, rm_kq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_TQ2_0][i], "mul_mat_vec_tq2_0_f16_f32", arr_dmmv_tq2_0_f16_f32_len[reduc16], arr_dmmv_tq2_0_f16_f32_انکڑے[reduc16], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_kq, 1, 1}, {wg_ناپ_subgroup16, rm_kq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f16_f32",   arr_dmmv_iq1_s_f16_f32_len[reduc16],   arr_dmmv_iq1_s_f16_f32_انکڑے[reduc16],   "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f16_f32",   arr_dmmv_iq1_m_f16_f32_len[reduc16],   arr_dmmv_iq1_m_f16_f32_انکڑے[reduc16],   "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f16_f32", arr_dmmv_iq2_xxs_f16_f32_len[reduc16], arr_dmmv_iq2_xxs_f16_f32_انکڑے[reduc16], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f16_f32",  arr_dmmv_iq2_xs_f16_f32_len[reduc16],  arr_dmmv_iq2_xs_f16_f32_انکڑے[reduc16],  "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f16_f32",   arr_dmmv_iq2_s_f16_f32_len[reduc16],   arr_dmmv_iq2_s_f16_f32_انکڑے[reduc16],   "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f16_f32", arr_dmmv_iq3_xxs_f16_f32_len[reduc16], arr_dmmv_iq3_xxs_f16_f32_انکڑے[reduc16], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f16_f32",   arr_dmmv_iq3_s_f16_f32_len[reduc16],   arr_dmmv_iq3_s_f16_f32_انکڑے[reduc16],   "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f16_f32",  arr_dmmv_iq4_xs_f16_f32_len[reduc16],  arr_dmmv_iq4_xs_f16_f32_انکڑے[reduc16],  "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f16_f32",  arr_dmmv_iq4_nl_f16_f32_len[reduc16],  arr_dmmv_iq4_nl_f16_f32_انکڑے[reduc16],  "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f16_f32",   OCP_DMMV_LEN(arr_dmmv_mxfp4_f16_f32, reduc16), OCP_DMMV_انکڑے(arr_dmmv_mxfp4_f16_f32, reduc16), "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f16_f32",   OCP_DMMV_LEN(arr_dmmv_nvfp4_f16_f32, reduc16), OCP_DMMV_انکڑے(arr_dmmv_nvfp4_f16_f32, reduc16), "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq, i+1}, 1, سچ, use_subgroups16, force_subgroup_size16);

#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            إذا (device->integer_dot_ضرب_حاصل) {
                ثابت uint32_t subgroup_ناپ_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_ناپ_control) ? device->subgroup_min_ناپ : device->subgroup_ناپ;
                ثابت uint32_t wg_ناپ_subgroup_int = (w == DMMV_WG_ناپ_SUBGROUP) ? subgroup_ناپ_int : (subgroup_ناپ_int * 4);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_q8_1_f32", arr_dmmv_q2_0_q8_1_f32_len[reduc], arr_dmmv_q2_0_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_int_n(2*rm_kq_int, i), 1, 1}, {wg_ناپ_subgroup_int, rm_int_n(2*rm_kq_int, i), i+1}, 1, سچ, use_subgroups, subgroup_ناپ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_q8_1_f32", arr_dmmv_q4_0_q8_1_f32_len[reduc], arr_dmmv_q4_0_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_ناپ_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, سچ, use_subgroups, subgroup_ناپ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_q8_1_f32", arr_dmmv_q4_1_q8_1_f32_len[reduc], arr_dmmv_q4_1_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_ناپ_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, سچ, use_subgroups, subgroup_ناپ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_q8_1_f32", arr_dmmv_q5_0_q8_1_f32_len[reduc], arr_dmmv_q5_0_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_ناپ_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, سچ, use_subgroups, subgroup_ناپ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_q8_1_f32", arr_dmmv_q5_1_q8_1_f32_len[reduc], arr_dmmv_q5_1_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_ناپ_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, سچ, use_subgroups, subgroup_ناپ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_q8_1_f32", arr_dmmv_q8_0_q8_1_f32_len[reduc], arr_dmmv_q8_0_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_ناپ_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, سچ, use_subgroups, subgroup_ناپ_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_MXFP4][i], "mul_mat_vec_mxfp4_q8_1_f32", arr_dmmv_mxfp4_q8_1_f32_len[reduc], arr_dmmv_mxfp4_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_int_n(2*rm_stdq_int, i), 1, 1}, {wg_ناپ_subgroup_int, rm_int_n(2*rm_stdq_int, i), i+1}, 1, سچ, use_subgroups, subgroup_ناپ_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_q8_1_f32", arr_dmmv_q2_k_q8_1_f32_len[reduc], arr_dmmv_q2_k_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_int_n(2*rm_kq_int, i), 1, 1}, {wg_ناپ_subgroup_int, rm_int_n(2*rm_kq_int, i), i+1}, 1, سچ, use_subgroups, subgroup_ناپ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_q8_1_f32", arr_dmmv_q3_k_q8_1_f32_len[reduc], arr_dmmv_q3_k_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_ناپ_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, سچ, use_subgroups, subgroup_ناپ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_q8_1_f32", arr_dmmv_q4_k_q8_1_f32_len[reduc], arr_dmmv_q4_k_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_ناپ_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, سچ, use_subgroups, subgroup_ناپ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_q8_1_f32", arr_dmmv_q5_k_q8_1_f32_len[reduc], arr_dmmv_q5_k_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_ناپ_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, سچ, use_subgroups, subgroup_ناپ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_q8_1_f32", arr_dmmv_q6_k_q8_1_f32_len[reduc], arr_dmmv_q6_k_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_ناپ_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, سچ, use_subgroups, subgroup_ناپ_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_S][i], "mul_mat_vec_iq1_s_q8_1_f32", arr_dmmv_iq1_s_q8_1_f32_len[reduc], arr_dmmv_iq1_s_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {1*rm_iq_int(i), 1, 1}, {wg_ناپ_subgroup_int, 1*rm_iq_int(i), i+1}, 1, سچ, use_subgroups, subgroup_ناپ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_M][i], "mul_mat_vec_iq1_m_q8_1_f32", arr_dmmv_iq1_m_q8_1_f32_len[reduc], arr_dmmv_iq1_m_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {1*rm_iq_int(i), 1, 1}, {wg_ناپ_subgroup_int, 1*rm_iq_int(i), i+1}, 1, سچ, use_subgroups, subgroup_ناپ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ4_XS][i], "mul_mat_vec_iq4_xs_q8_1_f32", arr_dmmv_iq4_xs_q8_1_f32_len[reduc], arr_dmmv_iq4_xs_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_ڈالو_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_ناپ_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, سچ, use_subgroups, subgroup_ناپ_int);

            }
#جے_انت // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
        }

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F32 ], "mul_mat_vec_id_f32_f32",        arr_dmmv_id_f32_f32_f32_len[reduc],     arr_dmmv_id_f32_f32_f32_انکڑے[reduc],     "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {1, 1, 1}, {wg_ناپ_subgroup, 1}, 1, خطأ, use_subgroups, force_subgroup_ناپ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F16 ], "mul_mat_vec_id_f16_f32",        arr_dmmv_id_f16_f32_f32_len[reduc],     arr_dmmv_id_f16_f32_f32_انکڑے[reduc],     "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {2, 1, 1}, {wg_ناپ_subgroup, 2}, 1, خطأ, use_subgroups, force_subgroup_ناپ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_BF16], "mul_mat_vec_id_bf16_f32",       arr_dmmv_id_bf16_f32_f32_len[reduc],    arr_dmmv_id_bf16_f32_f32_انکڑے[reduc],    "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {2, 1, 1}, {wg_ناپ_subgroup, 2}, 1, خطأ, use_subgroups, force_subgroup_ناپ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q1_0], "mul_mat_vec_id_q1_0_f32",       arr_dmmv_id_q1_0_f32_f32_len[reduc],    arr_dmmv_id_q1_0_f32_f32_انکڑے[reduc],    "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {2*rm_stdq, 1, 1}, {wg_ناپ_subgroup, 2*rm_stdq}, 1, سچ, use_subgroups, force_subgroup_ناپ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q2_0], "mul_mat_vec_id_q2_0_f32",       arr_dmmv_id_q2_0_f32_f32_len[reduc],    arr_dmmv_id_q2_0_f32_f32_انکڑے[reduc],    "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {2*rm_stdq, 1, 1}, {wg_ناپ_subgroup, 2*rm_stdq}, 1, سچ, use_subgroups, force_subgroup_ناپ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_f32",       arr_dmmv_id_q4_0_f32_f32_len[reduc],    arr_dmmv_id_q4_0_f32_f32_انکڑے[reduc],    "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {2*rm_stdq, 1, 1}, {wg_ناپ_subgroup, 2*rm_stdq}, 1, سچ, use_subgroups, force_subgroup_ناپ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_f32",       arr_dmmv_id_q4_1_f32_f32_len[reduc],    arr_dmmv_id_q4_1_f32_f32_انکڑے[reduc],    "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {2*rm_stdq, 1, 1}, {wg_ناپ_subgroup, 2*rm_stdq}, 1, سچ, use_subgroups, force_subgroup_ناپ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_f32",       arr_dmmv_id_q5_0_f32_f32_len[reduc],    arr_dmmv_id_q5_0_f32_f32_انکڑے[reduc],    "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {2*rm_stdq, 1, 1}, {wg_ناپ_subgroup, 2*rm_stdq}, 1, سچ, use_subgroups, force_subgroup_ناپ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_f32",       arr_dmmv_id_q5_1_f32_f32_len[reduc],    arr_dmmv_id_q5_1_f32_f32_انکڑے[reduc],    "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {2*rm_stdq, 1, 1}, {wg_ناپ_subgroup, 2*rm_stdq}, 1, سچ, use_subgroups, force_subgroup_ناپ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_f32",       arr_dmmv_id_q8_0_f32_f32_len[reduc],    arr_dmmv_id_q8_0_f32_f32_انکڑے[reduc],    "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {1*rm_stdq, 1, 1}, {wg_ناپ_subgroup, 1*rm_stdq}, 1, سچ, use_subgroups, force_subgroup_ناپ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_f32",       arr_dmmv_id_q2_k_f32_f32_len[reduc16],    arr_dmmv_id_q2_k_f32_f32_انکڑے[reduc16],    "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_kq, 1, 1}, {wg_ناپ_subgroup16, rm_kq}, 1, سچ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_f32",       arr_dmmv_id_q3_k_f32_f32_len[reduc16],    arr_dmmv_id_q3_k_f32_f32_انکڑے[reduc16],    "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_kq, 1, 1}, {wg_ناپ_subgroup16, rm_kq}, 1, سچ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_f32",       arr_dmmv_id_q4_k_f32_f32_len[reduc16],    arr_dmmv_id_q4_k_f32_f32_انکڑے[reduc16],    "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_kq, 1, 1}, {wg_ناپ_subgroup16, rm_kq}, 1, سچ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_f32",       arr_dmmv_id_q5_k_f32_f32_len[reduc16],    arr_dmmv_id_q5_k_f32_f32_انکڑے[reduc16],    "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_kq, 1, 1}, {wg_ناپ_subgroup16, rm_kq}, 1, سچ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_f32",       arr_dmmv_id_q6_k_f32_f32_len[reduc16],    arr_dmmv_id_q6_k_f32_f32_انکڑے[reduc16],    "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_kq, 1, 1}, {wg_ناپ_subgroup16, rm_kq}, 1, سچ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_TQ1_0], "mul_mat_vec_id_tq1_0_f32",     arr_dmmv_id_tq1_0_f32_f32_len[reduc16],   arr_dmmv_id_tq1_0_f32_f32_انکڑے[reduc16],   "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_kq, 1, 1}, {wg_ناپ_subgroup16, rm_kq}, 1, سچ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_TQ2_0], "mul_mat_vec_id_tq2_0_f32",     arr_dmmv_id_tq2_0_f32_f32_len[reduc16],   arr_dmmv_id_tq2_0_f32_f32_انکڑے[reduc16],   "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_kq, 1, 1}, {wg_ناپ_subgroup16, rm_kq}, 1, سچ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_S],   "mul_mat_vec_id_iq1_s_f32",   arr_dmmv_id_iq1_s_f32_f32_len[reduc16],   arr_dmmv_id_iq1_s_f32_f32_انکڑے[reduc16],   "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq}, 1, سچ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_M],   "mul_mat_vec_id_iq1_m_f32",   arr_dmmv_id_iq1_m_f32_f32_len[reduc16],   arr_dmmv_id_iq1_m_f32_f32_انکڑے[reduc16],   "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq}, 1, سچ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XXS], "mul_mat_vec_id_iq2_xxs_f32", arr_dmmv_id_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq2_xxs_f32_f32_انکڑے[reduc16], "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq}, 1, سچ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XS],  "mul_mat_vec_id_iq2_xs_f32",  arr_dmmv_id_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq2_xs_f32_f32_انکڑے[reduc16],  "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq}, 1, سچ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_S],   "mul_mat_vec_id_iq2_s_f32",   arr_dmmv_id_iq2_s_f32_f32_len[reduc16],   arr_dmmv_id_iq2_s_f32_f32_انکڑے[reduc16],   "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq}, 1, سچ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_XXS], "mul_mat_vec_id_iq3_xxs_f32", arr_dmmv_id_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq3_xxs_f32_f32_انکڑے[reduc16], "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq}, 1, سچ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_S],   "mul_mat_vec_id_iq3_s_f32",   arr_dmmv_id_iq3_s_f32_f32_len[reduc16],   arr_dmmv_id_iq3_s_f32_f32_انکڑے[reduc16],   "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq}, 1, سچ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_XS],  "mul_mat_vec_id_iq4_xs_f32",  arr_dmmv_id_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq4_xs_f32_f32_انکڑے[reduc16],  "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq}, 1, سچ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_NL],  "mul_mat_vec_id_iq4_nl_f32",  arr_dmmv_id_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_id_iq4_nl_f32_f32_انکڑے[reduc16],  "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq}, 1, سچ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_MXFP4],   "mul_mat_vec_id_mxfp4_f32",   OCP_DMMV_LEN(arr_dmmv_id_mxfp4_f32_f32, reduc16), OCP_DMMV_انکڑے(arr_dmmv_id_mxfp4_f32_f32, reduc16), "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq}, 1, سچ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_NVFP4],   "mul_mat_vec_id_nvfp4_f32",   OCP_DMMV_LEN(arr_dmmv_id_nvfp4_f32_f32, reduc16), OCP_DMMV_انکڑے(arr_dmmv_id_nvfp4_f32_f32, reduc16), "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_iq, 1, 1}, {wg_ناپ_subgroup16, rm_iq}, 1, سچ, use_subgroups16, force_subgroup_size16);

#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        إذا (device->integer_dot_ضرب_حاصل) {
            ثابت uint32_t subgroup_ناپ_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_ناپ_control) ? device->subgroup_min_ناپ : device->subgroup_ناپ;
            ثابت uint32_t wg_ناپ_subgroup_int = (w == DMMV_WG_ناپ_SUBGROUP) ? subgroup_ناپ_int : (subgroup_ناپ_int * 4);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q2_0], "mul_mat_vec_id_q2_0_q8_1_f32", arr_dmmv_id_q2_0_q8_1_f32_len[reduc], arr_dmmv_id_q2_0_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_id(2*rm_kq_int), 1, 1}, {wg_ناپ_subgroup_int, rm_id(2*rm_kq_int)}, 1, سچ, use_subgroups, subgroup_ناپ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_q8_1_f32", arr_dmmv_id_q4_0_q8_1_f32_len[reduc], arr_dmmv_id_q4_0_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_ناپ_subgroup_int, rm_id(1*rm_stdq_int)}, 1, سچ, use_subgroups, subgroup_ناپ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_q8_1_f32", arr_dmmv_id_q4_1_q8_1_f32_len[reduc], arr_dmmv_id_q4_1_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_ناپ_subgroup_int, rm_id(1*rm_stdq_int)}, 1, سچ, use_subgroups, subgroup_ناپ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_q8_1_f32", arr_dmmv_id_q5_0_q8_1_f32_len[reduc], arr_dmmv_id_q5_0_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_ناپ_subgroup_int, rm_id(1*rm_stdq_int)}, 1, سچ, use_subgroups, subgroup_ناپ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_q8_1_f32", arr_dmmv_id_q5_1_q8_1_f32_len[reduc], arr_dmmv_id_q5_1_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_ناپ_subgroup_int, rm_id(1*rm_stdq_int)}, 1, سچ, use_subgroups, subgroup_ناپ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_q8_1_f32", arr_dmmv_id_q8_0_q8_1_f32_len[reduc], arr_dmmv_id_q8_0_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_ناپ_subgroup_int, rm_id(1*rm_stdq_int)}, 1, سچ, use_subgroups, subgroup_ناپ_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_MXFP4], "mul_mat_vec_id_mxfp4_q8_1_f32", arr_dmmv_id_mxfp4_q8_1_f32_len[reduc], arr_dmmv_id_mxfp4_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_id(2*rm_stdq_int), 1, 1}, {wg_ناپ_subgroup_int, rm_id(2*rm_stdq_int)}, 1, سچ, use_subgroups, subgroup_ناپ_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_q8_1_f32", arr_dmmv_id_q2_k_q8_1_f32_len[reduc], arr_dmmv_id_q2_k_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_id(2*rm_kq_int), 1, 1}, {wg_ناپ_subgroup_int, rm_id(2*rm_kq_int)}, 1, سچ, use_subgroups, subgroup_ناپ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_q8_1_f32", arr_dmmv_id_q3_k_q8_1_f32_len[reduc], arr_dmmv_id_q3_k_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_ناپ_subgroup_int, rm_id(1*rm_kq_int)}, 1, سچ, use_subgroups, subgroup_ناپ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_q8_1_f32", arr_dmmv_id_q4_k_q8_1_f32_len[reduc], arr_dmmv_id_q4_k_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_ناپ_subgroup_int, rm_id(1*rm_kq_int)}, 1, سچ, use_subgroups, subgroup_ناپ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_q8_1_f32", arr_dmmv_id_q5_k_q8_1_f32_len[reduc], arr_dmmv_id_q5_k_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_ناپ_subgroup_int, rm_id(1*rm_kq_int)}, 1, سچ, use_subgroups, subgroup_ناپ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_q8_1_f32", arr_dmmv_id_q6_k_q8_1_f32_len[reduc], arr_dmmv_id_q6_k_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_ناپ_subgroup_int, rm_id(1*rm_kq_int)}, 1, سچ, use_subgroups, subgroup_ناپ_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_S], "mul_mat_vec_id_iq1_s_q8_1_f32", arr_dmmv_id_iq1_s_q8_1_f32_len[reduc], arr_dmmv_id_iq1_s_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {1*rm_iq_int(0), 1, 1}, {wg_ناپ_subgroup_int, 1*rm_iq_int(0)}, 1, سچ, use_subgroups, subgroup_ناپ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_M], "mul_mat_vec_id_iq1_m_q8_1_f32", arr_dmmv_id_iq1_m_q8_1_f32_len[reduc], arr_dmmv_id_iq1_m_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {1*rm_iq_int(0), 1, 1}, {wg_ناپ_subgroup_int, 1*rm_iq_int(0)}, 1, سچ, use_subgroups, subgroup_ناپ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ4_XS], "mul_mat_vec_id_iq4_xs_q8_1_f32", arr_dmmv_id_iq4_xs_q8_1_f32_len[reduc], arr_dmmv_id_iq4_xs_q8_1_f32_انکڑے[reduc], "main", mul_mat_vec_id_عدد_bindings, حجم(vk_mat_vec_id_ڈالو_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_ناپ_subgroup_int, rm_id(1*rm_stdq_int)}, 1, سچ, use_subgroups, subgroup_ناپ_int);
        }
#جے_انت // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
    }

#تعریف_رد OCP_DMMV_انکڑے
#تعریف_رد OCP_DMMV_LEN

#إذا !defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    GGML_بن_ورتیا(rm_stdq_int);
    GGML_بن_ورتیا(rm_kq_int);
    GGML_بن_ورتیا(is_rdna3_or_4);
    GGML_بن_ورتیا(rm_int_n);
    GGML_بن_ورتیا(rm_id);
    GGML_بن_ورتیا(rm_iq_int);
#جے_انت

    // dequant shaders
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_F32 ], "f32_to_f16",   dequant_f32_len,  dequant_f32_انکڑے,  "main", 2, 5 * حجم(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q1_0], "dequant_q1_0", dequant_q1_0_len, dequant_q1_0_انکڑے, "main", 2, 5 * حجم(uint32_t), {256 * 8, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_0], "dequant_q2_0", dequant_q2_0_len, dequant_q2_0_انکڑے, "main", 2, 5 * حجم(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_0], "dequant_q4_0", dequant_q4_0_len, dequant_q4_0_انکڑے, "main", 2, 5 * حجم(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_1], "dequant_q4_1", dequant_q4_1_len, dequant_q4_1_انکڑے, "main", 2, 5 * حجم(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_0], "dequant_q5_0", dequant_q5_0_len, dequant_q5_0_انکڑے, "main", 2, 5 * حجم(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_1], "dequant_q5_1", dequant_q5_1_len, dequant_q5_1_انکڑے, "main", 2, 5 * حجم(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q8_0], "dequant_q8_0", dequant_q8_0_len, dequant_q8_0_انکڑے, "main", 2, 5 * حجم(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_transpose[GGML_TYPE_Q8_0], "dequant_q8_0_transpose", dequant_q8_0_transpose_len, dequant_q8_0_transpose_انکڑے, "main", 2, 5 * حجم(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_K], "dequant_q2_k", dequant_q2_k_len, dequant_q2_k_انکڑے, "main", 2, 5 * حجم(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q3_K], "dequant_q3_k", dequant_q3_k_len, dequant_q3_k_انکڑے, "main", 2, 5 * حجم(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_K], "dequant_q4_k", dequant_q4_k_len, dequant_q4_k_انکڑے, "main", 2, 5 * حجم(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_K], "dequant_q5_k", dequant_q5_k_len, dequant_q5_k_انکڑے, "main", 2, 5 * حجم(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q6_K], "dequant_q6_k", dequant_q6_k_len, dequant_q6_k_انکڑے, "main", 2, 5 * حجم(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_TQ1_0], "dequant_tq1_0", dequant_tq1_0_len, dequant_tq1_0_انکڑے, "main", 2, 5 * حجم(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_TQ2_0], "dequant_tq2_0", dequant_tq2_0_len, dequant_tq2_0_انکڑے, "main", 2, 5 * حجم(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_S],   "dequant_iq1_s",   dequant_iq1_s_len,   dequant_iq1_s_انکڑے,   "main", 2, 5 * حجم(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_M],   "dequant_iq1_m",   dequant_iq1_m_len,   dequant_iq1_m_انکڑے,   "main", 2, 5 * حجم(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XXS], "dequant_iq2_xxs", dequant_iq2_xxs_len, dequant_iq2_xxs_انکڑے, "main", 2, 5 * حجم(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XS],  "dequant_iq2_xs",  dequant_iq2_xs_len,  dequant_iq2_xs_انکڑے,  "main", 2, 5 * حجم(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_S],   "dequant_iq2_s",   dequant_iq2_s_len,   dequant_iq2_s_انکڑے,   "main", 2, 5 * حجم(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_XXS], "dequant_iq3_xxs", dequant_iq3_xxs_len, dequant_iq3_xxs_انکڑے, "main", 2, 5 * حجم(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_S],   "dequant_iq3_s",   dequant_iq3_s_len,   dequant_iq3_s_انکڑے,   "main", 2, 5 * حجم(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_XS],  "dequant_iq4_xs",  dequant_iq4_xs_len,  dequant_iq4_xs_انکڑے,  "main", 2, 5 * حجم(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_NL],  "dequant_iq4_nl",  dequant_iq4_nl_len,  dequant_iq4_nl_انکڑے,  "main", 2, 5 * حجم(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_MXFP4],   "dequant_mxfp4",   dequant_mxfp4_len,   dequant_mxfp4_انکڑے,   "main", 2, 5 * حجم(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_NVFP4],   "dequant_nvfp4",   dequant_nvfp4_len,   dequant_nvfp4_انکڑے,   "main", 2, 5 * حجم(uint32_t), {256 * 16, 1, 1}, {}, 1);

    // get_rows
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_F32 ], "get_rows_f32",  get_صفاں_f32_len,  get_صفاں_f32_انکڑے,  "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_F16 ], "get_rows_f16",  get_صفاں_f16_len,  get_صفاں_f16_انکڑے,  "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_BF16], "get_rows_bf16", get_صفاں_bf16_len, get_صفاں_bf16_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_Q1_0], "get_rows_q1_0", get_صفاں_q1_0_len, get_صفاں_q1_0_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_Q2_0], "get_rows_q2_0", get_صفاں_q2_0_len, get_صفاں_q2_0_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_Q4_0], "get_rows_q4_0", get_صفاں_q4_0_len, get_صفاں_q4_0_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_Q4_1], "get_rows_q4_1", get_صفاں_q4_1_len, get_صفاں_q4_1_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_Q5_0], "get_rows_q5_0", get_صفاں_q5_0_len, get_صفاں_q5_0_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_Q5_1], "get_rows_q5_1", get_صفاں_q5_1_len, get_صفاں_q5_1_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_Q8_0], "get_rows_q8_0", get_صفاں_q8_0_len, get_صفاں_q8_0_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_Q2_K], "get_rows_q2_k", get_صفاں_q2_k_len, get_صفاں_q2_k_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_Q3_K], "get_rows_q3_k", get_صفاں_q3_k_len, get_صفاں_q3_k_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_Q4_K], "get_rows_q4_k", get_صفاں_q4_k_len, get_صفاں_q4_k_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_Q5_K], "get_rows_q5_k", get_صفاں_q5_k_len, get_صفاں_q5_k_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_Q6_K], "get_rows_q6_k", get_صفاں_q6_k_len, get_صفاں_q6_k_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_TQ1_0], "get_rows_tq1_0", get_صفاں_tq1_0_len, get_صفاں_tq1_0_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_TQ2_0], "get_rows_tq2_0", get_صفاں_tq2_0_len, get_صفاں_tq2_0_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_IQ1_S],   "get_rows_iq1_s",   get_صفاں_iq1_s_len,   get_صفاں_iq1_s_انکڑے,   "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_IQ1_M],   "get_rows_iq1_m",   get_صفاں_iq1_m_len,   get_صفاں_iq1_m_انکڑے,   "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs", get_صفاں_iq2_xxs_len, get_صفاں_iq2_xxs_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs",  get_صفاں_iq2_xs_len,  get_صفاں_iq2_xs_انکڑے,  "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_IQ2_S],   "get_rows_iq2_s",   get_صفاں_iq2_s_len,   get_صفاں_iq2_s_انکڑے,   "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs", get_صفاں_iq3_xxs_len, get_صفاں_iq3_xxs_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_IQ3_S],   "get_rows_iq3_s",   get_صفاں_iq3_s_len,   get_صفاں_iq3_s_انکڑے,   "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs",  get_صفاں_iq4_xs_len,  get_صفاں_iq4_xs_انکڑے,  "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl",  get_صفاں_iq4_nl_len,  get_صفاں_iq4_nl_انکڑے,  "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_MXFP4],   "get_rows_mxfp4",   get_صفاں_mxfp4_len,   get_صفاں_mxfp4_انکڑے,   "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_NVFP4],   "get_rows_nvfp4",   get_صفاں_nvfp4_len,   get_صفاں_nvfp4_انکڑے,   "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں[GGML_TYPE_I32],     "get_rows_i32",     get_صفاں_i32_len,     get_صفاں_i32_انکڑے,     "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_F32 ], "get_rows_f32_f32",  get_صفاں_f32_f32_len,  get_صفاں_f32_f32_انکڑے,  "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_F16 ], "get_rows_f16_f32",  get_صفاں_f16_f32_len,  get_صفاں_f16_f32_انکڑے,  "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_BF16], "get_rows_bf16_f32", get_صفاں_bf16_f32_len, get_صفاں_bf16_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_Q1_0], "get_rows_q1_0_f32", get_صفاں_q1_0_f32_len, get_صفاں_q1_0_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_Q2_0], "get_rows_q2_0_f32", get_صفاں_q2_0_f32_len, get_صفاں_q2_0_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_Q4_0], "get_rows_q4_0_f32", get_صفاں_q4_0_f32_len, get_صفاں_q4_0_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_Q4_1], "get_rows_q4_1_f32", get_صفاں_q4_1_f32_len, get_صفاں_q4_1_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_Q5_0], "get_rows_q5_0_f32", get_صفاں_q5_0_f32_len, get_صفاں_q5_0_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_Q5_1], "get_rows_q5_1_f32", get_صفاں_q5_1_f32_len, get_صفاں_q5_1_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_Q8_0], "get_rows_q8_0_f32", get_صفاں_q8_0_f32_len, get_صفاں_q8_0_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_Q2_K], "get_rows_q2_k_f32", get_صفاں_q2_k_f32_len, get_صفاں_q2_k_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_Q3_K], "get_rows_q3_k_f32", get_صفاں_q3_k_f32_len, get_صفاں_q3_k_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_Q4_K], "get_rows_q4_k_f32", get_صفاں_q4_k_f32_len, get_صفاں_q4_k_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_Q5_K], "get_rows_q5_k_f32", get_صفاں_q5_k_f32_len, get_صفاں_q5_k_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_Q6_K], "get_rows_q6_k_f32", get_صفاں_q6_k_f32_len, get_صفاں_q6_k_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_TQ1_0], "get_rows_tq1_0_f32", get_صفاں_tq1_0_f32_len, get_صفاں_tq1_0_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_TQ2_0], "get_rows_tq2_0_f32", get_صفاں_tq2_0_f32_len, get_صفاں_tq2_0_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_IQ1_S],   "get_rows_iq1_s_f32",   get_صفاں_iq1_s_f32_len,   get_صفاں_iq1_s_f32_انکڑے,   "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_IQ1_M],   "get_rows_iq1_m_f32",   get_صفاں_iq1_m_f32_len,   get_صفاں_iq1_m_f32_انکڑے,   "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs_f32", get_صفاں_iq2_xxs_f32_len, get_صفاں_iq2_xxs_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs_f32",  get_صفاں_iq2_xs_f32_len,  get_صفاں_iq2_xs_f32_انکڑے,  "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_IQ2_S],   "get_rows_iq2_s_f32",   get_صفاں_iq2_s_f32_len,   get_صفاں_iq2_s_f32_انکڑے,   "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs_f32", get_صفاں_iq3_xxs_f32_len, get_صفاں_iq3_xxs_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_IQ3_S],   "get_rows_iq3_s_f32",   get_صفاں_iq3_s_f32_len,   get_صفاں_iq3_s_f32_انکڑے,   "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs_f32",  get_صفاں_iq4_xs_f32_len,  get_صفاں_iq4_xs_f32_انکڑے,  "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl_f32",  get_صفاں_iq4_nl_f32_len,  get_صفاں_iq4_nl_f32_انکڑے,  "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_MXFP4],   "get_rows_mxfp4_f32",   get_صفاں_mxfp4_f32_len,   get_صفاں_mxfp4_f32_انکڑے,   "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_f32[GGML_TYPE_NVFP4],   "get_rows_nvfp4_f32",   get_صفاں_nvfp4_f32_len,   get_صفاں_nvfp4_f32_انکڑے,   "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_صفاں_پٹھ_f32, "get_rows_back_f32", get_صفاں_پٹھ_f32_len, get_صفاں_پٹھ_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {256, 1, 1}, {}, 1, سچ);

    ggml_vk_create_pipeline(device, device->pipeline_matmul_ونڈو_k_reduce, "split_k_reduce", ونڈو_k_reduce_len, ونڈو_k_reduce_انکڑے, "main", 2, 2 * حجم(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_flash_attn_ونڈو_k_reduce, "fa_split_k_reduce", fa_ونڈو_k_reduce_len, fa_ونڈو_k_reduce_انکڑے, "main", 3, حجم(vk_op_flash_attn_ونڈو_k_reduce_ڈالو_constants), {1, device->subgroup_ناپ, 1}, {device->subgroup_ناپ}, 1, سچ);

#إذا defined(VK_KHR_cooperative_میٹرکس) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    إذا (device->vendor_id == VK_VENDOR_ID_INTEL && (device->architecture == INTEL_XE2 || (device->architecture == INTEL_XE1 && device->coopmat_support && device->uma))) {
        خودکار اتلا_power_دا_2 = [&](uint32_t وچ) {
            GGML_ASSERT(وچ != 0);
            إذا (وچ <= 1) أرجع 1u;
            uint32_t ret = وچ - 1;
            ret |= ret >> 1;
            ret |= ret >> 2;
            ret |= ret >> 4;
            ret |= ret >> 8;
            ret |= ret >> 16;
            أرجع ret + 1;
            };

        uint32_t xe_native_sub_گروہ_ناپ = 16;
        إذا (device->architecture == INTEL_XE1) {
            xe_native_sub_گروہ_ناپ = 8;
        }

        لكل (خودکار& it : device->pipeline_xe_fa_decode_dual_phases) {
            ثابت uint32_t ونڈو_p_chunk = 32;
            خودکار HdQk = it.پہلا;
            خودکار& pipelines = it.second;
            uint32_t سر_dim_qk = std::get<0>(HdQk);
            uint32_t سر_dim_pv = std::get<1>(HdQk);
            uint32_t gqa_تناسب = std::get<2>(HdQk);
            uint32_t q_len = std::get<3>(HdQk);
            ثابت uint32_t باہرلا_dim_فی_wg = gqa_تناسب > 16 ? 8 : 16;
            uint32_t aligned_q_len = اتلا_power_دا_2(q_len);
            uint32_t گروہ_sz_ph1 = std::min(std::max(aligned_q_len * xe_native_sub_گروہ_ناپ, 64u), 256u);
            uint32_t باہرلا_فی_wg_ph1 = std::min(q_len, 256u / xe_native_sub_گروہ_ناپ);
            uint32_t aligned_gqa_تناسب = اتلا_power_دا_2(gqa_تناسب);
            uint32_t ونڈو_p_فی_iter_ph2 = 256;
            uint32_t ونڈو_p_فی_warp = 16;
            uint32_t گروہ_sz_ph2 = (ونڈو_p_فی_iter_ph2 / ونڈو_p_فی_warp) * xe_native_sub_گروہ_ناپ;
            uint32_t باہرلا_فی_wg_ph2 = std::min(std::max(16u / aligned_gqa_تناسب, 1u), q_len);
            ggml_vk_create_pipeline(device, pipelines.پہلا, "xe_fa_decode_ph1", fa_decode_ph1_cm1_len, fa_decode_ph1_cm1_انکڑے, "main", 5, حجم(vk_fa_xe_opt_ڈالو_constants), { 1, 32, 1 }, { گروہ_sz_ph1, gqa_تناسب, سر_dim_qk, xe_native_sub_گروہ_ناپ, ونڈو_p_chunk, باہرلا_فی_wg_ph1 }, 1, خطأ, سچ, xe_native_sub_گروہ_ناپ);
            ggml_vk_create_pipeline(device, pipelines.second, "xe_fa_decode_ph2", fa_decode_ph2_cm1_len, fa_decode_ph2_cm1_انکڑے, "main", 5, حجم(vk_fa_xe_opt_ڈالو_constants), { 1, 1, 1 }, { گروہ_sz_ph2, gqa_تناسب, سر_dim_pv, باہرلا_فی_wg_ph2, xe_native_sub_گروہ_ناپ, ونڈو_p_فی_iter_ph2, ونڈو_p_chunk, باہرلا_dim_فی_wg }, 1, خطأ, سچ, xe_native_sub_گروہ_ناپ);
        }
    }
#جے_انت

    لكل (خودکار &it : device->pipeline_fa_نقاب_opt) {
        خودکار BrBc = it.پہلا;
        ggml_vk_create_pipeline(device, it.second, "fa_mask_opt", fa_نقاب_opt_len, fa_نقاب_opt_انکڑے, "main", 2, حجم(vk_op_flash_attn_نقاب_opt_ڈالو_constants), {1, 1, 1}, {128, 128 / device->subgroup_ناپ, BrBc.پہلا, BrBc.second}, 1, سچ, سچ, device->subgroup_ناپ);
    }

    {
        // Large workgroup so the per-row KV scan parallelizes; capped to device limits.
        ثابت uint32_t compact_max = std::min({1024u, device->properties.limits.max_Compute_Work_گروہ_Invocations, device->properties.limits.max_Compute_Work_گروہ_ناپ[0]});

        // Fast ballot prefix-sum path when the device supports full subgroups; otherwise
        // a shared-memory prefix-sum fallback. Both emit a deterministic ascending list.
        device->fa_sparse_compact_use_subgroups = device->subgroup_ballot && device->subgroup_require_بھریا_support;
        إذا (device->fa_sparse_compact_use_subgroups) {
            ثابت uint32_t compact_wg = std::max(device->subgroup_ناپ, (compact_max / device->subgroup_ناپ) * device->subgroup_ناپ);
            ثابت uint32_t compact_عدد_sg = compact_wg / device->subgroup_ناپ;
            ggml_vk_create_pipeline(device, device->pipeline_fa_sparse_compact_subgroup, "fa_sparse_compact_subgroup", fa_sparse_compact_subgroup_len, fa_sparse_compact_subgroup_انکڑے, "main", 2, حجم(vk_op_flash_attn_sparse_compact_ڈالو_constants), {1, 1, 1}, {compact_wg, compact_عدد_sg}, 1, سچ, سچ, device->subgroup_ناپ);
        } وإلا {
            ggml_vk_create_pipeline(device, device->pipeline_fa_sparse_compact, "fa_sparse_compact", fa_sparse_compact_len, fa_sparse_compact_انکڑے, "main", 2, حجم(vk_op_flash_attn_sparse_compact_ڈالو_constants), {1, 1, 1}, {compact_max}, 1, سچ);
        }
    }

    إذا (device->subgroup_clustered && device->subgroup_require_بھریا_support) {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_subgroup_len, quantize_q8_1_x4_subgroup_انکڑے, "main", 2, حجم(vk_quantize_q8_1_ڈالو_constants), {32 * device->subgroup_ناپ / 8, 1, 1}, { device->subgroup_ناپ }, 1, سچ, سچ);
    } وإلا {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_len, quantize_q8_1_x4_انکڑے, "main", 2, حجم(vk_quantize_q8_1_ڈالو_constants), {32 * device->subgroup_ناپ / 8, 1, 1}, { device->subgroup_ناپ }, 1);
    }

    لكل (uint32_t i = 0; i < p021_max_gqa_تناسب; ++i) {
        إذا (device->subgroup_arithmetic && device->subgroup_require_بھریا_support) {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_subgroup_جوڑو_len, mul_mat_vec_p021_f16_f32_subgroup_جوڑو_انکڑے, "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_p021_ڈالو_constants), {1, 1, 1}, {device->subgroup_ناپ, i + 1}, 1, سچ, سچ);
        } وإلا {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_len,              mul_mat_vec_p021_f16_f32_انکڑے,              "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_p021_ڈالو_constants), {1, 1, 1}, {device->subgroup_ناپ, i + 1}, 1, سچ);
        }
    }
    ggml_vk_create_pipeline(device, device->pipeline_mul_mat_vec_nc_f16_f32, "mul_mat_vec_nc_f16_f32", mul_mat_vec_nc_f16_f32_len, mul_mat_vec_nc_f16_f32_انکڑے, "main", mul_mat_vec_عدد_bindings, حجم(vk_mat_vec_nc_ڈالو_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_norm_f32, "norm_f32", norm_f32_len, norm_f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_گروہ_norm_f32, "group_norm_f32", گروہ_norm_f32_len, گروہ_norm_f32_انکڑے, "main", 2, حجم(vk_op_ڈالو_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_f32, "rms_norm_f32", rms_norm_f32_len, rms_norm_f32_انکڑے, "main", 4, حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, {0, 0}, 1, سچ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_f32, "rms_norm_mul_f32", rms_norm_f32_len, rms_norm_f32_انکڑے, "main", 4, حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, {0, 1}, 1, سچ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_جوڑو_f32, "rms_norm_mul_add_f32", rms_norm_mul_جوڑو_f32_len, rms_norm_mul_جوڑو_f32_انکڑے, "main", 5, حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, {0, 1, 0}, 1, سچ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_جوڑو_mul_f32, "rms_norm_mul_add_mul_f32", rms_norm_mul_جوڑو_f32_len, rms_norm_mul_جوڑو_f32_انکڑے, "main", 5, حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, {0, 1, 1}, 1, سچ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_جوڑو_partials_f32, "rms_norm_mul_add_partials_f32", rms_norm_mul_جوڑو_partials_f32_len, rms_norm_mul_جوڑو_partials_f32_انکڑے, "main", 6, حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, {0, 1, 0}, 1, سچ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_جوڑو_mul_partials_f32, "rms_norm_mul_add_mul_partials_f32", rms_norm_mul_جوڑو_partials_f32_len, rms_norm_mul_جوڑو_partials_f32_انکڑے, "main", 6, حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, {0, 1, 1}, 1, سچ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_سیٹ_صفاں_f32_f32, "rms_norm_set_rows_f32_f32", rms_norm_سیٹ_صفاں_f32_f32_len, rms_norm_سیٹ_صفاں_f32_f32_انکڑے, "main", 4, حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, {0, 0}, 1, سچ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_سیٹ_صفاں_f32_f16, "rms_norm_set_rows_f32_f16", rms_norm_سیٹ_صفاں_f32_f16_len, rms_norm_سیٹ_صفاں_f32_f16_انکڑے, "main", 4, حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, {0, 0}, 1, سچ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_partials_f32, "rms_norm_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_انکڑے, "main", 4, حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, {0, 0}, 1, سچ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_partials_f32, "rms_norm_mul_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_انکڑے, "main", 4, حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, {0, 1}, 1, سچ);

    إذا (حجم(vk_op_rms_norm_mul_rope_ڈالو_constants) <= device->properties.limits.max_ڈالو_Constants_ناپ) {
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f32, "rms_norm_mul_rope_f32_f32", rms_norm_mul_rope_f32_f32_len, rms_norm_mul_rope_f32_f32_انکڑے, "main", 7, حجم(vk_op_rms_norm_mul_rope_ڈالو_constants), {1, 1, 1}, {0, 1}, 1, سچ);
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f16, "rms_norm_mul_rope_f32_f16", rms_norm_mul_rope_f32_f16_len, rms_norm_mul_rope_f32_f16_انکڑے, "main", 7, حجم(vk_op_rms_norm_mul_rope_ڈالو_constants), {1, 1, 1}, {0, 1}, 1, سچ);
    }

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_پٹھ_f32, "rms_norm_back_f32", rms_norm_پٹھ_f32_len, rms_norm_پٹھ_f32_انکڑے, "main", 3, حجم(vk_op_ڈالو_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_l2_norm_f32, "l2_norm_f32", l2_norm_f32_len, l2_norm_f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_32, "cpy_32", cpy_32_len, cpy_32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_f16, "cpy_f32_f16", cpy_f32_f16_len, cpy_f32_f16_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_16, "cpy_16", cpy_16_len, cpy_16_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f16_f32, "cpy_f16_f32", cpy_f16_f32_len, cpy_f16_f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_bf16,"cpy_f32_bf16",cpy_f32_bf16_len,cpy_f32_bf16_انکڑے,"main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_bf16_f32,"cpy_bf16_f32",cpy_bf16_f32_len,cpy_bf16_f32_انکڑے,"main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_bf16_f16,"cpy_bf16_f16",cpy_bf16_f16_len,cpy_bf16_f16_انکڑے,"main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_i32_f32, "cpy_i32_f32", cpy_i32_f32_len, cpy_i32_f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_i32, "cpy_f32_i32", cpy_f32_i32_len, cpy_f32_i32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_32, "contig_cpy_32", contig_cpy_32_len, contig_cpy_32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_f16, "contig_cpy_f32_f16", contig_cpy_f32_f16_len, contig_cpy_f32_f16_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_16, "contig_cpy_16", contig_cpy_16_len, contig_cpy_16_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f16_f32, "contig_cpy_f16_f32", contig_cpy_f16_f32_len, contig_cpy_f16_f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_bf16,"contig_cpy_f32_bf16",contig_cpy_f32_bf16_len,contig_cpy_f32_bf16_انکڑے,"main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_bf16_f32,"contig_cpy_bf16_f32",contig_cpy_bf16_f32_len,contig_cpy_bf16_f32_انکڑے,"main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_bf16_f16,"contig_cpy_bf16_f16",contig_cpy_bf16_f16_len,contig_cpy_bf16_f16_انکڑے,"main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_i32_f32, "contig_cpy_i32_f32", contig_cpy_i32_f32_len, contig_cpy_i32_f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_i32, "contig_cpy_f32_i32", contig_cpy_f32_i32_len, contig_cpy_f32_i32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_32, "cpy_transpose_32", cpy_transpose_32_len, cpy_transpose_32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_16, "cpy_transpose_16", cpy_transpose_16_len, cpy_transpose_16_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_02_32, "cpy_transpose_02_32", cpy_transpose_02_32_len, cpy_transpose_02_32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_02_16, "cpy_transpose_02_16", cpy_transpose_02_16_len, cpy_transpose_02_16_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q1_0], "cpy_f32_q1_0", cpy_f32_q1_0_len, cpy_f32_q1_0_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q2_0], "cpy_f32_q2_0", cpy_f32_q2_0_len, cpy_f32_q2_0_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_0], "cpy_f32_q4_0", cpy_f32_q4_0_len, cpy_f32_q4_0_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_1], "cpy_f32_q4_1", cpy_f32_q4_1_len, cpy_f32_q4_1_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_0], "cpy_f32_q5_0", cpy_f32_q5_0_len, cpy_f32_q5_0_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_1], "cpy_f32_q5_1", cpy_f32_q5_1_len, cpy_f32_q5_1_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q8_0], "cpy_f32_q8_0", cpy_f32_q8_0_len, cpy_f32_q8_0_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_IQ4_NL], "cpy_f32_iq4_nl", cpy_f32_iq4_nl_len, cpy_f32_iq4_nl_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {32, 1, 1}, {}, 1);

#تعریف_کرو سیٹ_صفاں_2(src_idx, src, itype) \
        ggml_vk_create_pipeline(device, device->pipeline_سیٹ_صفاں ## itype [src_idx][GGML_TYPE_F32],  "set_rows_" #src "_f32" #itype,  سیٹ_صفاں_3 ## src ## _f32 ## itype ## _len,  سیٹ_صفاں_3 ## src ## _f32 ## itype ## انکڑے_2,  "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, {1}, 1, سچ); \
        ggml_vk_create_pipeline(device, device->pipeline_سیٹ_صفاں ## itype [src_idx][GGML_TYPE_F16],  "set_rows_" #src "_f16" #itype,  سیٹ_صفاں_3 ## src ## _f16 ## itype ## _len,  سیٹ_صفاں_3 ## src ## _f16 ## itype ## انکڑے_2,  "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, {1}, 1, سچ); \
        ggml_vk_create_pipeline(device, device->pipeline_سیٹ_صفاں ## itype [src_idx][GGML_TYPE_BF16], "set_rows_" #src "_bf16" #itype, سیٹ_صفاں_3 ## src ## _bf16 ## itype ## _len, سیٹ_صفاں_3 ## src ## _bf16 ## itype ## انکڑے_2, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, {1}, 1, سچ); \
        ggml_vk_create_pipeline(device, device->pipeline_سیٹ_صفاں ## itype [src_idx][GGML_TYPE_Q1_0], "set_rows_" #src "_q1_0" #itype, سیٹ_صفاں_3 ## src ## _q1_0 ## itype ## _len, سیٹ_صفاں_3 ## src ## _q1_0 ## itype ## انکڑے_2, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, {1}, 1, سچ); \
        ggml_vk_create_pipeline(device, device->pipeline_سیٹ_صفاں ## itype [src_idx][GGML_TYPE_Q2_0], "set_rows_" #src "_q2_0" #itype, سیٹ_صفاں_3 ## src ## _q2_0 ## itype ## _len, سیٹ_صفاں_3 ## src ## _q2_0 ## itype ## انکڑے_2, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, {1}, 1, سچ); \
        ggml_vk_create_pipeline(device, device->pipeline_سیٹ_صفاں ## itype [src_idx][GGML_TYPE_Q4_0], "set_rows_" #src "_q4_0" #itype, سیٹ_صفاں_3 ## src ## _q4_0 ## itype ## _len, سیٹ_صفاں_3 ## src ## _q4_0 ## itype ## انکڑے_2, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, {1}, 1, سچ); \
        ggml_vk_create_pipeline(device, device->pipeline_سیٹ_صفاں ## itype [src_idx][GGML_TYPE_Q4_1], "set_rows_" #src "_q4_1" #itype, سیٹ_صفاں_3 ## src ## _q4_1 ## itype ## _len, سیٹ_صفاں_3 ## src ## _q4_1 ## itype ## انکڑے_2, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, {1}, 1, سچ); \
        ggml_vk_create_pipeline(device, device->pipeline_سیٹ_صفاں ## itype [src_idx][GGML_TYPE_Q5_0], "set_rows_" #src "_q5_0" #itype, سیٹ_صفاں_3 ## src ## _q5_0 ## itype ## _len, سیٹ_صفاں_3 ## src ## _q5_0 ## itype ## انکڑے_2, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, {1}, 1, سچ); \
        ggml_vk_create_pipeline(device, device->pipeline_سیٹ_صفاں ## itype [src_idx][GGML_TYPE_Q5_1], "set_rows_" #src "_q5_1" #itype, سیٹ_صفاں_3 ## src ## _q5_1 ## itype ## _len, سیٹ_صفاں_3 ## src ## _q5_1 ## itype ## انکڑے_2, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, {1}, 1, سچ); \
        ggml_vk_create_pipeline(device, device->pipeline_سیٹ_صفاں ## itype [src_idx][GGML_TYPE_Q8_0], "set_rows_" #src "_q8_0" #itype, سیٹ_صفاں_3 ## src ## _q8_0 ## itype ## _len, سیٹ_صفاں_3 ## src ## _q8_0 ## itype ## انکڑے_2, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, {1}, 1, سچ); \
        ggml_vk_create_pipeline(device, device->pipeline_سیٹ_صفاں ## itype [src_idx][GGML_TYPE_IQ4_NL], "set_rows_" #src "_iq4_nl" #itype, سیٹ_صفاں_3 ## src ## _iq4_nl ## itype ## _len, سیٹ_صفاں_3 ## src ## _iq4_nl ## itype ## انکڑے_2, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, {1}, 1, سچ);

    سیٹ_صفاں_2(0, f32, _i32)
    سیٹ_صفاں_2(0, f32, _i64)
    سیٹ_صفاں_2(1, f16, _i32)
    سیٹ_صفاں_2(1, f16, _i64)
#تعریف_رد سیٹ_صفاں_2


    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q1_0], "cpy_q1_0_f32", cpy_q1_0_f32_len, cpy_q1_0_f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {(uint32_t)ggml_blck_ناپ(GGML_TYPE_Q1_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q2_0], "cpy_q2_0_f32", cpy_q2_0_f32_len, cpy_q2_0_f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {(uint32_t)ggml_blck_ناپ(GGML_TYPE_Q2_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_0], "cpy_q4_0_f32", cpy_q4_0_f32_len, cpy_q4_0_f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {(uint32_t)ggml_blck_ناپ(GGML_TYPE_Q4_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_1], "cpy_q4_1_f32", cpy_q4_1_f32_len, cpy_q4_1_f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {(uint32_t)ggml_blck_ناپ(GGML_TYPE_Q4_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_0], "cpy_q5_0_f32", cpy_q5_0_f32_len, cpy_q5_0_f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {(uint32_t)ggml_blck_ناپ(GGML_TYPE_Q5_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_1], "cpy_q5_1_f32", cpy_q5_1_f32_len, cpy_q5_1_f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {(uint32_t)ggml_blck_ناپ(GGML_TYPE_Q5_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q8_0], "cpy_q8_0_f32", cpy_q8_0_f32_len, cpy_q8_0_f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {(uint32_t)ggml_blck_ناپ(GGML_TYPE_Q8_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_IQ4_NL], "cpy_iq4_nl_f32", cpy_iq4_nl_f32_len, cpy_iq4_nl_f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {(uint32_t)ggml_blck_ناپ(GGML_TYPE_IQ4_NL), 1, 1}, {}, 1);

    خودکار get_suffix = [](منطقي src0_f16, منطقي src1_f16, منطقي dst_f16) {
        std::string s;
        s += std::string(src0_f16 ? "_f16" : "_f32");
        s += std::string(src1_f16 ? "_f16" : "_f32");
        s += std::string(dst_f16 ? "_f16" : "_f32");
        أرجع s;
    };

#تعریف_کرو CREATE_ثنائی(name, namemod, spec, bindings) \
    لكل (صحيح s0 : {0,1}) لكل (صحيح s1 : {0,1}) لكل (صحيح d : {0,1}) \
        ggml_vk_create_pipeline2(device, device->pipeline_ ## name ## namemod[s0][s1][d], \
                                #name + get_suffix(s0, s1, d) + #namemod, name ## _len[s0][s1][d], name ## انکڑے_2[s0][s1][d], \
                                "main", (bindings), حجم(vk_op_ثنائی_ڈالو_constants), {512, 1, 1}, spec, 1);

    CREATE_ثنائی(جوڑو_2, , {0}, 4)
    CREATE_ثنائی(جوڑو_2, _norepeat, {1}, 4)
    CREATE_ثنائی(sub, , {0}, 3)
    CREATE_ثنائی(sub, _norepeat, {1}, 3)
    CREATE_ثنائی(mul, , {0}, 3)
    CREATE_ثنائی(mul, _norepeat, {1}, 3)
    CREATE_ثنائی(div, , {0}, 3)
    CREATE_ثنائی(div, _norepeat, {1}, 3)
    CREATE_ثنائی(جوڑو_rms, , {0}, 4)
    CREATE_ثنائی(جوڑو_rms, _norepeat, {1}, 4)
#تعریف_رد CREATE_ثنائی

    إذا (device->multi_جوڑو) {
        لكل (uint32_t i = 0; i < MAX_FUSED_ADDS; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_multi_جوڑو[i],     "multi_add_f32_"     + std::to_string(i+1), multi_جوڑو_f32_len,     multi_جوڑو_f32_انکڑے,     "main", MAX_PARAMETER_گنتی, حجم(vk_op_multi_جوڑو_ڈالو_constants), {512, 1, 1}, {i+2}, 1);
            ggml_vk_create_pipeline2(device, device->pipeline_multi_جوڑو_rms[i], "multi_add_rms_f32_" + std::to_string(i+1), multi_جوڑو_rms_f32_len, multi_جوڑو_rms_f32_انکڑے, "main", MAX_PARAMETER_گنتی, حجم(vk_op_multi_جوڑو_ڈالو_constants), {512, 1, 1}, {i+2}, 1);
        }
    }

    ggml_vk_create_pipeline(device, device->pipeline_جوڑو_id_f32, "add_id_f32", جوڑو_id_f32_len, جوڑو_id_f32_انکڑے, "main", 4, حجم(vk_op_جوڑو_id_ڈالو_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_acc_f32, "acc_f32", acc_f32_len, acc_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {512, 1, 1}, {0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_سیٹ_f32, "set_f32", acc_f32_len, acc_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {512, 1, 1}, {0, 0}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_concat_i8, "concat_i8", concat_i8_len, concat_i8_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i16, "concat_i16", concat_i16_len, concat_i16_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i32, "concat_i32", concat_i32_len, concat_i32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i64, "concat_i64", concat_i64_len, concat_i64_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_upscale_سبھ_توں_نیڑے_f32, "upscale_f32", upscale_f32_len, upscale_f32_انکڑے, "main", 2, حجم(vk_op_upscale_ڈالو_constants), {512, 1, 1}, {GGML_SCALE_منوال_سبھ_توں_نیڑے}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_f32, "upscale_f32", upscale_f32_len, upscale_f32_انکڑے, "main", 2, حجم(vk_op_upscale_ڈالو_constants), {512, 1, 1}, {GGML_SCALE_منوال_BILINEAR}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bicubic_f32, "upscale_f32", upscale_f32_len, upscale_f32_انکڑے, "main", 2, حجم(vk_op_upscale_ڈالو_constants), {512, 1, 1}, {GGML_SCALE_منوال_BICUBIC}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_antialias_f32, "upscale_f32", upscale_f32_len, upscale_f32_انکڑے, "main", 2, حجم(vk_op_upscale_ڈالو_constants), {512, 1, 1}, {GGML_SCALE_منوال_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_scale_f32, "scale_f32", scale_f32_len, scale_f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_log[0], "log_f32", log_f32_len, log_f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_log[1], "log_f16", log_f16_len, log_f16_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_tri[0], "tri_f32", tri_f32_len, tri_f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_tri[1], "tri_f16", tri_f16_len, tri_f16_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag[0], "diag_f32", diag_f32_len, diag_f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_diag[1], "diag_f16", diag_f16_len, diag_f16_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pad_f32, "pad_f32", pad_f32_len, pad_f32_انکڑے, "main", 2, حجم(vk_op_pad_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_pad_reflect_1d_f32, "pad_reflect_1d_f32", pad_reflect_1d_f32_len, pad_reflect_1d_f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_roll_f32, "roll_f32", roll_f32_len, roll_f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_repeat_i32, "repeat_i32", repeat_i32_len, repeat_i32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_repeat_پٹھ_f32, "repeat_back_f32", repeat_پٹھ_f32_len, repeat_پٹھ_f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_repeat_i16, "repeat_i16", repeat_i16_len, repeat_i16_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);

#تعریف_کرو CREATE_UNARY(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## f32_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## f16_انکڑے, "main", 2, حجم(vk_op_unary_ڈالو_constants), {512, 1, 1}, {}, 1);

    CREATE_UNARY(elu)
    CREATE_UNARY(gelu)
    CREATE_UNARY(gelu_erf)
    CREATE_UNARY(gelu_تیز)
    CREATE_UNARY(silu)
    CREATE_UNARY(relu)
    CREATE_UNARY(sqr)
    CREATE_UNARY(جذر)
    CREATE_UNARY(جیب)
    CREATE_UNARY(جیب_تمام)
    CREATE_UNARY(clamp)
    CREATE_UNARY(leaky_relu)
    CREATE_UNARY(xielu)
    CREATE_UNARY(neg)
    CREATE_UNARY(tanh)
    CREATE_UNARY(sigmoid)
    CREATE_UNARY(hardsigmoid)
    CREATE_UNARY(hardswish)
    CREATE_UNARY(مطلق_قدر)
    CREATE_UNARY(softplus)
    CREATE_UNARY(قدم)
    CREATE_UNARY(round)
    CREATE_UNARY(ceil)
    CREATE_UNARY(floor)
    CREATE_UNARY(trunc)
    CREATE_UNARY(sgn)
    CREATE_UNARY(قوت_نما)
    CREATE_UNARY(expm1)
#تعریف_رد CREATE_UNARY

// spec constants: {norepeat, op_on_b}
#تعریف_کرو CREATE_UNARY_MUL(name, idx) \
    لكل (صحيح dt = 0; dt < 2; ++dt) { \
        ثابت size_t len_ = dt ? name ## _mul_f16_len : name ## _mul_f32_len; \
        ثابت بنانشانی حرف * انکڑے_3 = dt ? name ## mul_f16_انکڑے : name ## mul_f32_انکڑے; \
        ثابت std::string dts_ = dt ? "f16" : "f32"; \
        لكل (صحيح ob = 0; ob < 2; ++ob) \
            لكل (صحيح nr = 0; nr < 2; ++nr) \
                ggml_vk_create_pipeline(device, device->pipeline_unary_mul[(idx)][dt][nr][ob], \
                    (#name "_mul" + std::string(ob ? "_b" : "") + "_" + dts_ + (nr ? "_norepeat" : "")).c_str(), \
                    len_, انکڑے_3, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {512, 1, 1}, \
                    { (uint32_t) nr, (uint32_t) ob }, 1); \
    }

    CREATE_UNARY_MUL(gelu, 0)
    CREATE_UNARY_MUL(sigmoid, 1)
    CREATE_UNARY_MUL(silu, 2)
    CREATE_UNARY_MUL(softplus, 3)
#تعریف_رد CREATE_UNARY_MUL

    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f16, "add1_f16_f16", add1_f16_f16_len, add1_f16_f16_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f32, "add1_f16_f32", add1_f16_f32_len, add1_f16_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f32_f32, "add1_f32_f32", add1_f32_f32_len, add1_f32_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_arange_f32, "arange_f32", arange_f32_len, arange_f32_انکڑے, "main", 1, حجم(vk_op_ڈالو_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_fill_f32, "fill_f32", fill_f32_len, fill_f32_انکڑے, "main", 1, حجم(vk_op_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_fill_f16, "fill_f16", fill_f16_len, fill_f16_انکڑے, "main", 1, حجم(vk_op_ڈالو_constants), {512, 1, 1}, {}, 1);

#تعریف_کرو CREATE_GLU(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## f32_انکڑے, "main", 3, حجم(vk_op_glu_ڈالو_constants), {512, 1, 1}, {}, 1, سچ);   \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## f16_انکڑے, "main", 3, حجم(vk_op_glu_ڈالو_constants), {512, 1, 1}, {}, 1, سچ);

    CREATE_GLU(geglu)
    CREATE_GLU(reglu)
    CREATE_GLU(swiglu)
    CREATE_GLU(swiglu_oai)
    CREATE_GLU(swiglu_clamp)
    CREATE_GLU(geglu_erf)
    CREATE_GLU(geglu_تیز)
#تعریف_رد CREATE_GLU

    ggml_vk_create_pipeline(device, device->pipeline_silu_پٹھ_f32, "silu_back_f32", silu_پٹھ_f32_len, silu_پٹھ_f32_انکڑے, "main", 3, حجم(vk_op_ڈالو_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag_نقاب_inf_f32, "diag_mask_inf_f32", diag_نقاب_inf_f32_len, diag_نقاب_inf_f32_انکڑے, "main", 2, حجم(vk_op_diag_نقاب_ڈالو_constants), {1, 512, 1}, {}, 1, سچ);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32, "soft_max_f32", soft_max_f32_len, soft_max_f32_انکڑے, "main", 4, حجم(vk_op_soft_max_ڈالو_constants), {1, 1, 1}, { device->subgroup_ناپ }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_wg512, "soft_max_f32_wg512", soft_max_f32_len, soft_max_f32_انکڑے, "main", 4, حجم(vk_op_soft_max_ڈالو_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16, "soft_max_f32_f16", soft_max_f32_f16_len, soft_max_f32_f16_انکڑے, "main", 4, حجم(vk_op_soft_max_ڈالو_constants), {1, 1, 1}, { device->subgroup_ناپ }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16_wg512, "soft_max_f32_f16_wg512", soft_max_f32_f16_len, soft_max_f32_f16_انکڑے, "main", 4, حجم(vk_op_soft_max_ڈالو_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_پٹھ_f32, "soft_max_back_f32", soft_max_پٹھ_f32_len, soft_max_پٹھ_f32_انکڑے, "main", 3, حجم(vk_op_ڈالو_constants), {1, 1, 1}, { device->subgroup_ناپ }, 1, سچ);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32,     "soft_max_large1_f32",     soft_max_large1_f32_len,     soft_max_large1_f32_انکڑے,     "main", 6, حجم(vk_op_soft_max_ڈالو_constants), {1, 1, 1}, { 128, 4 }, 1, سچ);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32,     "soft_max_large2_f32",     soft_max_large2_f32_len,     soft_max_large2_f32_انکڑے,     "main", 6, حجم(vk_op_soft_max_ڈالو_constants), {1, 1, 1}, { 128, 4 }, 1, سچ);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32,     "soft_max_large3_f32",     soft_max_large3_f32_len,     soft_max_large3_f32_انکڑے,     "main", 6, حجم(vk_op_soft_max_ڈالو_constants), {1, 1, 1}, { 128, 4 }, 1, سچ);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32_f16, "soft_max_large1_f32_f16", soft_max_large1_f32_f16_len, soft_max_large1_f32_f16_انکڑے, "main", 6, حجم(vk_op_soft_max_ڈالو_constants), {1, 1, 1}, { 128, 4 }, 1, سچ);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32_f16, "soft_max_large2_f32_f16", soft_max_large2_f32_f16_len, soft_max_large2_f32_f16_انکڑے, "main", 6, حجم(vk_op_soft_max_ڈالو_constants), {1, 1, 1}, { 128, 4 }, 1, سچ);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32_f16, "soft_max_large3_f32_f16", soft_max_large3_f32_f16_len, soft_max_large3_f32_f16_انکڑے, "main", 6, حجم(vk_op_soft_max_ڈالو_constants), {1, 1, 1}, { 128, 4 }, 1, سچ);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32, "rope_norm_f32", rope_norm_f32_len, rope_norm_f32_انکڑے, "main", 5, حجم(vk_op_rope_ڈالو_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32, "rope_neox_f32", rope_neox_f32_len, rope_neox_f32_انکڑے, "main", 5, حجم(vk_op_rope_ڈالو_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32, "rope_multi_f32", rope_multi_f32_len, rope_multi_f32_انکڑے, "main", 5, حجم(vk_op_rope_ڈالو_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f32, "rope_vision_f32", rope_vision_f32_len, rope_vision_f32_انکڑے, "main", 5, حجم(vk_op_rope_ڈالو_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f16, "rope_norm_f16", rope_norm_f16_len, rope_norm_f16_انکڑے, "main", 5, حجم(vk_op_rope_ڈالو_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f16, "rope_neox_f16", rope_neox_f16_len, rope_neox_f16_انکڑے, "main", 5, حجم(vk_op_rope_ڈالو_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f16, "rope_multi_f16", rope_multi_f16_len, rope_multi_f16_انکڑے, "main", 5, حجم(vk_op_rope_ڈالو_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f16, "rope_vision_f16", rope_vision_f16_len, rope_vision_f16_انکڑے, "main", 5, حجم(vk_op_rope_ڈالو_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32_f16, "rope_norm_f32_f16", rope_norm_f32_f16_len, rope_norm_f32_f16_انکڑے, "main", 5, حجم(vk_op_rope_ڈالو_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32_f16, "rope_neox_f32_f16", rope_neox_f32_f16_len, rope_neox_f32_f16_انکڑے, "main", 5, حجم(vk_op_rope_ڈالو_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32_f16, "rope_multi_f32_f16", rope_multi_f32_f16_len, rope_multi_f32_f16_انکڑے, "main", 5, حجم(vk_op_rope_ڈالو_constants), {1, 512, 1}, {}, 1);

    لكل (uint32_t i = 0; i < عدد_argsort_pipelines; ++i) {
        uint32_t بلاک_ناپ_2 = 1u << std::min(i, device->max_workgroup_ناپ_log2);
        إذا (i <= device->max_workgroup_ناپ_log2 &&
            2 * حجم(صحيح) * بلاک_ناپ_2 <= device->properties.limits.max_Compute_Shared_حافظہ_ناپ) {
            ثابت uint32_t NCOLS_PADDED_LOG2 = i;
            ggml_vk_create_pipeline2(device, device->pipeline_argsort_f32[i], "argsort_f32_"+std::to_string(i), argsort_f32_len, argsort_f32_انکڑے, "main", 3, حجم(vk_op_argsort_ڈالو_constants), {بلاک_ناپ_2, 1, 1}, {بلاک_ناپ_2, NCOLS_PADDED_LOG2}, 1, سچ);
        }
        ثابت uint32_t WG_UNROLL_ضارب = بلاک_ناپ_2 > 1 ? 2 : 1;
        بلاک_ناپ_2 /= WG_UNROLL_ضارب;
        ggml_vk_create_pipeline2(device, device->pipeline_argsort_large_f32[i], "argsort_large_f32_"+std::to_string(i), argsort_large_f32_len, argsort_large_f32_انکڑے, "main", 3, حجم(vk_op_argsort_ڈالو_constants), {بلاک_ناپ_2 * WG_UNROLL_ضارب, 1, 1}, {بلاک_ناپ_2, WG_UNROLL_ضارب}, 1, سچ);
    }

    لكل (uint32_t i = 0; i < عدد_topk_pipelines; ++i) {
        ثابت uint32_t بلاک_ناپ_2 = 1u << i;
        ثابت uint32_t NCOLS_PADDED_LOG2 = i;
        إذا (i <= device->max_workgroup_ناپ_log2) {
            uint32_t nary_shmem = 2 * حجم(صحيح) * بلاک_ناپ_2 +
                                  حجم(صحيح) * device->subgroup_ناپ +
                                  2 * حجم(صحيح) +
                                  2 * (بلاک_ناپ_2 / device->subgroup_ناپ) * حجم(صحيح);
            إذا (device->subgroup_arithmetic && device->subgroup_require_بھریا_support && device->subgroup_shuffle && device->subgroup_ballot &&
                nary_shmem <= device->properties.limits.max_Compute_Shared_حافظہ_ناپ) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_nary_تلاش_f32_len, topk_nary_تلاش_f32_انکڑے, "main", 2, حجم(vk_op_topk_ڈالو_constants), {بلاک_ناپ_2, 1, 1}, {بلاک_ناپ_2, device->subgroup_ناپ, device->subgroup_ناپ_log2}, 1, سچ, سچ, device->subgroup_ناپ);
            } وإلا إذا (2 * حجم(صحيح) * بلاک_ناپ_2 <= device->properties.limits.max_Compute_Shared_حافظہ_ناپ) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_argsort_f32_len, topk_argsort_f32_انکڑے, "main", 2, حجم(vk_op_topk_ڈالو_constants), {بلاک_ناپ_2, 1, 1}, {بلاک_ناپ_2, NCOLS_PADDED_LOG2}, 1, سچ);
            }
        }
    }

    // large-k fallback: one workgroup per row, radix-select instead of a full sort. The QSA
    // variant (spec constant 1) additionally gathers the qwen4 indexer input on the fly.
    {
        ثابت uint32_t بلاک_ناپ_2 = 1u << std::min(10u, device->max_workgroup_ناپ_log2);
        ggml_vk_create_pipeline2(device, device->pipeline_topk_radix_f32, "topk_radix_f32", topk_radix_select_f32_len, topk_radix_select_f32_انکڑے, "main", 5, حجم(vk_op_topk_radix_ڈالو_constants), {بلاک_ناپ_2, 1, 1}, {بلاک_ناپ_2, 0}, 1, سچ);
        ggml_vk_create_pipeline2(device, device->pipeline_topk_radix_qsa, "topk_radix_qsa", topk_radix_select_f32_len, topk_radix_select_f32_انکڑے, "main", 5, حجم(vk_op_topk_radix_ڈالو_constants), {بلاک_ناپ_2, 1, 1}, {بلاک_ناپ_2, 1}, 1, سچ);
    }

    ggml_vk_create_pipeline(device, device->pipeline_argmax_f32, "argmax_f32", argmax_f32_len, argmax_f32_انکڑے, "main", 2, حجم(vk_op_ڈالو_constants), {1, 1, 1}, { device->subgroup_ناپ }, 1);

    ggml_vk_create_pipeline(device, device->pipeline_جوڑ_صفاں_f32, "sum_rows_f32", جوڑ_صفاں_f32_len, جوڑ_صفاں_f32_انکڑے, "main", 2, حجم(vk_op_جوڑ_صفاں_ڈالو_constants), {1, 1, 1}, { device->subgroup_ناپ }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_f32, "cross_entropy_loss_f32", cross_entropy_loss_f32_len, cross_entropy_loss_f32_انکڑے, "main", 3, حجم(vk_op_ڈالو_constants), {1, 1, 1}, { device->subgroup_ناپ }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_f32_wg512, "cross_entropy_loss_f32_wg512", cross_entropy_loss_f32_len, cross_entropy_loss_f32_انکڑے, "main", 3, حجم(vk_op_ڈالو_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_پٹھ_f32, "cross_entropy_loss_back_f32", cross_entropy_loss_پٹھ_f32_len, cross_entropy_loss_پٹھ_f32_انکڑے, "main", 4, حجم(vk_op_ڈالو_constants), {1, 1, 1}, { device->subgroup_ناپ }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_پٹھ_f32_wg512, "cross_entropy_loss_back_f32_wg512", cross_entropy_loss_پٹھ_f32_len, cross_entropy_loss_پٹھ_f32_انکڑے, "main", 4, حجم(vk_op_ڈالو_constants), {1, 1, 1}, { 512 }, 1);
    // Intel Windows driver in range [32.0.101.8509, 32.0.101.8860) will crash when using fwht kernels so we gate that here
    ثابت منطقي can_use_fwht = device->driver_id != vk::DriverId::eIntelProprietaryWindows ||
        !ggml_vk_intel_windows_driver_وچ_range(device->properties.driver_نسخہ_2, 101, 8509, 101, 8860);
    إذا (can_use_fwht && device->subgroup_basic && device->subgroup_shuffle) {
        صحيح idx = 0;
        لكل (uint32_t n : {64, 128, 256, 512}) {
            إذا (device->subgroup_ناپ <= n) {
                ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_f32", fwht_f32_len, fwht_f32_انکڑے, "main", 2, حجم(vk_op_fwht_ڈالو_constants), {1, 1, 1}, { device->subgroup_ناپ, n }, 1, سچ, سچ, device->subgroup_ناپ);
            }
            ++idx;
        }
    } وإلا إذا (can_use_fwht) {
        صحيح idx = 0;
        لكل (uint32_t n : {64, 128, 256, 512}) {
            ثابت uint32_t بلاک_ناپ_3 = std::min(device->subgroup_ناپ, n);
            ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_shmem_f32", fwht_shmem_f32_len, fwht_shmem_f32_انکڑے, "main", 2, حجم(vk_op_fwht_ڈالو_constants), {1, 1, 1}, { بلاک_ناپ_3, n }, 1);
            ++idx;
        }
    }

    ثابت uint32_t cumsum_elem_فی_thread = (device->vendor_id == VK_VENDOR_ID_AMD || device->vendor_id == VK_VENDOR_ID_INTEL) ? 2 : 4;
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_f32,       "cumsum_f32", cumsum_f32_len, cumsum_f32_انکڑے, "main", 2, حجم(vk_op_جوڑ_صفاں_ڈالو_constants), {1, 1, 1}, { 256, device->subgroup_ناپ, cumsum_elem_فی_thread }, 1, سچ, سچ, device->subgroup_ناپ);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_small_f32, "cumsum_f32", cumsum_f32_len, cumsum_f32_انکڑے, "main", 2, حجم(vk_op_جوڑ_صفاں_ڈالو_constants), {1, 1, 1}, { 128, device->subgroup_ناپ, 1 }, 1, سچ, سچ, device->subgroup_ناپ);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass1_f32, "cumsum_multipass1_f32", cumsum_multipass1_f32_len, cumsum_multipass1_f32_انکڑے, "main", 3, حجم(vk_op_جوڑ_صفاں_ڈالو_constants), {256, 1, 1}, { 256, device->subgroup_ناپ }, 1, سچ, سچ, device->subgroup_ناپ);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass2_f32, "cumsum_multipass2_f32", cumsum_multipass2_f32_len, cumsum_multipass2_f32_انکڑے, "main", 3, حجم(vk_op_جوڑ_صفاں_ڈالو_constants), {256, 1, 1}, { 256, device->subgroup_ناپ }, 1, سچ, سچ, device->subgroup_ناپ);

    ggml_vk_create_pipeline(device, device->pipeline_گنتی_equal_i32, "count_equal_i32", گنتی_equal_i32_len, گنتی_equal_i32_انکڑے, "main", 3, حجم(vk_op_ڈالو_constants), {512, 1, 1}, { device->subgroup_ناپ }, 1);

    إذا (device->subgroup_arithmetic && device->subgroup_vote && device->subgroup_require_بھریا_support) {
        ggml_vk_create_pipeline(device, device->pipeline_گنتی_experts, "count_experts", گنتی_experts_subgroup_len, گنتی_experts_subgroup_انکڑے, "main", 2, حجم(vk_op_گنتی_experts_ڈالو_constants), {1, 1, 1}, {}, 1, سچ, سچ);
    } وإلا {
        ggml_vk_create_pipeline(device, device->pipeline_گنتی_experts, "count_experts", گنتی_experts_len, گنتی_experts_انکڑے, "main", 2, حجم(vk_op_گنتی_experts_ڈالو_constants), {1, 1, 1}, {}, 1, سچ);
    }

    // comb holds a token's 4x4 matrix in one 16-lane slice of a subgroup, so it
    // needs at least 16 lanes, pinned to a known size.
    إذا (device->subgroup_basic && device->subgroup_shuffle && device->subgroup_require_بھریا_support && device->subgroup_ناپ >= 16) {
        ثابت uint32_t ٹوکن_فی_workgroup = 4 * (device->subgroup_ناپ / 16);
        ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_comb_f32, "dsv4_hc_comb_f32", dsv4_hc_comb_f32_len, dsv4_hc_comb_f32_انکڑے, "main", 4, حجم(vk_op_dsv4_hc_comb_ڈالو_constants), {ٹوکن_فی_workgroup, 1, 1}, { device->subgroup_ناپ }, 1, سچ, سچ, device->subgroup_ناپ);
    }

    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_pre_f32,        "dsv4_hc_pre_f32",        dsv4_hc_pre_f32_len,  dsv4_hc_pre_f32_انکڑے,  "main", 3, حجم(vk_op_dsv4_hc_pre_ڈالو_constants),  {256, 1, 1}, { 256, 0 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_pre_gated_f32,  "dsv4_hc_pre_gated_f32",  dsv4_hc_pre_f32_len,  dsv4_hc_pre_f32_انکڑے,  "main", 3, حجم(vk_op_dsv4_hc_pre_ڈالو_constants),  {256, 1, 1}, { 256, 1 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_post_f32,       "dsv4_hc_post_f32",       dsv4_hc_post_f32_len, dsv4_hc_post_f32_انکڑے, "main", 5, حجم(vk_op_dsv4_hc_post_ڈالو_constants), {256, 1, 1}, { 256, 1 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_post_nocomb_f32,"dsv4_hc_post_nocomb_f32",dsv4_hc_post_f32_len, dsv4_hc_post_f32_انکڑے, "main", 5, حجم(vk_op_dsv4_hc_post_ڈالو_constants), {256, 1, 1}, { 256, 0 }, 1);

    لكل (خودکار &s : device->pipeline_solve_tri_f32) {
        ثابت vk_solve_tri_pipeline_state &state = s.پہلا;

        // Max number of rows to load at a time, limited by shared memory
        ثابت uint32_t batch_N = device->properties.limits.max_Compute_Shared_حافظہ_ناپ / ((state.N + state.K) * حجم(عائم));
        // Need at least K invocations, and prefer a minimum of 128 to spread out loading shared memory
        ثابت uint32_t بلاک_ناپ_3 = std::max(128u, 1u << (uint32_t)ceilf(log2f(عائم(state.K))));

        ggml_vk_create_pipeline(
            device, s.second, "solve_tri_f32",
            solve_tri_f32_len, solve_tri_f32_انکڑے, "main", 3,
            حجم(vk_op_ثنائی_ڈالو_constants), {1, 1, 1}, { 0, state.N, state.K, batch_N, بلاک_ناپ_3 }, 1, سچ);
    }

#تعریف_کرو IM2_خانہ(bda) \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32, "im2col_f32", im2col_f32 ## bda ## _len, im2col_f32 ## bda ## انکڑے_2, "main", 2, حجم(vk_op_im2col_ڈالو_constants), {512, 1, 1}, { device->subgroup_ناپ }, 1, سچ);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32, "im2col_3d_f32", im2col_3d_f32 ## bda ## _len, im2col_3d_f32 ## bda ## انکڑے_2, "main", 2, حجم(vk_op_im2col_3d_ڈالو_constants), {512, 1, 1}, { 512 }, 1, سچ);      \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32_f16, "im2col_f32_f16", im2col_f32_f16 ## bda ## _len, im2col_f32_f16 ## bda ## انکڑے_2, "main", 2, حجم(vk_op_im2col_ڈالو_constants), {512, 1, 1}, { device->subgroup_ناپ }, 1, سچ);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32_f16, "im2col_3d_f32_f16", im2col_3d_f32_f16 ## bda ## _len, im2col_3d_f32_f16 ## bda ## انکڑے_2, "main", 2, حجم(vk_op_im2col_3d_ڈالو_constants), {512, 1, 1}, { 512 }, 1, سچ);
    إذا (device->shader_int64 && device->بفر_device_پتہ) {
        IM2_خانہ(_bda)
    } وإلا {
        IM2_خانہ()
    }

    ggml_vk_create_pipeline(device, device->pipeline_timestep_embedding_f32, "timestep_embedding_f32", timestep_embedding_f32_len, timestep_embedding_f32_انکڑے, "main", 2, حجم(vk_op_timestep_embedding_ڈالو_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_conv_transpose_1d_f32, "conv_transpose_1d_f32", conv_transpose_1d_f32_len, conv_transpose_1d_f32_انکڑے, "main", 3, حجم(vk_op_conv_transpose_1d_ڈالو_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_f32,  "col2im_1d_f32",  col2im_1d_f32_len,  col2im_1d_f32_انکڑے,  "main", 2, حجم(vk_op_col2im_1d_ڈالو_constants), {256, 1, 1}, {}, 1, سچ);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_f16,  "col2im_1d_f16",  col2im_1d_f16_len,  col2im_1d_f16_انکڑے,  "main", 2, حجم(vk_op_col2im_1d_ڈالو_constants), {256, 1, 1}, {}, 1, سچ);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_bf16, "col2im_1d_bf16", col2im_1d_bf16_len, col2im_1d_bf16_انکڑے, "main", 2, حجم(vk_op_col2im_1d_ڈالو_constants), {256, 1, 1}, {}, 1, سچ);

    ggml_vk_create_pipeline(device, device->pipeline_باہرلا_prod_f32, "out_prod_f32", باہرلا_prod_f32_len, باہرلا_prod_f32_انکڑے, "main", 3, حجم(vk_op_ثنائی_ڈالو_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_snake_f32,  "snake_f32",  snake_f32_len,  snake_f32_انکڑے,  "main", 4, حجم(vk_op_snake_ڈالو_constants), {256, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_snake_f16,  "snake_f16",  snake_f16_len,  snake_f16_انکڑے,  "main", 4, حجم(vk_op_snake_ڈالو_constants), {256, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_snake_bf16, "snake_bf16", snake_bf16_len, snake_bf16_انکڑے, "main", 4, حجم(vk_op_snake_ڈالو_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pool1d_f32, "pool1d_f32", pool1d_f32_len, pool1d_f32_انکڑے, "main", 2, حجم(vk_op_pool1d_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_pool2d_f32, "pool2d_f32", pool2d_f32_len, pool2d_f32_انکڑے, "main", 2, حجم(vk_op_pool2d_ڈالو_constants), {1, 1, 1}, {512}, 1);
    إذا (device->subgroup_arithmetic && device->subgroup_require_بھریا_support) {
        ggml_vk_create_pipeline(device, device->pipeline_pool2d_subgroup_f32, "pool2d_subgroup_f32", pool2d_subgroup_f32_len, pool2d_subgroup_f32_انکڑے, "main", 2, حجم(vk_op_pool2d_ڈالو_constants), {1, 1, 1}, {128}, 1, خطأ, سچ, device->subgroup_ناپ);
    }

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv6_f32, "rwkv_wkv6_f32", rwkv_wkv6_f32_len, rwkv_wkv6_f32_انکڑے, "main", 7, حجم(vk_op_rwkv_wkv6_ڈالو_constants), {1, 1, 1}, {device->subgroup_ناپ}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv7_f32, "rwkv_wkv7_f32", rwkv_wkv7_f32_len, rwkv_wkv7_f32_انکڑے, "main", 8, حجم(vk_op_rwkv_wkv7_ڈالو_constants), {1, 1, 1}, {device->subgroup_ناپ}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_gated_خطی_attn_f32, "gated_linear_attn_f32", gated_خطی_attn_f32_len, gated_خطی_attn_f32_انکڑے, "main", 6, حجم(vk_op_gated_خطی_attn_ڈالو_constants), {1, 1, 1}, {}, 1);

    لكل (ggml_type k_type : lightning_indexer_k_types) {
        ثابت std::string name = "lightning_indexer_" + std::string(ggml_type_name(k_type)) + "_k_f32";
        ggml_vk_create_pipeline(device, device->pipeline_lightning_indexer_f32[k_type], name.c_str(), lightning_indexer_f32_len, lightning_indexer_f32_انکڑے, "main", 5, حجم(vk_op_lightning_indexer_ڈالو_constants), {1, 1, 1}, {(uint32_t)k_type, fa_بلاک_بائٹاں(k_type)}, 1, سچ);
    }

    {
        ثابت uint32_t gdn_sizes[] = {16, 32, 64, 128};
        ثابت حرف * gdn_ناں[][2] = {
            {"gated_delta_net_f32_d16",     "gated_delta_net_f32_d16_kda"},
            {"gated_delta_net_f32_d32",     "gated_delta_net_f32_d32_kda"},
            {"gated_delta_net_f32_d64",     "gated_delta_net_f32_d64_kda"},
            {"gated_delta_net_f32_d128",    "gated_delta_net_f32_d128_kda"},
        };
        لكل (uint32_t si = 0; si < 4; si++) {
            ثابت uint32_t S_V = gdn_sizes[si];
            GGML_ASSERT(is_pow2(S_V));

            // Intel Xe regresses at SIMD32 for this scan; prefer a narrower subgroup.
            uint32_t gdn_subgroup_ناپ = device->subgroup_ناپ;
            إذا (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_ناپ_control &&
                device->subgroup_min_ناپ <= 16u && device->subgroup_max_ناپ >= 16u) {
                gdn_subgroup_ناپ = 16u;
            }
            uint32_t lanes_فی_کالم;
            إذا (device->vendor_id == VK_VENDOR_ID_INTEL) {
                // Intel Xe: full-width reduction (min rows/lane) ~10x over the COLS_PER_WG=8 rule.
                lanes_فی_کالم = std::min(gdn_subgroup_ناپ, S_V);
            } وإلا إذا (S_V >= 128u && device->subgroup_clustered) {
                // COLS_PER_WG=8: measured optimum (Ampere sg32->4, Vega20 sg64->8).
                lanes_فی_کالم = std::max(1u, gdn_subgroup_ناپ / 8u);
            } وإلا {
                // Use largest power-of-two that divides both S_V and subgroup_size so that
                // (1) S_V % lanes_per_column == 0 and (2) S_V % (subgroup_size / lanes_per_column) == 0.
                // This means we don't need extra bounds checking logic in the shader.
                lanes_فی_کالم = std::min(S_V, gdn_subgroup_ناپ);
            }

            // gated_delta_net.comp relies on S_V % COLS_PER_WG == 0 and
            // S_V % LANES_PER_COLUMN == 0 to avoid bounds checks.
            طالما (lanes_فی_کالم > 1u) {
                ثابت منطقي valid_lanes = (gdn_subgroup_ناپ % lanes_فی_کالم) == 0 &&
                                         (S_V % lanes_فی_کالم) == 0;
                ثابت uint32_t خانے_گنتی_فی_wg = valid_lanes ? gdn_subgroup_ناپ / lanes_فی_کالم : 0;
                إذا (valid_lanes && خانے_گنتی_فی_wg > 0 && (S_V % خانے_گنتی_فی_wg) == 0) {
                    اكسر;
                }
                lanes_فی_کالم >>= 1u;
            }

            GGML_ASSERT((gdn_subgroup_ناپ % lanes_فی_کالم) == 0);
            GGML_ASSERT((S_V % lanes_فی_کالم) == 0);
            GGML_ASSERT((S_V % (gdn_subgroup_ناپ / lanes_فی_کالم)) == 0);

            ثابت منطقي need_partial_subgroup_reduce = lanes_فی_کالم != 1u && lanes_فی_کالم < gdn_subgroup_ناپ;
            ثابت منطقي use_clustered_reduce = device->subgroup_arithmetic && device->subgroup_clustered && need_partial_subgroup_reduce;
            ثابت منطقي use_subgroup_reduce = device->subgroup_arithmetic && !need_partial_subgroup_reduce;
            ثابت منطقي use_subgroup_ops = use_clustered_reduce || use_subgroup_reduce;
            size_t gdn_len;
            ثابت فراغ * gdn_انکڑے;
            إذا (use_clustered_reduce) {
                gdn_len = gated_delta_net_f32_len;
                gdn_انکڑے = (ثابت فراغ *)gated_delta_net_f32_انکڑے;
            } وإلا إذا (use_subgroup_reduce) {
                gdn_len = gated_delta_net_f32_nocluster_len;
                gdn_انکڑے = (ثابت فراغ *)gated_delta_net_f32_nocluster_انکڑے;
            } وإلا {
                gdn_len = gated_delta_net_f32_shmem_len;
                gdn_انکڑے = (ثابت فراغ *)gated_delta_net_f32_shmem_انکڑے;
            }

            ثابت uint32_t خانے_گنتی_فی_wg = gdn_subgroup_ناپ / lanes_فی_کالم;
            ثابت std::array<uint32_t, 3> wg_denoms = {1u, 1u, خانے_گنتی_فی_wg};

            لكل (uint32_t kda = 0; kda < 2; kda++) {
                ggml_vk_create_pipeline(device, device->pipeline_gated_delta_net[si][kda],
                    gdn_ناں[si][kda], gdn_len, gdn_انکڑے, "main", 7, حجم(vk_op_gated_delta_net_ڈالو_constants),
                    wg_denoms, {S_V, kda, gdn_subgroup_ناپ, lanes_فی_کالم}, 1, سچ, use_subgroup_ops, gdn_subgroup_ناپ);
            }
        }
    }

    إذا (device->subgroup_arithmetic && device->subgroup_require_بھریا_support) {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_جانچ_f32_d128, "ssm_scan_128_f32", ssm_جانچ_subgroup_f32_len, ssm_جانچ_subgroup_f32_انکڑے, "main", 8, حجم(vk_op_ssm_جانچ_ڈالو_constants), {1, 1, 1}, {128, device->subgroup_ناپ}, 1, سچ, سچ);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_جانچ_f32_d256, "ssm_scan_256_f32", ssm_جانچ_subgroup_f32_len, ssm_جانچ_subgroup_f32_انکڑے, "main", 8, حجم(vk_op_ssm_جانچ_ڈالو_constants), {1, 1, 1}, {256, device->subgroup_ناپ}, 1, سچ, سچ);
    } وإلا {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_جانچ_f32_d128, "ssm_scan_128_f32", ssm_جانچ_f32_len, ssm_جانچ_f32_انکڑے, "main", 8, حجم(vk_op_ssm_جانچ_ڈالو_constants), {1, 1, 1}, {128, device->subgroup_ناپ, 16}, 1, سچ, سچ);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_جانچ_f32_d256, "ssm_scan_256_f32", ssm_جانچ_f32_len, ssm_جانچ_f32_انکڑے, "main", 8, حجم(vk_op_ssm_جانچ_ڈالو_constants), {1, 1, 1}, {256, device->subgroup_ناپ, 16}, 1, سچ, سچ);
    }

    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_f32,           "ssm_conv_f32",           ssm_conv_f32_len, ssm_conv_f32_انکڑے, "main", 4, حجم(vk_op_ssm_conv_ڈالو_constants), {32, 16, 1}, {32, 16, 0, 0}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_silu_f32,      "ssm_conv_silu_f32",      ssm_conv_f32_len, ssm_conv_f32_انکڑے, "main", 4, حجم(vk_op_ssm_conv_ڈالو_constants), {32, 16, 1}, {32, 16, 0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_bias_silu_f32, "ssm_conv_bias_silu_f32", ssm_conv_f32_len, ssm_conv_f32_انکڑے, "main", 4, حجم(vk_op_ssm_conv_ڈالو_constants), {32, 16, 1}, {32, 16, 1, 1}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_قدم_adamw_f32, "opt_step_adamw_f32", opt_قدم_adamw_f32_len, opt_قدم_adamw_f32_انکڑے, "main", 5, حجم(vk_op_ڈالو_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_قدم_sgd_f32, "opt_step_sgd_f32", opt_قدم_sgd_f32_len, opt_قدم_sgd_f32_انکڑے, "main", 3, حجم(vk_op_ڈالو_constants), {512, 1, 1}, {}, 1);

    // conv2d, conv_transpose_2d, conv3d
    لكل (uint32_t s = 0; s < CONV_SHAPE_گنتی; ++s) {
        // smaller WG for the small-tile fallback gives more concurrent WGs per SM
        uint32_t conv2d_WG_ناپ  = (s == CONV_SHAPE_64x32) ? 128 : 256;
        uint32_t use_collectives = 0;  // Enables subgroup ops for preventing the re-calculation of indices.
        uint32_t conv2d_TS_K     = (s == CONV_SHAPE_64x32) ? 4 : 8;
        uint32_t conv2d_SHMEM_PAD = 4;
        vk_conv_بلاک_ناپ conv2d_BS = vk_conv_بلاک_sizes[s];
        منطقي conv2d_UNROLL = سچ;

#إذا defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        إذا (device->coopmat2) {
            conv2d_SHMEM_PAD = 8; // 8 float16_t
        }
#جے_انت

        إذا (device->vendor_id == VK_VENDOR_ID_INTEL) {
            conv2d_SHMEM_PAD = 0;
            conv2d_UNROLL = خطأ;
        } وإلا إذا (device->vendor_id == VK_VENDOR_ID_AMD) {
            conv2d_SHMEM_PAD = device->architecture == vk_device_architecture::AMD_GCN ? 1 : 4;
            إذا (s == CONV_SHAPE_128x128 && device->architecture != vk_device_architecture::AMD_GCN) {
                conv2d_UNROLL = خطأ;
            }
        }

        // Use collectives on pre-Turing NVIDIA GPUs and GCN AMD cards, which had slower integer math.
        منطقي allow_collectives_nv = device->vendor_id != VK_VENDOR_ID_NVIDIA ||
                                    device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        منطقي allow_collectives_amd = device->vendor_id != VK_VENDOR_ID_AMD ||
                                     device->architecture == vk_device_architecture::AMD_GCN;

        إذا (device->subgroup_shuffle &&
            device->vendor_id != VK_VENDOR_ID_INTEL &&   // Do not enable collectives on Intel, see PR 14316.
            allow_collectives_nv &&
            allow_collectives_amd) {
            use_collectives = 1;
            conv2d_BS.CRS   = std::min(
                device->subgroup_ناپ,
                conv2d_BS.CRS);  // CRS block size should be capped at subgroup size for correctness when shuffle is used.
        }

        // cm1 is used only when cm2 is unavailable; capped at 64x128 (due to shared memory size).
        // Requires 16x16x16 f16-acc since that's the fragment shape hard-coded in the shader.
        // Subgroup size must be 32 or 64 (to keep WG_SIZE sane) and we need
        // subgroup_size_control to force the driver to actually use it.
        منطقي conv2d_use_cm1 = خطأ;
#إذا defined(VK_KHR_cooperative_میٹرکس) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        conv2d_use_cm1 = !device->coopmat2 &&
                         device->coopmat_support && device->coopmat_support_16x16x16_f16acc &&
                         device->subgroup_ناپ_control &&
                         (device->subgroup_ناپ == 32 || device->subgroup_ناپ == 64) &&
                         s != CONV_SHAPE_128x128;
#جے_انت

        ثابت uint32_t conv2d_cm1_shmem_pad = 8;

        خودکار shmem_req = [&](uint32_t pad, منطقي csh_store, منطقي fp16_shmem) {
            ثابت uint32_t elem_ناپ = fp16_shmem ? (uint32_t)حجم(uint16_t) : (uint32_t)حجم(عائم);
            ثابت uint32_t csh_elems = csh_store ? conv2d_BS.K * conv2d_BS.NPQ : 0u;
            أرجع (conv2d_BS.K * (conv2d_BS.CRS + pad) + conv2d_BS.CRS * (conv2d_BS.NPQ + pad) + csh_elems) * elem_ناپ;
        };

        // 2D, transpose-2D, and 3D conv use the same KxCRS @ CRSxNPQ shmem
        // layout. cm1 needs Csh for output, so check before applying cm1 params.
        إذا (conv2d_use_cm1 && device->properties.limits.max_Compute_Shared_حافظہ_ناپ < shmem_req(conv2d_cm1_shmem_pad, سچ, سچ)) {
            conv2d_use_cm1 = خطأ;
        }

        uint32_t conv2d_WM = 16, conv2d_WN = 16;  // cm1 subgroup tile, ignored otherwise
        إذا (conv2d_use_cm1) {
            conv2d_SHMEM_PAD = conv2d_cm1_shmem_pad;
            // 16x16x16 fragments; pick WM/WN to keep WG_SIZE at 256
            // (i.e. 8 subgroups for sg=32, 4 subgroups for sg=64).
            ثابت منطقي sg64 = (device->subgroup_ناپ == 64);
            بدل (s) {
                حالة CONV_SHAPE_64x32:   conv2d_WM = sg64 ? 32 : 16; conv2d_WN = 16; اكسر;
                حالة CONV_SHAPE_64x128:  conv2d_WM = 32; conv2d_WN = sg64 ? 64 : 32; اكسر;
                حالة CONV_SHAPE_32x256:  conv2d_WM = sg64 ? 16 : 32; conv2d_WN = sg64 ? 128 : 32; اكسر;
                افتراضي: اكسر;
            }
            ثابت uint32_t warps_M = conv2d_BS.K / conv2d_WM;
            ثابت uint32_t warps_N = conv2d_BS.NPQ / conv2d_WN;
            conv2d_WG_ناپ         = warps_M * warps_N * device->subgroup_ناپ;
        }

        // stage cm2 accumulator through shmem for coalesced global stores;
        // skipped on 128x128 where the extra Csh footprint hurts occupancy.
        // cm1 always uses the staged path.
        uint32_t conv2d_csh_store = (device->coopmat2 && s != CONV_SHAPE_128x128) ? 1u : 0u;
        إذا (conv2d_use_cm1) {
            conv2d_csh_store = 1;
        }

        // shmem is fp16 on cm2/cm1 (matches Csh), fp32 on scalar
        ثابت منطقي conv2d_use_fp16_shmem = device->coopmat2 || conv2d_use_cm1;

        // shrink CRS if the non-cm1 config still doesn't fit
        إذا (device->properties.limits.max_Compute_Shared_حافظہ_ناپ < shmem_req(conv2d_SHMEM_PAD, conv2d_csh_store, conv2d_use_fp16_shmem)) {
            GGML_ASSERT(!conv2d_use_cm1);
            conv2d_BS.CRS = 8;
            إذا (use_collectives) {
                conv2d_BS.CRS = std::min(device->subgroup_ناپ, conv2d_BS.CRS);
            }
            conv2d_csh_store = 0;
        }

        std::array<uint32_t, 3> wg_denoms = { conv2d_BS.K, 1, 1 };
        std::vector<uint32_t> spec_constants = { conv2d_WG_ناپ, conv2d_BS.K, conv2d_BS.CRS, conv2d_BS.NPQ, conv2d_TS_K, use_collectives, conv2d_SHMEM_PAD };

        // cm1 needs a fixed subgroup width to match the WG_SIZE we computed
        ثابت uint32_t conv2d_required_subgroup_ناپ = conv2d_use_cm1 ? device->subgroup_ناپ : 0;

#تعریف_کرو CREATE_CONV(name, type_suffix, spv_suffix) \
        لكل (خودکار &c : device->pipeline_##name##type_suffix[s]) { \
            ثابت vk_conv2d_pipeline_state &state = c.پہلا;  \
            std::vector<uint32_t> spec_constants_cpy = spec_constants; \
            spec_constants_cpy.push_back(state.s0); \
            spec_constants_cpy.push_back(state.s1); \
            spec_constants_cpy.push_back(state.p0); \
            spec_constants_cpy.push_back(state.p1); \
            spec_constants_cpy.push_back(state.d0); \
            spec_constants_cpy.push_back(state.d1); \
            spec_constants_cpy.push_back(state.KW); \
            spec_constants_cpy.push_back(state.KH); \
            spec_constants_cpy.push_back(state.aligned); \
            spec_constants_cpy.push_back(conv2d_csh_store); \
            spec_constants_cpy.push_back(conv2d_WM); \
            spec_constants_cpy.push_back(conv2d_WN); \
            ggml_vk_create_pipeline( \
                device, c.second, #name #type_suffix, \
                name##type_suffix##spv_suffix##_len, name##type_suffix##spv_suffix##_data, "main", 3, \
                حجم(vk_op_conv2d_ڈالو_constants), wg_denoms, spec_constants_cpy, 1, سچ, use_collectives || conv2d_required_subgroup_ناپ, conv2d_required_subgroup_ناپ);    \
        }
#تعریف_کرو CREATE_CONVS(spv_suffix) \
        CREATE_CONV(conv2d, _f32, spv_suffix) \
        CREATE_CONV(conv2d, _f16_f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f16_f32, spv_suffix)
#إذا defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        إذا (device->coopmat2) {
            CREATE_CONVS(_cm2)
        } وإلا
#جے_انت
#إذا defined(VK_KHR_cooperative_میٹرکس) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        إذا (conv2d_use_cm1) {
            CREATE_CONVS(_cm1)
        } وإلا
#جے_انت
        إذا (conv2d_UNROLL) {
            CREATE_CONVS(_unroll)
        } وإلا {
            CREATE_CONVS( )
        }
#تعریف_رد CREATE_CONV
#تعریف_رد CREATE_CONVS

        std::vector<uint32_t> conv3d_spec_constants = { conv2d_WG_ناپ, conv2d_BS.K, conv2d_BS.CRS, conv2d_BS.NPQ, conv2d_TS_K, conv2d_SHMEM_PAD };
#تعریف_کرو CREATE_CONV3D(type_suffix, spv_suffix) \
        لكل (خودکار &c : device->pipeline_conv3d##type_suffix[s]) { \
            ثابت vk_conv3d_pipeline_state &state = c.پہلا; \
            std::vector<uint32_t> spec_constants_cpy = conv3d_spec_constants; \
            spec_constants_cpy.push_back(state.s0); \
            spec_constants_cpy.push_back(state.s1); \
            spec_constants_cpy.push_back(state.s2); \
            spec_constants_cpy.push_back(state.p0); \
            spec_constants_cpy.push_back(state.p1); \
            spec_constants_cpy.push_back(state.p2); \
            spec_constants_cpy.push_back(state.d0); \
            spec_constants_cpy.push_back(state.d1); \
            spec_constants_cpy.push_back(state.d2); \
            spec_constants_cpy.push_back(state.KW); \
            spec_constants_cpy.push_back(state.KH); \
            spec_constants_cpy.push_back(state.KD); \
            spec_constants_cpy.push_back(state.aligned); \
            spec_constants_cpy.push_back(conv2d_csh_store); \
            spec_constants_cpy.push_back(conv2d_WM); \
            spec_constants_cpy.push_back(conv2d_WN); \
            ggml_vk_create_pipeline( \
                device, c.second, "conv3d" #type_suffix, \
                conv3d##type_suffix##spv_suffix##_len, conv3d##type_suffix##spv_suffix##_data, "main", 3, \
                حجم(vk_op_conv3d_ڈالو_constants), wg_denoms, spec_constants_cpy, 1, سچ, conv2d_required_subgroup_ناپ != 0, conv2d_required_subgroup_ناپ); \
        }
#إذا defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        إذا (device->coopmat2) {
            CREATE_CONV3D(_f32, _cm2)
            CREATE_CONV3D(_f16_f32, _cm2)
        } وإلا
#جے_انت
#إذا defined(VK_KHR_cooperative_میٹرکس) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        إذا (conv2d_use_cm1) {
            CREATE_CONV3D(_f32, _cm1)
            CREATE_CONV3D(_f16_f32, _cm1)
        } وإلا
#جے_انت
        إذا (conv2d_UNROLL) {
            CREATE_CONV3D(_f32, _unroll)
            CREATE_CONV3D(_f16_f32, _unroll)
        } وإلا {
            CREATE_CONV3D(_f32, )
            CREATE_CONV3D(_f16_f32, )
        }
#تعریف_رد CREATE_CONV3D
    }

    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f32, "conv2d_dw_whcn_f32", conv2d_dw_whcn_f32_len, conv2d_dw_whcn_f32_انکڑے, "main", 3, حجم(vk_op_conv2d_dw_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f32, "conv2d_dw_cwhn_f32", conv2d_dw_cwhn_f32_len, conv2d_dw_cwhn_f32_انکڑے, "main", 3, حجم(vk_op_conv2d_dw_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f16_f32, "conv2d_dw_whcn_f16_f32", conv2d_dw_whcn_f16_f32_len, conv2d_dw_whcn_f16_f32_انکڑے, "main", 3, حجم(vk_op_conv2d_dw_ڈالو_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f16_f32, "conv2d_dw_cwhn_f16_f32", conv2d_dw_cwhn_f16_f32_len, conv2d_dw_cwhn_f16_f32_انکڑے, "main", 3, حجم(vk_op_conv2d_dw_ڈالو_constants), {512, 1, 1}, {}, 1);

    لكل (uint32_t use_ڈالو = 0; use_ڈالو < 2; ++use_ڈالو) {
        لكل (uint32_t i = 0; i < عدد_topk_moe_pipelines; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_topk_moe[i][use_ڈالو], "topk_moe_f32_"+std::to_string(i), topk_moe_f32_len, topk_moe_f32_انکڑے, "main", 4, حجم(vk_op_topk_moe_ڈالو_constants), {1, 1, 1}, {device->subgroup_ناپ, 1u<<i, use_ڈالو}, 1, سچ, سچ, device->subgroup_ناپ);
        }
    }

    // Drop compile_mutex so other threads can walk while we compile.
    compile_جندرا.unlock();

    // Compile what we claimed; create_pipeline_func reacquires compile_mutex
    // at the end to flip compile_pending/compiled and notify waiters.
    إذا (has_claimed_task) {
        خودکار & task = claimed_task;
        ggml_vk_create_pipeline_func(device, task.pipeline, task.spv_ناپ, task.spv_انکڑے,
                                     task.entrypoint, task.parameter_گنتی, task.wg_denoms,
                                     task.specialization_constants, task.disable_robustness,
                                     task.require_بھریا_subgroups, task.required_subgroup_ناپ);
    }

    // Another thread may be compiling the pipeline we need; block on it here.
    إذا (wait_pipeline) {
        std::unique_lock<std::mutex> wait_جندرا(device->compile_mutex);
        device->compile_cv.wait(wait_جندرا, [&] {
            أرجع wait_pipeline->compiled.load();
        });
    }
}

ساكن منطقي ggml_vk_external_حافظہ_host_usable(vk::DriverId driver_id) {
    // Disable external_memory_host until https://github.com/KhronosGroup/MoltenVK/pull/2622
    // is available in the Vulkan SDK.
    أرجع driver_id != vk::DriverId::eMoltenvk;
}

ساكن منطقي ggml_vk_host_بفر_import_usable(uint32_t vendor_id) {
    // amdgpu re-faults all pages of every imported host allocation on each submission
    أرجع vendor_id != VK_VENDOR_ID_AMD;
}

vk_device ggml_vk_get_device(size_t idx) {
    VK_LOG_DEBUG("ggml_vk_get_device(" << idx << ")");

    ساكن std::mutex mutex;
    std::lock_guard<std::mutex> جندرا(mutex);

    إذا (vk_instance.devices[idx] == خالی_سوچک) {
        VK_LOG_DEBUG("Initializing new vk_device");
        vk_device device = std::make_shared<vk_device_struct>();

        device->حافظہ_logger = std::unique_ptr<vk_حافظہ_logger>(جديد vk_حافظہ_logger());

        size_t dev_عدد = vk_instance.device_indices[idx];

        std::vector<vk::PhysicalDevice> physical_devices = vk_instance.instance.enumeratePhysicalDevices();

        إذا (dev_عدد >= physical_devices.size()) {
            std::cerr << "ggml_vulkan: Device with index " << dev_عدد << " does not exist." << std::سطر_انت;
            ارم std::runtime_error("Device not found");
        }

        device->physical_device = physical_devices[dev_عدد];
        ثابت std::vector<vk::ExtensionProperties> ext_props = device->physical_device.enumerateDeviceExtensionProperties();

        device->architecture = get_device_architecture(device->physical_device);

        ثابت حرف* GGML_VK_PREFER_HOST_حافظہ = getenv("GGML_VK_PREFER_HOST_MEMORY");
        device->prefer_host_حافظہ = GGML_VK_PREFER_HOST_حافظہ != خالی_سوچک;

        ثابت حرف* GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM = getenv("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM");
        device->disable_host_visible_vidmem = GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM != خالی_سوچک;

        ثابت حرف* GGML_VK_ALLOW_SYSMEM_FALLBACK = getenv("GGML_VK_ALLOW_SYSMEM_FALLBACK");
        device->allow_sysmem_fallback = GGML_VK_ALLOW_SYSMEM_FALLBACK != خالی_سوچک;

        ثابت حرف* GGML_VK_DISABLE_گراف_OPTIMIZE = getenv("GGML_VK_DISABLE_GRAPH_OPTIMIZE");
        device->disable_گراف_optimize = GGML_VK_DISABLE_گراف_OPTIMIZE != خالی_سوچک;

        منطقي fp16_storage = خطأ;
        منطقي fp16_compute = خطأ;
        منطقي maintenance4_support = خطأ;
        منطقي sm_builtins = خطأ;
        منطقي amd_shader_core_properties2 = خطأ;
        منطقي pipeline_robustness = خطأ;
        منطقي coopmat2_support = خطأ;
        منطقي coopmat2_decode_سمتیہ_support = خطأ;
        منطقي pipeline_executable_properties_support = خطأ;
        منطقي internally_sync_support = خطأ;
        device->coopmat_support = خطأ;
        device->integer_dot_ضرب_حاصل = خطأ;
        device->shader_64b_indexing = خطأ;
        منطقي bfloat16_support = خطأ;
        منطقي dot2_f16_support = خطأ;
        منطقي ocp_microscaling_extension = خطأ;
        منطقي shader_float8_extension = خطأ;

        لكل (ثابت خودکار& properties : ext_props) {
            إذا (strcmp("VK_KHR_maintenance4", properties.extensionName) == 0) {
                maintenance4_support = سچ;
            } وإلا إذا (strcmp("VK_KHR_16bit_storage", properties.extensionName) == 0) {
                fp16_storage = سچ;
            } وإلا إذا (strcmp("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
                fp16_compute = سچ;
            } وإلا إذا (strcmp("VK_NV_shader_sm_builtins", properties.extensionName) == 0) {
                sm_builtins = سچ;
            } وإلا إذا (strcmp("VK_AMD_shader_core_properties2", properties.extensionName) == 0) {
                amd_shader_core_properties2 = سچ;
            } وإلا إذا (strcmp("VK_EXT_pipeline_robustness", properties.extensionName) == 0) {
                pipeline_robustness = سچ;
            } وإلا إذا (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                device->subgroup_ناپ_control = سچ;
#إذا defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            } وإلا إذا (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT")) {
                device->coopmat_support = سچ;
                device->coopmat_m = 0;
                device->coopmat_n = 0;
                device->coopmat_k = 0;
#جے_انت
#إذا defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            } وإلا إذا (strcmp("VK_NV_cooperative_matrix2", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT2")) {
                coopmat2_support = سچ;
#جے_انت
            } وإلا إذا (strcmp(VK_NV_COOPERATIVE_میٹرکس_DECODE_سمتیہ_EXTENSION_NAME, properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR")) {
                coopmat2_decode_سمتیہ_support = سچ;
#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            } وإلا إذا (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
                device->integer_dot_ضرب_حاصل = سچ;
#جے_انت
#إذا defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
            } وإلا إذا (strcmp("VK_KHR_shader_bfloat16", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_BFLOAT16")) {
                bfloat16_support = سچ;
#جے_انت
#إذا defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT)
            } وإلا إذا (strcmp(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME, properties.extensionName) == 0) {
                ocp_microscaling_extension = سچ;
#جے_انت
#إذا defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
            } وإلا إذا (strcmp(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, properties.extensionName) == 0) {
                shader_float8_extension = سچ;
#جے_انت
            } وإلا إذا (strcmp("VK_VALVE_shader_mixed_float_dot_product", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_DOT2")) {
                dot2_f16_support = سچ;
            } وإلا إذا (strcmp("VK_KHR_pipeline_executable_properties", properties.extensionName) == 0) {
                pipeline_executable_properties_support = سچ;
            } وإلا إذا (strcmp("VK_EXT_memory_priority", properties.extensionName) == 0 &&
                       getenv("GGML_VK_ENABLE_MEMORY_PRIORITY")) {
                device->حافظہ_ترجیح = سچ;
            } وإلا إذا (strcmp("VK_EXT_external_memory_host", properties.extensionName) == 0) {
                device->external_حافظہ_host = سچ;
#إذا defined(VK_EXT_shader_64bit_indexing)
            } وإلا إذا (strcmp("VK_EXT_shader_64bit_indexing", properties.extensionName) == 0) {
                device->shader_64b_indexing = سچ;
#جے_انت
            } وإلا إذا (strcmp(VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME, properties.extensionName) == 0) {
                internally_sync_support = سچ;
            } وإلا إذا (strcmp("VK_EXT_device_fault", properties.extensionName) == 0) {
                device->device_fault = سچ;
            }
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::PhysicalDeviceMaintenance3Properties props3;
        vk::PhysicalDeviceMaintenance4Properties props4;
        vk::PhysicalDeviceSubgroupProperties subgroup_props;
        vk::PhysicalDeviceDriverProperties driver_props;
        vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV sm_props;
        vk::PhysicalDeviceShaderCoreProperties2AMD amd_shader_core_properties2_props;
        vk::PhysicalDeviceVulkan11Properties vk11_props;
        vk::PhysicalDeviceVulkan12Properties vk12_props;
        vk::Physical_Device_Subgroup_ناپ_Control_Properties_EXT subgroup_ناپ_control_props;
        vk::Physical_Device_Shader_Integer_Dot_ضرب_حاصل_Properties_KHR shader_integer_dot_ضرب_حاصل_props;
        vk::Physical_Device_External_حافظہ_Host_Properties_EXT external_حافظہ_host_props;

        props2.p_اگلا = &props3;
        props3.p_اگلا = &subgroup_props;
        subgroup_props.p_اگلا = &driver_props;
        driver_props.p_اگلا = &vk11_props;
        vk11_props.p_اگلا = &vk12_props;

        Vk_ادھار_باہرلا_Structure * آخری_struct = (Vk_ادھار_باہرلا_Structure *)&vk12_props;

        إذا (maintenance4_support) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&props4;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&props4;
        }
        إذا (sm_builtins) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&sm_props;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&sm_props;
        }
        إذا (amd_shader_core_properties2) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&amd_shader_core_properties2_props;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&amd_shader_core_properties2_props;
        }
        إذا (device->subgroup_ناپ_control) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&subgroup_ناپ_control_props;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&subgroup_ناپ_control_props;
        }

#إذا defined(VK_NV_cooperative_matrix2)
        vk::PhysicalDeviceCooperativeMatrix2PropertiesNV coopmat2_props;
        إذا (coopmat2_support) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&coopmat2_props;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&coopmat2_props;
        }
#جے_انت

        إذا (device->integer_dot_ضرب_حاصل) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&shader_integer_dot_ضرب_حاصل_props;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&shader_integer_dot_ضرب_حاصل_props;
        }

        إذا (device->external_حافظہ_host) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&external_حافظہ_host_props;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&external_حافظہ_host_props;
        }

        device->physical_device.getProperties2(&props2);
        device->properties = props2.properties;
        device->vendor_id = device->properties.vendorID;
        device->driver_id = driver_props.driverID;

        device->external_حافظہ_host = device->external_حافظہ_host && ggml_vk_external_حافظہ_host_usable(device->driver_id);

        // Implementing the async backend interfaces seems broken on older Intel HW,
        // see https://github.com/ggml-org/llama.cpp/issues/17302.
        device->support_async = (device->vendor_id != VK_VENDOR_ID_INTEL ||
                                 std::string(device->properties.deviceName.انکڑے()).find("(DG1)") == std::string::npos) &&
                                getenv("GGML_VK_DISABLE_ASYNC") == خالی_سوچک;

        إذا (!device->support_async) {
            GGML_LOG_DEBUG("ggml_vulkan: WARNING: Async execution disabled on certain Intel devices.\n");
        }

        ثابت حرف* GGML_VK_FORCE_MAX_ونڈ_ناپ = getenv("GGML_VK_FORCE_MAX_ALLOCATION_SIZE");

        إذا (GGML_VK_FORCE_MAX_ونڈ_ناپ != خالی_سوچک) {
            device->max_حافظہ_ونڈ_ناپ = std::stoull(GGML_VK_FORCE_MAX_ونڈ_ناپ);
        } وإلا إذا (maintenance4_support) {
            device->max_حافظہ_ونڈ_ناپ = std::min(props3.max_حافظہ_ونڈ_ناپ_2, props4.max_بفر_ناپ_2);
        } وإلا {
            device->max_حافظہ_ونڈ_ناپ = props3.max_حافظہ_ونڈ_ناپ_2;
        }

        ثابت حرف* GGML_VK_FORCE_MAX_بفر_ناپ = getenv("GGML_VK_FORCE_MAX_BUFFER_SIZE");

        إذا (GGML_VK_FORCE_MAX_بفر_ناپ != خالی_سوچک) {
            device->max_بفر_ناپ = std::stoull(GGML_VK_FORCE_MAX_بفر_ناپ);
        } وإلا إذا (maintenance4_support) {
            device->max_بفر_ناپ = props4.max_بفر_ناپ_2;
        } وإلا {
            device->max_بفر_ناپ = device->max_حافظہ_ونڈ_ناپ;
        }

        ثابت حرف* GGML_VK_SUBALLOCATION_بلاک_ناپ = getenv("GGML_VK_SUBALLOCATION_BLOCK_SIZE");

        إذا (GGML_VK_SUBALLOCATION_بلاک_ناپ != خالی_سوچک) {
            device->suballocation_بلاک_ناپ = std::stoull(GGML_VK_SUBALLOCATION_بلاک_ناپ);
        } وإلا {
            // Limit batching of allocations to 1GB by default to avoid fragmentation issues
            device->suballocation_بلاک_ناپ = 1024*1024*1024;
        }
        device->suballocation_بلاک_ناپ = std::min(device->suballocation_بلاک_ناپ, device->max_حافظہ_ونڈ_ناپ);

        device->subgroup_ناپ = subgroup_props.subgroup_ناپ_2;
        device->subgroup_ناپ_log2 = uint32_t(log2f(عائم(device->subgroup_ناپ)));
        device->uma = device->properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
        إذا (sm_builtins) {
            device->shader_core_گنتی = sm_props.shaderSMCount;
        } وإلا إذا (amd_shader_core_properties2) {
            device->shader_core_گنتی = amd_shader_core_properties2_props.سرگرم_Compute_Unit_گنتی;
        } وإلا إذا (device->vendor_id == VK_VENDOR_ID_INTEL) {
            device->shader_core_گنتی = ggml_vk_intel_shader_core_گنتی(device->physical_device);
        } وإلا {
            device->shader_core_گنتی = 0;
        }
        device->float_controls_rte_fp16 = vk12_props.shader_Rounding_منوال_RTEFloat16;
        device->float_controls_denorm_preserve_fp16 = vk12_props.shaderDenormPreserveFloat16;

        device->subgroup_basic = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_بٹ::eCompute) &&
                                 (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_بٹ::eBasic);
        device->subgroup_arithmetic = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_بٹ::eCompute) &&
                                      (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_بٹ::eArithmetic);
#جے_تعریف_شدہ __APPLE__
        // Workaround for subgroup arithmetic failing on MoltenVK with AMD GPUs (issue 15846)
        إذا (device->vendor_id == VK_VENDOR_ID_AMD) {
            device->subgroup_arithmetic = خطأ;
        }
#جے_انت
        device->subgroup_shuffle = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_بٹ::eCompute) &&
                                   (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_بٹ::eShuffle);
#جے_تعریف_شدہ __APPLE__
        إذا (device->vendor_id == VK_VENDOR_ID_AMD) {
            device->subgroup_shuffle = خطأ;
        }
#جے_انت
        device->subgroup_clustered = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_بٹ::eCompute) &&
                                     (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_بٹ::eClustered);

        device->subgroup_ballot = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_بٹ::eCompute) &&
                                  (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_بٹ::eBallot);

        device->subgroup_vote = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_بٹ::eCompute) &&
                                (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_بٹ::eVote);

        // Submit at least every 100 nodes, in case there are workloads without as much matmul.
        device->max_nodes_فی_submit = 100;
        ثابت حرف* GGML_VK_MAX_NODES_فی_SUBMIT = getenv("GGML_VK_MAX_NODES_PER_SUBMIT");
        إذا (GGML_VK_MAX_NODES_فی_SUBMIT != خالی_سوچک) {
            uint32_t max_nodes_فی_submit = std::stoul(GGML_VK_MAX_NODES_فی_SUBMIT);
            device->max_nodes_فی_submit = std::max(max_nodes_فی_submit, 1u);
        }

        ثابت منطقي force_disable_f16 = getenv("GGML_VK_DISABLE_F16") != خالی_سوچک;

        device->fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

        إذا (!ggml_vk_khr_cooperative_میٹرکس_support(device->properties, driver_props, device->architecture)) {
            device->coopmat_support = خطأ;
        }

        device->integer_dot_ضرب_حاصل = device->integer_dot_ضرب_حاصل && shader_integer_dot_ضرب_حاصل_props.integer_Dot_Product4x8_ثنائی_ہندسہ_Packed_Signed_Accelerated;

        device->min_imported_host_اشارہ_گر_alignment = external_حافظہ_host_props.min_Imported_Host_اشارہ_گر_Alignment;

        device->max_workgroup_ناپ_log2 = uint32_t(log2f(عائم(device->properties.limits.max_Compute_Work_گروہ_Invocations)));

        std::vector<vk::قطار_Family_Properties> قطار_family_props = device->physical_device.get_قطار_Family_Properties();

        // Try to find a non-graphics compute queue and transfer-focused queues
        // Allow overriding avoiding the graphics queue because it can increase performance on RADV
        ثابت منطقي allow_graphics_قطار = (getenv("GGML_VK_ALLOW_GRAPHICS_QUEUE") != خالی_سوچک);
        ثابت vk::قطار_Flag_بٹ graphics_flag = allow_graphics_قطار ? (vk::قطار_Flag_بٹ)0 : vk::قطار_Flag_بٹ::eGraphics;
        ثابت uint32_t compute_قطار_family_اشاریہ = ggml_vk_لبھو_قطار_family_اشاریہ(قطار_family_props, vk::قطار_Flag_بٹ::eCompute, graphics_flag, -1, 1);
        ثابت uint32_t transfer_قطار_family_اشاریہ = ggml_vk_لبھو_قطار_family_اشاریہ(قطار_family_props, vk::قطار_Flag_بٹ::eTransfer, vk::قطار_Flag_بٹ::eCompute | graphics_flag, compute_قطار_family_اشاریہ, 1);

        ثابت عائم priorities[] = { 1.0f, 1.0f };
        device->single_قطار = compute_قطار_family_اشاریہ == transfer_قطار_family_اشاریہ && قطار_family_props[compute_قطار_family_اشاریہ].قطار_گنتی == 1;

        std::vector<vk::Device_قطار_Create_Info> device_قطار_create_infos;
        vk::DeviceCreateInfo device_create_info{};
        std::vector<ثابت حرف *> device_extensions;
        vk::PhysicalDeviceFeatures device_features = device->physical_device.getFeatures();

        VkPhysicalDeviceFeatures2 device_features2;
        device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        device_features2.p_اگلا = خالی_سوچک;
        device_features2.features = (VkPhysicalDeviceFeatures)device_features;

        VkPhysicalDeviceVulkan11Features vk11_features;
        vk11_features.p_اگلا = خالی_سوچک;
        vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        device_features2.p_اگلا = &vk11_features;

        VkPhysicalDeviceVulkan12Features vk12_features;
        vk12_features.p_اگلا = خالی_سوچک;
        vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vk11_features.p_اگلا = &vk12_features;

        آخری_struct = (Vk_ادھار_باہرلا_Structure *)&vk12_features;

        VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR internally_synchronized_queues_features{};
        internally_synchronized_queues_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR;
        internally_synchronized_queues_features.p_اگلا = خالی_سوچک;
        internally_synchronized_queues_features.internallySynchronizedQueues = VK_FALSE;

        إذا (internally_sync_support) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&internally_synchronized_queues_features;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&internally_synchronized_queues_features;
            device_extensions.push_back(VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME);
        }

        VkPhysicalDevicePipelineRobustnessFeaturesEXT pl_robustness_features;
        pl_robustness_features.p_اگلا = خالی_سوچک;
        pl_robustness_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES_EXT;
        pl_robustness_features.pipelineRobustness = VK_FALSE;

        إذا (pipeline_robustness) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&pl_robustness_features;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&pl_robustness_features;
            device_extensions.push_back("VK_EXT_pipeline_robustness");
        }

        Vk_Physical_Device_حافظہ_ترجیح_Features_EXT حافظہ_ترجیح_features;
        حافظہ_ترجیح_features.p_اگلا = خالی_سوچک;
        حافظہ_ترجیح_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_حافظہ_ترجیح_FEATURES_EXT;
        حافظہ_ترجیح_features.حافظہ_ترجیح_2 = VK_FALSE;
        إذا (device->حافظہ_ترجیح) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&حافظہ_ترجیح_features;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&حافظہ_ترجیح_features;
            device_extensions.push_back("VK_EXT_memory_priority");
        }

        Vk_Physical_Device_Subgroup_ناپ_Control_Features_EXT subgroup_ناپ_control_features;
        subgroup_ناپ_control_features.p_اگلا = خالی_سوچک;
        subgroup_ناپ_control_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_ناپ_CONTROL_FEATURES_EXT;
        subgroup_ناپ_control_features.compute_بھریا_Subgroups = خطأ;
        subgroup_ناپ_control_features.subgroup_ناپ_Control = خطأ;

        إذا (device->subgroup_ناپ_control) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&subgroup_ناپ_control_features;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&subgroup_ناپ_control_features;
        }

#إذا defined(VK_KHR_cooperative_میٹرکس)
        Vk_Physical_Device_Cooperative_میٹرکس_Features_KHR coopmat_features;
        coopmat_features.p_اگلا = خالی_سوچک;
        coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_میٹرکس_FEATURES_KHR;
        coopmat_features.cooperative_میٹرکس_2 = VK_FALSE;

        إذا (device->coopmat_support) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&coopmat_features;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&coopmat_features;
        }
#جے_انت

#إذا defined(VK_NV_cooperative_matrix2)
        VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features {};
        coopmat2_features.p_اگلا = خالی_سوچک;
        coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_میٹرکس_2_FEATURES_NV;
        إذا (coopmat2_support) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&coopmat2_features;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&coopmat2_features;
            device_extensions.push_back("VK_NV_cooperative_matrix2");
        }
#جے_انت

        Vk_Physical_Device_Cooperative_میٹرکس_Decode_سمتیہ_Features_NV coopmat2_decode_سمتیہ_features {};
        coopmat2_decode_سمتیہ_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_میٹرکس_DECODE_سمتیہ_FEATURES_NV;
        إذا (coopmat2_decode_سمتیہ_support) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&coopmat2_decode_سمتیہ_features;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&coopmat2_decode_سمتیہ_features;
            device_extensions.push_back(VK_NV_COOPERATIVE_میٹرکس_DECODE_سمتیہ_EXTENSION_NAME);
        }

#إذا defined(VK_KHR_shader_bfloat16)
        VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features {};
        bfloat16_features.p_اگلا = خالی_سوچک;
        bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
        إذا (bfloat16_support) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&bfloat16_features;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&bfloat16_features;
            device_extensions.push_back("VK_KHR_shader_bfloat16");
        }
#جے_انت

        VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT ocp_microscaling_features {};
        ocp_microscaling_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT;
        إذا (ocp_microscaling_extension) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&ocp_microscaling_features;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&ocp_microscaling_features;
            device_extensions.push_back(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME);
        }

        VkPhysicalDeviceShaderFloat8FeaturesEXT shader_float8_features {};
        shader_float8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
        إذا (shader_float8_extension) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&shader_float8_features;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&shader_float8_features;
            device_extensions.push_back(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
        }

        VkPhysicalDeviceMaintenance4Features maint4_features {};
        maint4_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
        إذا (maintenance4_support) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&maint4_features;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&maint4_features;
            device_extensions.push_back("VK_KHR_maintenance4");
        }

        Vk_Physical_Device_Shader_Integer_Dot_ضرب_حاصل_Features_KHR shader_integer_dot_ضرب_حاصل_features {};
        shader_integer_dot_ضرب_حاصل_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_ضرب_حاصل_FEATURES_KHR;
        إذا (device->integer_dot_ضرب_حاصل) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&shader_integer_dot_ضرب_حاصل_features;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&shader_integer_dot_ضرب_حاصل_features;
            device_extensions.push_back("VK_KHR_shader_integer_dot_product");
        }

        Vk_Physical_Device_Shader_Mixed_Float_Dot_ضرب_حاصل_Features_VALVE dot2_features {};
        dot2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_ضرب_حاصل_FEATURES_VALVE;
        إذا (dot2_f16_support) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&dot2_features;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&dot2_features;
            device_extensions.push_back("VK_VALVE_shader_mixed_float_dot_product");
        }

        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pep_features {};
        pep_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR;
        إذا (pipeline_executable_properties_support) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&pep_features;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&pep_features;
            device_extensions.push_back("VK_KHR_pipeline_executable_properties");
        }

        إذا (device->external_حافظہ_host) {
            device_extensions.push_back("VK_EXT_external_memory_host");
        }

#إذا defined(VK_EXT_shader_64bit_indexing)
        Vk_Physical_Device_Shader64_ثنائی_ہندسہ_Indexing_Features_EXT shader_64bit_indexing_features {};
        shader_64bit_indexing_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_64_ثنائی_ہندسہ_INDEXING_FEATURES_EXT;
        إذا (device->shader_64b_indexing) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&shader_64bit_indexing_features;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&shader_64bit_indexing_features;
            device_extensions.push_back("VK_EXT_shader_64bit_indexing");
        }
#جے_انت

        VkPhysicalDeviceFaultFeaturesEXT fault_features {};
        fault_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT;
        إذا (device->device_fault) {
            آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&fault_features;
            آخری_struct = (Vk_ادھار_باہرلا_Structure *)&fault_features;
            device_extensions.push_back("VK_EXT_device_fault");
        }

        vkGetPhysicalDeviceFeatures2(device->physical_device, &device_features2);

        device->device_fault = device->device_fault && fault_features.deviceFault;

        device->has_internally_synchronized_queues = internally_synchronized_queues_features.internallySynchronizedQueues;

        // Build queue create infos only after querying whether internally synchronized queues are enabled.
        // getQueue2() later uses the same flag, so creation/retrieval must stay consistent.
        vk::Device_قطار_Create_Flags قطار_flags = device->has_internally_synchronized_queues ?
                                                eInternallySynchronizedKHR :
                                                vk::Device_قطار_Create_Flags();

        إذا (compute_قطار_family_اشاریہ != transfer_قطار_family_اشاریہ) {
            device_قطار_create_infos.push_back({قطار_flags, compute_قطار_family_اشاریہ, 1, priorities});
            device_قطار_create_infos.push_back({قطار_flags, transfer_قطار_family_اشاریہ, 1, priorities + 1});
        } وإلا إذا(!device->single_قطار) {
            device_قطار_create_infos.push_back({قطار_flags, compute_قطار_family_اشاریہ, 2, priorities});
        } وإلا {
            device_قطار_create_infos.push_back({قطار_flags, compute_قطار_family_اشاریہ, 1, priorities});
        }

        device->pipeline_executable_properties_support = pipeline_executable_properties_support;

        device->fp16 = device->fp16 && vk12_features.shaderFloat16;

#إذا defined(VK_KHR_shader_bfloat16)
        device->bf16 = bfloat16_support && bfloat16_features.shaderBFloat16Type;
#وإلا
        device->bf16 = خطأ;
#جے_انت

        device->dot2_f16 = dot2_f16_support && dot2_features.shader_Mixed_Float_Dot_ضرب_حاصل_Float16_Acc_Float32;
        device->ocp_fp4 = ocp_microscaling_extension && ocp_microscaling_features.shaderFloat4 &&
                          shader_float8_extension && shader_float8_features.shaderFloat8 &&
                          !getenv("GGML_VK_DISABLE_OCP_FP4");

        device->pipeline_robustness = pl_robustness_features.pipelineRobustness;

        device->multi_جوڑو = vk12_props.shader_Rounding_منوال_RTEFloat16 &&
                            device->properties.limits.max_ڈالو_Constants_ناپ >= حجم(vk_op_multi_جوڑو_ڈالو_constants) &&
                            getenv("GGML_VK_DISABLE_MULTI_ADD") == خالی_سوچک;

        device->shader_int64 = device_features2.features.shaderInt64;
        device->بفر_device_پتہ = vk12_features.بفر_Device_پتہ;
        device->vulkan_حافظہ_model = vk12_features.vulkan_حافظہ_Model;

        إذا (device->subgroup_ناپ_control) {
            device->subgroup_min_ناپ = subgroup_ناپ_control_props.min_Subgroup_ناپ;
            device->subgroup_max_ناپ = subgroup_ناپ_control_props.max_Subgroup_ناپ;
            device_extensions.push_back("VK_EXT_subgroup_size_control");
        }

        device->subgroup_ناپ_control = device->subgroup_ناپ_control &&
                (subgroup_ناپ_control_props.required_Subgroup_ناپ_Stages & vk::Shader_Stage_Flag_بٹ::eCompute) &&
                subgroup_ناپ_control_features.subgroup_ناپ_Control;

        device->subgroup_require_بھریا_support = subgroup_ناپ_control_features.compute_بھریا_Subgroups;

#إذا defined(VK_KHR_cooperative_میٹرکس)
        device->coopmat_support = device->coopmat_support && coopmat_features.cooperative_میٹرکس_2;
        device->coopmat1_fa_support = device->coopmat_support && device->subgroup_require_بھریا_support;
#جے_انت

        إذا (coopmat2_support) {
#إذا defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            إذا (coopmat2_features.cooperative_میٹرکس_Workgroup_Scope &&
                coopmat2_features.cooperative_میٹرکس_Flexible_Dimensions &&
                coopmat2_features.cooperative_میٹرکس_Reductions &&
                coopmat2_features.cooperative_میٹرکس_Conversions &&
                coopmat2_features.cooperative_میٹرکس_فی_Element_Operations &&
                coopmat2_features.cooperative_میٹرکس_Tensor_Addressing &&
                coopmat2_features.cooperative_میٹرکس_بلاک_Loads &&
                vk12_features.بفر_Device_پتہ) {

                std::vector<Vk_Cooperative_میٹرکس_Flexible_Dimensions_Properties_NV> flexible_dimensions;
                uint32_t count = 0;

                PFN_vk_Get_Physical_Device_Cooperative_میٹرکس_Flexible_Dimensions_Properties_NV
                    vk_Get_Physical_Device_Cooperative_میٹرکس_Flexible_Dimensions_Properties_NV =
                        (PFN_vk_Get_Physical_Device_Cooperative_میٹرکس_Flexible_Dimensions_Properties_NV)
                        vk_instance.instance.getProcAddr("vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV");

                vk_Get_Physical_Device_Cooperative_میٹرکس_Flexible_Dimensions_Properties_NV(device->physical_device, &count, خالی_سوچک);

                Vk_Cooperative_میٹرکس_Flexible_Dimensions_Properties_NV خالی_تھاواں_prop {};
                خالی_تھاواں_prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_میٹرکس_FLEXIBLE_DIMENSIONS_PROPERTIES_NV;
                flexible_dimensions.resize(count, خالی_تھاواں_prop);

                vk_Get_Physical_Device_Cooperative_میٹرکس_Flexible_Dimensions_Properties_NV(device->physical_device, &count, flexible_dimensions.انکڑے());

                منطقي لبھیا_fp16_128 = خطأ,
                     لبھیا_fp16_256 = خطأ,
                     لبھیا_fp32_128 = خطأ,
                     لبھیا_fp32_256 = خطأ;
                منطقي لبھیا_bf16_128 = خطأ,
                     لبھیا_bf16_256 = خطأ;
                // need to support fp16*fp16 with fp16/fp32 accumulator, for workgroupsize 128
                // with 32x16x16 and 256 with 32x32x16.
                لكل (خودکار &prop : flexible_dimensions) {
                    إذا (prop.saturatingAccumulation == VK_FALSE &&
                        prop.scope == VK_SCOPE_WORKGROUP_KHR) {

                        إذا (prop.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                            prop.BType == VK_COMPONENT_TYPE_FLOAT16_KHR) {

                            إذا (prop.workgroupInvocations == 128 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 16 &&
                                prop.KGranularity <= 16) {
                                إذا (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    prop.نتیجہ_Type == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                                    لبھیا_fp16_128 = سچ;
                                }
                                إذا (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                    prop.نتیجہ_Type == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                                    لبھیا_fp32_128 = سچ;
                                }
                            }
                            إذا (prop.workgroupInvocations == 256 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 32 &&
                                prop.KGranularity <= 16) {
                                إذا (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    prop.نتیجہ_Type == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                                    لبھیا_fp16_256 = سچ;
                                }
                                إذا (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                    prop.نتیجہ_Type == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                                    لبھیا_fp32_256 = سچ;
                                }
                            }
                        }

#إذا defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                        إذا (bfloat16_support &&
                            prop.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                            prop.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                            prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                            prop.نتیجہ_Type == VK_COMPONENT_TYPE_FLOAT32_KHR) {

                            إذا (prop.workgroupInvocations == 128 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 16 &&
                                prop.KGranularity <= 16) {
                                لبھیا_bf16_128 = سچ;
                            }
                            إذا (prop.workgroupInvocations == 256 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 32 &&
                                prop.KGranularity <= 16) {
                                لبھیا_bf16_256 = سچ;
                            }
                        }
#جے_انت
                    }
                }
                إذا (لبھیا_fp16_128 && لبھیا_fp16_256 &&
                    لبھیا_fp32_128 && لبھیا_fp32_256 &&
                    coopmat2_props.cooperative_میٹرکس_Flexible_Dimensions_Max_Dimension >= 512) {
                    device->coopmat2 = سچ;
                    device->coopmat2_bf16_support = لبھیا_bf16_128 && لبھیا_bf16_256;
                    device->coopmat2_decode_سمتیہ = coopmat2_decode_سمتیہ_support && coopmat2_decode_سمتیہ_features.cooperative_میٹرکس_Decode_سمتیہ;
                }
            }
#جے_انت
        }

        إذا (!vk11_features.storage_Buffer16_ثنائی_ہندسہ_Access) {
            std::cerr << "ggml_vulkan: device " << GGML_VK_NAME << idx << " does not support 16-bit storage." << std::سطر_انت;
            ارم std::runtime_error("Unsupported device");
        }

        device_extensions.push_back("VK_KHR_16bit_storage");

#جے_تعریف_شدہ GGML_VULKAN_VALIDATE
        device_extensions.push_back("VK_KHR_shader_non_semantic_info");
#جے_انت

        إذا (device->fp16) {
            device_extensions.push_back("VK_KHR_shader_float16_int8");
        }

#إذا defined(VK_KHR_cooperative_میٹرکس)
        إذا (device->coopmat_support) {
            // Query supported shapes
            std::vector<Vk_Cooperative_میٹرکس_Properties_KHR> cm_props;

            PFN_vk_Get_Physical_Device_Cooperative_میٹرکس_Properties_KHR pfn_vk_Get_Physical_Device_Cooperative_میٹرکس_Properties_KHR =
                (PFN_vk_Get_Physical_Device_Cooperative_میٹرکس_Properties_KHR)vkGetInstanceProcAddr(vk_instance.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");

            uint32_t cm_props_عدد;

            pfn_vk_Get_Physical_Device_Cooperative_میٹرکس_Properties_KHR(device->physical_device, &cm_props_عدد, خالی_سوچک);

            cm_props.resize(cm_props_عدد);

            لكل (خودکار& prop : cm_props) {
                prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_میٹرکس_PROPERTIES_KHR;
            }

            pfn_vk_Get_Physical_Device_Cooperative_میٹرکس_Properties_KHR(device->physical_device, &cm_props_عدد, cm_props.انکڑے());

            VK_LOG_DEBUG("ggml_vulkan: Cooperative Matrix Shapes: " << cm_props.size());

            لكل (خودکار& prop : cm_props) {
                VK_LOG_DEBUG("ggml_vulkan: M: " << prop.MSize << " N: " << prop.NSize << " K: " << prop.KSize << " A: " << vk::to_string((vk::ComponentTypeKHR)prop.AType) << " B: " << vk::to_string((vk::ComponentTypeKHR)prop.BType) << " C: " << vk::to_string((vk::ComponentTypeKHR)prop.CType) << " Result: " << vk::to_string((vk::ComponentTypeKHR)prop.نتیجہ_Type) << " saturatingAccumulation: " << prop.saturatingAccumulation << " scope: " << vk::to_string((vk::ScopeKHR)prop.scope));

                إذا ((vk::ComponentTypeKHR)prop.AType == vk::ComponentTypeKHR::eFloat16 &&
                    (vk::ComponentTypeKHR)prop.BType == vk::ComponentTypeKHR::eFloat16 &&
                    (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
                ) {
                    إذا ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat32 &&
                        (vk::ComponentTypeKHR)prop.نتیجہ_Type == vk::ComponentTypeKHR::eFloat32) {
                        // coopmat sizes not set yet
                        إذا (device->coopmat_m == 0) {
                            device->coopmat_acc_f32_support = سچ;
                            device->coopmat_m = prop.MSize;
                            device->coopmat_n = prop.NSize;
                            device->coopmat_k = prop.KSize;
                        } وإلا إذا (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                            // Only enable if shape is identical
                            device->coopmat_acc_f32_support = سچ;
                        }
                        إذا (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
                            device->coopmat_support_16x16x16_f32acc = سچ;
                        }
                    } وإلا إذا ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat16 &&
                               (vk::ComponentTypeKHR)prop.نتیجہ_Type == vk::ComponentTypeKHR::eFloat16) {
                        // coopmat sizes not set yet
                        إذا (device->coopmat_m == 0) {
                            device->coopmat_acc_f16_support = سچ;
                            device->coopmat_m = prop.MSize;
                            device->coopmat_n = prop.NSize;
                            device->coopmat_k = prop.KSize;
                        } وإلا إذا (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                            // Only enable if shape is identical
                            device->coopmat_acc_f16_support = سچ;
                        }
                        إذا (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
                            device->coopmat_support_16x16x16_f16acc = سچ;
                        }
                    }
                } وإلا إذا ((vk::ComponentTypeKHR)prop.AType      == vk::ComponentTypeKHR::eSint8 &&
                           (vk::ComponentTypeKHR)prop.BType      == vk::ComponentTypeKHR::eSint8 &&
                           (vk::ComponentTypeKHR)prop.CType      == vk::ComponentTypeKHR::eSint32 &&
                           (vk::ComponentTypeKHR)prop.نتیجہ_Type == vk::ComponentTypeKHR::eSint32 &&
                           (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup &&
                           device->coopmat_int_m == 0
                ) {
                    device->coopmat_int_support = سچ;
                    device->coopmat_int_m = prop.MSize;
                    device->coopmat_int_n = prop.NSize;
                    device->coopmat_int_k = prop.KSize;
                }
#إذا defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                إذا (bfloat16_support &&
                    prop.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    prop.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    prop.نتیجہ_Type == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
                ) {
                    // coopmat sizes not set yet
                    إذا (device->coopmat_m == 0) {
                        device->coopmat_bf16_support = سچ;
                        device->coopmat_m = prop.MSize;
                        device->coopmat_n = prop.NSize;
                        device->coopmat_k = prop.KSize;
                    } وإلا إذا (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                        // Only enable if shape is identical
                        device->coopmat_bf16_support = سچ;
                    }
                }
#جے_انت
            }

            إذا (device->coopmat_m == 0 || !device->coopmat_acc_f32_support) {
                // No suitable matmul mode found
                GGML_LOG_DEBUG("ggml_vulkan: WARNING: No suitable matrix core mode found. Disabling matrix cores.\n");
                device->coopmat_support = خطأ;
            }
        }

        إذا (device->coopmat_support) {
            device_extensions.push_back("VK_KHR_cooperative_matrix");
        }
#جے_انت
        device->name = GGML_VK_NAME + std::to_string(idx);

        device_create_info
            .سیٹ_Flags(vk::DeviceCreateFlags())
            .سیٹ_قطار_Create_Infos(device_قطار_create_infos)
            .سیٹ_PEnabled_Extension_ناں(device_extensions);
        device_create_info.سیٹ_PNext(&device_features2);
        device->device = device->physical_device.createDevice(device_create_info);

        إذا (device->device_fault) {
            device->pfn_vkGetDeviceFaultInfoEXT = (PFN_vkGetDeviceFaultInfoEXT)
                vkGetDeviceProcAddr(device->device, "vkGetDeviceFaultInfoEXT");
        }

        // Queues
        device->compute_قطار = ggml_vk_create_قطار(device, compute_قطار_family_اشاریہ, 0, { vk::Pipeline_Stage_Flag_بٹ::eComputeShader | vk::Pipeline_Stage_Flag_بٹ::eTransfer }, خطأ);

        // Shaders
        // Disable matmul tile sizes early if performance low or not supported
        لكل (uint32_t i = 0; i < GGML_TYPE_گنتی; ++i) {
            بدل (device->vendor_id) {
#جے_بنا_تعریف GGML_VULKAN_لگاتار_TESTS
            حالة VK_VENDOR_ID_AMD:
                device->mul_mat_l[i]    = device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary;
                device->mul_mat_m[i]    = سچ;
                device->mul_mat_s[i]    = سچ;
                device->mul_mat_id_l[i] = خطأ;
                device->mul_mat_id_m[i] = سچ;
                device->mul_mat_id_s[i] = سچ;
                اكسر;
            حالة VK_VENDOR_ID_INTEL: {
                // Current Windows driver does not expose BF16 support.
                // We only want to use l_warptile if coopmat is available
                ثابت منطقي use_l_warptile = (i == GGML_TYPE_BF16) ? (device->coopmat_bf16_support && device->coopmat_support) : device->coopmat_support;
                device->mul_mat_l[i] = use_l_warptile;
                device->mul_mat_id_l[i] = use_l_warptile;
                device->mul_mat_m[i] = سچ;
                device->mul_mat_s[i] = سچ;
                device->mul_mat_id_m[i] = سچ;
                device->mul_mat_id_s[i] = سچ;
                اكسر;
            }
            حالة VK_VENDOR_ID_APPLE:
                device->mul_mat_l[i] = خطأ;
                device->mul_mat_m[i] = سچ;
                device->mul_mat_s[i] = خطأ;
                device->mul_mat_id_l[i] = خطأ;
                device->mul_mat_id_m[i] = سچ;
                device->mul_mat_id_s[i] = خطأ;
                اكسر;
            حالة VK_VENDOR_ID_QUALCOMM:
                device->mul_mat_l[i] = خطأ;
                device->mul_mat_m[i] = سچ;
                device->mul_mat_s[i] = !device->coopmat_support;
                device->mul_mat_id_l[i] = خطأ;
                device->mul_mat_id_m[i] = سچ;
                device->mul_mat_id_s[i] = !device->coopmat_support;
                اكسر;
            حالة VK_VENDOR_ID_SAMSUNG:
                device->mul_mat_l[i] = خطأ;
                device->mul_mat_m[i] = سچ;
                device->mul_mat_s[i] = سچ;
                device->mul_mat_id_l[i] = خطأ;
                device->mul_mat_id_m[i] = سچ;
                device->mul_mat_id_s[i] = سچ;
                اكسر;
#جے_انت
            افتراضي:
                device->mul_mat_l[i] = سچ;
                device->mul_mat_m[i] = سچ;
                device->mul_mat_s[i] = سچ;
                device->mul_mat_id_l[i] = سچ;
                device->mul_mat_id_m[i] = سچ;
                device->mul_mat_id_s[i] = سچ;
                اكسر;
            }

#إذا VK_سرخی_نسخہ >= 287
            // Honeykrisp driver for Asahi Linux doesn't report VK_VENDOR_ID_APPLE.
            // Check for Honeykrisp driver and force same configuration as the VK_VENDOR_ID_APPLE case.
            إذا (device->driver_id == vk::DriverId::eMesaHoneykrisp) {
                device->mul_mat_l[i] = خطأ;
                device->mul_mat_m[i] = سچ;
                device->mul_mat_s[i] = خطأ;
                device->mul_mat_id_l[i] = خطأ;
                device->mul_mat_id_m[i] = سچ;
                device->mul_mat_id_s[i] = خطأ;
            }
#جے_انت

            device->mul_mat_l_int[i]    = device->mul_mat_l[i];
            device->mul_mat_m_int[i]    = device->mul_mat_m[i];
            device->mul_mat_s_int[i]    = device->mul_mat_s[i];
            device->mul_mat_id_l_int[i] = device->mul_mat_id_l[i];
            device->mul_mat_id_m_int[i] = device->mul_mat_id_m[i];
            device->mul_mat_id_s_int[i] = device->mul_mat_id_s[i];
        }


        std::vector<vk::Descriptor_سیٹ_Layout_Binding> dsl_binding;
        std::vector<vk::DescriptorBindingFlags> dsl_binding_flags;
        لكل (uint32_t i = 0; i < MAX_PARAMETER_گنتی; i++) {
            dsl_binding.push_back({i, vk::DescriptorType::e_Storage_بفر, 1, vk::Shader_Stage_Flag_بٹ::eCompute});
            dsl_binding_flags.push_back({});
        }

        vk::Descriptor_سیٹ_Layout_Binding_Flags_Create_Info dslbfci = { dsl_binding_flags };

        vk::Descriptor_سیٹ_Layout_Create_Info descriptor_سیٹ_layout_create_info(
            {},
            dsl_binding);
        descriptor_سیٹ_layout_create_info.سیٹ_PNext(&dslbfci);
        device->dsl = device->device.create_Descriptor_سیٹ_Layout(descriptor_سیٹ_layout_create_info);

        ggml_vk_لدو_shaders(device);

        // Prefer a dedicated transfer queue on AMD dGPUs (non-GCN) when graphics queue use is disabled.
        ثابت منطقي prefers_transfer_قطار =
            device->vendor_id == VK_VENDOR_ID_AMD &&
            device->architecture != AMD_GCN &&
            !device->uma &&
            !allow_graphics_قطار;

        إذا (!device->single_قطار) {
            ثابت uint32_t transfer_قطار_اشاریہ = compute_قطار_family_اشاریہ == transfer_قطار_family_اشاریہ ? 1 : 0;
            device->transfer_قطار = ggml_vk_create_قطار(device, transfer_قطار_family_اشاریہ, transfer_قطار_اشاریہ, { vk::Pipeline_Stage_Flag_بٹ::eTransfer }, سچ);

            device->async_use_transfer_قطار = prefers_transfer_قطار || (getenv("GGML_VK_ASYNC_USE_TRANSFER_QUEUE") != خالی_سوچک);
        } وإلا {
            device->transfer_قطار = ggml_vk_create_aliased_قطار(device, device->compute_قطار);

            device->async_use_transfer_قطار = خطأ;
        }

        device->fence = device->device.createFence({});

        device->بفر_type = {
            /* .iface    = */ ggml_backend_vk_بفر_type_interface,
            /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), idx),
            /* .context  = */ جديد ggml_backend_vk_بفر_type_context{ device->name, device },
        };

        device->idx = idx;

        device->serialize_submissions = getenv("GGML_VK_SERIALIZE_SUBMISSIONS") != خالی_سوچک;

        device->disable_fusion = getenv("GGML_VK_DISABLE_FUSION") != خالی_سوچک;

        device->disable_descriptor_reuse = getenv("GGML_VK_DISABLE_DESCRIPTOR_REUSE") != خالی_سوچک;

        device->جوڑو_rms_fusion = !device->disable_fusion &&
                                 device->subgroup_arithmetic &&
                                 device->vendor_id != VK_VENDOR_ID_INTEL;
        device->partials_binding_alignment =
            std::max(4u, (uint32_t)device->properties.limits.min_Storage_بفر_وتھ_Alignment);

        device->mmvq_منوال = 0;
        إذا (getenv("GGML_VK_DISABLE_MMVQ")) {
            device->mmvq_منوال = -1;
        } وإلا إذا (getenv("GGML_VK_FORCE_MMVQ")) {
            device->mmvq_منوال = 1;
        }

        // publish the device only when it is fully built
        vk_instance.devices[idx] = device;

        أرجع device;
    }

    أرجع vk_instance.devices[idx];
}

ساكن فراغ ggml_vk_print_gpu_info(size_t idx) {
    GGML_ASSERT(idx < vk_instance.device_indices.size());
    size_t dev_عدد = vk_instance.device_indices[idx];
    VK_LOG_DEBUG("ggml_vk_print_gpu_info(" << dev_عدد << ")");
    GGML_ASSERT(vk_instance_initialized);

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    إذا (dev_عدد >= devices.size()) {
        std::cerr << "ggml_vulkan: Device with index " << dev_عدد << " does not exist." << std::سطر_انت;
        ارم std::runtime_error("Device not found");
    }

    vk::PhysicalDevice physical_device = devices[dev_عدد];
    std::vector<vk::ExtensionProperties> ext_props = physical_device.enumerateDeviceExtensionProperties();

    منطقي fp16_storage = خطأ;
    منطقي fp16_compute = خطأ;
    منطقي coopmat_support = خطأ;
    منطقي coopmat2_support = خطأ;
    منطقي coopmat2_decode_سمتیہ_support = خطأ;
    منطقي integer_dot_ضرب_حاصل = خطأ;
    منطقي bfloat16_support = خطأ;
    منطقي dot2_f16_support = خطأ;
    منطقي ocp_microscaling_extension = خطأ;
    منطقي shader_float8_extension = خطأ;

    لكل (خودکار properties : ext_props) {
        إذا (strcmp("VK_KHR_16bit_storage", properties.extensionName) == 0) {
            fp16_storage = سچ;
        } وإلا إذا (strcmp("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
            fp16_compute = سچ;
#إذا defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
       } وإلا إذا (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT")) {
            coopmat_support = سچ;
#جے_انت
#إذا defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        } وإلا إذا (strcmp("VK_NV_cooperative_matrix2", properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT2")) {
            coopmat2_support = سچ;
#جے_انت
        } وإلا إذا (strcmp(VK_NV_COOPERATIVE_میٹرکس_DECODE_سمتیہ_EXTENSION_NAME, properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR")) {
            coopmat2_decode_سمتیہ_support = سچ;
#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        } وإلا إذا (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
            integer_dot_ضرب_حاصل = سچ;
#جے_انت
#إذا defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        } وإلا إذا (strcmp("VK_KHR_shader_bfloat16", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_BFLOAT16")) {
            bfloat16_support = سچ;
#جے_انت
#إذا defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT)
        } وإلا إذا (strcmp(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME, properties.extensionName) == 0) {
            ocp_microscaling_extension = سچ;
#جے_انت
#إذا defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        } وإلا إذا (strcmp(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, properties.extensionName) == 0) {
            shader_float8_extension = سچ;
#جے_انت
        } وإلا إذا (strcmp("VK_VALVE_shader_mixed_float_dot_product", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_DOT2")) {
            dot2_f16_support = سچ;
        }
    }

    ثابت vk_device_architecture device_architecture = get_device_architecture(physical_device);

    ثابت حرف* GGML_VK_DISABLE_F16 = getenv("GGML_VK_DISABLE_F16");
    منطقي force_disable_f16 = GGML_VK_DISABLE_F16 != خالی_سوچک;

    منطقي fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

    vk::PhysicalDeviceProperties2 props2;
    vk::PhysicalDeviceMaintenance3Properties props3;
    vk::PhysicalDeviceSubgroupProperties subgroup_props;
    vk::PhysicalDeviceDriverProperties driver_props;
    vk::Physical_Device_Shader_Integer_Dot_ضرب_حاصل_Properties_KHR shader_integer_dot_ضرب_حاصل_props;
    props2.p_اگلا = &props3;
    props3.p_اگلا = &subgroup_props;
    subgroup_props.p_اگلا = &driver_props;

    // Pointer to the last chain element
    Vk_ادھار_باہرلا_Structure * آخری_struct = (Vk_ادھار_باہرلا_Structure *)&driver_props;

    إذا (integer_dot_ضرب_حاصل) {
        آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&shader_integer_dot_ضرب_حاصل_props;
        آخری_struct = (Vk_ادھار_باہرلا_Structure *)&shader_integer_dot_ضرب_حاصل_props;
    }

    physical_device.getProperties2(&props2);

    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    device_features2.p_اگلا = خالی_سوچک;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.p_اگلا = خالی_سوچک;
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.p_اگلا = &vk11_features;

    VkPhysicalDeviceVulkan12Features vk12_features;
    vk12_features.p_اگلا = خالی_سوچک;
    vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk11_features.p_اگلا = &vk12_features;

    // Pointer to the last chain element
    آخری_struct = (Vk_ادھار_باہرلا_Structure *)&vk12_features;

#إذا defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    Vk_Physical_Device_Cooperative_میٹرکس_Features_KHR coopmat_features;
    coopmat_features.p_اگلا = خالی_سوچک;
    coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_میٹرکس_FEATURES_KHR;
    coopmat_features.cooperative_میٹرکس_2 = VK_FALSE;

    إذا (coopmat_support) {
        آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&coopmat_features;
        آخری_struct = (Vk_ادھار_باہرلا_Structure *)&coopmat_features;
    }
#جے_انت

    Vk_Physical_Device_Shader_Integer_Dot_ضرب_حاصل_Features_KHR shader_integer_dot_ضرب_حاصل_features {};
    shader_integer_dot_ضرب_حاصل_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_ضرب_حاصل_FEATURES_KHR;
    إذا (integer_dot_ضرب_حاصل) {
        آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&shader_integer_dot_ضرب_حاصل_features;
        آخری_struct = (Vk_ادھار_باہرلا_Structure *)&shader_integer_dot_ضرب_حاصل_features;
    }

#إذا defined(VK_KHR_shader_bfloat16)
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features {};
    bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
    إذا (bfloat16_support) {
        آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&bfloat16_features;
        آخری_struct = (Vk_ادھار_باہرلا_Structure *)&bfloat16_features;
    }
#جے_انت

#إذا defined(VK_NV_cooperative_matrix2)
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features {};
    coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_میٹرکس_2_FEATURES_NV;
    إذا (coopmat2_support) {
        آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&coopmat2_features;
        آخری_struct = (Vk_ادھار_باہرلا_Structure *)&coopmat2_features;
    }
#جے_انت

    Vk_Physical_Device_Cooperative_میٹرکس_Decode_سمتیہ_Features_NV coopmat2_decode_سمتیہ_features {};
    coopmat2_decode_سمتیہ_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_میٹرکس_DECODE_سمتیہ_FEATURES_NV;
    إذا (coopmat2_decode_سمتیہ_support) {
        آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&coopmat2_decode_سمتیہ_features;
        آخری_struct = (Vk_ادھار_باہرلا_Structure *)&coopmat2_decode_سمتیہ_features;
    }

    Vk_Physical_Device_Shader_Mixed_Float_Dot_ضرب_حاصل_Features_VALVE dot2_features {};
    dot2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_ضرب_حاصل_FEATURES_VALVE;
    إذا (dot2_f16_support) {
        آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&dot2_features;
        آخری_struct = (Vk_ادھار_باہرلا_Structure *)&dot2_features;
    }

#إذا defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT ocp_microscaling_features {};
    ocp_microscaling_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT;
    VkPhysicalDeviceShaderFloat8FeaturesEXT shader_float8_features {};
    shader_float8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
    إذا (ocp_microscaling_extension) {
        آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&ocp_microscaling_features;
        آخری_struct = (Vk_ادھار_باہرلا_Structure *)&ocp_microscaling_features;
    }
    إذا (shader_float8_extension) {
        آخری_struct->p_اگلا = (Vk_ادھار_باہرلا_Structure *)&shader_float8_features;
        آخری_struct = (Vk_ادھار_باہرلا_Structure *)&shader_float8_features;
    }
#جے_انت

    vkGetPhysicalDeviceFeatures2(physical_device, &device_features2);

    fp16 = fp16 && vk12_features.shaderFloat16;

#إذا defined(VK_KHR_shader_bfloat16)
    منطقي bf16 = bfloat16_support && bfloat16_features.shaderBFloat16Type;
#وإلا
    منطقي bf16 = خطأ;
#جے_انت

    uint32_t default_subgroup_ناپ = get_subgroup_ناپ("", device_architecture);
    ثابت size_t subgroup_ناپ = (default_subgroup_ناپ != 0) ? default_subgroup_ناپ : subgroup_props.subgroup_ناپ_2;
    ثابت منطقي uma = props2.properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

    integer_dot_ضرب_حاصل = integer_dot_ضرب_حاصل
                       && shader_integer_dot_ضرب_حاصل_props.integer_Dot_Product4x8_ثنائی_ہندسہ_Packed_Signed_Accelerated
                       && shader_integer_dot_ضرب_حاصل_features.shader_Integer_Dot_ضرب_حاصل;

    coopmat_support = coopmat_support
#إذا defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                   && coopmat_features.cooperative_میٹرکس_2
#جے_انت
                   && ggml_vk_khr_cooperative_میٹرکس_support(props2.properties, driver_props, device_architecture);

#إذا defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    coopmat2_support = coopmat2_support &&
                       coopmat2_features.cooperative_میٹرکس_Workgroup_Scope &&
                       coopmat2_features.cooperative_میٹرکس_Flexible_Dimensions &&
                       coopmat2_features.cooperative_میٹرکس_Reductions &&
                       coopmat2_features.cooperative_میٹرکس_Conversions &&
                       coopmat2_features.cooperative_میٹرکس_فی_Element_Operations &&
                       coopmat2_features.cooperative_میٹرکس_Tensor_Addressing &&
                       coopmat2_features.cooperative_میٹرکس_بلاک_Loads;
#وإلا
    coopmat2_support = خطأ;
#جے_انت

    coopmat2_decode_سمتیہ_support = coopmat2_decode_سمتیہ_support && coopmat2_decode_سمتیہ_features.cooperative_میٹرکس_Decode_سمتیہ;
#إذا !defined(GGML_VULKAN_COOPMAT2_DECODE_سمتیہ_GLSLC_SUPPORT)
    coopmat2_decode_سمتیہ_support = خطأ;
#جے_انت

    std::string میٹرکس_cores = coopmat2_support ? (coopmat2_decode_سمتیہ_support ? "NV_coopmat2v" : "NV_coopmat2")
                             : coopmat_support  ? "KHR_coopmat"
                             : "none";

    منطقي dot2_f16 = dot2_f16_support && dot2_features.shader_Mixed_Float_Dot_ضرب_حاصل_Float16_Acc_Float32;
    ثابت حرف *fp16_str = fp16 ? (dot2_f16 ? "dot2" : "1") : "0";
#إذا defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    ثابت منطقي fp4 = ocp_microscaling_extension && ocp_microscaling_features.shaderFloat4 &&
                     shader_float8_extension && shader_float8_features.shaderFloat8 &&
                     !getenv("GGML_VK_DISABLE_OCP_FP4");
#وإلا
    GGML_بن_ورتیا(ocp_microscaling_extension);
    GGML_بن_ورتیا(shader_float8_extension);
    ثابت منطقي fp4 = خطأ;
#جے_انت

    std::string device_name = props2.properties.deviceName.انکڑے();
    GGML_LOG_DEBUG("ggml_vulkan: %zu = %s (%s) | uma: %d | fp16: %s | bf16: %d | fp4: %d | warp size: %zu | shared memory: %d | int dot: %d | matrix cores: %s\n",
              idx, device_name.c_str(), driver_props.driverName.انکڑے(), uma, fp16_str, bf16, fp4, subgroup_ناپ,
              props2.properties.limits.max_Compute_Shared_حافظہ_ناپ, integer_dot_ضرب_حاصل, میٹرکس_cores.c_str());

    إذا (props2.properties.deviceType == vk::PhysicalDeviceType::eCpu) {
        GGML_LOG_DEBUG("ggml_vulkan: Warning: Device type is CPU. This is probably not the device you want.\n");
    }
}

ساكن DispatchLoaderDynamic ggml_vk_default_dispatcher_instance;

DispatchLoaderDynamic & ggml_vk_default_dispatcher() {
    أرجع ggml_vk_default_dispatcher_instance;
}

فراغ ggml_vk_instance_init() {
    إذا (vk_instance_initialized) {
        أرجع;
    }
    VK_LOG_DEBUG("ggml_vk_instance_init()");

    // See https://github.com/KhronosGroup/Vulkan-Hpp?tab=readme-ov-file#extensions--per-device-function-pointers-
    ggml_vk_default_dispatcher_instance.init(vkGetInstanceProcAddr);

    // vkEnumerateInstanceVersion is Vulkan 1.1. A null value indicated Vulkan 1.0.
    إذا (ggml_vk_default_dispatcher_instance.vk_Enumerate_Instance_نسخہ == خالی_سوچک) {
        GGML_LOG_ERROR("ggml_vulkan: Error: Vulkan 1.2 required.");
        ارم vk::SystemError(vk::نتیجہ_2::eErrorFeatureNotPresent, "Vulkan 1.2 required");
    }

    uint32_t api_نسخہ = vk::enumerate_Instance_نسخہ();

    إذا (api_نسخہ < VK_API_نسخہ_1_2) {
        GGML_LOG_ERROR("ggml_vulkan: Error: Vulkan 1.2 required.");
        ارم vk::SystemError(vk::نتیجہ_2::eErrorFeatureNotPresent, "Vulkan 1.2 required");
    }

    vk::ApplicationInfo app_info{ "ggml-vulkan", 1, خالی_سوچک, 0, api_نسخہ };

    ثابت std::vector<vk::ExtensionProperties> instance_extensions = vk::enumerateInstanceExtensionProperties();
    ثابت منطقي layer_settings = ggml_vk_instance_layer_settings_دستیاب();
#جے_تعریف_شدہ __APPLE__
    ثابت منطقي portability_enumeration_ext = ggml_vk_instance_portability_enumeration_ext_دستیاب(instance_extensions);
#جے_انت
    ثابت منطقي debug_utils_ext = ggml_vk_instance_debug_utils_ext_دستیاب(instance_extensions) && getenv("GGML_VK_DEBUG_MARKERS") != خالی_سوچک;
    std::vector<ثابت حرف*> layers;

    إذا (layer_settings) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }
    std::vector<ثابت حرف*> extensions;
    إذا (layer_settings) {
        extensions.push_back("VK_EXT_layer_settings");
    }
#جے_تعریف_شدہ __APPLE__
    إذا (portability_enumeration_ext) {
        extensions.push_back("VK_KHR_portability_enumeration");
    }
#جے_انت
    إذا (debug_utils_ext) {
        extensions.push_back("VK_EXT_debug_utils");
    }
    VkBool32 enable_بہترین_practice = layer_settings;
    std::vector<vk::LayerSettingEXT> settings = {
        {
            "VK_LAYER_KHRONOS_validation",
            "validate_best_practices",
            vk::LayerSettingTypeEXT::eBool32,
            1,
            &enable_بہترین_practice
        },
    };
    vk::LayerSettingsCreateInfoEXT layer_setting_info(settings);
    vk::InstanceCreateInfo instance_create_info(vk::InstanceCreateFlags{}, &app_info, layers, extensions, &layer_setting_info);
#جے_تعریف_شدہ __APPLE__
    إذا (portability_enumeration_ext) {
        instance_create_info.flags |= vk::Instance_Create_Flag_بٹ::eEnumeratePortabilityKHR;
    }
#جے_انت

    vk_instance.instance = vk::createInstance(instance_create_info);
    vk_instance_initialized = سچ;

    إذا (debug_utils_ext) {
        vk_instance.debug_utils_support              = سچ;
        vk_instance.pfn_vk_سیٹ_Debug_Utils_Object_Name_EXT = (PFN_vk_سیٹ_Debug_Utils_Object_Name_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkSetDebugUtilsObjectNameEXT");
        vk_instance.pfn_vk_قطار_Begin_Debug_Utils_Label_EXT = (PFN_vk_قطار_Begin_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkQueueBeginDebugUtilsLabelEXT");
        vk_instance.pfn_vk_قطار_سرا_Debug_Utils_Label_EXT = (PFN_vk_قطار_سرا_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkQueueEndDebugUtilsLabelEXT");
        vk_instance.pfn_vkCmdBeginDebugUtilsLabelEXT = (PFN_vkCmdBeginDebugUtilsLabelEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdBeginDebugUtilsLabelEXT");
        vk_instance.pfn_vk_Cmd_سرا_Debug_Utils_Label_EXT =   (PFN_vk_Cmd_سرا_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdEndDebugUtilsLabelEXT");
        vk_instance.pfn_vk_Cmd_داخل_کرو_Debug_Utils_Label_EXT = (PFN_vk_Cmd_داخل_کرو_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdInsertDebugUtilsLabelEXT");
    }

    vk_perf_logger_enabled = getenv("GGML_VK_PERF_LOGGER") != خالی_سوچک;
    vk_perf_logger_concurrent = getenv("GGML_VK_PERF_LOGGER_CONCURRENT") != خالی_سوچک;
    vk_enable_sync_logger = getenv("GGML_VK_SYNC_LOGGER") != خالی_سوچک;
    vk_حافظہ_logger_enabled = getenv("GGML_VK_MEMORY_LOGGER") != خالی_سوچک;
    ثابت حرف* GGML_VK_PIPELINE_STATS = getenv("GGML_VK_PIPELINE_STATS");
    إذا (GGML_VK_PIPELINE_STATS != خالی_سوچک) {
        vk_pipeline_stats_filter = GGML_VK_PIPELINE_STATS;
    }
    ثابت حرف* GGML_VK_PERF_LOGGER_FREQUENCY = getenv("GGML_VK_PERF_LOGGER_FREQUENCY");

    إذا (GGML_VK_PERF_LOGGER_FREQUENCY != خالی_سوچک) {
        vk_perf_logger_frequency = std::stoul(GGML_VK_PERF_LOGGER_FREQUENCY);
    }

    // See https://github.com/KhronosGroup/Vulkan-Hpp?tab=readme-ov-file#extensions--per-device-function-pointers-
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_instance.instance);

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    // Emulate behavior of CUDA_VISIBLE_DEVICES for Vulkan
    حرف * devices_env = getenv("GGML_VK_VISIBLE_DEVICES");
    إذا (devices_env != خالی_سوچک) {
        size_t عدد_دستیاب_devices = devices.size();

        std::string devices(devices_env);
        std::replace(devices.begin(), devices.end(), ',', ' ');

        std::stringstream ss(devices);
        size_t tmp;
        طالما (ss >> tmp) {
            إذا(tmp >= عدد_دستیاب_devices) {
                std::cerr << "ggml_vulkan: Invalid device index " << tmp << " in GGML_VK_VISIBLE_DEVICES." << std::سطر_انت;
                ارم std::runtime_error("Invalid Vulkan device index");
            }
            vk_instance.device_indices.push_back(tmp);
        }
    } وإلا {
        // If no vulkan devices are found, return early
        إذا (devices.خالی_تھاواں()) {
            GGML_LOG_INFO("ggml_vulkan: No devices found.\n");
            أرجع;
        }

        // Default to using all dedicated GPUs
        لكل (size_t i = 0; i < devices.size(); i++) {
            vk::PhysicalDeviceProperties2 new_props;
            vk::PhysicalDeviceDriverProperties new_driver;
            vk::PhysicalDeviceIDProperties new_id;
            new_props.p_اگلا = &new_driver;
            new_driver.p_اگلا = &new_id;
            devices[i].getProperties2(&new_props);

            إذا ((new_props.properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu || new_props.properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) && ggml_vk_device_is_supported(devices[i])) {
                // Check if there are two physical devices corresponding to the same GPU
                // This handles the case where the same GPU appears with different drivers (e.g., RADV + AMDVLK on Linux),
                // see https://github.com/ggml-org/llama.cpp/pull/7582 for original deduplication.
                // MoltenVK on macOS may report the same UUID for distinct GPUs on multi-GPU cards,
                // see https://github.com/KhronosGroup/MoltenVK/issues/2683. Skip when both old/new
                // driver is MoltenVK
                خودکار old_device = std::find_if(
                    vk_instance.device_indices.begin(),
                    vk_instance.device_indices.end(),
                    [&devices, &new_id, &new_driver](ثابت size_t k){
                        vk::PhysicalDeviceProperties2 old_props;
                        vk::PhysicalDeviceDriverProperties old_driver;
                        vk::PhysicalDeviceIDProperties old_id;
                        old_props.p_اگلا = &old_driver;
                        old_driver.p_اگلا = &old_id;
                        devices[k].getProperties2(&old_props);

                        منطقي same_uuid = std::equal(std::begin(old_id.deviceUUID), std::end(old_id.deviceUUID), std::begin(new_id.deviceUUID));
                        same_uuid = same_uuid || (
                            old_id.deviceLUIDValid && new_id.deviceLUIDValid &&
                            std::equal(std::begin(old_id.deviceLUID), std::end(old_id.deviceLUID), std::begin(new_id.deviceLUID))
                        );
                        منطقي both_molten_vk = (new_driver.driverID == vk::DriverId::eMoltenvk && old_driver.driverID == vk::DriverId::eMoltenvk);

                        أرجع same_uuid && !both_molten_vk;
                    }
                );
                إذا (old_device == vk_instance.device_indices.end()) {
                    vk_instance.device_indices.push_back(i);
                } وإلا {
                    // There can be two physical devices corresponding to the same GPU if there are 2 different drivers
                    // This can cause error when splitting layers aross the devices, need to keep only 1
                    VK_LOG_DEBUG("Device " << i << " and device " << *old_device << " have the same deviceUUID");

                    vk::PhysicalDeviceProperties2 old_props;
                    vk::PhysicalDeviceDriverProperties old_driver;
                    old_props.p_اگلا = &old_driver;
                    devices[*old_device].getProperties2(&old_props);

                    std::map<vk::DriverId, صحيح> driver_priorities {};
                    صحيح old_ترجیح = std::numeric_limits<صحيح>::max();
                    صحيح new_ترجیح = std::numeric_limits<صحيح>::max();

                    // Check https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDriverId.html for the list of driver id
                    // Smaller number -> higher priority
                    بدل (old_props.properties.vendorID) {
                        حالة VK_VENDOR_ID_AMD:
                            driver_priorities[vk::DriverId::eMesaRadv] = 1;
                            driver_priorities[vk::DriverId::e_Amd_Open_سروت] = 2;
                            driver_priorities[vk::DriverId::eAmdProprietary] = 3;
                            اكسر;
                        حالة VK_VENDOR_ID_INTEL:
                            driver_priorities[vk::DriverId::e_Intel_Open_سروت_MESA] = 1;
                            driver_priorities[vk::DriverId::eIntelProprietaryWindows] = 2;
                            اكسر;
                        حالة VK_VENDOR_ID_NVIDIA:
                            driver_priorities[vk::DriverId::eNvidiaProprietary] = 1;
#إذا defined(VK_API_نسخہ_1_3) && VK_سرخی_نسخہ >= 235
                            driver_priorities[vk::DriverId::eMesaNvk] = 2;
#جے_انت
                            اكسر;
                        حالة VK_VENDOR_ID_QUALCOMM:
                            driver_priorities[vk::DriverId::eQualcommProprietary] = 1;
                            driver_priorities[vk::DriverId::eMesaTurnip] = 2;
                            اكسر;
                    }
                    driver_priorities[vk::DriverId::eMesaDozen] = 100;

                    إذا (driver_priorities.count(old_driver.driverID)) {
                        old_ترجیح = driver_priorities[old_driver.driverID];
                    }
                    إذا (driver_priorities.count(new_driver.driverID)) {
                        new_ترجیح = driver_priorities[new_driver.driverID];
                    }

                    إذا (new_ترجیح < old_ترجیح) {
                        خودکار r = std::remove(vk_instance.device_indices.begin(), vk_instance.device_indices.end(), *old_device);
                        vk_instance.device_indices.erase(r, vk_instance.device_indices.end());
                        vk_instance.device_indices.push_back(i);

                        VK_LOG_DEBUG("Prioritize device " << i << " driver " << new_driver.driverName << " over device " << *old_device << " driver " << old_driver.driverName);
                    }
                    وإلا {
                        VK_LOG_DEBUG("Prioritize device " << *old_device << " driver " << old_driver.driverName << " over device " << i << " driver " << new_driver.driverName << std::سطر_انت);
                    }
                }
            }
        }

        // If no GPUs found, fall back to the first non-CPU device.
        // If only CPU devices are available, return without devices.
        إذا (vk_instance.device_indices.خالی_تھاواں()) {
            لكل (size_t i = 0; i < devices.size(); i++) {
                إذا (devices[i].getProperties().deviceType != vk::PhysicalDeviceType::eCpu) {
                    vk_instance.device_indices.push_back(i);
                    اكسر;
                }
            }
        }

        إذا (vk_instance.device_indices.خالی_تھاواں()) {
            GGML_LOG_INFO("ggml_vulkan: No devices found.\n");
            أرجع;
        }
    }
    GGML_LOG_DEBUG("ggml_vulkan: Found %zu Vulkan devices:\n", vk_instance.device_indices.size());

    size_t host_بفر_import_گنتی = 0;
    لكل (size_t i = 0; i < vk_instance.device_indices.size(); i++) {
        vk::PhysicalDevice vkdev = devices[vk_instance.device_indices[i]];
        std::vector<vk::ExtensionProperties> extensionprops = vkdev.enumerateDeviceExtensionProperties();

        منطقي membudget_supported = خطأ;
        منطقي external_حافظہ_host_supported = خطأ;
        لكل (ثابت خودکار & ext : extensionprops) {
            إذا (strcmp(VK_EXT_حافظہ_BUDGET_EXTENSION_NAME, ext.extensionName) == 0) {
                membudget_supported = سچ;
            } وإلا إذا (strcmp("VK_EXT_external_memory_host", ext.extensionName) == 0) {
                external_حافظہ_host_supported = سچ;
            }
        }

        vk_instance.device_supports_membudget.push_back(membudget_supported);

        منطقي host_بفر_import = خطأ;
        إذا (external_حافظہ_host_supported) {
            vk::PhysicalDeviceProperties2 props2;
            vk::PhysicalDeviceDriverProperties driver_props;
            vk::Physical_Device_External_حافظہ_Host_Properties_EXT ext_props;
            props2.p_اگلا = &driver_props;
            driver_props.p_اگلا = &ext_props;
            vkdev.getProperties2(&props2);
            host_بفر_import = ggml_vk_external_حافظہ_host_usable(driver_props.driverID) &&
                                 ggml_vk_host_بفر_import_usable(props2.properties.vendorID);
            إذا (host_بفر_import) {
                vk_instance.host_import_alignment = std::max<size_t>(vk_instance.host_import_alignment, ext_props.min_Imported_Host_اشارہ_گر_Alignment);
                host_بفر_import_گنتی++;
            }
        }
        vk_instance.device_host_بفر_import.push_back(host_بفر_import);

        ggml_vk_print_gpu_info(i);
    }

    vk_instance.multi_device = host_بفر_import_گنتی > 1 && vk_instance.device_host_بفر_import[0];
}

فراغ ggml_vk_init(ggml_backend_vk_context * ctx, size_t idx) {
    VK_LOG_DEBUG("ggml_vk_init(" << ctx->name << ", " << idx << ")");
    ggml_vk_instance_init();
    GGML_ASSERT(idx < vk_instance.device_indices.size());

    ctx->name = GGML_VK_NAME + std::to_string(idx);

    ctx->device = ggml_vk_get_device(idx);

    ctx->semaphore_idx = 0;
    ctx->event_idx = 0;

    ctx->prealloc_ناپ_x = 0;
    ctx->prealloc_ناپ_y = 0;
    ctx->prealloc_ناپ_ونڈو_k = 0;
    // Fixed size of 1KB, for deterministic behavior
    ctx->prealloc_ناپ_جوڑو_rms_partials = 1024;

    ctx->fence = ctx->device->device.createFence({});
    ctx->almost_ready_fence = ctx->device->device.createFence({});

    ctx->compute_cmd_pool.init(ctx->device, ctx->device->compute_قطار.get());
    إذا (ctx->device->async_use_transfer_قطار) {
        vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
        vk::SemaphoreCreateInfo ci{};
        ci.سیٹ_PNext(&tci);
        ctx->transfer_semaphore.s = ctx->device->device.createSemaphore(ci);
        ctx->transfer_semaphore.value = 0;

        ctx->transfer_cmd_pool.init(ctx->device, ctx->device->transfer_قطار.get());
    }

    إذا (vk_perf_logger_enabled) {
        ctx->perf_logger = std::unique_ptr<vk_perf_logger>(جديد vk_perf_logger());
    }

#جے_تعریف_شدہ GGML_VULKAN_CHECK_RESULTS
    ثابت حرف* skip_checks = getenv("GGML_VULKAN_SKIP_CHECKS");
    vk_skip_checks = (skip_checks == عدم ? 0 : atoi(skip_checks));
    ثابت حرف* output_tensor = getenv("GGML_VULKAN_OUTPUT_TENSOR");
    vk_output_tensor = (output_tensor == عدم ? 0 : atoi(output_tensor));
#جے_انت
}

vk_pipeline ggml_vk_get_منزل_fp16(ggml_backend_vk_context * ctx, ggml_type type) {
    VK_LOG_DEBUG("ggml_vk_get_to_fp16()");
    بدل (type) {
        حالة GGML_TYPE_F32:
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_IQ1_S:
        حالة GGML_TYPE_IQ1_M:
        حالة GGML_TYPE_IQ2_XXS:
        حالة GGML_TYPE_IQ2_XS:
        حالة GGML_TYPE_IQ2_S:
        حالة GGML_TYPE_IQ3_XXS:
        حالة GGML_TYPE_IQ3_S:
        حالة GGML_TYPE_IQ4_XS:
        حالة GGML_TYPE_IQ4_NL:
        حالة GGML_TYPE_MXFP4:
        حالة GGML_TYPE_NVFP4:
        حالة GGML_TYPE_TQ1_0:
        حالة GGML_TYPE_TQ2_0:
            اكسر;
        افتراضي:
            أرجع خالی_سوچک;
    }

    أرجع ctx->device->pipeline_dequant[type];
}

ساكن vk_pipeline ggml_vk_get_dequantize_mul_mat_vec(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type, uint32_t عدد_خانے_گنتی, uint32_t m, uint32_t k) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec()");
    GGML_ASSERT(b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_F16 || b_type == GGML_TYPE_Q8_1);
    GGML_ASSERT(عدد_خانے_گنتی >= 1 && عدد_خانے_گنتی <= mul_mat_vec_max_خانے_گنتی);

    إذا (b_type == GGML_TYPE_Q8_1) {
        بدل (a_type) {
            حالة GGML_TYPE_Q2_0:
            حالة GGML_TYPE_Q4_0:
            حالة GGML_TYPE_Q4_1:
            حالة GGML_TYPE_Q5_0:
            حالة GGML_TYPE_Q5_1:
            حالة GGML_TYPE_Q8_0:
            حالة GGML_TYPE_MXFP4:
            حالة GGML_TYPE_Q2_K:
            حالة GGML_TYPE_Q3_K:
            حالة GGML_TYPE_Q4_K:
            حالة GGML_TYPE_Q5_K:
            حالة GGML_TYPE_Q6_K:
            حالة GGML_TYPE_IQ1_S:
            حالة GGML_TYPE_IQ1_M:
            حالة GGML_TYPE_IQ4_XS:
                اكسر;
            افتراضي:
                أرجع خالی_سوچک;
        }
    }

    بدل (a_type) {
        حالة GGML_TYPE_F32:
        حالة GGML_TYPE_F16:
        حالة GGML_TYPE_BF16:
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_IQ1_S:
        حالة GGML_TYPE_IQ1_M:
        حالة GGML_TYPE_IQ2_XXS:
        حالة GGML_TYPE_IQ2_XS:
        حالة GGML_TYPE_IQ2_S:
        حالة GGML_TYPE_IQ3_XXS:
        حالة GGML_TYPE_IQ3_S:
        حالة GGML_TYPE_IQ4_XS:
        حالة GGML_TYPE_IQ4_NL:
        حالة GGML_TYPE_MXFP4:
        حالة GGML_TYPE_NVFP4:
        حالة GGML_TYPE_TQ1_0:
        حالة GGML_TYPE_TQ2_0:
            اكسر;
        افتراضي:
            أرجع خالی_سوچک;
    }

    // heuristic to choose workgroup size
    uint32_t dmmv_wg = DMMV_WG_ناپ_SUBGROUP;
    إذا ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA && ctx->device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) || ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Prefer larger workgroups when M is small, to spread the work out more
        // and keep more SMs busy.
        // q6_k seems to prefer small workgroup size even for "medium" values of M.
        إذا (a_type == GGML_TYPE_Q6_K) {
            إذا (m < 4096 && k >= 1024) {
                dmmv_wg = DMMV_WG_ناپ_LARGE;
            }
        } وإلا {
            إذا (m <= 8192 && k >= 1024) {
                dmmv_wg = DMMV_WG_ناپ_LARGE;
            }
        }
    }

    إذا (b_type == GGML_TYPE_Q8_1) {
        إذا (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
            dmmv_wg = DMMV_WG_ناپ_SUBGROUP;
        }
        أرجع ctx->device->pipeline_dequant_mul_mat_vec_q8_1_f32[dmmv_wg][a_type][عدد_خانے_گنتی-1];
    }

    أرجع b_type == GGML_TYPE_F32 ? ctx->device->pipeline_dequant_mul_mat_vec_f32_f32[dmmv_wg][a_type][عدد_خانے_گنتی-1] : ctx->device->pipeline_dequant_mul_mat_vec_f16_f32[dmmv_wg][a_type][عدد_خانے_گنتی-1];
}

ساكن vk_pipeline ggml_vk_get_dequantize_mul_mat_vec_id(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type, uint32_t m, uint32_t k) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec_id()");
    GGML_ASSERT(b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_Q8_1);

    إذا (b_type == GGML_TYPE_Q8_1) {
        بدل (a_type) {
            حالة GGML_TYPE_Q2_0:
            حالة GGML_TYPE_Q4_0:
            حالة GGML_TYPE_Q4_1:
            حالة GGML_TYPE_Q5_0:
            حالة GGML_TYPE_Q5_1:
            حالة GGML_TYPE_Q8_0:
            حالة GGML_TYPE_MXFP4:
            حالة GGML_TYPE_Q2_K:
            حالة GGML_TYPE_Q3_K:
            حالة GGML_TYPE_Q4_K:
            حالة GGML_TYPE_Q5_K:
            حالة GGML_TYPE_Q6_K:
            حالة GGML_TYPE_IQ1_S:
            حالة GGML_TYPE_IQ1_M:
            حالة GGML_TYPE_IQ4_XS:
                اكسر;
            افتراضي:
                أرجع خالی_سوچک;
        }
    }

    بدل (a_type) {
        حالة GGML_TYPE_F32:
        حالة GGML_TYPE_F16:
        حالة GGML_TYPE_BF16:
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_IQ1_S:
        حالة GGML_TYPE_IQ1_M:
        حالة GGML_TYPE_IQ2_XXS:
        حالة GGML_TYPE_IQ2_XS:
        حالة GGML_TYPE_IQ2_S:
        حالة GGML_TYPE_IQ3_XXS:
        حالة GGML_TYPE_IQ3_S:
        حالة GGML_TYPE_IQ4_XS:
        حالة GGML_TYPE_IQ4_NL:
        حالة GGML_TYPE_MXFP4:
        حالة GGML_TYPE_NVFP4:
        حالة GGML_TYPE_TQ1_0:
        حالة GGML_TYPE_TQ2_0:
            اكسر;
        افتراضي:
            أرجع خالی_سوچک;
    }

    // heuristic to choose workgroup size
    uint32_t dmmv_wg = DMMV_WG_ناپ_SUBGROUP;
    إذا ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA && ctx->device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) || ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Prefer larger workgroups when M is small, to spread the work out more
        // and keep more SMs busy.
        // q6_k seems to prefer small workgroup size even for "medium" values of M.
        إذا (a_type == GGML_TYPE_Q6_K) {
            إذا (m < 4096 && k >= 1024) {
                dmmv_wg = DMMV_WG_ناپ_LARGE;
            }
        } وإلا {
            إذا (m <= 8192 && k >= 1024) {
                dmmv_wg = DMMV_WG_ناپ_LARGE;
            }
        }
    }

    إذا (b_type == GGML_TYPE_Q8_1) {
        إذا (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
            dmmv_wg = DMMV_WG_ناپ_SUBGROUP;
        }
        أرجع ctx->device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[dmmv_wg][a_type];
    }

    أرجع ctx->device->pipeline_dequant_mul_mat_vec_id_f32[dmmv_wg][a_type];
}

vk_subbuffer ggml_vk_tensor_subbuffer(
    ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * tensor, منطقي allow_misalign) {

    vk_بفر بفر = خالی_سوچک;
    size_t وتھ = 0;
    إذا (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, tensor->انکڑے, بفر, وتھ);
    }
    إذا (!بفر) {
        خودکار buf_ctx = (ggml_backend_vk_بفر_context *)tensor->بفر->context;
        بفر = buf_ctx->dev_بفر;
        وتھ = vk_tensor_وتھ(tensor) + tensor->view_offs;
    }
    GGML_ASSERT(بفر != خالی_سوچک);

    size_t size = ggml_nbytes(tensor);

    ثابت size_t descriptor_وتھ = ggml_vk_descriptor_وتھ(
        وتھ, ctx->device->properties.limits.min_Storage_بفر_وتھ_Alignment, ggml_type_ناپ(tensor->type));
    ثابت size_t misalign_بائٹاں = وتھ - descriptor_وتھ;
    // The shader must support misaligned offsets when indexing into the buffer
    GGML_ASSERT(allow_misalign || misalign_بائٹاں == 0);
    وتھ = descriptor_وتھ;
    size += misalign_بائٹاں;

    أرجع vk_subbuffer{بفر, وتھ, size};
}

ساكن vk_command_بفر* ggml_vk_get_or_create_cmd_بفر(vk_device& device, vk_command_pool& pool) {
    لكل (خودکار& cmd_بفر : pool.cmd_buffers) {
        إذا (!cmd_بفر.وچ_use) {
            cmd_بفر.use_counter++;
            cmd_بفر.وچ_use = سچ;
            أرجع &cmd_بفر;
        }
    }
    أرجع ggml_vk_create_cmd_بفر(device, pool);
}

ساكن vk_submission ggml_vk_begin_submission(vk_device& device, vk_command_pool& p, منطقي one_ویلا = سچ) {
    vk_submission s;
    s.بفر = ggml_vk_get_or_create_cmd_بفر(device, p);
    إذا (one_ویلا) {
        s.بفر->buf.begin({ vk::Command_بفر_Usage_Flag_بٹ::e_One_ویلا_Submit });
    } وإلا {
        s.بفر->buf.begin({ vk::Command_بفر_Usage_Flags{} });
    }

    أرجع s;
}

فراغ ggml_vk_cmd_label_begin(vk::Command_بفر buf, ثابت حرف * name) {
    vk::DebugUtilsLabelEXT label = {};
    label.pLabelName = name;
    label.color = std::array<عائم, 4>{1.0f, 1.0f, 1.0f, 1.0f};
    vk_instance.pfn_vkCmdBeginDebugUtilsLabelEXT(buf, reinterpret_cast<VkDebugUtilsLabelEXT *>(&label));
}

فراغ ggml_vk_ctx_سرا(vk_context& ctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_end(" << ctx << ", " << ctx->seqs.size() << ")");
    إذا (ctx->s == خالی_سوچک) {
        أرجع;
    }

    // close open labels so this buffer is balanced; reopened in ggml_vk_ctx_begin
    إذا (vk_instance.debug_utils_support) {
        لكل (size_t i = 0; i < ctx->debug_labels.size(); i++) {
            vk_instance.pfn_vk_Cmd_سرا_Debug_Utils_Label_EXT(ctx->s->بفر->buf);
        }
        // the enclosing per-command-buffer region
        vk_instance.pfn_vk_Cmd_سرا_Debug_Utils_Label_EXT(ctx->s->بفر->buf);
    }

    ctx->s->بفر->buf.end();
    ctx->s = خالی_سوچک;
}

فراغ ggml_vk_ctx_begin(vk_device& device, vk_context& subctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_begin(" << device->name << ")");
    إذا (subctx->s != خالی_سوچک) {
        ggml_vk_ctx_سرا(subctx);
    }

    subctx->seqs.push_back({ ggml_vk_begin_submission(device, *subctx->p) });
    subctx->s = subctx->seqs[subctx->seqs.size() - 1].انکڑے();

    إذا (vk_instance.debug_utils_support) {
        // outermost region, one per command buffer, so the gaps between submits stand out
        ثابت std::string name = "submit " + std::to_string(device->debug_cmdbuf_idx++);
        ggml_vk_cmd_label_begin(subctx->s->بفر->buf, name.c_str());

        // reopen labels left open when the previous command buffer was submitted
        لكل (ثابت std::string & label : subctx->debug_labels) {
            ggml_vk_cmd_label_begin(subctx->s->بفر->buf, label.c_str());
        }
    }
}

vk_context ggml_vk_get_compute_ctx(ggml_backend_vk_context * ctx) {
    vk_context نتیجہ;
    إذا (!ctx->compute_ctx.expired()) {
        نتیجہ = ctx->compute_ctx.جندرا();
    } وإلا {
        نتیجہ = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);

        ctx->compute_ctx = نتیجہ;
        ggml_vk_ctx_begin(ctx->device, نتیجہ);
    }

    إذا (ctx->device->async_use_transfer_قطار && ctx->transfer_semaphore_آخری_submitted < ctx->transfer_semaphore.value) {
        نتیجہ->s->wait_semaphores.push_back(ctx->transfer_semaphore);
        ctx->transfer_semaphore_آخری_submitted = ctx->transfer_semaphore.value;
    }

    أرجع نتیجہ;
}

vk_context ggml_vk_get_transfer_ctx(ggml_backend_vk_context * ctx) {
    vk_context نتیجہ;
    إذا (!ctx->transfer_ctx.expired()) {
        نتیجہ = ctx->transfer_ctx.جندرا();
    } وإلا {
        نتیجہ = ggml_vk_create_context(ctx, ctx->transfer_cmd_pool);

        ctx->transfer_ctx = نتیجہ;
        ggml_vk_ctx_begin(ctx->device, نتیجہ);
    }

    أرجع نتیجہ;
}

منطقي ggml_vk_submit_transfer_ctx(ggml_backend_vk_context * ctx) {
    إذا (!ctx->device->async_use_transfer_قطار || ctx->transfer_ctx.expired()) {
        أرجع خطأ;
    }

    vk_context cpy_ctx = ctx->transfer_ctx.جندرا();
    ggml_vk_ctx_سرا(cpy_ctx);

    لكل (خودکار& cpy : cpy_ctx->وچ_memcpys) {
        حافظہ_نقل(cpy.dst, cpy.src, cpy.n);
    }

    ctx->transfer_semaphore.value++;
    cpy_ctx->seqs.back().back().signal_semaphores.push_back(ctx->transfer_semaphore);

    ggml_vk_submit(cpy_ctx, {});
    ctx->transfer_ctx.reset();
    أرجع سچ;
}

size_t ggml_vk_align_ناپ(size_t چوڑائی, size_t align) {
    VK_LOG_DEBUG("ggml_vk_align_size(" << چوڑائی << ", " << align << ")");
    أرجع CEIL_DIV(چوڑائی, align) * align;
}

فراغ deferred_memcpy(فراغ * dst, ثابت فراغ * src, size_t size, std::vector<vk_staging_memcpy>* memcpys) {
    إذا (memcpys == خالی_سوچک) {
        حافظہ_نقل(dst, src, size);
    } وإلا {
        memcpys->emplace_back(dst, src, size);
    }
}

فراغ deferred_memset(فراغ * dst, uint32_t val, size_t size, std::vector<vk_staging_memset>* memsets) {
    إذا (memsets == خالی_سوچک) {
        memset(dst, val, size);
    } وإلا {
        memsets->emplace_back(dst, val, size);
    }
}

ساكن uint32_t ggml_vk_guess_ونڈو_k(ggml_backend_vk_context * ctx, uint32_t m, uint32_t n, uint32_t k, منطقي disable_ونڈو_k, ثابت vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_vk_guess_split_k(" << m << ", " << n << ", " << k << ", " << disable_ونڈو_k << ")");

    إذا (disable_ونڈو_k) {
        أرجع 1;
    }

    uint32_t ونڈو_k = 1;
    إذا (ctx->device->shader_core_گنتی != 0 && n >= pipeline->wg_denoms[1]) {
        // If k is 'large' and the SMs will fill less than halfway, use split_k.
        uint32_t m_tiles = CEIL_DIV(m, pipeline->wg_denoms[0]);
        uint32_t n_tiles = CEIL_DIV(n, pipeline->wg_denoms[1]);

        إذا (k >= 2048) {
            إذا (m_tiles * n_tiles <= ctx->device->shader_core_گنتی / 2) {
                ونڈو_k = ctx->device->shader_core_گنتی / (m_tiles * n_tiles);
            } وإلا إذا (m_tiles * n_tiles <= ctx->device->shader_core_گنتی * 2 / 3) {
                ونڈو_k = 3;
            }
            // Cap the split at 8x. Unless k is huge this is a lot of overhead.
            ونڈو_k = std::min(ونڈو_k, 8u);

            // ggml_vk_matmul will align the splits to be a multiple of 256.
            // If this rounded up size would cause the last split to be empty,
            // then reduce the split count.
            طالما (سچ) {
                إذا (ونڈو_k == 1) {
                    اكسر;
                }
                uint32_t k_ونڈو = CEIL_DIV(k, ونڈو_k);
                k_ونڈو = ROUNDUP_POW2(k_ونڈو, 256);
                إذا (k_ونڈو * (ونڈو_k - 1) < k) {
                    اكسر;
                }
                ونڈو_k--;
            }
        }
    }

    أرجع ونڈو_k;
}

فراغ ggml_vk_matmul(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& ونڈو_k_بفر,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t ونڈو_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3,
        uint32_t padded_n, uint64_t b_خانہ_بائٹاں, uint64_t d_خانہ_بائٹاں) {
        VK_LOG_DEBUG("ggml_vk_matmul(a: (" << a.بفر->بفر << ", " << a.وتھ << ", " << a.size << "), b: (" << b.بفر->بفر << ", " << b.وتھ << ", " << b.size << "), d: (" << d.بفر->بفر << ", " << d.وتھ << ", " << d.size << "), split_k: (" << (ونڈو_k_بفر.بفر != خالی_سوچک ? ونڈو_k_بفر.بفر->بفر : VK_NULL_HANDLE) << ", " << ونڈو_k_بفر.وتھ << ", " << ونڈو_k_بفر.size << "), m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", split_k: " << ونڈو_k << ", batch: " << batch << ", ne02: " << ne02 << ", ne12: " << ne12 << ", broadcast2: " << broadcast2 << ", broadcast3: " << broadcast3 << ", padded_n: " << padded_n << ")");
    إذا (ونڈو_k == 1) {
        // a small tile can still produce more workgroups than the device allows on the n axis:
        // split the dispatch into n-chunks, shifting the B and D column windows
        ثابت uint64_t max_n = (uint64_t)ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[1] * pipeline->wg_denoms[1];
        ثابت uint64_t n_chunks = CEIL_DIV(n, max_n);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, n_chunks * CEIL_DIV(batch, ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[2]));

        لكل (uint64_t n_ادھار = 0; n_ادھار < n; n_ادھار += max_n) {
            ثابت uint32_t n_chunk = (uint32_t) std::min(n - n_ادھار, max_n);
            GGML_ASSERT(n_ادھار * b_خانہ_بائٹاں < b.size && n_ادھار * d_خانہ_بائٹاں < d.size);
            GGML_ASSERT(b.size != VK_سارا_ناپ && d.size != VK_سارا_ناپ);
            vk_subbuffer b_chunk = b;
            vk_subbuffer d_chunk = d;
            b_chunk.وتھ += n_ادھار * b_خانہ_بائٹاں;
            b_chunk.size   -= n_ادھار * b_خانہ_بائٹاں;
            d_chunk.وتھ += n_ادھار * d_خانہ_بائٹاں;
            d_chunk.size   -= n_ادھار * d_خانہ_بائٹاں;

            uint32_t ادھار_work_گروہ_z = 0;
            طالما (ادھار_work_گروہ_z < batch) {
                uint32_t groups_z = std::min(batch - ادھار_work_گروہ_z, ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[2]);

                ثابت vk_mat_mat_ڈالو_constants pc = { m, n_chunk, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, ادھار_work_گروہ_z, batch, k, ne02, ne12, broadcast2, broadcast3, uint32_t(padded_n - n_ادھار) };
                ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b_chunk, d_chunk }, pc, { m, n_chunk, groups_z });
                ادھار_work_گروہ_z += groups_z;
            }
        }
        أرجع;
    }

    إذا (ctx->prealloc_ونڈو_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    GGML_ASSERT(batch_stride_d == m * n);

    // Round the split size up to a multiple of 256 (k-quant alignment)
    uint32_t k_ونڈو = CEIL_DIV(k, ونڈو_k);
    k_ونڈو = ROUNDUP_POW2(k_ونڈو, 256);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, CEIL_DIV(batch, ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[2]));

    uint32_t ادھار_work_گروہ_z = 0;
    طالما (ادھار_work_گروہ_z < batch) {
        uint32_t groups_z = std::min(batch - ادھار_work_گروہ_z, ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[2]);

        ثابت vk_mat_mat_ڈالو_constants pc1 = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, ادھار_work_گروہ_z, batch, k_ونڈو, ne02, ne12, broadcast2, broadcast3, padded_n };
        // Make sure enough workgroups get assigned for split k to work
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, ونڈو_k_بفر }, pc1, { (CEIL_DIV(m, pipeline->wg_denoms[0]) * pipeline->wg_denoms[0]) * ونڈو_k, n, groups_z });
        ادھار_work_گروہ_z += groups_z;
    }
    ggml_vk_sync_buffers(ctx, subctx);
    ثابت std::array<uint32_t, 2> pc2 = { (uint32_t)(m * n * batch), ونڈو_k };
    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_matmul_ونڈو_k_reduce, { ونڈو_k_بفر, d }, pc2, { m * n * batch, 1, 1 });
    ctx->prealloc_ونڈو_k_need_sync = سچ;
}

ساكن منطقي ggml_vk_get_mul_mat_mat_f16acc(ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec) {
    إذا (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16) أرجع خطأ;
    إذا (src1_type == GGML_TYPE_Q8_1) أرجع خطأ;
    إذا (src0_type == GGML_TYPE_F16) {
        أرجع prec == GGML_PREC_DEFAULT && ctx->device->fp16 && !(ctx->device->coopmat_support && !ctx->device->coopmat_acc_f16_support);
    }
    // quant types
    إذا (ctx->device->coopmat2) {
        أرجع prec == GGML_PREC_DEFAULT;
    }
    إذا (ctx->device->coopmat_support) {
        أرجع ctx->device->fp16 && ctx->device->coopmat_acc_f16_support && prec == GGML_PREC_DEFAULT;
    }
    أرجع ctx->device->fp16 && prec == GGML_PREC_DEFAULT;
}

ساكن ثابت std::vector<vk_matmul_pipeline_pair>* ggml_vk_get_mul_mat_mat_pipeline_map(
        ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec, منطقي mul_mat_id = خطأ) {
    منطقي f16acc = ggml_vk_get_mul_mat_mat_f16acc(ctx, src0_type, src1_type, prec);
    vk_matmul_pipeline_کلید کلید{src0_type, src1_type, mul_mat_id, f16acc};
    خودکار it = ctx->device->pipeline_matmul.find(کلید);
    إذا (it == ctx->device->pipeline_matmul.end() || it->second.خالی_تھاواں()) {
        // Try without f16acc
        إذا (f16acc) {
            کلید.f16acc = خطأ;
            it = ctx->device->pipeline_matmul.find(کلید);
            إذا (it != ctx->device->pipeline_matmul.end() && !it->second.خالی_تھاواں()) أرجع &it->second;
        }
        أرجع خالی_سوچک;
    }
    أرجع &it->second;
}

ساكن vk_pipeline ggml_vk_guess_matmul_pipeline_map(ggml_backend_vk_context * ctx,
        ثابت std::vector<vk_matmul_pipeline_pair>& configs,
        uint32_t m, uint32_t n, منطقي aligned, منطقي mul_mat_id) {
    خودکار& selector = mul_mat_id ? ctx->device->matmul_id_tile_selector : ctx->device->matmul_tile_selector;
    uint32_t idx = selector(m, n, 0, ctx->device->shader_core_گنتی, configs);
    إذا (idx >= configs.size()) idx = (uint32_t)configs.size() - 1;
    أرجع (aligned && configs[idx].aligned) ? configs[idx].aligned : configs[idx].unaligned;
}

ساكن uint32_t ggml_vk_guess_matmul_pipeline_align_map(ggml_backend_vk_context * ctx,
        ثابت std::vector<vk_matmul_pipeline_pair>& configs,
        uint32_t m, uint32_t n, منطقي mul_mat_id) {
    خودکار& selector = mul_mat_id ? ctx->device->matmul_id_tile_selector : ctx->device->matmul_tile_selector;
    uint32_t idx = selector(m, n, 0, ctx->device->shader_core_گنتی, configs);
    إذا (idx >= configs.size()) idx = (uint32_t)configs.size() - 1;
    أرجع configs[idx].align;
}

ساكن uint64_t ggml_vk_mul_mat_id_max_tiles(uint64_t n_experts, uint64_t n_صفاں, uint32_t bn) {
    // Each nonempty expert can add at most one partial tile to the combined grid.
    أرجع std::min(n_صفاں, CEIL_DIV(n_صفاں, bn) + std::min(n_experts, n_صفاں) - 1);
}

ساكن فراغ ggml_vk_matmul_id(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& ids, ثابت vk_subbuffer & expert_گنتی_buf,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t n_as, uint32_t nei0, uint32_t nei1, uint32_t nbi1, uint32_t ne11,
        uint32_t صف_ids_وتھ, uint32_t max_tiles) {
    VK_LOG_DEBUG("ggml_vk_matmul_id(a: (" << a.بفر->بفر << ", " << a.وتھ << ", " << a.size << "), b: (" << b.بفر->بفر << ", " << b.وتھ << ", " << b.size << "), d: (" << d.بفر->بفر << ", " << d.وتھ << ", " << d.size << "), ids: (" << ids.بفر->بفر << ", " << ids.وتھ << ", " << ids.size << "), expert_count: (" << expert_گنتی_buf.بفر->بفر << ", " << expert_گنتی_buf.وتھ << ", " << expert_گنتی_buf.size << "), " <<
        "m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", " <<
        "batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", " <<
        "n_as: " << n_as << ", nei0: " << nei0 << ", nei1: " << nei1 << ", nbi1: " << nbi1 << ", ne11: " << ne11 << ")");
    ثابت vk_mat_mat_id_ڈالو_constants pc = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d,
                                              nei0, nei1, nbi1, ne11, n_as, صف_ids_وتھ };
    ثابت uint32_t tiles_y = std::min(max_tiles, ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[1]);
    ثابت std::array<uint32_t, 3> elements = { m, tiles_y * pipeline->wg_denoms[1], uint32_t(CEIL_DIV(max_tiles, tiles_y)) };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, d, ids, expert_گنتی_buf }, pc, elements);
}

منطقي ggml_vk_dim01_contiguous(ثابت ggml_tensor * tensor) {
    أرجع
        tensor->nb[0] == ggml_type_ناپ(tensor->type) &&
        tensor->nb[1] == (tensor->nb[0]*tensor->ne[0])/ggml_blck_ناپ(tensor->type) &&
        (tensor->ne[3] == 1 || tensor->nb[3] == tensor->nb[2]*tensor->ne[2]);
}

// Batch stride in elements of a tensor read in place.
ساكن uint32_t ggml_vk_batch_stride(ثابت ggml_tensor * tensor) {
    أرجع (uint32_t)(tensor->nb[2] / ggml_type_ناپ(tensor->type) * ggml_blck_ناپ(tensor->type));
}

vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, ثابت ggml_tensor * src, ثابت ggml_tensor * dst, ggml_type منزل) {

    // Choose "contiguous copy" shader if src/dst are contiguous
    منطقي contig = ggml_is_contiguous(src) && (!dst || ggml_is_contiguous(dst));

    // Use optimized "transpose" shader if src dim1 is the innermost dimension.
    منطقي transpose = dst && src->nb[1] == ggml_type_ناپ(منزل) && ggml_are_same_shape(dst, src);

    إذا (transpose && src->type == منزل) {
        إذا (ggml_type_ناپ(منزل) == 4) {
            أرجع ctx->device->pipeline_cpy_transpose_32;
        } وإلا إذا (ggml_type_ناپ(منزل) == 2) {
            أرجع ctx->device->pipeline_cpy_transpose_16;
        }
    }

    // Same, for a 0<->2 swap: src dim2 is the innermost dimension.
    منطقي transpose02 = dst && !contig && src->nb[2] == ggml_type_ناپ(منزل) &&
                       ggml_is_contiguous(dst) && ggml_are_same_shape(dst, src);

    إذا (transpose02 && src->type == منزل) {
        إذا (ggml_type_ناپ(منزل) == 4) {
            أرجع ctx->device->pipeline_cpy_transpose_02_32;
        } وإلا إذا (ggml_type_ناپ(منزل) == 2) {
            أرجع ctx->device->pipeline_cpy_transpose_02_16;
        }
    }

    إذا (src->type == GGML_TYPE_F32 && منزل == GGML_TYPE_F16) {
        إذا (contig) {
            أرجع ctx->device->pipeline_contig_cpy_f32_f16;
        } وإلا {
            أرجع ctx->device->pipeline_cpy_f32_f16;
        }
    }
    إذا (src->type == GGML_TYPE_F16 && منزل == GGML_TYPE_F32) {
        إذا (contig) {
            أرجع ctx->device->pipeline_contig_cpy_f16_f32;
        } وإلا {
            أرجع ctx->device->pipeline_cpy_f16_f32;
        }
    }
    إذا (src->type == GGML_TYPE_F32 && منزل == GGML_TYPE_BF16) {
        إذا (contig) {
            أرجع ctx->device->pipeline_contig_cpy_f32_bf16;
        } وإلا {
            أرجع ctx->device->pipeline_cpy_f32_bf16;
        }
    }
    إذا (src->type == GGML_TYPE_BF16 && منزل == GGML_TYPE_F32) {
        إذا (contig) {
            أرجع ctx->device->pipeline_contig_cpy_bf16_f32;
        } وإلا {
            أرجع ctx->device->pipeline_cpy_bf16_f32;
        }
    }
    إذا (src->type == GGML_TYPE_BF16 && منزل == GGML_TYPE_F16) {
        إذا (contig) {
            أرجع ctx->device->pipeline_contig_cpy_bf16_f16;
        } وإلا {
            أرجع ctx->device->pipeline_cpy_bf16_f16;
        }
    }
    إذا (src->type == GGML_TYPE_F32 && منزل == GGML_TYPE_I32) {
        إذا (contig) {
            أرجع ctx->device->pipeline_contig_cpy_f32_i32;
        } وإلا {
            أرجع ctx->device->pipeline_cpy_f32_i32;
        }
    }
    إذا (src->type == GGML_TYPE_I32 && منزل == GGML_TYPE_F32) {
        إذا (contig) {
            أرجع ctx->device->pipeline_contig_cpy_i32_f32;
        } وإلا {
            أرجع ctx->device->pipeline_cpy_i32_f32;
        }
    }
    إذا (src->type == GGML_TYPE_F32) {
        بدل (منزل) {
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_IQ4_NL:
            أرجع ctx->device->pipeline_cpy_f32_quant[منزل];
        افتراضي:
            اكسر;
        }
    }

    إذا (منزل == GGML_TYPE_F32) {
        بدل (src->type) {
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_IQ4_NL:
            أرجع ctx->device->pipeline_cpy_quant_f32[src->type];
        افتراضي:
            اكسر;
        }
    }

    إذا (src->type == منزل) {
        // Copy two or four bytes at a time, depending on block size.
        // For quantized types, we scale by block size/type size. But
        // this path is also used for bf16->bf16 for example, where the
        // type size must be exactly 2 or 4.
        GGML_ASSERT(ggml_is_quantized(منزل) || ggml_type_ناپ(src->type) == 2 || ggml_type_ناپ(src->type) == 4);
        إذا ((ggml_type_ناپ(src->type) % 4) == 0) {
            إذا (contig) {
                أرجع ctx->device->pipeline_contig_cpy_32;
            } وإلا {
                أرجع ctx->device->pipeline_cpy_32;
            }
        } وإلا {
            إذا (contig) {
                أرجع ctx->device->pipeline_contig_cpy_16;
            } وإلا {
                أرجع ctx->device->pipeline_cpy_16;
            }
        }
    }

    std::cerr << "Missing CPY op for types: " << ggml_type_name(src->type) << " " << ggml_type_name(منزل) << std::سطر_انت;
    GGML_ABORT("fatal error");
}

فراغ ggml_vk_cpy_منزل_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, ثابت ggml_tensor * tensor, ثابت vk_subbuffer & وچ, ثابت vk_subbuffer & باہرلا) {
    VK_LOG_DEBUG("ggml_vk_cpy_to_contiguous((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::cerr << "buffer in size=" << وچ.بفر->size << ", buffer out size=" << باہرلا.بفر->size << ")");

    ثابت uint32_t ne = ggml_nelements(tensor);
    std::array<uint32_t, 3> elements;

    إذا (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } وإلا إذا (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } وإلا {
        elements = { ne, 1, 1 };
    }

    vk_op_unary_ڈالو_constants pc = vk_op_unary_ڈالو_constants_init(tensor, tensor, ne);
    pc.nb10 = 1;
    pc.nb11 = (uint32_t)tensor->ne[0];
    pc.nb12 = (uint32_t)(tensor->ne[0] * tensor->ne[1]);
    pc.nb13 = (uint32_t)(tensor->ne[0] * tensor->ne[1] * tensor->ne[2]);
    init_pushconst_fastdiv(pc);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { وچ, باہرلا }, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
}

ساكن فراغ ggml_vk_cpy_منزل_strided(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, ثابت ggml_tensor * tensor,
        ثابت vk_subbuffer & وچ, ثابت vk_subbuffer & باہرلا,
        uint32_t nb10, uint32_t nb11, uint32_t nb12, uint32_t nb13) {
    VK_LOG_DEBUG("ggml_vk_cpy_to_strided((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::cerr << "dst_nb=(" << nb10 << ", " << nb11 << ", " << nb12 << ", " << nb13 << "), buffer in size=" << وچ.بفر->size << ", buffer out size=" << باہرلا.بفر->size << ")");

    ثابت uint32_t ne = ggml_nelements(tensor);
    std::array<uint32_t, 3> elements;

    إذا (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } وإلا إذا (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } وإلا {
        elements = { ne, 1, 1 };
    }

    vk_op_unary_ڈالو_constants pc = vk_op_unary_ڈالو_constants_init(tensor, tensor, ne);
    pc.nb10 = nb10;
    pc.nb11 = nb11;
    pc.nb12 = nb12;
    pc.nb13 = nb13;
    init_pushconst_fastdiv(pc);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { وچ, باہرلا }, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
}

vk_pipeline ggml_vk_get_quantize_pipeline(ggml_backend_vk_context * ctx, ggml_type type) {
    بدل(type) {
        حالة GGML_TYPE_Q8_1:
            أرجع ctx->device->pipeline_quantize_q8_1_x4;
        افتراضي:
            std::cerr << "Missing quantize pipeline for type: " << ggml_type_name(type) << std::سطر_انت;
            GGML_ABORT("fatal error");
    }
}

فراغ ggml_vk_quantize_q8_1(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت vk_subbuffer & وچ, ثابت vk_subbuffer & باہرلا, uint32_t ne) {
    VK_LOG_DEBUG("ggml_vk_quantize_q8_1(" << "buffer in size=" << وچ.بفر->size << ", buffer out size=" << باہرلا.بفر->size << ", " << ne << ")");

    vk_pipeline pipeline = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);

    ثابت uint32_t عدد_بلاکاں = CEIL_DIV(ne, pipeline->wg_denoms[0]);
    // clamp the number of elements to the max workgroup count. The shader will iterate over the total number of blocks.
    ثابت uint64_t max_elements = std::min<uint64_t>(uint64_t{ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[0]} * pipeline->wg_denoms[0], std::numeric_limits<uint32_t>::max());
    ثابت uint32_t elements = std::min(ne, static_cast<uint32_t>(max_elements));

    ثابت vk_quantize_q8_1_ڈالو_constants pc = {
        ne,
        عدد_بلاکاں,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { وچ, باہرلا }, pc, { elements, 1, 1 });
    ggml_vk_sync_buffers(ctx, subctx);
}

ساكن vk_pipeline ggml_vk_get_64b_indexing_pipeline(ggml_backend_vk_context * ctx, vk_pipeline &pipeline) {
    GGML_بن_ورتیا(ctx);
#إذا defined(VK_EXT_shader_64bit_indexing)
    vk_pipeline *ptr = &pipeline;
    طالما (*ptr) {
        إذا ((*ptr)->is_64b_indexing) {
            أرجع *ptr;
        }
        ptr = &(*ptr)->اگلا;
    }
#جے_انت
    أرجع pipeline;
}

// perf-heuristic for choosing mul_mat b type, based on shader availability
ساكن ggml_type ggml_vk_mul_mat_b_type(
        ggml_backend_vk_context * ctx, ggml_type src0_type, ثابت ggml_tensor * src1,
        ggml_type f16_type, ggml_prec prec, منطقي mul_mat_id, منطقي can_quantize_y, منطقي prefer_f16_b) {
    خودکار دستیاب = [&](ggml_type b) {
        أرجع ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0_type, b, prec, mul_mat_id) != خالی_سوچک;
    };
    إذا (can_quantize_y && دستیاب(GGML_TYPE_Q8_1))             أرجع GGML_TYPE_Q8_1;
    إذا (prefer_f16_b   && دستیاب(f16_type))                   أرجع f16_type;
    إذا (ggml_vk_dim01_contiguous(src1) && دستیاب(src1->type)) أرجع src1->type;
    إذا (src1->type == GGML_TYPE_BF16 && دستیاب(GGML_TYPE_F32)) أرجع GGML_TYPE_F32;
    أرجع f16_type;
}

ساكن فراغ ggml_vk_mul_mat_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst, منطقي disable_ونڈو_k) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_q_f16((" << src0 << ", name=" << src0->name << ", type=" << ggml_type_name(src0->type) << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << ggml_type_name(src1->type) << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << ggml_type_name(dst->type) << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    ثابت uint64_t ne00 = src0->ne[0];
    ثابت uint64_t ne01 = src0->ne[1];
    ثابت uint64_t ne02 = src0->ne[2];
    ثابت uint64_t ne03 = src0->ne[3];

    ثابت uint64_t ne10 = src1->ne[0];
    ثابت uint64_t ne11 = src1->ne[1];
    ثابت uint64_t ne12 = src1->ne[2];
    ثابت uint64_t ne13 = src1->ne[3];

    ثابت uint64_t ne21 = dst->ne[1];
    ثابت uint32_t stride_d = dst->nb[1] / ggml_type_ناپ(dst->type);
    ثابت uint32_t stride_batch_d = stride_d*ne21;

    ثابت uint64_t r2 = ne12 / ne02;
    ثابت uint64_t r3 = ne13 / ne03;

    ggml_backend_vk_بفر_context * dst_buf_ctx = (ggml_backend_vk_بفر_context *)dst->بفر->context;
    ggml_backend_vk_بفر_context * src0_buf_ctx = (ggml_backend_vk_بفر_context *)src0->بفر->context;
    ggml_backend_vk_بفر_context * src1_buf_ctx = (ggml_backend_vk_بفر_context *)src1->بفر->context;

    vk_بفر d_Qx = خالی_سوچک;
    size_t qx_buf_وتھ = 0;
    vk_بفر d_Qy = خالی_سوچک;
    size_t qy_buf_وتھ = 0;

    منطقي src0_uma = خطأ;
    منطقي src1_uma = خطأ;

    إذا (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->انکڑے, d_Qx, qx_buf_وتھ);
        ggml_vk_host_get(ctx->device, src1->انکڑے, d_Qy, qy_buf_وتھ);
        src0_uma = d_Qx != خالی_سوچک;
        src1_uma = d_Qy != خالی_سوچک;
    }

    ثابت منطقي x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    // If src0 is BF16, try to use a BF16 x BF16 multiply
    ggml_type f16_type = src0->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;

    ثابت منطقي is_coopmat = ctx->device->coopmat_support || ctx->device->coopmat2;
    ثابت منطقي can_quantize_y = (ctx->device->integer_dot_ضرب_حاصل || ctx->device->coopmat_int_support) &&
                                src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && (ne11 * ne10) % 4 == 0;
    // coopmat shaders are fast enough that memory bandwidth reduction from f16 activations helps
    ثابت منطقي prefer_f16_b = is_coopmat && ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32;

    ثابت ggml_type y_kernel_type = ggml_vk_mul_mat_b_type(ctx, src0->type, src1, f16_type,
                                        (ggml_prec)dst->op_params[0], خطأ, can_quantize_y, prefer_f16_b);

    ثابت منطقي quantize_y       = y_kernel_type == GGML_TYPE_Q8_1;
    ثابت منطقي y_f32_kernel     = y_kernel_type == GGML_TYPE_F32;
    ثابت منطقي y_needs_reformat = !quantize_y && (!ggml_vk_dim01_contiguous(src1) || y_kernel_type != src1->type);

    ثابت std::vector<vk_matmul_pipeline_pair>* mmp_map =
        ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_kernel_type, (ggml_prec)dst->op_params[0]);

    ثابت منطقي qx_needs_dequant = mmp_map == خالی_سوچک || x_non_contig;
    ثابت منطقي qy_needs_dequant = y_needs_reformat;

    إذا (qx_needs_dequant) {
        // dequant src0 to f16
        mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, f16_type, y_kernel_type, (ggml_prec)dst->op_params[0]);
    }

    GGML_ASSERT(mmp_map != خالی_سوچک);

    ثابت uint32_t kpad = quantize_y ? 0 : ggml_vk_align_ناپ(ne10, ggml_vk_guess_matmul_pipeline_align_map(ctx, *mmp_map, ne01, ne11, خطأ));
    ثابت منطقي aligned = !quantize_y && ne10 == kpad && ne01 > 8 && ne11 > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline_map(ctx, *mmp_map, ne01, ne11, aligned, خطأ);

    إذا (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_بفر_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    // Reserve extra storage in the N dimension for the Y matrix, so we can avoid bounds-checking
    uint32_t padded_n = qy_needs_dequant ? ROUNDUP_POW2(ne11, pipeline->wg_denoms[1]) : ne11;
    ثابت uint64_t x_ne = ggml_nelements(src0);
    // 128 elements per Q8_1 x4 block
    ثابت uint64_t y_ne = padded_n * ne10 * ne12 * ne13;
    ثابت uint64_t d_ne = ggml_nelements(dst);

    ثابت uint32_t ونڈو_k = ggml_vk_guess_ونڈو_k(ctx, ne01, ne11, ne10, disable_ونڈو_k, pipeline);

    ثابت uint64_t qx_sz = ggml_type_ناپ(src0->type) * x_ne / ggml_blck_ناپ(src0->type);
    ثابت uint64_t qy_sz = ggml_type_ناپ(src1->type) * y_ne / ggml_blck_ناپ(src1->type);
    ثابت uint64_t x_sz = !qx_needs_dequant ? qx_sz : حجم(ggml_fp16_t) * x_ne;
    ثابت uint64_t y_sz = quantize_y ? (ggml_vk_align_ناپ(y_ne, 128) * ggml_type_ناپ(GGML_TYPE_Q8_1) / ggml_blck_ناپ(GGML_TYPE_Q8_1)) : (y_f32_kernel ? حجم(عائم) * y_ne : حجم(ggml_fp16_t) * y_ne);
    ثابت uint64_t d_sz = حجم(عائم) * d_ne;

    vk_pipeline منزل_fp16_vk_0 = خالی_سوچک;
    vk_pipeline منزل_fp16_vk_1 = خالی_سوچک;
    vk_pipeline منزل_q8_1 = خالی_سوچک;

    إذا (x_non_contig) {
        منزل_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, خالی_سوچک, f16_type);
    } وإلا {
        منزل_fp16_vk_0 = ggml_vk_get_منزل_fp16(ctx, src0->type);
    }
    إذا (y_needs_reformat) {
        منزل_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, خالی_سوچک, y_kernel_type);
    } وإلا {
        منزل_fp16_vk_1 = ggml_vk_get_منزل_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || منزل_fp16_vk_0 != خالی_سوچک);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || منزل_fp16_vk_1 != خالی_سوچک);  // NOLINT

    إذا (quantize_y) {
        منزل_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    {
        ثابت uint64_t ونڈو_k_ناپ = ونڈو_k > 1 ? d_sz * ونڈو_k : 0;
        إذا (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_بفر_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_بفر_Range) ||
                (ونڈو_k > 1 && ونڈو_k_ناپ > ctx->device->properties.limits.max_Storage_بفر_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        إذا (qx_needs_dequant && ctx->prealloc_ناپ_x < x_sz) {
            ctx->prealloc_ناپ_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا ((qy_needs_dequant || quantize_y) && ctx->prealloc_ناپ_y < y_sz) {
            ctx->prealloc_ناپ_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا (ونڈو_k > 1 && ctx->prealloc_ناپ_ونڈو_k < ونڈو_k_ناپ) {
            ctx->prealloc_ناپ_ونڈو_k = ونڈو_k_ناپ;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        إذا (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, منزل_fp16_vk_0, 1);
        }
        إذا (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, منزل_fp16_vk_1, 1);
        }
        إذا (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, منزل_q8_1, 1);
        }
        إذا (ونڈو_k > 1) {
            ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_ونڈو_k_reduce, 1);
        }
    }

    vk_بفر d_D = dst_buf_ctx->dev_بفر;
    ثابت uint64_t d_buf_وتھ = vk_tensor_وتھ(dst) + dst->view_offs;
    GGML_ASSERT(d_D != خالی_سوچک);
    GGML_ASSERT(d_D->size >= d_buf_وتھ + d_sz);
    vk_بفر d_X;
    uint64_t x_buf_وتھ = 0;
    vk_بفر d_Y;
    uint64_t y_buf_وتھ = 0;
    إذا (!src0_uma) {
        d_Qx = src0_buf_ctx->dev_بفر;
        qx_buf_وتھ = vk_tensor_وتھ(src0) + src0->view_offs;
        GGML_ASSERT(d_Qx != خالی_سوچک);
    }
    إذا (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_بفر;
        qy_buf_وتھ = vk_tensor_وتھ(src1) + src1->view_offs;
        GGML_ASSERT(d_Qy != خالی_سوچک);
    }
    إذا (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz);
    } وإلا {
        d_X = d_Qx;
        x_buf_وتھ = qx_buf_وتھ;
        GGML_ASSERT(qx_sz == x_sz);
    }
    إذا (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz);
    } وإلا إذا (quantize_y) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= CEIL_DIV(y_sz, 144) * 144);
    } وإلا {
        d_Y = d_Qy;
        y_buf_وتھ = qy_buf_وتھ;
        GGML_ASSERT(qy_sz == y_sz);
    }

    إذا (x_non_contig || qx_needs_dequant) {
        إذا (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    إذا (x_non_contig) {
        ggml_vk_cpy_منزل_contiguous(ctx, subctx, منزل_fp16_vk_0, src0, ggml_vk_subbuffer(ctx, d_Qx, qx_buf_وتھ), ggml_vk_subbuffer(ctx, d_X, 0));
    } وإلا إذا (qx_needs_dequant) {
        ثابت std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_dispatch_pipeline(ctx, subctx, منزل_fp16_vk_0, { vk_subbuffer{ d_Qx, qx_buf_وتھ, qx_sz }, vk_subbuffer{ d_X, 0, x_sz } }, pc, { (uint32_t)(x_ne), 1, 1});
        ggml_vk_sync_buffers(ctx, subctx);
    }
    إذا (y_needs_reformat) {
        إذا (ctx->prealloc_y_آخری_pipeline_ورتیا != منزل_fp16_vk_1.get() ||
            ctx->prealloc_y_آخری_tensor_ورتیا != src1 ||
            ctx->prealloc_y_آخری_k_padded) {
            إذا (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_منزل_contiguous(ctx, subctx, منزل_fp16_vk_1, src1, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_وتھ), ggml_vk_subbuffer(ctx, d_Y, 0));
            ctx->prealloc_y_آخری_pipeline_ورتیا = منزل_fp16_vk_1.get();
            ctx->prealloc_y_آخری_tensor_ورتیا = src1;
            ctx->prealloc_y_آخری_k_padded = خطأ;
        }
    }
    إذا (quantize_y) {
        إذا (ctx->prealloc_y_آخری_pipeline_ورتیا != منزل_q8_1.get() ||
            ctx->prealloc_y_آخری_tensor_ورتیا != src1 ||
            ctx->prealloc_y_آخری_k_padded) {
            إذا (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_وتھ), ggml_vk_subbuffer(ctx, d_Y, 0), y_ne);
            ctx->prealloc_y_آخری_pipeline_ورتیا = منزل_q8_1.get();
            ctx->prealloc_y_آخری_tensor_ورتیا = src1;
            ctx->prealloc_y_آخری_k_padded = خطأ;
        }
    }

    // The partial tiles of a quantized A rely on the bound range to read zeros past the last row,
    // so the range stays exact: the strided extent of a tensor read in place, the staged size otherwise.
    ثابت uint64_t x_range = qx_needs_dequant ? x_sz : ggml_nbytes(src0);
    ثابت uint64_t y_range = (qy_needs_dequant || quantize_y) ? y_sz : ggml_nbytes(src1);

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_batch_y = (qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1);

    إذا (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_ناپ(src0->type);
    }

    إذا (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant && !quantize_y) {
        stride_batch_y = src1->nb[0] / ggml_type_ناپ(src1->type);
    }

    // column stride in bytes of the B buffer: raw src1 keeps its own stride, staging buffers are tightly packed
    ثابت uint64_t b_خانہ_بائٹاں = (y_needs_reformat || qy_needs_dequant || quantize_y)
        ? (quantize_y ? (uint64_t)(ne10 / ggml_blck_ناپ(GGML_TYPE_Q8_1)) * ggml_type_ناپ(GGML_TYPE_Q8_1) : (uint64_t)ne10 * حجم(ggml_fp16_t))
        : (uint64_t)src1->nb[1];

    // compute
    ggml_vk_matmul(
        ctx, subctx, pipeline,
        { d_X, x_buf_وتھ, x_range }, { d_Y, y_buf_وتھ, y_range },
        ggml_vk_subbuffer(ctx, d_D, d_buf_وتھ), { ctx->prealloc_ونڈو_k, 0, d_sz * ونڈو_k },
        ne01, ne11, ne10,
        ne10, ne10, stride_d, stride_batch_x, stride_batch_y, stride_batch_d,
        ونڈو_k, ne12*ne13, ne02, ne12, r2, r3, padded_n,
        b_خانہ_بائٹاں, (uint64_t)dst->nb[1]
    );  // NOLINT

    إذا (x_non_contig || qx_needs_dequant) {
        ctx->prealloc_x_need_sync = سچ;
    }
    إذا (y_needs_reformat || quantize_y) {
        ctx->prealloc_y_need_sync = سچ;
    }
}

ساكن منطقي ggml_vk_should_use_mmvq(ثابت vk_device& device, uint32_t m, uint32_t n, uint32_t k, ggml_type src0_type) {
    إذا (device->mmvq_منوال == 1) {
        أرجع سچ;
    } وإلا إذا (device->mmvq_منوال == -1) {
        أرجع خطأ;
    }

    // q6_k only has 2-byte alignment which makes it somewhat problematic,
    // using MMVQ is only a win on Intel.
    منطقي mmvq_q6 = device->vendor_id == VK_VENDOR_ID_INTEL;
    إذا (src0_type == GGML_TYPE_Q6_K && !mmvq_q6) {
        أرجع خطأ;
    }

    // MMVQ is generally good for batches
    إذا (n > 1) {
        أرجع سچ;
    }

    // Quantization overhead is not worth it for small k
    بدل (device->vendor_id) {
    حالة VK_VENDOR_ID_NVIDIA:
        إذا (src0_type == GGML_TYPE_Q2_0 || src0_type == GGML_TYPE_Q2_K || src0_type == GGML_TYPE_Q3_K || src0_type == GGML_TYPE_IQ1_S || src0_type == GGML_TYPE_IQ1_M) {
            أرجع سچ;
        }

        إذا (k <= 4096) {
            أرجع خطأ;
        }

        بدل (src0_type) {
        حالة GGML_TYPE_MXFP4:
        حالة GGML_TYPE_Q8_0:
            أرجع device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        افتراضي:
            أرجع سچ;
        }
    حالة VK_VENDOR_ID_AMD:
        إذا (k < 2048) {
            أرجع خطأ;
        }

        بدل (src0_type) {
        حالة GGML_TYPE_Q8_0:
            أرجع device->architecture == vk_device_architecture::AMD_GCN;
        افتراضي:
            أرجع سچ;
        }
    حالة VK_VENDOR_ID_INTEL:
        إذا (device->architecture == vk_device_architecture::INTEL_XE2) {
            إذا (src0_type == GGML_TYPE_Q2_0 || src0_type == GGML_TYPE_Q2_K || src0_type == GGML_TYPE_Q3_K || src0_type == GGML_TYPE_Q6_K) {
                أرجع سچ;
            }
        }

        إذا (device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            // Intel Windows proprietary driver MMVQ performance for !Q2/Q3/Q6 is worse than fp16,
            // see https://github.com/ggml-org/llama.cpp/issues/17628 and
            // https://github.com/ggml-org/llama.cpp/pull/23056
            أرجع خطأ;
        }

        إذا (k < 2048) {
            أرجع خطأ;
        }

        بدل (src0_type) {
        // From tests on A770 Linux, may need more tuning
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_IQ4_XS:
            أرجع خطأ;
        افتراضي:
            أرجع سچ;
        }
    حالة VK_VENDOR_ID_QUALCOMM:
        أرجع خطأ;
    افتراضي:
        أرجع سچ;
    }

    GGML_بن_ورتیا(m);
}

ساكن فراغ ggml_vk_mul_mat_vec_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گرہ_idx, منطقي وٹاندرا_inputs = خطأ) {
    ggml_tensor * dst = cgraph->nodes[گرہ_idx];
    ثابت ggml_tensor * src0 = dst->src[وٹاندرا_inputs ? 1 : 0];
    ثابت ggml_tensor * src1 = dst->src[وٹاندرا_inputs ? 0 : 1];

    VK_LOG_DEBUG("ggml_vk_mul_mat_vec_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << ")),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    ثابت uint64_t ne00 = src0->ne[0];
    ثابت uint64_t ne01 = src0->ne[1];
    ثابت uint64_t ne02 = src0->ne[2];
    ثابت uint64_t ne03 = src0->ne[3];

    ثابت uint64_t ne10 = src1->ne[0];
    ثابت uint64_t ne11 = src1->ne[1];
    ثابت uint64_t ne12 = src1->ne[2];
    ثابت uint64_t ne13 = src1->ne[3];

    ثابت uint64_t ne20 = dst->ne[وٹاندرا_inputs ? 1 : 0];
    ثابت uint64_t ne21 = dst->ne[وٹاندرا_inputs ? 0 : 1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    ثابت uint64_t r2 = ne12 / ne02;
    ثابت uint64_t r3 = ne13 / ne03;

    // batch_n indicates that we need to compute a few vector results, and this assumes
    // ne12 and ne13 are 1. It overloads the batch_strides to hold the row strides.
    GGML_ASSERT(ne11 == 1 || ne12 * ne13 == 1);
    منطقي batch_n = ne11 > 1;

    ثابت منطقي x_non_contig = !ggml_vk_dim01_contiguous(src0);

    // BF16 src1 has no mul_mat_vec B-type; widen to F32
    ثابت منطقي widen_bf16 = src1->type == GGML_TYPE_BF16;
    ثابت منطقي y_non_contig = !ggml_vk_dim01_contiguous(src1) || widen_bf16;

    ثابت ggml_type y_kernel_type = widen_bf16 ? GGML_TYPE_F32 : src1->type;
    ثابت منطقي f16_f32_kernel = y_kernel_type == GGML_TYPE_F32;
    منطقي quantize_y = ctx->device->integer_dot_ضرب_حاصل && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0 && ggml_vk_should_use_mmvq(ctx->device, ne01, ne11, ne10, src0->type);

    vk_pipeline منزل_fp16_vk_0 = خالی_سوچک;
    vk_pipeline منزل_fp16_vk_1 = خالی_سوچک;
    إذا (x_non_contig) {
        منزل_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, خالی_سوچک, src0->type);
    }
    إذا (y_non_contig) {
        منزل_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, خالی_سوچک, y_kernel_type);
    } وإلا {
        منزل_fp16_vk_1 = ggml_vk_get_منزل_fp16(ctx, src1->type);
    }

    // Check for mmq first
    vk_pipeline dmmv = quantize_y ? ggml_vk_get_dequantize_mul_mat_vec(ctx, src0->type, GGML_TYPE_Q8_1, ne11, ne20, ne00) : خالی_سوچک;
    vk_pipeline منزل_q8_1 = خالی_سوچک;

    إذا (dmmv == خالی_سوچک) {
        // Fall back to f16 dequant mul mat
        dmmv = ggml_vk_get_dequantize_mul_mat_vec(ctx, src0->type, y_kernel_type, ne11, ne20, ne00);
        quantize_y = خطأ;
    }

    إذا (quantize_y) {
        منزل_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    إذا (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_بفر_Range) {
        dmmv = ggml_vk_get_64b_indexing_pipeline(ctx, dmmv);
    }

    ثابت منطقي qx_needs_dequant = x_non_contig;
    ثابت منطقي qy_needs_dequant = !quantize_y && ((src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig);

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT

    GGML_ASSERT(!qx_needs_dequant || منزل_fp16_vk_0 != خالی_سوچک);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || منزل_fp16_vk_1 != خالی_سوچک);  // NOLINT
    GGML_ASSERT(dmmv != خالی_سوچک);

    ثابت uint64_t x_ne = ggml_nelements(src0);
    ثابت uint64_t y_ne = ggml_nelements(src1);

    ثابت uint64_t qx_sz = ggml_vk_align_ناپ(ggml_type_ناپ(src0->type) * x_ne / ggml_blck_ناپ(src0->type), ctx->device->properties.limits.min_Storage_بفر_وتھ_Alignment);
    ثابت uint64_t x_sz = x_non_contig ? ggml_vk_align_ناپ(ggml_type_ناپ(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_بفر_وتھ_Alignment) : qx_sz;
    ثابت uint64_t y_sz = quantize_y ? (ggml_vk_align_ناپ(y_ne, 128) * ggml_type_ناپ(GGML_TYPE_Q8_1) / ggml_blck_ناپ(GGML_TYPE_Q8_1)) :
                         (f16_f32_kernel ? حجم(عائم) * y_ne : حجم(ggml_fp16_t) * y_ne);

    {
        إذا (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_بفر_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_بفر_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        إذا (qx_needs_dequant && ctx->prealloc_ناپ_x < x_sz) {
            ctx->prealloc_ناپ_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا ((qy_needs_dequant || quantize_y) && ctx->prealloc_ناپ_y < y_sz) {
            ctx->prealloc_ناپ_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        إذا (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, منزل_fp16_vk_0, 1);
        }
        إذا (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, منزل_fp16_vk_1, 1);
        }
        إذا (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, منزل_q8_1, 1);
        }
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[گرہ_idx + ctx->عدد_additional_fused_ops]);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer d_X, d_Y;

    إذا (qx_needs_dequant) {
        d_X = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    } وإلا {
        d_X = d_Qx;
        GGML_ASSERT(qx_sz == x_sz);
    }
    إذا (qy_needs_dequant || quantize_y) {
        d_Y = { ctx->prealloc_y, 0, ctx->prealloc_y->size };
    } وإلا {
        d_Y = d_Qy;
    }

    إذا (x_non_contig) {
        إذا (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        GGML_ASSERT(x_sz == ggml_vk_align_ناپ(ggml_type_ناپ(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_بفر_وتھ_Alignment));
        ggml_vk_cpy_منزل_contiguous(ctx, subctx, منزل_fp16_vk_0, src0, d_Qx, d_X);
    }
    إذا (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_type_ناپ(y_kernel_type) * y_ne);
        إذا (ctx->prealloc_y_آخری_pipeline_ورتیا != منزل_fp16_vk_1.get() ||
            ctx->prealloc_y_آخری_tensor_ورتیا != src1 ||
            ctx->prealloc_y_آخری_k_padded) {
            إذا (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_منزل_contiguous(ctx, subctx, منزل_fp16_vk_1, src1, d_Qy, d_Y);
            ctx->prealloc_y_آخری_pipeline_ورتیا = منزل_fp16_vk_1.get();
            ctx->prealloc_y_آخری_tensor_ورتیا = src1;
            ctx->prealloc_y_آخری_k_padded = خطأ;
        }
    }
    إذا (quantize_y) {
        إذا (ctx->prealloc_y_آخری_pipeline_ورتیا != منزل_q8_1.get() ||
            ctx->prealloc_y_آخری_tensor_ورتیا != src1 ||
            ctx->prealloc_y_آخری_k_padded) {
            إذا (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, d_Qy, d_Y, y_ne);
            ctx->prealloc_y_آخری_pipeline_ورتیا = منزل_q8_1.get();
            ctx->prealloc_y_آخری_tensor_ورتیا = src1;
            ctx->prealloc_y_آخری_k_padded = خطأ;
        }
    }

    // For batch_n, the A matrix is the same for each batch, and B/D use the row stride as the batch stride
    uint32_t stride_batch_x = batch_n ? 0 : (qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0));
    uint32_t stride_batch_y = batch_n ? ne10 : ((qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1));
    uint32_t stride_batch_d = batch_n ? ne20 : (ne20*ne21);

    إذا (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_ناپ(src0->type);
    }

    إذا (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[0] / ggml_type_ناپ(src1->type);
    }

    ثابت uint32_t max_groups_x = ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    إذا (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = CEIL_DIV(groups_x, groups_z);
    }

    uint32_t fusion_flags = 0;

    vk_subbuffer d_F0 = d_D;
    إذا (ctx->عدد_additional_fused_ops > 0) {
        ثابت ggml_tensor * جوڑو_2 = cgraph->nodes[گرہ_idx + 1];
        ثابت ggml_tensor * bias = جوڑو_2->src[0] == dst ? جوڑو_2->src[1] : جوڑو_2->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    إذا (ctx->عدد_additional_fused_ops == 2) {
        ثابت ggml_tensor * جوڑو_2 = cgraph->nodes[گرہ_idx + 2];
        ثابت ggml_tensor * bias = جوڑو_2->src[0] == cgraph->nodes[گرہ_idx + 1] ? جوڑو_2->src[1] : جوڑو_2->src[0];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    ggml_pipeline_request_descriptor_sets(ctx, dmmv, CEIL_DIV(ne12 * ne13, ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[1]));

    uint32_t ادھار_work_گروہ_y = 0;
    طالما (ادھار_work_گروہ_y < ne12 * ne13) {

        uint32_t groups_y = std::min((uint32_t)(ne12 * ne13) - ادھار_work_گروہ_y, ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[1]);
        ثابت vk_mat_vec_ڈالو_constants pc = {
            (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
            stride_batch_x, stride_batch_y, stride_batch_d,
            fusion_flags, ادھار_work_گروہ_y,
            (uint32_t)ne02, (uint32_t)ne12, (uint32_t)r2, (uint32_t)r3,
        };
        ggml_vk_dispatch_pipeline(ctx, subctx, dmmv,
                                  {
                                    d_X,
                                    d_Y,
                                    d_D,
                                    d_F0,
                                    d_F1,
                                  },
                                  pc, { groups_x, groups_y, groups_z });
        ادھار_work_گروہ_y += groups_y;
    }

    إذا (x_non_contig) {
        ctx->prealloc_x_need_sync = سچ;
    }
    إذا (y_non_contig || quantize_y) {
        ctx->prealloc_y_need_sync = سچ;
    }
}

ساكن فراغ ggml_vk_mul_mat_vec_p021_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گرہ_idx) {
    ggml_tensor * dst = cgraph->nodes[گرہ_idx];
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat_p021_f16_f32(" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]);  // NOLINT
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]);  // NOLINT
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ثابت uint64_t ne00 = src0->ne[0];
    ثابت uint64_t ne01 = src0->ne[1];
    ثابت uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    //const uint64_t ne10 = src1->ne[0];
    ثابت uint64_t ne11 = src1->ne[1];
    ثابت uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    GGML_ASSERT(ne11 == 1);

    // With grouped query attention there are > 1 Q matrices per K, V matrix.
    uint32_t gqa_تناسب = (uint32_t)ne12 / (uint32_t)ne02;
    إذا (gqa_تناسب > 8 || gqa_تناسب == 0 || ne12 != ne02 * gqa_تناسب) {
        gqa_تناسب = 1;
    }

    vk_pipeline pipeline = ctx->device->pipeline_mul_mat_vec_p021_f16_f32[gqa_تناسب - 1];

    إذا (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_بفر_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    {
        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[گرہ_idx + ctx->عدد_additional_fused_ops], سچ);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1, سچ);

    vk_subbuffer d_F0 = d_D;

    uint32_t fusion_flags = 0;

    إذا (ctx->عدد_additional_fused_ops > 0) {
        ثابت ggml_tensor * جوڑو_2 = cgraph->nodes[گرہ_idx + 1];
        ثابت ggml_tensor * bias = جوڑو_2->src[0] == dst ? جوڑو_2->src[1] : جوڑو_2->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    إذا (ctx->عدد_additional_fused_ops > 1) {
        ثابت ggml_tensor * bias = cgraph->nodes[گرہ_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    // compute

    vk_mat_vec_p021_ڈالو_constants pc = {
        (uint32_t)ne00, (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne12,
        0, 0, fusion_flags
    };

    init_pushconst_tensor_offsets(ctx, pc, src0, src1, خالی_سوچک, خالی_سوچک, cgraph->nodes[گرہ_idx + ctx->عدد_additional_fused_ops]);

    uint32_t workgroups_z = (uint32_t)ne12;
    // When gqa_ratio > 1, each invocation does multiple rows and we can launch fewer workgroups
    إذا (gqa_تناسب > 1) {
        workgroups_z /= gqa_تناسب;
    }

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            d_Qx,
            d_Qy,
            d_D,
            d_F0,
            d_F1,
        }, pc, { 1, (uint32_t)ne01, workgroups_z });
}

ساكن فراغ ggml_vk_mul_mat_vec_nc_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گرہ_idx) {
    ggml_tensor * dst = cgraph->nodes[گرہ_idx];
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat_nc_f16_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ثابت uint64_t ne00 = src0->ne[0];
    ثابت uint64_t ne01 = src0->ne[1];
    ثابت uint64_t ne02 = src0->ne[2];
    ثابت uint64_t ne03 = src0->ne[3];

    ثابت uint64_t nb01 = src0->nb[1];
    ثابت uint64_t nb02 = src0->nb[2];

    ثابت uint64_t nb12 = src1->nb[2];

    // const uint64_t ne10 = src1->ne[0];
    ثابت uint64_t ne11 = src1->ne[1];
    ثابت uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    ثابت uint32_t nb03 = (uint32_t)(src0->nb[3] / حجم(ggml_fp16_t));
    ثابت uint32_t nb13 = (uint32_t)(src1->nb[3] / حجم(عائم));
    ثابت uint32_t nb23 = (uint32_t)(dst->nb[3] / حجم(عائم));

    GGML_ASSERT(ne11 == 1);
    GGML_ASSERT(src0->ne[3] == src1->ne[3]); // checked in supports_op

    ثابت uint32_t صف_stride_x = nb01 / حجم(ggml_fp16_t);
    ثابت uint32_t channel_stride_x = nb02 / حجم(ggml_fp16_t);
    ثابت uint32_t channel_stride_y = nb12 / حجم(عائم);

    vk_pipeline pipeline = ctx->device->pipeline_mul_mat_vec_nc_f16_f32;
    إذا (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_بفر_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    {
        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[گرہ_idx + ctx->عدد_additional_fused_ops], سچ);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1, سچ);
    vk_subbuffer d_F0 = d_D;

    uint32_t fusion_flags = 0;

    إذا (ctx->عدد_additional_fused_ops > 0) {
        ثابت ggml_tensor * جوڑو_2 = cgraph->nodes[گرہ_idx + 1];
        ثابت ggml_tensor * bias = جوڑو_2->src[0] == dst ? جوڑو_2->src[1] : جوڑو_2->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    إذا (ctx->عدد_additional_fused_ops > 1) {
        ثابت ggml_tensor * bias = cgraph->nodes[گرہ_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    // compute
    vk_mat_vec_nc_ڈالو_constants pc = {
        (uint32_t)ne00, (uint32_t)ne01,
        صف_stride_x, channel_stride_x, channel_stride_y,
        (uint32_t)(ne12 / ne02), (uint32_t)ne12,
        0, 0,
        nb03, nb13, nb23, fusion_flags
    };

    init_pushconst_tensor_offsets(ctx, pc, src0, src1, خالی_سوچک, خالی_سوچک, cgraph->nodes[گرہ_idx + ctx->عدد_additional_fused_ops]);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            d_Qx,
            d_Qy,
            d_D,
            d_F0,
            d_F1,
        }, pc, { (uint32_t)ne03, (uint32_t)ne01, (uint32_t)ne12 });
}

ساكن صحيح ggml_vk_fwht_pipeline_idx(int64_t n) {
    بدل (n) {
        حالة 64:  أرجع 0;
        حالة 128: أرجع 1;
        حالة 256: أرجع 2;
        حالة 512: أرجع 3;
        افتراضي:  أرجع -1;
    }
}

منطقي ggml_vk_can_use_fwht(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * src1, ثابت ggml_tensor * dst) {
    إذا (ctx->عدد_additional_fused_ops != 0) {
        أرجع خطأ;
    }

    إذا (ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        أرجع خطأ;
    }

    ثابت صحيح idx = ggml_vk_fwht_pipeline_idx(src1->ne[0]);
    إذا (idx < 0 || ctx->device->pipeline_fwht_f32[idx] == خالی_سوچک) {
        أرجع خطأ;
    }

    إذا (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }

    إذا (!ggml_is_contiguous(src1)) {
        أرجع خطأ;
    }
    GGML_ASSERT(ggml_is_contiguous(dst));

    أرجع سچ;
}

فراغ ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src, ggml_tensor * dst) {
    ثابت صحيح idx = ggml_vk_fwht_pipeline_idx(src->ne[0]);
    vk_pipeline pipeline = ctx->device->pipeline_fwht_f32[idx];

    ثابت uint32_t صفاں_فی_workgroup = 4;
    ثابت uint32_t n_صفاں = (uint32_t)ggml_nrows(src);
    ثابت uint32_t max_workgroups_x = ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[0];

    ثابت uint32_t کل_workgroups = CEIL_DIV(n_صفاں, صفاں_فی_workgroup);
    ثابت uint32_t workgroups_x = std::min(کل_workgroups, max_workgroups_x);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ثابت vk_subbuffer src_buf = ggml_vk_tensor_subbuffer(ctx, src, سچ);
    ثابت vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, سچ);

    vk_op_fwht_ڈالو_constants pc = {
        n_صفاں,
        0,
        0,
        1.0f / std::جذر((عائم)src->ne[0]),
    };
    init_pushconst_tensor_offsets(ctx, pc, src, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf }, pc, { workgroups_x, 1, 1 });
}

ساكن uint32_t ggml_vk_nb_elem(ثابت ggml_tensor * t, صحيح i) {
    أرجع (uint32_t)(t->nb[i] / ggml_type_ناپ(t->type));
}

فراغ ggml_vk_dsv4_hc_comb(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * mixes, ثابت ggml_tensor * scale, ثابت ggml_tensor * ادھار, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_comb(" << mixes << ", " << scale << ", " << ادھار << ", " << dst << ")");

    vk_pipeline pipeline = ctx->device->pipeline_dsv4_hc_comb_f32;
    GGML_ASSERT(pipeline != خالی_سوچک);

    ثابت uint32_t n_ٹوکن = (uint32_t)mixes->ne[1];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ثابت vk_subbuffer mixes_buf = ggml_vk_tensor_subbuffer(ctx, mixes, سچ);
    ثابت vk_subbuffer scale_buf = ggml_vk_tensor_subbuffer(ctx, scale, سچ);
    ثابت vk_subbuffer ادھار_buf  = ggml_vk_tensor_subbuffer(ctx, ادھار,  سچ);
    ثابت vk_subbuffer dst_buf   = ggml_vk_tensor_subbuffer(ctx, dst,   سچ);

    vk_op_dsv4_hc_comb_ڈالو_constants pc = {
        n_ٹوکن,
        ggml_vk_nb_elem(mixes, 0), ggml_vk_nb_elem(mixes, 1),
        ggml_vk_nb_elem(scale, 0),
        ggml_vk_nb_elem(ادھار,  0),
        ggml_vk_nb_elem(dst,   0), ggml_vk_nb_elem(dst, 1), ggml_vk_nb_elem(dst, 2),
        0, 0, 0, 0,
        ggml_get_op_params_f32(dst, 0),
        (uint32_t)ggml_get_op_params_i32(dst, 1),
    };
    init_pushconst_tensor_offsets(ctx, pc, mixes, scale, ادھار, خالی_سوچک, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { mixes_buf, scale_buf, ادھار_buf, dst_buf }, pc, { n_ٹوکن, 1, 1 });
}

فراغ ggml_vk_dsv4_hc_pre(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * x, ثابت ggml_tensor * weights, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_pre(" << x << ", " << weights << ", " << dst << ")");

    ثابت عائم scale = ggml_get_op_params_f32(dst, 0);
    ثابت منطقي  gated = ggml_get_op_params_i32(dst, 1) != 0;

    vk_pipeline pipeline = gated ? ctx->device->pipeline_dsv4_hc_pre_gated_f32 : ctx->device->pipeline_dsv4_hc_pre_f32;
    GGML_ASSERT(pipeline != خالی_سوچک);

    ثابت uint32_t n_embd   = (uint32_t)x->ne[0];
    ثابت uint32_t n_ٹوکن = (uint32_t)x->ne[2];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ثابت vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x,       سچ);
    ثابت vk_subbuffer w_buf = ggml_vk_tensor_subbuffer(ctx, weights, سچ);
    ثابت vk_subbuffer d_buf = ggml_vk_tensor_subbuffer(ctx, dst,     سچ);

    vk_op_dsv4_hc_pre_ڈالو_constants pc = {
        n_embd, n_ٹوکن,
        ggml_vk_nb_elem(x, 0), ggml_vk_nb_elem(x, 1), ggml_vk_nb_elem(x, 2),
        ggml_vk_nb_elem(weights, 0), ggml_vk_nb_elem(weights, 1), ggml_vk_nb_elem(weights, 2),
        ggml_vk_nb_elem(dst, 0), ggml_vk_nb_elem(dst, 1),
        0, 0, 0,
        scale,
    };
    init_pushconst_tensor_offsets(ctx, pc, x, weights, خالی_سوچک, خالی_سوچک, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, w_buf, d_buf }, pc, { n_embd, n_ٹوکن, 1 });
}

فراغ ggml_vk_dsv4_hc_post(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * x, ثابت ggml_tensor * residual, ثابت ggml_tensor * post, ثابت ggml_tensor * comb, ggml_tensor * dst, ثابت ggml_tensor * gate_scale_وچ) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_post(" << x << ", " << residual << ", " << post << ", " << comb << ", " << dst << ")");

    vk_pipeline pipeline = comb ? ctx->device->pipeline_dsv4_hc_post_f32 : ctx->device->pipeline_dsv4_hc_post_nocomb_f32;
    GGML_ASSERT(pipeline != خالی_سوچک);

    ثابت uint32_t n_embd   = (uint32_t)x->ne[0];
    ثابت uint32_t n_ٹوکن = (uint32_t)x->ne[1];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ثابت vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x,        سچ);
    ثابت vk_subbuffer r_buf = ggml_vk_tensor_subbuffer(ctx, residual, سچ);
    // with a fused gate, post is scale(sigmoid(scale(p_src))) and the shader applies it to p_src
    ثابت ggml_tensor * p_src = gate_scale_وچ ? gate_scale_وچ->src[0] : post;
    ثابت vk_subbuffer p_buf = ggml_vk_tensor_subbuffer(ctx, p_src,    سچ);
    ثابت vk_subbuffer c_buf = comb ? ggml_vk_tensor_subbuffer(ctx, comb, سچ) : x_buf;
    ثابت vk_subbuffer d_buf = ggml_vk_tensor_subbuffer(ctx, dst,      سچ);

    vk_op_dsv4_hc_post_ڈالو_constants pc = {
        n_embd, n_ٹوکن,
        ggml_vk_nb_elem(x, 0), ggml_vk_nb_elem(x, 1),
        ggml_vk_nb_elem(residual, 0), ggml_vk_nb_elem(residual, 1), ggml_vk_nb_elem(residual, 2),
        ggml_vk_nb_elem(p_src, 0), ggml_vk_nb_elem(p_src, 1),
        comb ? ggml_vk_nb_elem(comb, 0) : 0, comb ? ggml_vk_nb_elem(comb, 1) : 0, comb ? ggml_vk_nb_elem(comb, 2) : 0,
        ggml_vk_nb_elem(dst,  0), ggml_vk_nb_elem(dst,  1), ggml_vk_nb_elem(dst,  2),
        0, 0, 0, 0, 0,
        gate_scale_وچ ? 1u : 0u,
        gate_scale_وچ ? ggml_get_op_params_f32(gate_scale_وچ, 0) : 1.0f,
        gate_scale_وچ ? ggml_get_op_params_f32(post, 0) : 1.0f,
    };
    init_pushconst_tensor_offsets(ctx, pc, x, residual, p_src, comb, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, r_buf, p_buf, c_buf, d_buf }, pc, { n_embd, n_ٹوکن, 1 });
}

فراغ ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گرہ_idx) {
    ggml_tensor * dst = cgraph->nodes[گرہ_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat(" << src0 << ", " << src1 << ", " << dst << ")");

    // Handle huge A matrix by splitting the M dimensions. This works well for convolution use cases
    // where the M dimension is very large.
    // Split_k doesn't work with M splitting.
    // This only supports batchsize == 1.
    ثابت size_t nbytes = ggml_nbytes(src0);
    ثابت منطقي needs_ونڈو = dst->ne[2] == 1 && dst->ne[3] == 1 && nbytes > ctx->device->properties.limits.max_Storage_بفر_Range;
    إذا (needs_ونڈو) {
        // Choose the number of rows that can fit (and divide by two, to allow for any additional offsets)
        ثابت uint32_t M_ونڈو = ctx->device->properties.limits.max_Storage_بفر_Range / (2 * src0->nb[1]);
        uint32_t m_وتھ = 0;
        طالما (m_وتھ < dst->ne[0]) {
            ثابت uint32_t cur_M_ناپ = std::min(M_ونڈو, (uint32_t)(dst->ne[0] - m_وتھ));
            ggml_tensor dst2 = *dst;
            ggml_tensor src02 = *src0;

            dst2.view_src = dst->view_src ? dst->view_src : dst;
            src02.view_src = src0->view_src ? src0->view_src : src0;

            dst2.view_offs += m_وتھ * dst->nb[0];
            src02.view_offs += m_وتھ * src0->nb[1];
            dst2.ne[0] = cur_M_ناپ;
            src02.ne[1] = cur_M_ناپ;

            ggml_vk_mul_mat_q_f16(ctx, subctx, &src02, src1, &dst2, سچ);

            m_وتھ += cur_M_ناپ;
        }
    } وإلا إذا (ggml_vk_can_use_fwht(ctx, src1, dst)) {
        ggml_vk_fwht(ctx, subctx, src1, dst);
    } وإلا إذا (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && dst->ne[1] == 1 &&
        // detect 0213 permutation, and batch size of 1
        src0->nb[0] <= src0->nb[2] &&
        src0->nb[2] <= src0->nb[1] &&
        src0->nb[1] <= src0->nb[3] &&
        src1->nb[0] <= src1->nb[2] &&
        src1->nb[2] <= src1->nb[1] &&
        src1->nb[1] <= src1->nb[3] &&
        src0->ne[3] == 1 &&
        src1->ne[3] == 1 &&
        src0->ne[1] <= ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[1] &&
        src1->ne[2] <= ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[2]) {
        ggml_vk_mul_mat_vec_p021_f16_f32(ctx, subctx, cgraph, گرہ_idx);
    } وإلا إذا (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && dst->ne[1] == 1 &&
               !ggml_is_permuted(src0) && !ggml_is_permuted(src1) &&
               src0->ne[3] <= ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[0] &&
               src0->ne[1] <= ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[1] &&
               src1->ne[2] <= ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[2]) {
        ggml_vk_mul_mat_vec_nc_f16_f32(ctx, subctx, cgraph, گرہ_idx);
    // With one output row, B^T*A has the same flat output as A^T*B.
    } وإلا إذا (ctx->عدد_additional_fused_ops == 0 &&
               (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
               (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16 || ggml_is_quantized(src1->type)) &&
               dst->ne[0] == 1 && dst->ne[1] > mul_mat_vec_max_خانے_گنتی &&
               src0->ne[2] == 1 && src0->ne[3] == 1 &&
               src1->ne[2] == 1 && src1->ne[3] == 1 &&
               ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
               get_misalign_بائٹاں(ctx, src0) == 0 && get_misalign_بائٹاں(ctx, src1) == 0 && get_misalign_بائٹاں(ctx, dst) == 0) {
        ggml_vk_mul_mat_vec_q_f16(ctx, subctx, cgraph, گرہ_idx, سچ);
    // mul_mat_vec supports batching ne12*ne13 when ne11==1, or treating ne11 as the batch size (up to four)
    // when ne12 and ne13 are one.
    } وإلا إذا ((dst->ne[1] == 1 || (dst->ne[1] <= mul_mat_vec_max_خانے_گنتی && src1->ne[2] * src1->ne[3] == 1)) &&
               (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || ggml_is_quantized(src0->type))) {
        ggml_vk_mul_mat_vec_q_f16(ctx, subctx, cgraph, گرہ_idx);
    } وإلا {
        ggml_vk_mul_mat_q_f16(ctx, subctx, src0, src1, dst, خطأ);
    }
}

ساكن فراغ ggml_vk_mul_mat_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * ids, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1=" << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ثابت uint64_t ne00 = src0->ne[0];
    ثابت uint64_t ne01 = src0->ne[1];
    ثابت uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    ثابت uint64_t ne10 = src1->ne[0];
    ثابت uint64_t ne11 = src1->ne[1];
    ثابت uint64_t ne12 = src1->ne[2];
    ثابت uint64_t ne13 = src1->ne[3];

    ثابت uint64_t nei0 = ids->ne[0];
    ثابت uint64_t nei1 = ids->ne[1];

    ثابت uint32_t nbi0 = ids->nb[0];
    ثابت uint32_t nbi1 = ids->nb[1];

    ثابت uint64_t ne20 = dst->ne[0];
    ثابت uint64_t ne21 = dst->ne[1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    ثابت uint64_t n_as = ne02;

    ggml_backend_vk_بفر_context * dst_buf_ctx = (ggml_backend_vk_بفر_context *)dst->بفر->context;
    ggml_backend_vk_بفر_context * src0_buf_ctx = (ggml_backend_vk_بفر_context *)src0->بفر->context;
    ggml_backend_vk_بفر_context * src1_buf_ctx = (ggml_backend_vk_بفر_context *)src1->بفر->context;

    vk_بفر d_Qx = خالی_سوچک;
    size_t qx_buf_وتھ = 0;
    vk_بفر d_Qy = خالی_سوچک;
    size_t qy_buf_وتھ = 0;

    منطقي src0_uma = خطأ;
    منطقي src1_uma = خطأ;

    إذا (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->انکڑے, d_Qx, qx_buf_وتھ);
        ggml_vk_host_get(ctx->device, src1->انکڑے, d_Qy, qy_buf_وتھ);
        src0_uma = d_Qx != خالی_سوچک;
        src1_uma = d_Qy != خالی_سوچک;
    }

    // Reformat and convert to fp16 if non-contiguous, or for coopmat2 for better perf
    ثابت منطقي x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    // If src0 is BF16, try to use a BF16 x BF16 multiply
    ggml_type f16_type = src0->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;
#إذا defined(GGML_VULKAN_COOPMAT2_DECODE_سمتیہ_GLSLC_SUPPORT)
    // B must already be, or be convertible to, the matmul B type used by this path.
    ثابت منطقي y_decode_سمتیہ_supported = ctx->device->coopmat2_decode_سمتیہ &&
                                           (f16_type != GGML_TYPE_BF16 || ctx->device->coopmat2_bf16_support) &&
                                           (src1->type == GGML_TYPE_F32 || src1->type == f16_type);
    // If B is copied to prealloc_y, we can choose a 4-element-aligned row stride.
    ثابت منطقي y_decode_سمتیہ_ورتدا_prealloc = !ggml_vk_dim01_contiguous(src1) || src1->type != f16_type;
    // Direct B reads are safe only if row starts and the original buffer offset are 4-element aligned.
    ثابت منطقي y_decode_سمتیہ_aligned =
        (ne10 % 4 == 0) &&
        (y_decode_سمتیہ_ورتدا_prealloc || get_misalign_بائٹاں(ctx, src1) % (4 * ggml_type_ناپ(src1->type)) == 0);
    // Stage B only when decode-vector is available and direct B reads would be misaligned.
    ثابت منطقي y_decode_سمتیہ_staging = y_decode_سمتیہ_supported && !y_decode_سمتیہ_aligned;
#وإلا
    ثابت منطقي y_decode_سمتیہ_staging = خطأ;
#جے_انت
    ثابت منطقي can_quantize_y = ctx->device->integer_dot_ضرب_حاصل && src1->type == GGML_TYPE_F32 &&
                                ggml_is_contiguous(src1) && (ne11 * ne10) % 4 == 0;
    // Intel coopmat1 matrix cores prefer f16 activations for quantized weights
    ثابت منطقي prefer_f16_b = ctx->device->coopmat_support && !ctx->device->coopmat2 &&
                              ctx->device->vendor_id == VK_VENDOR_ID_INTEL &&
                              ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32;

    ثابت ggml_type y_kernel_type = ggml_vk_mul_mat_b_type(ctx, src0->type, src1, f16_type,
                                        (ggml_prec)dst->op_params[0], سچ, can_quantize_y, prefer_f16_b);

    ثابت منطقي quantize_y   = y_kernel_type == GGML_TYPE_Q8_1;
    ثابت منطقي y_f32_kernel = y_kernel_type == GGML_TYPE_F32;

    ثابت std::vector<vk_matmul_pipeline_pair>* mmp_map =
        ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_kernel_type, (ggml_prec)dst->op_params[0], سچ);

    ثابت منطقي qx_needs_dequant = mmp_map == خالی_سوچک || x_non_contig;

    إذا (qx_needs_dequant) {
        // dequant src0 to f16
        mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, f16_type, y_kernel_type, (ggml_prec)dst->op_params[0], سچ);
    }

    GGML_ASSERT(mmp_map != خالی_سوچک);

    // Coopmat2 MUL_MAT_ID BK specialization constants in ggml_vk_load_shaders are at most 64.
    ثابت uint32_t y_staged_صف_stride = ctx->device->coopmat2 && !quantize_y ? ggml_vk_align_ناپ(ne10, 64) : ne10;
    ثابت منطقي y_needs_k_padding = ne10 != y_staged_صف_stride;

    ثابت منطقي y_needs_reformat = !quantize_y && (!ggml_vk_dim01_contiguous(src1) || y_kernel_type != src1->type ||
                                                  y_decode_سمتیہ_staging || y_needs_k_padding);
    ثابت منطقي qy_needs_dequant = y_needs_reformat;

    ثابت uint32_t kpad = quantize_y ? 0 : ggml_vk_align_ناپ(ne10, ggml_vk_guess_matmul_pipeline_align_map(ctx, *mmp_map, ne01, nei1, سچ));
    ثابت منطقي aligned = !quantize_y && ne10 == kpad && ne01 > 8 && nei1 > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline_map(ctx, *mmp_map, ne01, nei1, aligned, سچ);

    ثابت uint32_t bn = pipeline->wg_denoms[1];
    ثابت uint64_t max_tiles = ggml_vk_mul_mat_id_max_tiles(n_as, nei0 * nei1, bn);
    ثابت uint64_t صف_ids_وتھ = 1 + 2 * max_tiles;
    ثابت uint64_t صف_map_الفاظ = صف_ids_وتھ + max_tiles * bn;

    إذا (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_بفر_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }
    ثابت uint64_t x_ne = ggml_nelements(src0);
    ثابت uint64_t y_ne = (uint64_t)y_staged_صف_stride * ne11 * ne12 * ne13;
    ثابت uint64_t d_ne = ggml_nelements(dst);

    ثابت uint64_t qx_sz = ggml_type_ناپ(src0->type) * x_ne / ggml_blck_ناپ(src0->type);
    ثابت uint64_t qy_sz = ggml_type_ناپ(src1->type) * ggml_nelements(src1) / ggml_blck_ناپ(src1->type);
    ثابت uint64_t x_sz = !qx_needs_dequant ? qx_sz : حجم(ggml_fp16_t) * x_ne;
    ثابت uint64_t y_sz = quantize_y ? (ggml_vk_align_ناپ(y_ne, 128) * ggml_type_ناپ(GGML_TYPE_Q8_1) / ggml_blck_ناپ(GGML_TYPE_Q8_1)) : (y_f32_kernel ? حجم(عائم) * y_ne : حجم(ggml_fp16_t) * y_ne);
    ثابت uint64_t d_sz = حجم(عائم) * d_ne;

    vk_pipeline منزل_fp16_vk_0 = خالی_سوچک;
    vk_pipeline منزل_fp16_vk_1 = خالی_سوچک;
    vk_pipeline منزل_q8_1 = خالی_سوچک;

    خودکار make_y_staged_dst = [&]() {
        ggml_tensor y_staged_dst = *src1;
        y_staged_dst.type = f16_type;
        y_staged_dst.nb[0] = ggml_type_ناپ(f16_type);
        y_staged_dst.nb[1] = y_staged_dst.nb[0] * y_staged_صف_stride;
        y_staged_dst.nb[2] = y_staged_dst.nb[1] * ne11;
        y_staged_dst.nb[3] = y_staged_dst.nb[2] * y_staged_dst.ne[2];
        أرجع y_staged_dst;
    };

    إذا (x_non_contig) {
        منزل_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, خالی_سوچک, f16_type);
    } وإلا {
        منزل_fp16_vk_0 = ggml_vk_get_منزل_fp16(ctx, src0->type);
    }
    إذا (y_needs_reformat) {
        ggml_tensor y_staged_dst;
        ثابت ggml_tensor * y_staged_dst_ptr = خالی_سوچک;
        إذا (y_needs_k_padding) {
            y_staged_dst = make_y_staged_dst();
            y_staged_dst_ptr = &y_staged_dst;
        }

        منزل_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, y_staged_dst_ptr, f16_type);
    } وإلا {
        منزل_fp16_vk_1 = ggml_vk_get_منزل_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || منزل_fp16_vk_0 != خالی_سوچک);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || منزل_fp16_vk_1 != خالی_سوچک);  // NOLINT

    إذا (quantize_y) {
        منزل_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }
    vk_pipeline گنتی_experts = ctx->device->pipeline_گنتی_experts;

    ثابت size_t expert_انکڑے_ناپ = حجم(uint32_t) * صف_map_الفاظ;

    {
        إذا (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_بفر_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_بفر_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        إذا (qx_needs_dequant && ctx->prealloc_ناپ_x < x_sz) {
            ctx->prealloc_ناپ_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا ((qy_needs_dequant || quantize_y) && ctx->prealloc_ناپ_y < y_sz) {
            ctx->prealloc_ناپ_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا (ctx->prealloc_ناپ_ونڈو_k < expert_انکڑے_ناپ) {
            ctx->prealloc_ناپ_ونڈو_k = expert_انکڑے_ناپ;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        إذا (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, منزل_fp16_vk_0, 1);
        }
        إذا (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, منزل_fp16_vk_1, 1);
        }
        إذا (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, منزل_q8_1, 1);
        }
        ggml_pipeline_request_descriptor_sets(ctx, گنتی_experts, 1);
    }

    vk_بفر d_D = dst_buf_ctx->dev_بفر;
    ثابت uint64_t d_buf_وتھ = vk_tensor_وتھ(dst) + dst->view_offs;
    GGML_ASSERT(d_D != خالی_سوچک);
    vk_بفر d_X;
    uint64_t x_buf_وتھ = 0;
    vk_بفر d_Y;
    uint64_t y_buf_وتھ = 0;
    إذا (!src0_uma) {
        d_Qx = src0_buf_ctx->dev_بفر;
        qx_buf_وتھ = vk_tensor_وتھ(src0) + src0->view_offs;
        GGML_ASSERT(d_Qx != خالی_سوچک);
    }
    إذا (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_بفر;
        qy_buf_وتھ = vk_tensor_وتھ(src1) + src1->view_offs;
        GGML_ASSERT(d_Qy != خالی_سوچک);
    }
    إذا (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz);
    } وإلا {
        d_X = d_Qx;
        x_buf_وتھ = qx_buf_وتھ;
        GGML_ASSERT(qx_sz == x_sz);
    }
    إذا (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz);
    } وإلا إذا (quantize_y) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= CEIL_DIV(y_sz, 144) * 144);
    } وإلا {
        d_Y = d_Qy;
        y_buf_وتھ = qy_buf_وتھ;
        GGML_ASSERT(qy_sz == y_sz);
    }

    إذا (x_non_contig || qx_needs_dequant) {
        إذا (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }
    vk_subbuffer d_ids = ggml_vk_tensor_subbuffer(ctx, ids, سچ);

    // Count how many times each expert is used
    vk_subbuffer expert_گنتی_buf = { ctx->prealloc_ونڈو_k, 0, expert_انکڑے_ناپ };
    إذا (ctx->prealloc_ونڈو_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }
    {
        vk_op_گنتی_experts_ڈالو_constants pc = { (uint32_t)nei0,
                                           (uint32_t)nei1,
                                           (uint32_t)(nbi0 / ggml_type_ناپ(ids->type)),
                                           (uint32_t)(nbi1 / ggml_type_ناپ(ids->type)),
                                           (uint32_t)(get_misalign_بائٹاں(ctx, ids) / ggml_type_ناپ(ids->type)),
                                           (uint32_t)n_as,
                                           0, 0, bn, (uint32_t)صف_ids_وتھ };
        init_pushconst_fastdiv(pc);
        ggml_vk_dispatch_pipeline(ctx, subctx, گنتی_experts,
            { d_ids, expert_گنتی_buf }, pc,
            { 1, 1, 1 });
    }

    إذا (x_non_contig) {
        ggml_vk_cpy_منزل_contiguous(ctx, subctx, منزل_fp16_vk_0, src0, ggml_vk_subbuffer(ctx, d_Qx, qx_buf_وتھ), ggml_vk_subbuffer(ctx, d_X, 0));
    } وإلا إذا (qx_needs_dequant) {
        ثابت std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_dispatch_pipeline(ctx, subctx, منزل_fp16_vk_0,
            { vk_subbuffer{ d_Qx, qx_buf_وتھ, qx_sz }, vk_subbuffer{ d_X, 0, x_sz } }, pc, { (uint32_t)x_ne, 1, 1});
    }
    إذا (y_needs_reformat) {
        إذا (ctx->prealloc_y_آخری_pipeline_ورتیا != منزل_fp16_vk_1.get() ||
            ctx->prealloc_y_آخری_tensor_ورتیا != src1 ||
            ctx->prealloc_y_آخری_k_padded != y_needs_k_padding) {
            إذا (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            إذا (y_needs_k_padding) {
                GGML_ASSERT(y_sz % 4 == 0);
                // Zero B padding because clamping only A can produce 0 * Inf or NaN.
                subctx->s->بفر->buf.fill_بفر(d_Y->بفر, 0, y_sz, 0);
                ggml_vk_sync_buffers(ctx, subctx);
                ثابت ggml_tensor y_staged_dst = make_y_staged_dst();
                ثابت uint32_t y_staged_dst_type_ناپ = ggml_type_ناپ(y_staged_dst.type);
                ggml_vk_cpy_منزل_strided(
                    ctx, subctx, منزل_fp16_vk_1, src1,
                    ggml_vk_subbuffer(ctx, d_Qy, qy_buf_وتھ), ggml_vk_subbuffer(ctx, d_Y, 0),
                    (uint32_t)(y_staged_dst.nb[0] / y_staged_dst_type_ناپ),
                    (uint32_t)(y_staged_dst.nb[1] / y_staged_dst_type_ناپ),
                    (uint32_t)(y_staged_dst.nb[2] / y_staged_dst_type_ناپ),
                    (uint32_t)(y_staged_dst.nb[3] / y_staged_dst_type_ناپ));
            } وإلا {
                ggml_vk_cpy_منزل_contiguous(ctx, subctx, منزل_fp16_vk_1, src1, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_وتھ), ggml_vk_subbuffer(ctx, d_Y, 0));
            }
            ctx->prealloc_y_آخری_pipeline_ورتیا = منزل_fp16_vk_1.get();
            ctx->prealloc_y_آخری_tensor_ورتیا = src1;
            ctx->prealloc_y_آخری_k_padded = y_needs_k_padding;
        }
    }
    إذا (quantize_y) {
        إذا (ctx->prealloc_y_آخری_pipeline_ورتیا != منزل_q8_1.get() ||
            ctx->prealloc_y_آخری_tensor_ورتیا != src1 ||
            ctx->prealloc_y_آخری_k_padded) {
            إذا (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_وتھ), ggml_vk_subbuffer(ctx, d_Y, 0), y_ne);
            ctx->prealloc_y_آخری_pipeline_ورتیا = منزل_q8_1.get();
            ctx->prealloc_y_آخری_tensor_ورتیا = src1;
            ctx->prealloc_y_آخری_k_padded = خطأ;
        }
    }
    ggml_vk_sync_buffers(ctx, subctx);

    ثابت uint64_t x_range = qx_needs_dequant ? x_sz : ggml_nbytes(src0);
    ثابت uint64_t y_range = (qy_needs_dequant || quantize_y) ? y_sz : ggml_nbytes(src1);

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_b_y = y_needs_k_padding ? y_staged_صف_stride : ne10;
    uint32_t stride_batch_y = y_needs_k_padding ? y_staged_صف_stride * ne11 : ((qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1));

    إذا (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_ناپ(src0->type);
    }

    إذا (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant && !quantize_y) {
        stride_batch_y = src1->nb[0] / ggml_type_ناپ(src1->type);
    }

    // compute
    ggml_vk_matmul_id(
        ctx, subctx, pipeline,
        { d_X, x_buf_وتھ, x_range }, { d_Y, y_buf_وتھ, y_range },
        { d_D, d_buf_وتھ, d_sz }, std::move(d_ids), expert_گنتی_buf,
        ne01, ne21, ne10, ne10, stride_b_y, ne01,
        stride_batch_x, stride_batch_y, ne20*ne21,
        n_as, nei0, nei1, nbi1 / ggml_type_ناپ(ids->type), ne11, صف_ids_وتھ, max_tiles
    );  // NOLINT

    إذا (x_non_contig || qx_needs_dequant) {
        ctx->prealloc_x_need_sync = سچ;
    }
    إذا (y_needs_reformat || quantize_y) {
        ctx->prealloc_y_need_sync = سچ;
    }
    ctx->prealloc_ونڈو_k_need_sync = سچ;
}

ساكن فراغ ggml_vk_mul_mat_vec_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گرہ_idx) {
    ggml_tensor * dst = cgraph->nodes[گرہ_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    ggml_tensor * ids = dst->src[2];
    VK_LOG_DEBUG("ggml_vk_mul_mat_vec_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1=" << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ثابت uint64_t ne00 = src0->ne[0];
    ثابت uint64_t ne01 = src0->ne[1];
    // const uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    ثابت uint64_t ne10 = src1->ne[0];
    ثابت uint64_t ne11 = src1->ne[1];
    ثابت uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    ثابت uint64_t nei0 = ids->ne[0];
    ثابت uint64_t nei1 = ids->ne[1];
    ثابت uint32_t nbi1 = (uint32_t)(ids->nb[1] / حجم(صحيح));

    ثابت uint64_t ne20 = dst->ne[0];
    ثابت uint64_t ne21 = dst->ne[1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    ثابت منطقي x_non_contig = !ggml_vk_dim01_contiguous(src0);
    ثابت منطقي y_non_contig = !ggml_vk_dim01_contiguous(src1);

    ثابت منطقي f16_f32_kernel = src1->type == GGML_TYPE_F32;
    منطقي quantize_y = ctx->device->integer_dot_ضرب_حاصل && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0 && ggml_vk_should_use_mmvq(ctx->device, ne01, ne12, ne10, src0->type);

    vk_pipeline منزل_fp16_vk_0 = خالی_سوچک;
    vk_pipeline منزل_fp16_vk_1 = خالی_سوچک;
    إذا (x_non_contig) {
        منزل_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, خالی_سوچک, src0->type);
    }
    إذا (y_non_contig) {
        منزل_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, خالی_سوچک, src1->type);
    } وإلا {
        منزل_fp16_vk_1 = ggml_vk_get_منزل_fp16(ctx, src1->type);
    }

    // Check for mmq first
    vk_pipeline dmmv = quantize_y ? ggml_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, GGML_TYPE_Q8_1, ne20, ne00) : خالی_سوچک;
    vk_pipeline منزل_q8_1 = خالی_سوچک;

    إذا (dmmv == خالی_سوچک) {
        // Fall back to f16 dequant mul mat
        dmmv = ggml_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, src1->type, ne20, ne00);
        quantize_y = خطأ;
    }

    إذا (quantize_y) {
        منزل_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    ثابت منطقي qx_needs_dequant = x_non_contig;
    ثابت منطقي qy_needs_dequant = !quantize_y && ((src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig);

    إذا (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_بفر_Range) {
        dmmv = ggml_vk_get_64b_indexing_pipeline(ctx, dmmv);
    }

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT
    GGML_ASSERT(!qx_needs_dequant || منزل_fp16_vk_0 != خالی_سوچک);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || منزل_fp16_vk_1 != خالی_سوچک);  // NOLINT
    GGML_ASSERT(dmmv != خالی_سوچک);

    ثابت uint64_t x_ne = ggml_nelements(src0);
    ثابت uint64_t y_ne = ggml_nelements(src1);

    ثابت uint64_t qx_sz = ggml_vk_align_ناپ(ggml_type_ناپ(src0->type) * x_ne / ggml_blck_ناپ(src0->type), ctx->device->properties.limits.min_Storage_بفر_وتھ_Alignment);
    ثابت uint64_t x_sz = x_non_contig ? ggml_vk_align_ناپ(ggml_type_ناپ(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_بفر_وتھ_Alignment) : qx_sz;
    ثابت uint64_t y_sz = quantize_y ? (ggml_vk_align_ناپ(y_ne, 128) * ggml_type_ناپ(GGML_TYPE_Q8_1) / ggml_blck_ناپ(GGML_TYPE_Q8_1)) :
                                       (f16_f32_kernel ? حجم(عائم) * y_ne : حجم(ggml_fp16_t) * y_ne);

    {
        إذا (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_بفر_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_بفر_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        إذا (qx_needs_dequant && ctx->prealloc_ناپ_x < x_sz) {
            ctx->prealloc_ناپ_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا ((qy_needs_dequant || quantize_y) && ctx->prealloc_ناپ_y < y_sz) {
            ctx->prealloc_ناپ_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        إذا (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, منزل_fp16_vk_0, 1);
        }
        إذا (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, منزل_fp16_vk_1, 1);
        }
        إذا (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, منزل_q8_1, 1);
        }
        ggml_pipeline_request_descriptor_sets(ctx, dmmv, nei1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[گرہ_idx + ctx->عدد_additional_fused_ops]);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer d_ids = ggml_vk_tensor_subbuffer(ctx, ids);
    vk_subbuffer d_F0 = d_D;
    vk_subbuffer d_X, d_Y;

    إذا (qx_needs_dequant) {
        d_X = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    } وإلا {
        d_X = d_Qx;
    }
    إذا (qy_needs_dequant || quantize_y) {
        d_Y = { ctx->prealloc_y, 0, ctx->prealloc_y->size };
    } وإلا {
        d_Y = d_Qy;
    }

    إذا (x_non_contig) {
        إذا (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    إذا (x_non_contig) {
        GGML_ASSERT(x_sz == ggml_vk_align_ناپ(ggml_type_ناپ(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_بفر_وتھ_Alignment));
        ggml_vk_cpy_منزل_contiguous(ctx, subctx, منزل_fp16_vk_0, src0, d_Qx, d_X);
    }
    إذا (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_type_ناپ(src1->type) * y_ne);
        إذا (ctx->prealloc_y_آخری_pipeline_ورتیا != منزل_fp16_vk_1.get() ||
            ctx->prealloc_y_آخری_tensor_ورتیا != src1 ||
            ctx->prealloc_y_آخری_k_padded) {
            إذا (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_منزل_contiguous(ctx, subctx, منزل_fp16_vk_1, src1, d_Qy, d_Y);
            ctx->prealloc_y_آخری_pipeline_ورتیا = منزل_fp16_vk_1.get();
            ctx->prealloc_y_آخری_tensor_ورتیا = src1;
            ctx->prealloc_y_آخری_k_padded = خطأ;
        }
    }
    إذا (quantize_y) {
        إذا (ctx->prealloc_y_آخری_pipeline_ورتیا != منزل_q8_1.get() ||
            ctx->prealloc_y_آخری_tensor_ورتیا != src1 ||
            ctx->prealloc_y_آخری_k_padded) {
            إذا (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, d_Qy, d_Y, y_ne);
            ctx->prealloc_y_آخری_pipeline_ورتیا = منزل_q8_1.get();
            ctx->prealloc_y_آخری_tensor_ورتیا = src1;
            ctx->prealloc_y_آخری_k_padded = خطأ;
        }
    }

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_batch_y = (qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1);

    إذا (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[2] / ggml_type_ناپ(src1->type);
    }

    ثابت uint32_t max_groups_x = ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    إذا (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = CEIL_DIV(groups_x, groups_z);
    }

    uint32_t fusion_flags = 0;

    إذا (ctx->عدد_additional_fused_ops > 0) {
        ثابت ggml_tensor * bias = cgraph->nodes[گرہ_idx + 1]->src[1];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);

        إذا (cgraph->nodes[گرہ_idx + 1]->op == GGML_OP_MUL) {
            fusion_flags |= MAT_VEC_FUSION_FLAGS_SCALE0;
        } وإلا {
            GGML_ASSERT(cgraph->nodes[گرہ_idx + 1]->op == GGML_OP_جوڑو_ID);
            fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
        }
    }

    vk_subbuffer d_F1 = d_D;
    إذا (ctx->عدد_additional_fused_ops > 1) {
        ثابت ggml_tensor * scale = cgraph->nodes[گرہ_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, scale);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_SCALE1;
    }

    // Loop over the batch dimension
    لكل (uint32_t expert_i1 = 0; expert_i1 < nei1; ++expert_i1) {
        ثابت vk_mat_vec_id_ڈالو_constants pc = {
            (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
            stride_batch_x, stride_batch_y, (uint32_t)(ne20 * ne21),
            fusion_flags,
            (uint32_t)nei0, (uint32_t)ne11, expert_i1, nbi1
        };
        ggml_vk_dispatch_pipeline(ctx, subctx, dmmv,
            {
                d_X,
                d_Y,
                d_D,
                d_F0,
                d_F1,
                d_ids,
            },
            pc, { groups_x, (uint32_t)nei0, groups_z });
    }

    إذا (x_non_contig) {
        ctx->prealloc_x_need_sync = سچ;
    }
    إذا (y_non_contig || quantize_y) {
        ctx->prealloc_y_need_sync = سچ;
    }
}

ساكن منطقي ggml_vk_use_mul_mat_vec_id(ثابت ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * ids = dst->src[2];
    أرجع ids->ne[1] <= 8 && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type));
}

منطقي ggml_vk_use_mul_mat_vec_id(ثابت بنية ggml_cgraph * cgraph, صحيح گرہ_idx) {
    أرجع ggml_vk_use_mul_mat_vec_id(cgraph->nodes[گرہ_idx]);
}

فراغ ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گرہ_idx) {
    ggml_tensor * dst = cgraph->nodes[گرہ_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    ggml_tensor * src2 = dst->src[2];
    VK_LOG_DEBUG("ggml_vk_mul_mat_id(" << src0 << ", " << src1 << ", " << src2 << ", " << dst << ")");
    إذا (ggml_vk_use_mul_mat_vec_id(cgraph, گرہ_idx)) {
        ggml_vk_mul_mat_vec_id_q_f16(ctx, subctx, cgraph, گرہ_idx);
    } وإلا {
        ggml_vk_mul_mat_id_q_f16(ctx, subctx, src0, src1, src2, dst);
    }
}

منطقي ggml_vk_flash_attn_scalar_shmem_support(ثابت vk_device& device, ثابت vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, منطقي f32acc, ggml_type k_type, ggml_type v_type) {
    GGML_بن_ورتیا(f32acc);
    // Needs to be kept up to date on shader changes
    ثابت uint32_t wg_ناپ = params.workgroup_ناپ;
    ثابت uint32_t Br = params.بلاک_صفاں;
    ثابت uint32_t Bc = params.بلاک_خانے_گنتی;

    // BF16 uses the fp32 shader (FLOAT_TYPE=float)
    ثابت uint32_t float_type_ناپ = (device->fp16 && k_type != GGML_TYPE_BF16) ? حجم(ggml_fp16_t) : حجم(عائم);

    ثابت منطقي mmq = ggml_vk_fa_scalar_ورتدا_mmq(device, k_type, v_type);

    // tmpsh is overestimated slightly
    ثابت uint32_t tmpsh = wg_ناپ * حجم(عائم);
    ثابت uint32_t tmpshv4 = wg_ناپ * 4 * float_type_ناپ;

    ثابت uint32_t masksh = Bc * (Br + 1) * float_type_ناپ;
    // DATA_A_IQ4_NL is compiled into the FA shaders unconditionally, so its shared table is always allocated.
    ثابت uint32_t iq_shmem = 16 * float_type_ناپ;

    uint32_t Qf, kvsh, kblocksh_ناپ;
    إذا (mmq) {
        // block_b_cache: int32_t qs[8] + FLOAT_TYPEV2 ds
        ثابت uint32_t بلاک_b_ناپ = 8 * حجم(int32_t) + 2 * float_type_ناپ;
        Qf = Br * (hsk / 32) * بلاک_b_ناپ;

        // kvsh uses D = HSV (K goes through kblocksh instead)
        kvsh = params.shmem_staging ? Bc * (hsv / 4 + 1) * 4 * float_type_ناپ : 4 * float_type_ناپ;

        // The mixed MMQ shader uses a superset block_a_cache that fits every
        // FA-supported quant: int32_t qs[8] + uint32_t qh + FLOAT_TYPEV2 dm.
        // Single-scale types leave dm.y unused; non-Q5_* leave qh unused.
        ثابت uint32_t بلاک_a_ناپ = 8 * حجم(int32_t) + حجم(uint32_t) + 2 * float_type_ناپ;
        kblocksh_ناپ = params.shmem_staging ? Bc * (hsk / 32) * بلاک_a_ناپ : بلاک_a_ناپ;
    } وإلا {
        Qf = Br * (hsk / 4 + 1) * 4 * float_type_ناپ;

        ثابت uint32_t D = std::max(hsk, hsv);
        kvsh = params.shmem_staging ? Bc * (D / 4 + 1) * 4 * float_type_ناپ : 4 * float_type_ناپ;

        kblocksh_ناپ = 0;
    }

    ثابت uint32_t کل_ناپ = tmpsh + tmpshv4 + masksh + iq_shmem + Qf + kvsh + kblocksh_ناپ;
    ثابت منطقي supported = کل_ناپ <= device->properties.limits.max_Compute_Shared_حافظہ_ناپ;

    VK_LOG_DEBUG("ggml_vk_flash_attn_scalar_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", mmq=" << mmq << ", total_size=" << کل_ناپ << ", supported=" << supported);

    أرجع supported;
}

منطقي ggml_vk_flash_attn_coopmat_shmem_support(ثابت vk_device& device, ثابت vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, منطقي f32acc, ggml_type k_type, ggml_type v_type) {
    GGML_بن_ورتیا(v_type);
    // Needs to be kept up to date on shader changes
    ثابت uint32_t Br = params.بلاک_صفاں;
    ثابت uint32_t Bc = params.بلاک_خانے_گنتی;

    ثابت uint32_t MatBr = 16, MatBc = 16;

    ثابت uint32_t صف_ونڈو = Bc / MatBc;

    ثابت uint32_t hsk_pad = ROUNDUP_POW2(hsk, 16);
    ثابت uint32_t hsv_pad = ROUNDUP_POW2(hsv, 16);

    ثابت uint32_t acctype = f32acc ? 4 : 2;
    ثابت uint32_t f16vec4 = 8;

    ثابت uint32_t tmpsh = (Bc / MatBc) * حجم(عائم);
    // DATA_A_IQ4_NL is compiled into the FA shaders unconditionally, so its shared table is always allocated.
    ثابت uint32_t iq_shmem = 16 * حجم(ggml_fp16_t);

    ثابت uint32_t qstride = hsk_pad / 4 + 2;
    ثابت uint32_t Qf = Br * qstride * f16vec4;

    ثابت uint32_t psh_stride = Br / 4 + 2;
    ثابت uint32_t Psh = Bc * psh_stride * f16vec4;

    ثابت uint32_t sfshstride = (hsk <= 128) ? (Br + 8) : Br;
    ثابت uint32_t sfsh = Bc * sfshstride * acctype;

    ثابت uint32_t kvshstride = (params.shmem_staging ? std::max(hsk_pad, hsv_pad) : MatBr) / 4 + 2;
    ثابت uint32_t vsh_stride = MatBc / 4 * صف_ونڈو;
    ثابت uint32_t ksh = ((kvshstride >= vsh_stride) ? (Bc * kvshstride) : (Bc * vsh_stride)) * f16vec4;

    // BF16 PVMat accumulator is f32 (no bf16 accumulator support), so pvsh is vec4 (16 bytes)
    ثابت uint32_t pvsh_elem_ناپ = (k_type == GGML_TYPE_BF16) ? 16u : f16vec4;
    ثابت uint32_t osh_stride = params.صف_ونڈو * MatBr / 4;
    ثابت uint32_t pvsh = MatBc * osh_stride * pvsh_elem_ناپ;

    ثابت uint32_t ڈھلوان = Br * acctype;

    ثابت uint32_t کل_ناپ = tmpsh + iq_shmem + Qf + Psh + sfsh + ksh + pvsh + ڈھلوان;
    ثابت منطقي supported = کل_ناپ <= device->properties.limits.max_Compute_Shared_حافظہ_ناپ;

    VK_LOG_DEBUG("ggml_vk_flash_attn_coopmat_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", f32acc=" << f32acc << ", total_size=" << کل_ناپ << ", supported=" << supported);

    أرجع supported;
}

فراغ ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * q, ثابت ggml_tensor * k, ثابت ggml_tensor * v, ثابت ggml_tensor * نقاب, ثابت ggml_tensor * sinks, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_flash_attn((" << q << ", name=" << q->name << ", type=" << q->type << ", ne0=" << q->ne[0] << ", ne1=" << q->ne[1] << ", ne2=" << q->ne[2] << ", ne3=" << q->ne[3] << ", nb0=" << q->nb[0] << ", nb1=" << q->nb[1] << ", nb2=" << q->nb[2] << ", nb3=" << q->nb[3];
    std::cerr << "), (" << k << ", name=" << k->name << ", type=" << k->type << ", ne0=" << k->ne[0] << ", ne1=" << k->ne[1] << ", ne2=" << k->ne[2] << ", ne3=" << k->ne[3] << ", nb0=" << k->nb[0] << ", nb1=" << k->nb[1] << ", nb2=" << k->nb[2] << ", nb3=" << k->nb[3];
    std::cerr << "), (" << v << ", name=" << v->name << ", type=" << v->type << ", ne0=" << v->ne[0] << ", ne1=" << v->ne[1] << ", ne2=" << v->ne[2] << ", ne3=" << v->ne[3] << ", nb0=" << v->nb[0] << ", nb1=" << v->nb[1] << ", nb2=" << v->nb[2] << ", nb3=" << v->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    إذا (sinks) {
        std::cerr << "), (" << sinks << ", name=" << sinks->name << ", type=" << sinks->type << ", ne0=" << sinks->ne[0] << ", ne1=" << sinks->ne[1] << ", ne2=" << sinks->ne[2] << ", ne3=" << sinks->ne[3] << ", nb0=" << sinks->nb[0] << ", nb1=" << sinks->nb[1] << ", nb2=" << sinks->nb[2] << ", nb3=" << sinks->nb[3];
    }
    std::cerr << "))");

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ثابت uint32_t nem0 = نقاب ? نقاب->ne[0] : 0;
    ثابت uint32_t nem1 = نقاب ? نقاب->ne[1] : 0;
    ثابت uint32_t nem2 = نقاب ? نقاب->ne[2] : 0;
    ثابت uint32_t nem3 = نقاب ? نقاب->ne[3] : 0;

    ثابت uint32_t HSK = nek0;
    ثابت uint32_t HSV = nev0;
    uint32_t N = neq1;
    ثابت uint32_t KV = nek1;

    GGML_ASSERT(ne0 == HSV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_ناپ(q->type));
    GGML_ASSERT(nbk0 == ggml_type_ناپ(k->type));
    GGML_ASSERT(nbv0 == ggml_type_ناپ(v->type));

    GGML_ASSERT(neq0 == HSK);

    GGML_ASSERT(neq1 == N);

    GGML_ASSERT(nev1 == nek1);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(عائم));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    أكد(dst->type == GGML_TYPE_F32);
    أكد(q->type == GGML_TYPE_F32);
    uint32_t gqa_تناسب = 1;
    uint32_t qk_تناسب = neq2 / nek2;
    uint32_t workgroups_x = (uint32_t)neq1;
    uint32_t workgroups_y = (uint32_t)neq2;
    uint32_t workgroups_z = (uint32_t)neq3;

    ثابت منطقي f32acc = !ctx->device->fp16 || dst->op_params[3] == GGML_PREC_F32 || k->type == GGML_TYPE_BF16;

    // dequant K/V once into an f16 scratch, reordered KV layout so FA can read without a stride
    خودکار is_dense_kv_cache = [](ثابت ggml_tensor * t) {
        أرجع t->nb[0] == ggml_type_ناپ(t->type) &&
               t->nb[2] == ggml_صف_ناپ(t->type, t->ne[0]) &&
               t->nb[1] == t->nb[2] * t->ne[2] &&
               (t->ne[3] == 1 || t->nb[3] == t->nb[1] * t->ne[1]);
    };
    ثابت منطقي k_quant = k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_BF16 && k->type != GGML_TYPE_F32;
    ثابت منطقي v_quant = v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_BF16 && v->type != GGML_TYPE_F32;
    ثابت منطقي kv_raw = (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16) && k->type == v->type;
    // gated to AMD, except RDNA4, which regressed in benchmarks
    ثابت منطقي kv_raw_strided = ctx->device->vendor_id == VK_VENDOR_ID_AMD &&
                                ctx->device->architecture != vk_device_architecture::AMD_RDNA4 &&
                                neq1 >= 256 &&
                                kv_raw &&
                                (k->nb[1] != (uint64_t)HSK * ggml_type_ناپ(k->type) ||
                                 v->nb[1] != (uint64_t)HSV * ggml_type_ناپ(v->type)) &&
                                (HSK % 8) == 0 && (HSV % 8) == 0;
    ثابت size_t kv_scratch_ts = kv_raw_strided ? ggml_type_ناپ(k->type) : حجم(ggml_fp16_t);
    ثابت منطقي use_dequant_kv = ((k_quant && v_quant) || kv_raw_strided) && neq1 >= 64 &&
                                is_dense_kv_cache(k) && is_dense_kv_cache(v) &&
                                (uint64_t)ggml_nelements(k) * kv_scratch_ts <= ctx->device->properties.limits.max_Storage_بفر_Range &&
                                (uint64_t)ggml_nelements(v) * kv_scratch_ts <= ctx->device->properties.limits.max_Storage_بفر_Range &&
                                (kv_raw_strided ||
                                 (ctx->device->pipeline_dequant_transpose[k->type] != خالی_سوچک &&
                                  ctx->device->pipeline_dequant_transpose[v->type] != خالی_سوچک)) &&
                                // coopmat2 path does not benefit from the f16 scratch
                                !ctx->device->coopmat2 &&
                                // Intel Xe1 regresses, see PR 25494
                                (ctx->device->vendor_id != VK_VENDOR_ID_INTEL ||
                                 (ctx->device->coopmat_support && ctx->device->architecture != vk_device_architecture::INTEL_XE1));
    ثابت ggml_type k_type_eff = (use_dequant_kv && !kv_raw_strided) ? GGML_TYPE_F16 : k->type;
    ثابت ggml_type v_type_eff = (use_dequant_kv && !kv_raw_strided) ? GGML_TYPE_F16 : v->type;

    // For scalar/coopmat1 FA, we can use the "large" size to accommodate qga.
    // For coopmat2 FA, we always use the small size (which is still pretty large for gqa).
    vk_fa_tuning_params tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, 512, KV, k_type_eff, v_type_eff, f32acc);
    ثابت uint32_t max_gqa = std::min(tuning_params.بلاک_صفاں, 32u);

    إذا (N <= 8 && qk_تناسب > 1 && qk_تناسب <= max_gqa &&
        qk_تناسب * nek2 == neq2 && nek2 == nev2 && nem2 <= 1) {
        // grouped query attention - make the N dimension equal to gqa_ratio, reduce
        // workgroups proportionally in y dimension. The shader will detect gqa_ratio > 1
        // and change addressing calculations to index Q's dimension 2.
        gqa_تناسب = qk_تناسب;
        N = gqa_تناسب;
        workgroups_y /= gqa_تناسب;
    }

    tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, N, KV, k_type_eff, v_type_eff, f32acc);

    عائم scale         = 1.0f;
    عائم max_bias      = 0.0f;
    عائم logit_softcap = 0.0f;

    حافظہ_نقل(&scale,         (ثابت عائم *) dst->op_params + 0, حجم(عائم));
    حافظہ_نقل(&max_bias,      (ثابت عائم *) dst->op_params + 1, حجم(عائم));
    حافظہ_نقل(&logit_softcap, (ثابت عائم *) dst->op_params + 2, حجم(عائم));

    إذا (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    // Sparse mask hint (op_params[4]): compact the <= n_kv_max finite positions and gather only those.
    ثابت int32_t n_kv_max = نقاب ? ggml_get_op_params_i32(dst, 4) : 0;
    ساكن ثابت منطقي disable_sparse = getenv("GGML_VK_FA_SPARSE_DISABLE") != خالی_سوچک;
    ثابت منطقي kv_f16 = k_type_eff == GGML_TYPE_F16 && v_type_eff == GGML_TYPE_F16;
    // cm2 dense is fast, so it needs a larger reduction to win.
    // With quantized K/V, sparse only breaks even around 16x (measured on RDNA3/RDNA4).
    ثابت int64_t min_تناسب = tuning_params.path == FA_COOPMAT2 ? 4 : (kv_f16 ? 2 : 16);
    // coopmat2 vector decode requires 8B strides.
    خودکار sparse_gather_aligned = [](ثابت ggml_tensor * t) {
        أرجع (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) ||
               (t->nb[1] | t->nb[2] | t->nb[3]) % (4 * حجم(ggml_fp16_t)) == 0;
    };
    ثابت منطقي use_sparse = !disable_sparse && n_kv_max > 0 && نقاب &&
                            max_bias == 0.0f && logit_softcap == 0.0f &&
                            (tuning_params.path != FA_COOPMAT2 || (sparse_gather_aligned(k) && sparse_gather_aligned(v))) &&
                            nem0 == KV &&
                            (int64_t)KV >= std::max<int64_t>(4096, min_تناسب * (int64_t)n_kv_max) &&
                            (gqa_تناسب > 1 || (tuning_params.path == FA_SCALAR && N == 1));

    ثابت uint32_t q_stride = (uint32_t)(nbq1 / ggml_type_ناپ(q->type));
    uint32_t k_stride = (uint32_t)(nbk1 / ggml_type_ناپ(k->type));
    uint32_t v_stride = (uint32_t)(nbv1 / ggml_type_ناپ(v->type));

    // For F32, the shader treats it as a block of size 4 (for vec4 loads)
    إذا (k->type == GGML_TYPE_F32) {
        k_stride /= 4;
    }
    إذا (v->type == GGML_TYPE_F32) {
        v_stride /= 4;
    }

    uint32_t nbk2_eff = (uint32_t)nbk2, nbk3_eff = (uint32_t)nbk3;
    uint32_t nbv2_eff = (uint32_t)nbv2, nbv3_eff = (uint32_t)nbv3;
    إذا (use_dequant_kv) {
        k_stride = HSK;
        v_stride = HSV;
        nbk2_eff = (uint32_t)((uint64_t)HSK * KV * حجم(ggml_fp16_t));
        nbk3_eff = (uint32_t)((uint64_t)HSK * KV * nek2 * حجم(ggml_fp16_t));
        nbv2_eff = (uint32_t)((uint64_t)HSV * KV * حجم(ggml_fp16_t));
        nbv3_eff = (uint32_t)((uint64_t)HSV * KV * nev2 * حجم(ggml_fp16_t));
    }
    ثابت uint32_t alignment = tuning_params.بلاک_خانے_گنتی;
    منطقي aligned = (KV % alignment) == 0 &&
                   // the "aligned" shader variant will forcibly align strides, for performance
                   (q_stride & 7) == 0 && (k_stride & 7) == 0 && (v_stride & 7) == 0;

    // Need to use the coopmat2 variant that clamps loads when HSK/HSV aren't sufficiently aligned.
    إذا (((HSK | HSV) % 16) != 0 && tuning_params.path == FA_COOPMAT2) {
        aligned = خطأ;
    }

    // Only use mask opt when the mask is fairly large. This hasn't been tuned extensively.
    منطقي use_نقاب_opt = نقاب && !use_sparse && nem1 >= 32 && nem0 * nem1 > 32768 && nem0 >= tuning_params.بلاک_خانے_گنتی * 16
                        && (ctx->device->architecture != vk_device_architecture::AMD_GCN || HSK > 256 || HSV > 256);
    vk_fa_pipeline_state fa_pipeline_state = get_fa_pipeline_state(ctx->device, tuning_params, HSK, HSV, aligned, f32acc,
                                                                   نقاب != خالی_سوچک, use_نقاب_opt, logit_softcap != 0, use_sparse, k_type_eff, v_type_eff);

    vk_pipeline pipeline = خالی_سوچک;

    منطقي xe_fa_opt = خطأ;
    منطقي fa_copy_qstate = خطأ;
    منطقي xe_fa_supported_platform =
        (ctx->device.get()->architecture == INTEL_XE2 && ctx->device.get()->properties.deviceID != 0xFD80 && ctx->device.get()->properties.deviceID != 0xFD81) ||
        (ctx->device.get()->architecture == INTEL_XE1 && ctx->device.get()->coopmat_support && ctx->device.get()->uma);
    منطقي xe_fa_supported_usage = neq0 % 32 == 0 && nev0 % 16 == 0 && q->nb[1] > q->nb[2] && k->nb[1] > k->nb[2] && v->nb[1] > v->nb[2] && نقاب != خالی_سوچک;
    منطقي xe_fa_supported_dtype = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 && (نقاب != خالی_سوچک && نقاب->type == GGML_TYPE_F16);
    std::pair<vk_pipeline, vk_pipeline> xe_fa_pipeline_dual_phases = { خالی_سوچک , خالی_سوچک };
    vk_pipeline xe_fa_pipeline = خالی_سوچک;
    size_t ناپ_p = 0;
    size_t ناپ_گروہ_max = 0;

    {
        std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
        خودکار &pipelines = ctx->device->pipeline_flash_attn_f32_f16;
        خودکار it = pipelines.find(fa_pipeline_state);
        إذا (it != pipelines.end()) {
            pipeline = it->second;
        } وإلا {
            pipelines[fa_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
        }
    }

    أكد(pipeline);
    // Compile early to initialize wg_denoms.
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    uint32_t ونڈو_kv = KV;
    uint32_t ونڈو_k = 1;

    // Intel Alchemist prefers more workgroups
    ثابت uint32_t shader_core_گنتی_multiplier = (ctx->device->vendor_id == VK_VENDOR_ID_INTEL && ctx->device->architecture != INTEL_XE2) ? 2 : 1;

    // Use a placeholder core count if one isn't available. split_k is a big help for perf.
    ثابت uint32_t shader_core_گنتی = ctx->device->shader_core_گنتی ? ctx->device->shader_core_گنتی * shader_core_گنتی_multiplier : 16;

    ثابت uint32_t Br = fa_pipeline_state.Br;
    ثابت uint32_t Bc = fa_pipeline_state.Bc;

    GGML_ASSERT(Br == pipeline->wg_denoms[0]);
    ثابت uint32_t Tr = CEIL_DIV(N, Br);

    // Try to use split_k when KV is large enough to be worth the overhead.
    // Sparse: split_kv carries n_kv_max, split_k partitions its blocks for occupancy.
    إذا (use_sparse) {
        ونڈو_kv = (uint32_t)n_kv_max;
        ثابت uint32_t کل_بلاکاں = CEIL_DIV((uint32_t)n_kv_max, Bc);
        ثابت uint32_t ادھار_wgs = (gqa_تناسب > 1 ? workgroups_x : Tr) * workgroups_y * workgroups_z;
        إذا (ادھار_wgs < shader_core_گنتی * 2) {
            ونڈو_k = shader_core_گنتی * 2 / ادھار_wgs;
        }
        ونڈو_k = std::max(1u, std::min(ونڈو_k, کل_بلاکاں));
        // Match the shader's per-split block count so no split is empty.
        ثابت uint32_t فی_بلاکاں = CEIL_DIV(کل_بلاکاں, ونڈو_k);
        ونڈو_k = CEIL_DIV(کل_بلاکاں, فی_بلاکاں);
    } وإلا إذا (gqa_تناسب > 1 && workgroups_x <= Br) {
        ونڈو_k = shader_core_گنتی * 2 / (workgroups_x * workgroups_y * workgroups_z);
    } وإلا إذا (gqa_تناسب <= 1) {
        uint32_t کل_wgs_no_ونڈو = Tr * workgroups_y * workgroups_z;
        إذا (کل_wgs_no_ونڈو < shader_core_گنتی * 2) {
            ونڈو_k = shader_core_گنتی * 2 / کل_wgs_no_ونڈو;
        }
    }

    إذا (!use_sparse && ونڈو_k > 1) {
        // Try to evenly split KV into split_k chunks, but it needs to be a multiple
        // of "align", so recompute split_k based on that.
        ونڈو_kv = ROUNDUP_POW2(std::max(1u, KV / ونڈو_k), alignment);
        ونڈو_k = CEIL_DIV(KV, ونڈو_kv);
        xe_fa_opt = xe_fa_supported_platform && xe_fa_supported_usage && xe_fa_supported_dtype;
        إذا (xe_fa_opt) {
            ثابت uint32_t ونڈو_p_ناپ = 32;
            ثابت size_t max_dim = (nek1 + ونڈو_p_ناپ - 1) / ونڈو_p_ناپ;
            ثابت size_t p_dim = max_dim * ونڈو_p_ناپ;
#إذا defined(VK_KHR_cooperative_میٹرکس) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
            خودکار& pipelines = ctx->device->pipeline_xe_fa_decode_dual_phases;
            خودکار it = pipelines.find({ (uint32_t)neq0, (uint32_t)nev0, qk_تناسب, (uint32_t)neq1 });
            إذا (it != pipelines.end()) {
                xe_fa_pipeline_dual_phases = it->second;
            } وإلا {
                pipelines[{(uint32_t)neq0, (uint32_t)nev0, qk_تناسب, (uint32_t)neq1}] = xe_fa_pipeline_dual_phases = std::make_pair(std::make_shared<vk_pipeline_struct>(), std::make_shared<vk_pipeline_struct>());
            }
#جے_انت
            إذا (xe_fa_pipeline_dual_phases.پہلا == خالی_سوچک || xe_fa_pipeline_dual_phases.second == خالی_سوچک) {
                xe_fa_opt = خطأ;
                fa_copy_qstate = خطأ;
            } وإلا {
                ناپ_p = neq1 * neq2 * p_dim * neq3 * حجم(ggml_fp16_t);
                ناپ_گروہ_max = neq1 * neq2 * max_dim * neq3 * حجم(عائم);
                size_t عارضی_ناپ = ggml_nelements(q) * حجم(ggml_fp16_t) + ناپ_p + ناپ_گروہ_max;
                fa_copy_qstate = سچ;
                إذا (ctx->prealloc_ناپ_x < عارضی_ناپ) {
                    ctx->prealloc_ناپ_x = عارضی_ناپ;
                    ggml_vk_preallocate_buffers(ctx, subctx);
                }

                إذا (ctx->prealloc_x_need_sync) {
                    ggml_vk_sync_buffers(ctx, subctx);
                }
            }
        }
    }

    إذا (xe_fa_opt == سچ) {
        use_نقاب_opt = خطأ;
    }

    // Reserve space for split_k temporaries. For each split x batch, we need to store the O matrix (D x ne1)
    // and the per-row m and L values (ne1 rows). We store all the matrices first, followed by the rows.
    // For matrices, the order is (inner to outer) [HSV, ne1, k, ne2, ne3].
    // For L/M, the order is (inner to outer) [ne1, k, ne2, ne3].
    ثابت uint64_t ونڈو_k_ناپ = ونڈو_k > 1 ? (HSV * ne1 * حجم(عائم) + ne1 * حجم(عائم) * 2) * ونڈو_k * ne2 * ne3 : 0;
    إذا (ونڈو_k_ناپ > ctx->device->properties.limits.max_Storage_بفر_Range) {
        GGML_ABORT("Requested preallocation size is too large");
    }
    إذا (ctx->prealloc_ناپ_ونڈو_k < ونڈو_k_ناپ) {
        ctx->prealloc_ناپ_ونڈو_k = ونڈو_k_ناپ;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }

    ثابت uint32_t نقاب_opt_عدد_dwords = CEIL_DIV(nem0, 16 * Bc);
    ثابت uint64_t نقاب_opt_ناپ = حجم(uint32_t) * نقاب_opt_عدد_dwords * CEIL_DIV(nem1, Br) * nem2 * nem3;

    vk_pipeline pipeline_fa_نقاب_opt = خالی_سوچک;
    إذا (use_نقاب_opt) {
        {
            std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
            خودکار &pipelines = ctx->device->pipeline_fa_نقاب_opt;
            خودکار it = pipelines.find({Br, Bc});
            إذا (it != pipelines.end()) {
                pipeline_fa_نقاب_opt = it->second;
            } وإلا {
                pipelines[{Br, Bc}] = pipeline_fa_نقاب_opt = std::make_shared<vk_pipeline_struct>();
            }
        }
        أكد(pipeline_fa_نقاب_opt);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline_fa_نقاب_opt, 1);

        إذا (ctx->prealloc_ناپ_y < نقاب_opt_ناپ) {
            ctx->prealloc_ناپ_y = نقاب_opt_ناپ;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا (ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    // Sparse index scratch reuses prealloc_y (mutually exclusive with mask opt).
    ثابت uint64_t sparse_idx_ناپ = use_sparse
        ? حجم(int32_t) * (uint64_t)n_kv_max * nem1 * nem2 * nem3
        : 0;
    vk_pipeline sparse_compact_pipeline = ctx->device->fa_sparse_compact_use_subgroups
        ? ctx->device->pipeline_fa_sparse_compact_subgroup
        : ctx->device->pipeline_fa_sparse_compact;
    إذا (use_sparse) {
        ggml_pipeline_request_descriptor_sets(ctx, sparse_compact_pipeline, 1);
        إذا (ctx->prealloc_ناپ_y < sparse_idx_ناپ) {
            ctx->prealloc_ناپ_y = sparse_idx_ناپ;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا (ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    ثابت uint32_t n_سر_kv   = neq2;
    ثابت uint32_t n_سر_log2 = 1u << (uint32_t) floorf(log2f((عائم) n_سر_kv));
    ثابت عائم m0 = powf(2.0f, -(max_bias       ) / n_سر_log2);
    ثابت عائم m1 = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);

    vk_subbuffer q_buf = ggml_vk_tensor_subbuffer(ctx, q);
    vk_subbuffer k_buf = ggml_vk_tensor_subbuffer(ctx, k);
    vk_subbuffer v_buf = ggml_vk_tensor_subbuffer(ctx, v);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer نقاب_buf = نقاب ? ggml_vk_tensor_subbuffer(ctx, نقاب) : q_buf;
    vk_subbuffer sinks_buf = sinks ? ggml_vk_tensor_subbuffer(ctx, sinks) : q_buf;
    vk_subbuffer نقاب_opt_buf = use_نقاب_opt ? ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0) : q_buf;
    vk_subbuffer sparse_buf = use_sparse ? ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0) : q_buf;
    إذا (use_نقاب_opt || use_sparse) {
        // the mask opt bits and the sparse index list overwrite a matmul input converted into prealloc_y
        ctx->prealloc_y_آخری_pipeline_ورتیا = خالی_سوچک;
        ctx->prealloc_y_آخری_tensor_ورتیا = خالی_سوچک;
    }

    إذا (use_dequant_kv) {
        ثابت uint64_t k_sz = (uint64_t)ggml_nelements(k) * kv_scratch_ts;
        ثابت uint64_t v_sz = (uint64_t)ggml_nelements(v) * kv_scratch_ts;
        إذا (ctx->prealloc_ناپ_x < k_sz + v_sz) {
            ctx->prealloc_ناپ_x = k_sz + v_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        vk_subbuffer k_dst = vk_subbuffer{ ctx->prealloc_x, 0,    k_sz };
        vk_subbuffer v_dst = vk_subbuffer{ ctx->prealloc_x, k_sz, v_sz };
        ثابت uint32_t k_nel = (uint32_t)ggml_nelements(k);
        ثابت uint32_t v_nel = (uint32_t)ggml_nelements(v);
        إذا (kv_raw_strided) {
            vk_pipeline cp_k = ggml_vk_get_cpy_pipeline(ctx, k, خالی_سوچک, k->type);
            vk_pipeline cp_v = ggml_vk_get_cpy_pipeline(ctx, v, خالی_سوچک, v->type);
            ggml_pipeline_request_descriptor_sets(ctx, cp_k, 1);
            ggml_pipeline_request_descriptor_sets(ctx, cp_v, 1);
            إذا (ctx->prealloc_x_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            خودکار make_pc = [](uint32_t hs, uint32_t nh, uint32_t kv, uint32_t nel) {
                vk_op_unary_ڈالو_constants pc{};
                pc.ne = nel;
                pc.ne00 = hs; pc.ne01 = nh;      pc.ne02 = kv; pc.ne03 = nel / (hs * nh * kv);
                pc.nb00 = 1;  pc.nb01 = hs;      pc.nb02 = hs * nh; pc.nb03 = hs * nh * kv;
                pc.ne10 = hs; pc.ne11 = nh;      pc.ne12 = kv; pc.ne13 = pc.ne03;
                pc.nb10 = 1;  pc.nb11 = hs * kv; pc.nb12 = hs; pc.nb13 = hs * kv * nh;
                init_pushconst_fastdiv(pc);
                أرجع pc;
            };
            خودکار cpy_elems = [](uint32_t ne) -> std::array<uint32_t, 3> {
                إذا (ne > 262144) { أرجع { 512, 512, CEIL_DIV(ne, 262144) }; }
                إذا (ne > 512)    { أرجع { 512, CEIL_DIV(ne, 512), 1 }; }
                أرجع { ne, 1, 1 };
            };
            ggml_vk_dispatch_pipeline(ctx, subctx, cp_k, { k_buf, k_dst },
                make_pc((uint32_t)HSK, (uint32_t)nek2, (uint32_t)KV, k_nel), cpy_elems(k_nel));
            ggml_vk_dispatch_pipeline(ctx, subctx, cp_v, { v_buf, v_dst },
                make_pc((uint32_t)HSV, (uint32_t)nev2, (uint32_t)KV, v_nel), cpy_elems(v_nel));
        } وإلا {
            vk_pipeline tr_k = ctx->device->pipeline_dequant_transpose[k->type];
            vk_pipeline tr_v = ctx->device->pipeline_dequant_transpose[v->type];
            ggml_pipeline_request_descriptor_sets(ctx, tr_k, 1);
            ggml_pipeline_request_descriptor_sets(ctx, tr_v, 1);
            إذا (ctx->prealloc_x_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            { ثابت std::vector<uint32_t> pc = { (uint32_t)HSK, (uint32_t)nek2, (uint32_t)KV, 0, k_nel };
              ggml_vk_dispatch_pipeline(ctx, subctx, tr_k, { k_buf, k_dst }, pc, { k_nel, 1, 1 }); }
            { ثابت std::vector<uint32_t> pc = { (uint32_t)HSV, (uint32_t)nev2, (uint32_t)KV, 0, v_nel };
              ggml_vk_dispatch_pipeline(ctx, subctx, tr_v, { v_buf, v_dst }, pc, { v_nel, 1, 1 }); }
        }
        ggml_vk_sync_buffers(ctx, subctx);
        k_buf = k_dst;
        v_buf = v_dst;
    }

    uint32_t نقاب_n_سر_log2 = ((sinks != خالی_سوچک) << 24) | n_سر_log2;

    إذا (use_نقاب_opt)
    {
        ثابت vk_op_flash_attn_نقاب_opt_ڈالو_constants opt_pc = {
            nem0,
            nem1,
            nem2,
            (uint32_t)(نقاب->nb[1] / حجم(ggml_fp16_t)),
            (uint32_t)(نقاب->nb[2] / حجم(ggml_fp16_t)),
            (uint32_t)(نقاب->nb[3] / حجم(ggml_fp16_t)),
            نقاب_opt_عدد_dwords,
            نقاب_opt_عدد_dwords * CEIL_DIV(nem1, Br),
            نقاب_opt_عدد_dwords * CEIL_DIV(nem1, Br) * nem2,
        };

        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline_fa_نقاب_opt,
                                  { نقاب_buf, نقاب_opt_buf }, opt_pc,
                                  { نقاب_opt_عدد_dwords, CEIL_DIV(nem1, Br), nem2 * nem3 });
        ggml_vk_sync_buffers(ctx, subctx);
    }

    إذا (use_sparse)
    {
        ثابت vk_op_flash_attn_sparse_compact_ڈالو_constants sc_pc = {
            KV,
            nem1,
            nem2,
            (uint32_t)(نقاب->nb[1] / حجم(ggml_fp16_t)),
            (uint32_t)(نقاب->nb[2] / حجم(ggml_fp16_t)),
            (uint32_t)(نقاب->nb[3] / حجم(ggml_fp16_t)),
            (uint32_t)n_kv_max,
        };

        ggml_vk_dispatch_pipeline(ctx, subctx, sparse_compact_pipeline,
                                  { نقاب_buf, sparse_buf }, sc_pc,
                                  { nem1, nem2, nem3 });
        ggml_vk_sync_buffers(ctx, subctx);
    }

    ثابت vk_flash_attn_ڈالو_constants pc = { N, KV,
                                              (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
                                              (uint32_t)neq2, (uint32_t)neq3,
                                              (uint32_t)nek2, (uint32_t)nek3,
                                              (uint32_t)nev2, (uint32_t)nev3,
                                              nem1, nem2, nem3,
                                              q_stride, (uint32_t)nbq2, (uint32_t)nbq3,
                                              k_stride, nbk2_eff, nbk3_eff,
                                              v_stride, nbv2_eff, nbv3_eff,
                                              scale, max_bias, logit_softcap,
                                              نقاب_n_سر_log2, m0, m1,
                                              gqa_تناسب, ونڈو_kv, ونڈو_k };

    إذا (xe_fa_opt && ونڈو_k > 1) {
        خودکار اتلا_power_دا_2 = [&](uint32_t وچ) {
            GGML_ASSERT(وچ != 0);
            إذا (وچ <= 1) أرجع 1u;
            uint32_t ret = وچ - 1;
            ret |= ret >> 1;
            ret |= ret >> 2;
            ret |= ret >> 4;
            ret |= ret >> 8;
            ret |= ret >> 16;
            أرجع ret + 1;
        };
        خودکار منزل_fp16_vk_0 = ggml_vk_get_منزل_fp16(ctx, q->type);
        ثابت uint32_t باہرلا_dim_فی_wg = qk_تناسب > 16 ? 8 : 16;
        size_t x_ne = ggml_nelements(q);
        size_t عارضی_buf_وتھ = 0;
        uint32_t سر_stride_k = uint32_t(nbk2 / ggml_type_ناپ(k->type));
        uint32_t سر_stride_v = uint32_t(nbv2 / ggml_type_ناپ(v->type));
        uint32_t batch_stride_q = uint32_t(nbq3 / ggml_type_ناپ(q->type));
        uint32_t batch_stride_k = uint32_t(nbk3 / ggml_type_ناپ(k->type));
        uint32_t batch_stride_v = uint32_t(nbv3 / ggml_type_ناپ(v->type));
        uint32_t batch_stride_m = نقاب ? uint32_t(نقاب->nb[3] / ggml_type_ناپ(نقاب->type)) : 0u;
        uint32_t batch_stride_o = uint32_t(nb3 / ggml_type_ناپ(dst->type));
        vk_fa_xe_opt_ڈالو_constants pc_ph1 = { (uint32_t)nek1, (uint32_t)neq1, (uint32_t)neq2, (uint32_t)nek2, qk_تناسب, 1, (sinks != خالی_سوچک) ? 1u : 0u, (uint32_t)k_stride, سر_stride_k,
            batch_stride_q, batch_stride_k, batch_stride_v, batch_stride_m, batch_stride_o, scale };
        vk_fa_xe_opt_ڈالو_constants pc_ph2 = pc_ph1;
        pc_ph2.nbkv_tok = v_stride;
        pc_ph2.nbkv_سر = سر_stride_v;
        vk_subbuffer q_عارضی_buf = fa_copy_qstate ? ggml_vk_subbuffer(ctx, ctx->prealloc_x, عارضی_buf_وتھ) : q_buf;
        عارضی_buf_وتھ += fa_copy_qstate ? x_ne * حجم(ggml_fp16_t) : 0;
        vk_subbuffer p_عارضی_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_x, عارضی_buf_وتھ);
        عارضی_buf_وتھ += ناپ_p;
        vk_subbuffer max_عارضی_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_x, عارضی_buf_وتھ);
        عارضی_buf_وتھ += ناپ_گروہ_max;
        uint32_t xe_native_sub_گروہ_ناپ = ctx->device.get()->architecture == INTEL_XE1 ? 8 : 16;
        uint32_t aligned_gqa_تناسب = اتلا_power_دا_2(qk_تناسب);
        uint32_t باہرلا_فی_wg_ph1 = std::min(256u / xe_native_sub_گروہ_ناپ, (uint32_t)neq1);
        uint32_t باہرلا_فی_wg_ph2 = std::min(std::max(16u / aligned_gqa_تناسب, 1u), (uint32_t)neq1);
        uint32_t ph1_wg = ((neq1 + باہرلا_فی_wg_ph1 - 1) / باہرلا_فی_wg_ph1) * nek2;
        uint32_t ph2_wg = ((neq1 + باہرلا_فی_wg_ph2 - 1) / باہرلا_فی_wg_ph2) * ne0 / باہرلا_dim_فی_wg;
        إذا (fa_copy_qstate) {
            ثابت std::vector<uint32_t> pc_cpy_fp16 =
            { (uint32_t)q->ne[0], (uint32_t)q->ne[1], (uint32_t)q->ne[2], (uint32_t)q->ne[3], (uint32_t)(x_ne) };
            ggml_vk_sync_buffers(ctx, subctx);
            ggml_pipeline_request_descriptor_sets(ctx, منزل_fp16_vk_0, 1);
            ggml_vk_dispatch_pipeline(ctx, subctx, منزل_fp16_vk_0, { q_buf, q_عارضی_buf }, pc_cpy_fp16, { (uint32_t)(x_ne), 1, 1 });
        }

        ggml_vk_sync_buffers(ctx, subctx);
        ggml_pipeline_request_descriptor_sets(ctx, xe_fa_pipeline_dual_phases.پہلا, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, xe_fa_pipeline_dual_phases.پہلا,
            { q_عارضی_buf, k_buf, نقاب_buf, p_عارضی_buf, max_عارضی_buf },
            pc_ph1, { (uint32_t)ph1_wg, (uint32_t)nek1, (uint32_t)neq3 });

        ggml_vk_sync_buffers(ctx, subctx);
        ggml_pipeline_request_descriptor_sets(ctx, xe_fa_pipeline_dual_phases.second, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, xe_fa_pipeline_dual_phases.second,
            { p_عارضی_buf, v_buf, max_عارضی_buf, sinks_buf, dst_buf },
            pc_ph2, { (uint32_t)ph2_wg, (uint32_t)nev2, (uint32_t)neq3 });

        ctx->prealloc_x_need_sync = سچ;
    } وإلا إذا (ونڈو_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_flash_attn_ونڈو_k_reduce, 1);

        إذا (ctx->prealloc_ونڈو_k_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        // We reuse workgroups_x to mean the number of splits, so we need to
        // cancel out the divide by wg_denoms[0].
        uint32_t dispatch_x;
        إذا (gqa_تناسب > 1) {
            workgroups_x *= pipeline->wg_denoms[0];
            dispatch_x = ونڈو_k * workgroups_x;
        } وإلا {
            dispatch_x = Tr * ونڈو_k * pipeline->wg_denoms[0];
        }

        vk_subbuffer ونڈو_k_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_ونڈو_k, 0);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                    {q_buf, k_buf, v_buf, نقاب_buf, sinks_buf, ونڈو_k_buf, نقاب_opt_buf, sparse_buf},
                                    pc, { dispatch_x, workgroups_y, workgroups_z });

        ggml_vk_sync_buffers(ctx, subctx);
        ثابت vk_op_flash_attn_ونڈو_k_reduce_ڈالو_constants pc2 = { HSV, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3, ونڈو_k, (sinks != خالی_سوچک) };
        ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_flash_attn_ونڈو_k_reduce,
                                    {ونڈو_k_buf, sinks_buf, dst_buf},
                                    pc2, { (uint32_t)ne1, HSV, (uint32_t)(ne2 * ne3) });
        ctx->prealloc_ونڈو_k_need_sync = سچ;
    } وإلا {
        إذا (gqa_تناسب > 1) {
            // When using gqa, we want one actual workgroup per batch, so cancel out wg_denoms
            workgroups_x *= pipeline->wg_denoms[0];
        }
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                    {q_buf, k_buf, v_buf, نقاب_buf, sinks_buf, dst_buf, نقاب_opt_buf, sparse_buf},
                                    pc, { workgroups_x, workgroups_y, workgroups_z });
    }

    إذا (use_dequant_kv) {
        ctx->prealloc_x_need_sync = سچ;
    }
    إذا (use_نقاب_opt || use_sparse) {
        ctx->prealloc_y_need_sync = سچ;
    }
}

ساكن vk_conv_shapes ggml_vk_conv_select_shape(ggml_backend_vk_context * ctx, uint32_t K, uint32_t NPQ) {
    خودکار n_tiles = [&](vk_conv_shapes s) {
        أرجع CEIL_DIV(K, vk_conv_بلاک_sizes[s].K)
            * CEIL_DIV(NPQ, vk_conv_بلاک_sizes[s].NPQ);
    };

    // We can't query number of shader cores on Intel, use 32 as a placeholder
    // so small convolutions will still choose a smaller tile.
    ثابت uint32_t shader_core_گنتی = ctx->device->shader_core_گنتی > 0 ? ctx->device->shader_core_گنتی : 32;

    // 128x128 isn't used with cm1 due to shared memory size; fall through to a smaller tile.
    منطقي allow_128x128 = سچ;
#إذا defined(VK_KHR_cooperative_میٹرکس) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    إذا (!ctx->device->coopmat2 && ctx->device->coopmat_support && ctx->device->coopmat_support_16x16x16_f16acc) {
        allow_128x128 = خطأ;
    }
#جے_انت

    إذا (allow_128x128 && K > 64 && n_tiles(CONV_SHAPE_128x128) >= shader_core_گنتی * 2) {
        أرجع CONV_SHAPE_128x128;
    } وإلا إذا (K <= 32 && n_tiles(CONV_SHAPE_32x256) >= shader_core_گنتی * 2) {
        أرجع CONV_SHAPE_32x256;
    } وإلا إذا (K <= 64 && n_tiles(CONV_SHAPE_64x128) >= shader_core_گنتی * 2) {
        أرجع CONV_SHAPE_64x128;
    } وإلا إذا (!allow_128x128 && K > 64 && n_tiles(CONV_SHAPE_64x128) >= shader_core_گنتی * 2) {
        // cm1 fallback for large K when 128x128 isn't available
        أرجع CONV_SHAPE_64x128;
    } وإلا {
        أرجع CONV_SHAPE_64x32;
    }
}

ساكن vk_pipeline ggml_vk_op_get_pipeline(ggml_backend_vk_context * ctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * dst, ggml_op op) {
    بدل (op) {
    حالة GGML_OP_GET_صفاں:
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        إذا (src0->type == GGML_TYPE_I32) {
            // i32 src only supports i32 result
            GGML_ASSERT(dst->type == GGML_TYPE_I32);
            أرجع ctx->device->pipeline_get_صفاں[src0->type];
        }
        إذا (dst->type == GGML_TYPE_F16) {
            أرجع ctx->device->pipeline_get_صفاں[src0->type];
        }
        إذا (dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_get_صفاں_f32[src0->type];
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_GET_صفاں_پٹھ:
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_get_صفاں_پٹھ_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_ACC:
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_acc_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_سیٹ:
        إذا (src0->type == src1->type && src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32)) {
            أرجع ctx->device->pipeline_سیٹ_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_جوڑو:
    حالة GGML_OP_SUB:
    حالة GGML_OP_MUL:
    حالة GGML_OP_DIV:
        إذا ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16)) {
            أرجع خالی_سوچک;
        }
        بدل (op) {
        حالة GGML_OP_جوڑو:
        {
            إذا (ctx->عدد_additional_fused_ops > 0) {
                إذا (ctx->do_جوڑو_rms_partials) {
                    أرجع ctx->device->pipeline_multi_جوڑو_rms[ctx->عدد_additional_fused_ops];
                } وإلا {
                    أرجع ctx->device->pipeline_multi_جوڑو[ctx->عدد_additional_fused_ops];
                }
            }
            إذا (ctx->do_جوڑو_rms_partials) {
                خودکار pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_جوڑو_rms_norepeat : ctx->device->pipeline_جوڑو_rms;
                أرجع pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
            } وإلا {
                خودکار pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_جوڑو_norepeat : ctx->device->pipeline_جوڑو;
                أرجع pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
            }
        }
        حالة GGML_OP_SUB:
        {
            خودکار pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_sub_norepeat : ctx->device->pipeline_sub;
            أرجع pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        حالة GGML_OP_MUL:
        {
            خودکار pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_mul_norepeat : ctx->device->pipeline_mul;
            أرجع pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        حالة GGML_OP_DIV:
        {
            خودکار pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_div_norepeat : ctx->device->pipeline_div;
            أرجع pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        افتراضي:
            اكسر;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_جوڑو_ID:
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_جوڑو_id_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_باہرلا_PROD:
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_باہرلا_prod_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_CONCAT: {
        إذا (!ggml_vk_concat_supported(src0, src1, dst)) {
            أرجع خالی_سوچک;
        }
        بدل (ggml_vk_concat_unit_ناپ(src0->type)) {
        حالة 1:
            أرجع ctx->device->pipeline_concat_i8;
        حالة 2:
            أرجع ctx->device->pipeline_concat_i16;
        حالة 4:
            أرجع ctx->device->pipeline_concat_i32;
        حالة 8:
            أرجع ctx->device->pipeline_concat_i64;
        افتراضي:
            أرجع خالی_سوچک;
        }
    }
    حالة GGML_OP_UPSCALE:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            uint32_t mode = (ggml_get_op_params_i32(dst, 0) & (0xFF | GGML_SCALE_FLAG_ANTIALIAS));
            بدل (mode) {
                حالة GGML_SCALE_منوال_سبھ_توں_نیڑے:
                    أرجع ctx->device->pipeline_upscale_سبھ_توں_نیڑے_f32;
                حالة GGML_SCALE_منوال_BILINEAR:
                    أرجع ctx->device->pipeline_upscale_bilinear_f32;
                حالة GGML_SCALE_منوال_BICUBIC:
                    أرجع ctx->device->pipeline_upscale_bicubic_f32;
                حالة GGML_SCALE_منوال_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS:
                    أرجع ctx->device->pipeline_upscale_bilinear_antialias_f32;
                افتراضي:
                    أرجع خالی_سوچک;
            }
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_SCALE:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_scale_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_SQR:
        إذا (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            أرجع ctx->device->pipeline_sqr[dst->type == GGML_TYPE_F16];
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_SQRT:
        إذا (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            أرجع ctx->device->pipeline_sqrt[dst->type == GGML_TYPE_F16];
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_SIN:
        إذا (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            أرجع ctx->device->pipeline_sin[dst->type == GGML_TYPE_F16];
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_COS:
        إذا (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            أرجع ctx->device->pipeline_cos[dst->type == GGML_TYPE_F16];
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_LOG:
        إذا (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            أرجع ctx->device->pipeline_log[dst->type == GGML_TYPE_F16];
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_TRI:
        إذا (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            أرجع ctx->device->pipeline_tri[dst->type == GGML_TYPE_F16];
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_DIAG:
        إذا (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            أرجع ctx->device->pipeline_diag[dst->type == GGML_TYPE_F16];
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_CLAMP:
        إذا (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            أرجع ctx->device->pipeline_clamp[dst->type == GGML_TYPE_F16];
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_PAD:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_pad_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_PAD_REFLECT_1D:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_pad_reflect_1d_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_ROLL:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_roll_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_REPEAT:
        إذا (ggml_type_ناپ(src0->type) == حجم(عائم) && ggml_type_ناپ(dst->type) == حجم(عائم)) {
            أرجع ctx->device->pipeline_repeat_i32;
        }
        إذا (ggml_type_ناپ(src0->type) == 2 && ggml_type_ناپ(dst->type) == 2) {
            أرجع ctx->device->pipeline_repeat_i16;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_REPEAT_پٹھ:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_repeat_پٹھ_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_CPY:
    حالة GGML_OP_CONT:
    حالة GGML_OP_DUP:
        أرجع ggml_vk_get_cpy_pipeline(ctx, src0, dst, dst->type);
    حالة GGML_OP_سیٹ_صفاں:
        {
            إذا (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
                أرجع خالی_سوچک;
            }
            ثابت صحيح src_idx = src0->type == GGML_TYPE_F16;
            إذا (src1->type == GGML_TYPE_I64) {
                أرجع ctx->device->pipeline_سیٹ_صفاں_i64[src_idx][dst->type];
            } وإلا إذا (src1->type == GGML_TYPE_I32) {
                أرجع ctx->device->pipeline_سیٹ_صفاں_i32[src_idx][dst->type];
            }
            أرجع خالی_سوچک;
        }
    حالة GGML_OP_SILU_پٹھ:
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_silu_پٹھ_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_NORM:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_norm_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_گروہ_NORM:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_گروہ_norm_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_RMS_NORM:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            إذا (ctx->do_جوڑو_rms_partials) {
                أرجع ctx->fused_rms_norm_منوال == RMS_NORM_MUL ? ctx->device->pipeline_rms_norm_mul_partials_f32 : ctx->device->pipeline_rms_norm_partials_f32;
            }
            أرجع ctx->fused_rms_norm_منوال == RMS_NORM_MUL ? ctx->device->pipeline_rms_norm_mul_f32 : ctx->device->pipeline_rms_norm_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_RMS_NORM_پٹھ:
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_rms_norm_پٹھ_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_L2_NORM:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_l2_norm_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_UNARY:
        إذا ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            أرجع خالی_سوچک;
        }

        بدل (ggml_get_unary_op(dst)) {
            حالة GGML_UNARY_OP_EXP:
                أرجع ctx->device->pipeline_exp[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_EXPM1:
                أرجع ctx->device->pipeline_expm1[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_ELU:
                أرجع ctx->device->pipeline_elu[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_SILU:
                أرجع ctx->device->pipeline_silu[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_GELU:
                أرجع ctx->device->pipeline_gelu[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_GELU_ERF:
                أرجع ctx->device->pipeline_gelu_erf[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_GELU_تیز:
                أرجع ctx->device->pipeline_gelu_تیز[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_RELU:
                أرجع ctx->device->pipeline_relu[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_XIELU:
                أرجع ctx->device->pipeline_xielu[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_NEG:
                أرجع ctx->device->pipeline_neg[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_TANH:
                أرجع ctx->device->pipeline_tanh[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_SIGMOID:
                أرجع ctx->device->pipeline_sigmoid[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_HARDSIGMOID:
                أرجع ctx->device->pipeline_hardsigmoid[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_HARDSWISH:
                أرجع ctx->device->pipeline_hardswish[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_ABS:
                أرجع ctx->device->pipeline_abs[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_SOFTPLUS:
                أرجع ctx->device->pipeline_softplus[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_قدم:
                أرجع ctx->device->pipeline_قدم[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_گیڑا:
                أرجع ctx->device->pipeline_گیڑا[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_CEIL:
                أرجع ctx->device->pipeline_ceil[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_FLOOR:
                أرجع ctx->device->pipeline_floor[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_TRUNC:
                أرجع ctx->device->pipeline_trunc[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_SGN:
                أرجع ctx->device->pipeline_sgn[dst->type == GGML_TYPE_F16];
            افتراضي:
                اكسر;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_GLU:
        إذا ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            أرجع خالی_سوچک;
        }

        بدل (ggml_get_glu_op(dst)) {
            حالة GGML_GLU_OP_GEGLU:
                أرجع ctx->device->pipeline_geglu[dst->type == GGML_TYPE_F16];
            حالة GGML_GLU_OP_REGLU:
                أرجع ctx->device->pipeline_reglu[dst->type == GGML_TYPE_F16];
            حالة GGML_GLU_OP_SWIGLU:
                أرجع ctx->device->pipeline_swiglu[dst->type == GGML_TYPE_F16];
            حالة GGML_GLU_OP_SWIGLU_OAI:
                أرجع ctx->device->pipeline_swiglu_oai[dst->type == GGML_TYPE_F16];
            حالة GGML_GLU_OP_SWIGLU_CLAMP:
                أرجع ctx->device->pipeline_swiglu_clamp[dst->type == GGML_TYPE_F16];
            حالة GGML_GLU_OP_GEGLU_ERF:
                أرجع ctx->device->pipeline_geglu_erf[dst->type == GGML_TYPE_F16];
            حالة GGML_GLU_OP_GEGLU_تیز:
                أرجع ctx->device->pipeline_geglu_تیز[dst->type == GGML_TYPE_F16];
            افتراضي:
                اكسر;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_DIAG_نقاب_INF:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_diag_نقاب_inf_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_SOFT_MAX:
        GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        GGML_ASSERT(!src2 || src2->type == GGML_TYPE_F32);

        إذا (ctx->عدد_additional_fused_ops) {
            uint32_t idx = (uint32_t)ceilf(log2f(عائم(dst->ne[0])));
            GGML_ASSERT(idx < عدد_topk_moe_pipelines);
            // use n_experts from push constant if it's not equal to the power of two spec constant
            منطقي use_ڈالو = dst->ne[0] != (1u << idx);
            أرجع ctx->device->pipeline_topk_moe[idx][use_ڈالو];
        }

        إذا (src0->type == GGML_TYPE_F32 && (src1 == خالی_سوچک || src1->type == GGML_TYPE_F32) && dst->type == GGML_TYPE_F32) {
            أرجع src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_wg512 : ctx->device->pipeline_soft_max_f32;
        }
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            أرجع src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_f16_wg512 : ctx->device->pipeline_soft_max_f32_f16;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_SOFT_MAX_پٹھ:
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_soft_max_پٹھ_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_ROPE:
    حالة GGML_OP_ROPE_پٹھ:
        {
            ثابت ggml_tensor *rope = ctx->عدد_additional_fused_ops == 2 ? dst->src[0]->src[0] : dst;
            ثابت صحيح mode = ((ثابت int32_t *) rope->op_params)[2];
            ثابت منطقي is_neox = mode & GGML_ROPE_TYPE_NEOX;
            ثابت منطقي is_mrope = mode & GGML_ROPE_TYPE_MROPE;
            ثابت منطقي is_vision = mode == GGML_ROPE_TYPE_VISION;

            إذا (is_neox) {
                إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    أرجع ctx->device->pipeline_rope_neox_f32;
                }
                إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    أرجع ctx->device->pipeline_rope_neox_f32_f16;
                }
                إذا (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    أرجع ctx->device->pipeline_rope_neox_f16;
                }
            } وإلا إذا (is_mrope && !is_vision) {
                إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    أرجع ctx->device->pipeline_rope_multi_f32;
                }
                إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    أرجع ctx->device->pipeline_rope_multi_f32_f16;
                }
                إذا (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    أرجع ctx->device->pipeline_rope_multi_f16;
                }
            } وإلا إذا (is_vision) {
                إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    أرجع ctx->device->pipeline_rope_vision_f32;
                }
                إذا (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    أرجع ctx->device->pipeline_rope_vision_f16;
                }
            } وإلا {
                إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    أرجع ctx->device->pipeline_rope_norm_f32;
                }
                إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    أرجع ctx->device->pipeline_rope_norm_f32_f16;
                }
                إذا (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    أرجع ctx->device->pipeline_rope_norm_f16;
                }
            }
            أرجع خالی_سوچک;
        }
    حالة GGML_OP_جوڑ:
    حالة GGML_OP_جوڑ_صفاں:
    حالة GGML_OP_اوسط:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_جوڑ_صفاں_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_CROSS_ENTROPY_LOSS:
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع src0->ne[0] > 1024 ? ctx->device->pipeline_cross_entropy_loss_f32_wg512 : ctx->device->pipeline_cross_entropy_loss_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_CROSS_ENTROPY_LOSS_پٹھ:
        // src0 is the scalar grad; src1 is logits
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2 && src2->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع src1->ne[0] > 1024 ? ctx->device->pipeline_cross_entropy_loss_پٹھ_f32_wg512 : ctx->device->pipeline_cross_entropy_loss_پٹھ_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_CUMSUM:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            إذا (src0->ne[0] <= 512) {
                أرجع ctx->device->pipeline_cumsum_small_f32;
            } وإلا {
                أرجع ctx->device->pipeline_cumsum_f32;
            }
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_SOLVE_TRI:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {

            vk_solve_tri_pipeline_state solve_tri_pipeline_state(src0->ne[0], src1->ne[0]);

            vk_pipeline pipeline = خالی_سوچک;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                خودکار it = ctx->device->pipeline_solve_tri_f32.find(solve_tri_pipeline_state);
                إذا (it != ctx->device->pipeline_solve_tri_f32.end()) {
                    pipeline = it->second;
                } وإلا {
                    ctx->device->pipeline_solve_tri_f32[solve_tri_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            أرجع pipeline;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_ARGMAX:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) {
            أرجع ctx->device->pipeline_argmax_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_گنتی_EQUAL:
        إذا (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I64) {
            أرجع ctx->device->pipeline_گنتی_equal_i32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_IM2_خانہ:
        إذا (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_im2col_f32;
        }
        إذا (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            أرجع ctx->device->pipeline_im2col_f32_f16;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_IM2_خانہ_3_D:
        إذا (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_im2col_3d_f32;
        }
        إذا (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            أرجع ctx->device->pipeline_im2col_3d_f32_f16;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_TIMESTEP_EMBEDDING:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_timestep_embedding_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_CONV_TRANSPOSE_1D:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_conv_transpose_1d_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_COL2IM_1D:
        بدل (src0->type) {
            حالة GGML_TYPE_F32:  أرجع ctx->device->pipeline_col2im_1d_f32;
            حالة GGML_TYPE_F16:  أرجع ctx->device->pipeline_col2im_1d_f16;
            حالة GGML_TYPE_BF16: أرجع ctx->device->pipeline_col2im_1d_bf16;
            افتراضي:             أرجع خالی_سوچک;
        }
    حالة GGML_OP_POOL_1D:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_pool1d_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_POOL_2D:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            // Wide windows use neighboring subgroup lanes to read neighboring inputs.
            إذا (ctx->device->pipeline_pool2d_subgroup_f32 && dst->op_params[1] >= int32_t(ctx->device->subgroup_ناپ)) {
                أرجع ctx->device->pipeline_pool2d_subgroup_f32;
            }
            أرجع ctx->device->pipeline_pool2d_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_RWKV_WKV6:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_rwkv_wkv6_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_RWKV_WKV7:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_rwkv_wkv7_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_GATED_خطی_ATTN:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_gated_خطی_attn_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_LIGHTNING_INDEXER:
        // only the k type selects a pipeline, the other types are fixed by ggml_lightning_indexer()
        إذا (ggml_vk_lightning_indexer_k_type_supported(src1->type)) {
            أرجع ctx->device->pipeline_lightning_indexer_f32[src1->type];
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_GATED_DELTA_NET:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ثابت uint32_t S_v = dst->src[2]->ne[0];
            ثابت uint32_t kda = (dst->src[3]->ne[0] == (int64_t)S_v) ? 1 : 0;
            uint32_t si;
            بدل (S_v) {
                حالة 16:  si = 0; اكسر;
                حالة 32:  si = 1; اكسر;
                حالة 64:  si = 2; اكسر;
                حالة 128: si = 3; اكسر;
                افتراضي: أرجع خالی_سوچک;
            }
            أرجع ctx->device->pipeline_gated_delta_net[si][kda];
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_SSM_جانچ:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ثابت uint32_t d_state = src0->ne[0];
            إذا (d_state == 128) {
                أرجع ctx->device->pipeline_ssm_جانچ_f32_d128;
            } وإلا إذا (d_state == 256) {
                أرجع ctx->device->pipeline_ssm_جانچ_f32_d256;
            }
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_SSM_CONV:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            بدل (ctx->عدد_additional_fused_ops) {
                حالة 0:  أرجع ctx->device->pipeline_ssm_conv_f32;
                حالة 1:  أرجع ctx->device->pipeline_ssm_conv_silu_f32;
                حالة 2:  أرجع ctx->device->pipeline_ssm_conv_bias_silu_f32;
                افتراضي: أرجع خالی_سوچک;
            }
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_OPT_قدم_ADAMW:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_opt_قدم_adamw_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_OPT_قدم_SGD:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_opt_قدم_sgd_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_LEAKY_RELU:
        إذا (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            أرجع ctx->device->pipeline_leaky_relu[dst->type == GGML_TYPE_F16];
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_CONV_2D:
    حالة GGML_OP_CONV_TRANSPOSE_2D:
        إذا (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            uint32_t K = dst->ne[2]; // Cout
            uint32_t NPQ = dst->ne[3] * dst->ne[1] * dst->ne[0]; // N * OH * OW
            vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, K, NPQ);

            منطقي transpose = dst->op == GGML_OP_CONV_TRANSPOSE_2D;
            uint32_t KW = (uint32_t)src0->ne[0];
            uint32_t KH = (uint32_t)src0->ne[1];
            uint32_t s0 = (uint32_t)(ggml_get_op_params_i32(dst, 0));
            uint32_t s1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 1) : s0;
            uint32_t p0 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 2) : 0;
            uint32_t p1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 3) : 0;
            uint32_t d0 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 4) : 1;
            uint32_t d1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 5) : 1;

            // tile-aligned shapes let the shader skip bounds checks
            ثابت uint32_t Cin = (uint32_t)src1->ne[2];
            ثابت uint32_t CRS = Cin * KW * KH;
            ثابت uint32_t BS_K   = vk_conv_بلاک_sizes[shape].K;
            ثابت uint32_t BS_CRS = vk_conv_بلاک_sizes[shape].CRS;
            ثابت uint32_t BS_NPQ = vk_conv_بلاک_sizes[shape].NPQ;
            ثابت uint32_t aligned = ((K   % BS_K   == 0) &&
                                      (CRS % BS_CRS == 0) &&
                                      (NPQ % BS_NPQ == 0)) ? 1u : 0u;

            vk_conv2d_pipeline_state conv2d_pipeline_state(s0, s1, p0, p1, d0, d1, KW, KH, aligned);

            std::map<vk_conv2d_pipeline_state, vk_pipeline> *pipelines = خالی_سوچک;
            إذا (op == GGML_OP_CONV_2D) {
                إذا (src0->type == GGML_TYPE_F32) {
                    pipelines = &ctx->device->pipeline_conv2d_f32[shape];
                } وإلا إذا (src0->type == GGML_TYPE_F16) {
                    pipelines = &ctx->device->pipeline_conv2d_f16_f32[shape];
                }
            } وإلا إذا (op == GGML_OP_CONV_TRANSPOSE_2D) {
                إذا (src0->type == GGML_TYPE_F32) {
                    pipelines = &ctx->device->pipeline_conv_transpose_2d_f32[shape];
                } وإلا إذا (src0->type == GGML_TYPE_F16) {
                    pipelines = &ctx->device->pipeline_conv_transpose_2d_f16_f32[shape];
                }
            }

            vk_pipeline pipeline = خالی_سوچک;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                خودکار it = pipelines->find(conv2d_pipeline_state);
                إذا (it != pipelines->end()) {
                    pipeline = it->second;
                } وإلا {
                    (*pipelines)[conv2d_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            أرجع pipeline;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_CONV_2D_DW:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            إذا (ggml_is_contiguous(src1)) {
                أرجع ctx->device->pipeline_conv2d_dw_whcn_f32;
            } وإلا إذا (ggml_is_contiguous_channels(src1)) {
                أرجع ctx->device->pipeline_conv2d_dw_cwhn_f32;
            }
        } وإلا إذا (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            إذا (ggml_is_contiguous(src1)) {
                أرجع ctx->device->pipeline_conv2d_dw_whcn_f16_f32;
            } وإلا إذا (ggml_is_contiguous_channels(src1)) {
                أرجع ctx->device->pipeline_conv2d_dw_cwhn_f16_f32;
            }
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_CONV_3D:
        إذا (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ثابت uint32_t OC = (uint32_t)ggml_get_op_params_i32(dst, 11);
            ثابت uint32_t IC = (uint32_t)ggml_get_op_params_i32(dst, 9);
            ثابت uint32_t N  = (uint32_t)ggml_get_op_params_i32(dst, 10);
            ثابت uint32_t NPQ = N * dst->ne[2] * dst->ne[1] * dst->ne[0];
            ثابت vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, OC, NPQ);

            ثابت uint32_t KW = (uint32_t)src0->ne[0];
            ثابت uint32_t KH = (uint32_t)src0->ne[1];
            ثابت uint32_t KD = (uint32_t)src0->ne[2];
            ثابت uint32_t s0 = (uint32_t)ggml_get_op_params_i32(dst, 0);
            ثابت uint32_t s1 = (uint32_t)ggml_get_op_params_i32(dst, 1);
            ثابت uint32_t s2 = (uint32_t)ggml_get_op_params_i32(dst, 2);
            ثابت uint32_t p0 = (uint32_t)ggml_get_op_params_i32(dst, 3);
            ثابت uint32_t p1 = (uint32_t)ggml_get_op_params_i32(dst, 4);
            ثابت uint32_t p2 = (uint32_t)ggml_get_op_params_i32(dst, 5);
            ثابت uint32_t d0 = (uint32_t)ggml_get_op_params_i32(dst, 6);
            ثابت uint32_t d1 = (uint32_t)ggml_get_op_params_i32(dst, 7);
            ثابت uint32_t d2 = (uint32_t)ggml_get_op_params_i32(dst, 8);

            ثابت uint32_t CRS = IC * KW * KH * KD;
            ثابت uint32_t BS_K   = vk_conv_بلاک_sizes[shape].K;
            ثابت uint32_t BS_CRS = vk_conv_بلاک_sizes[shape].CRS;
            ثابت uint32_t BS_NPQ = vk_conv_بلاک_sizes[shape].NPQ;
            ثابت uint32_t aligned = ((OC  % BS_K   == 0) &&
                                      (CRS % BS_CRS == 0) &&
                                      (NPQ % BS_NPQ == 0)) ? 1u : 0u;

            vk_conv3d_pipeline_state conv3d_pipeline_state(s0, s1, s2, p0, p1, p2, d0, d1, d2, KW, KH, KD, aligned);

            std::map<vk_conv3d_pipeline_state, vk_pipeline> *pipelines = خالی_سوچک;
            إذا (src0->type == GGML_TYPE_F32) {
                pipelines = &ctx->device->pipeline_conv3d_f32[shape];
            } وإلا إذا (src0->type == GGML_TYPE_F16) {
                pipelines = &ctx->device->pipeline_conv3d_f16_f32[shape];
            } وإلا {
                أرجع خالی_سوچک;
            }

            vk_pipeline pipeline = خالی_سوچک;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                خودکار it = pipelines->find(conv3d_pipeline_state);
                إذا (it != pipelines->end()) {
                    pipeline = it->second;
                } وإلا {
                    (*pipelines)[conv3d_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            أرجع pipeline;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_ADD1:
        إذا (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            أرجع ctx->device->pipeline_add1_f16_f16;
        }
        إذا (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            أرجع ctx->device->pipeline_add1_f16_f32;
        }
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_add1_f32_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_ARANGE:
        إذا (dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_arange_f32;
        }
        أرجع خالی_سوچک;
    حالة GGML_OP_FILL:
        إذا (dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_fill_f32;
        }
        إذا (dst->type == GGML_TYPE_F16) {
            أرجع ctx->device->pipeline_fill_f16;
        }
        أرجع خالی_سوچک;
    افتراضي:
        أرجع خالی_سوچک;
    }

    GGML_بن_ورتیا(src2);
}

سانچا<قسم_ناں PC>
ساكن فراغ ggml_vk_op_f32(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst, ggml_op op, PC&& pc, vk_pipeline pipeline_override = خالی_سوچک) {
    VK_LOG_DEBUG("ggml_vk_op_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    إذا (src1 != خالی_سوچک) {
        std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    }
    إذا (src2 != خالی_سوچک) {
        std::cerr << "), (" << src2 << ", name=" << src2->name << ", type=" << src2->type << ", ne0=" << src2->ne[0] << ", ne1=" << src2->ne[1] << ", ne2=" << src2->ne[2] << ", ne3=" << src2->ne[3] << ", nb0=" << src2->nb[0] << ", nb1=" << src2->nb[1] << ", nb2=" << src2->nb[2] << ", nb3=" << src2->nb[3];
    }
    إذا (src3 != خالی_سوچک) {
        std::cerr << "), (" << src3 << ", name=" << src3->name << ", type=" << src3->type << ", ne0=" << src3->ne[0] << ", ne1=" << src3->ne[1] << ", ne2=" << src3->ne[2] << ", ne3=" << src3->ne[3] << ", nb0=" << src3->nb[0] << ", nb1=" << src3->nb[1] << ", nb2=" << src3->nb[2] << ", nb3=" << src3->nb[3];
    }
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << ggml_op_name(op) << ")");
    GGML_ASSERT(op == GGML_OP_GET_صفاں || op == GGML_OP_CPY || op == GGML_OP_CONCAT || (!ggml_is_quantized(src0->type) && (src1 == خالی_سوچک || !ggml_is_quantized(src1->type))));  // NOLINT
    GGML_ASSERT(dst->بفر != خالی_سوچک);
    ثابت uint64_t ne00 = src0->ne[0];
    ثابت uint64_t ne01 = src0->ne[1];
    ثابت uint64_t ne02 = src0->ne[2];
    ثابت uint64_t ne03 = src0->ne[3];

    ثابت منطقي use_src1 = src1 != خالی_سوچک;
    ثابت uint64_t ne10 = use_src1 ? src1->ne[0] : 0;
    ثابت uint64_t ne11 = use_src1 ? src1->ne[1] : 0;
    ثابت uint64_t ne12 = use_src1 ? src1->ne[2] : 0;
    ثابت uint64_t ne13 = use_src1 ? src1->ne[3] : 0;

    ثابت منطقي use_src2 = src2 != خالی_سوچک;
    ثابت منطقي use_src3 = src3 != خالی_سوچک;

    init_pushconst_fastdiv(pc);

    vk_pipeline pipeline;
    إذا (pipeline_override) {
        pipeline = pipeline_override;
    } وإلا {
        pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, op);
    }

    إذا (pipeline == خالی_سوچک) {
        std::cerr << "ggml_vulkan: Error: Missing op: " << ggml_op_name(op) << " for " << ggml_type_name(src0->type);
        إذا (src1 != خالی_سوچک) {
            std::cerr << " and " << ggml_type_name(src1->type);
        }
        std::cerr << " to " << ggml_type_name(dst->type) << std::سطر_انت;
        GGML_ABORT("fatal error");
    }

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0, سچ);
    vk_subbuffer src1_buf = use_src1 ? ggml_vk_tensor_subbuffer(ctx, src1, سچ) : vk_subbuffer{};
    vk_subbuffer src2_buf = use_src2 ? ggml_vk_tensor_subbuffer(ctx, src2, سچ) : vk_subbuffer{};
    vk_subbuffer src3_buf = use_src3 ? ggml_vk_tensor_subbuffer(ctx, src3, سچ) : vk_subbuffer{};
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, سچ);

    // Compute misalignment offset for descriptors and store it in in push constants.
    init_pushconst_tensor_offsets(ctx, pc, src0, src1, src2, src3, dst);

    std::array<uint32_t, 3> elements;

    بدل (op) {
    حالة GGML_OP_NORM:
    حالة GGML_OP_RMS_NORM_پٹھ:
    حالة GGML_OP_L2_NORM:
    حالة GGML_OP_SOFT_MAX:
    حالة GGML_OP_SOFT_MAX_پٹھ:
    حالة GGML_OP_جوڑ_صفاں:
    حالة GGML_OP_CUMSUM:
    حالة GGML_OP_اوسط:
    حالة GGML_OP_ARGMAX:
        {
            ثابت uint32_t nr = ggml_nrows(src0);
            إذا (nr > 262144) {
                elements = { 512, 512, CEIL_DIV(nr, 262144) };
            } وإلا إذا (nr > 512) {
                elements = { 512, CEIL_DIV(nr, 512), 1 };
            } وإلا {
                elements = { nr, 1, 1 };
            }
        } اكسر;
    حالة GGML_OP_SOLVE_TRI:
        {
            uint32_t nr = (uint32_t)(ne02 * ne03);
            إذا (nr > 262144) {
                elements = { 512, 512, CEIL_DIV(nr, 262144) };
            } وإلا إذا (nr > 512) {
                elements = { 512, CEIL_DIV(nr, 512), 1 };
            } وإلا {
                elements = { nr, 1, 1 };
            }
        }
        اكسر;
    حالة GGML_OP_RMS_NORM:
        إذا (ctx->do_جوڑو_rms_partials) {
            // Run one element per thread, 128 threads per workgroup
            elements = { (uint32_t)CEIL_DIV(ne00, 128), 1, 1 };
        } وإلا {
            elements = { (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne03 };
            elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[1]);
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[2]);
        }
        اكسر;

    حالة GGML_OP_جوڑ:
        // We use GGML_OP_SUM_ROWS with 1 row.
        elements = { 1, 1, 1 };
        اكسر;
    حالة GGML_OP_گروہ_NORM:
        {
            ثابت uint32_t عدد_groups = dst->op_params[0];
            elements = { عدد_groups * (uint32_t)src0->ne[3], 1, 1 };
        } اكسر;
    حالة GGML_OP_DIAG_نقاب_INF:
        elements = { (uint32_t)ggml_nrows(src0), (uint32_t)ne00, 1 };
        اكسر;
    حالة GGML_OP_ROPE:
    حالة GGML_OP_ROPE_پٹھ:
        {
            uint32_t nrows = (uint32_t)ggml_nrows(src0);
            uint32_t z = 1;
            إذا (nrows > ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[0]) {
                z = CEIL_DIV(nrows, 32768);
                nrows = 32768;
            }
            elements = { nrows, (uint32_t)ne00, z };

        } اكسر;
    حالة GGML_OP_GET_صفاں:
        elements = { (uint32_t)ne00, (uint32_t)ne10, (uint32_t)(ne11 * ne12) };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[1]);
        elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[2]);
        اكسر;
    حالة GGML_OP_GET_صفاں_پٹھ:
        elements = { (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], 1 };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[1]);
        اكسر;
    حالة GGML_OP_ARGSORT:
        GGML_ASSERT(0);
        اكسر;
    حالة GGML_OP_IM2_خانہ:
        {
            ثابت منطقي is_2D = dst->op_params[6] == 1;

            ثابت uint32_t IC = src1->ne[is_2D ? 2 : 1];

            ثابت uint32_t KH = is_2D ? src0->ne[1] : 1;
            ثابت uint32_t KW =         src0->ne[0];

            ثابت uint32_t OH = is_2D ? dst->ne[2] : 1;
            ثابت uint32_t OW =         dst->ne[1];

            ثابت uint32_t batch = src1->ne[is_2D ? 3 : 2];

            ثابت uint32_t CHW = IC * KH * KW;
            // Cap X workgroups to limit concurrent IC channel reads.
            // The shader loops over X to cover the full CHW dimension.
            // AMD prefers a lower limit
            ثابت uint32_t min_cap = ctx->device->vendor_id == VK_VENDOR_ID_AMD ? 512u : 4096u;
            ثابت uint32_t x_elements = std::min(CHW, std::max(min_cap, OW * KH * KW));
            elements = { x_elements, OW, OH * batch };
            elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[1]);
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[2]);
        } اكسر;
    حالة GGML_OP_IM2_خانہ_3_D:
        {
            ثابت uint32_t IC = ((ثابت uint32_t *)(dst->op_params))[9];

            ثابت uint32_t N  = ne13 / IC;

            ثابت uint32_t KD = ne02;
            ثابت uint32_t KH = ne01;
            ثابت uint32_t KW = ne00;

            ثابت uint32_t OD = dst->ne[3] / N;
            ثابت uint32_t OH = dst->ne[2];
            ثابت uint32_t OW = dst->ne[1];

            ثابت uint32_t IC_KD_KH_KW = IC*KD*KH*KW;
            ثابت uint32_t N_OD_OH = N*OD*OH;

            elements = { IC_KD_KH_KW, OW, N_OD_OH };
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[2]);
        } اكسر;
    حالة GGML_OP_TIMESTEP_EMBEDDING:
        {
            ثابت uint32_t dim = dst->op_params[0];
            uint32_t half_ceil = (dim + 1) / 2;
            elements = { half_ceil, (uint32_t)src0->ne[0], 1 };
        } اكسر;
    حالة GGML_OP_CONV_TRANSPOSE_1D:
        {
            elements = {uint32_t(src0->ne[1]), 1, 1}; // parallelize in {Cout, 1, 1}
        } اكسر;
    حالة GGML_OP_COL2IM_1D:
        {
            elements = { uint32_t(dst->ne[0]), uint32_t(dst->ne[1]), 1 };
        } اكسر;
    حالة GGML_OP_POOL_1D:
        {
            ثابت uint32_t N = dst->ne[3] * dst->ne[2];
            ثابت uint32_t OC = dst->ne[1];
            ثابت uint32_t OL = dst->ne[0];
            elements = { N * OC * OL, 1, 1};
        } اكسر;
    حالة GGML_OP_POOL_2D:
        {
            ثابت uint32_t outputs = ggml_nelements(dst);
            ثابت uint32_t outputs_فی_workgroup = pipeline == ctx->device->pipeline_pool2d_subgroup_f32 ? 128 / ctx->device->subgroup_ناپ : 512;
            ثابت uint32_t workgroups = CEIL_DIV(outputs, outputs_فی_workgroup);
            ثابت uint32_t groups_y = CEIL_DIV(workgroups, ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[0]);
            ثابت uint32_t groups_x = CEIL_DIV(workgroups, groups_y);
            elements = { groups_x, groups_y, 1 };
        } اكسر;
    حالة GGML_OP_CONV_2D:
    حالة GGML_OP_CONV_TRANSPOSE_2D:
        إذا constexpr (std::is_same_v<PC, vk_op_conv2d_ڈالو_constants>) {
            ثابت uint32_t NPQ = pc.N * pc.OH * pc.OW;
            ثابت vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, pc.Cout, NPQ);
            ثابت uint32_t NPQ_بلاکاں = CEIL_DIV(NPQ, vk_conv_بلاک_sizes[shape].NPQ);

            elements = { pc.Cout, NPQ_بلاکاں, 1 };
            إذا (elements[1] > 512) {
                elements[2] = CEIL_DIV(elements[1], 512);
                elements[1] = 512;
            }
        } وإلا {
            GGML_ABORT("invalid push constant type for CONV_2D");
        }
        اكسر;
    حالة GGML_OP_CONV_3D:
        إذا constexpr (std::is_same_v<PC, vk_op_conv3d_ڈالو_constants>) {
            ثابت uint32_t NPQ = pc.N * pc.OD * pc.OH * pc.OW;
            ثابت vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, pc.OC, NPQ);
            ثابت uint32_t NPQ_بلاکاں = CEIL_DIV(NPQ, vk_conv_بلاک_sizes[shape].NPQ);

            elements = { pc.OC, NPQ_بلاکاں, 1 };
            إذا (elements[1] > 512) {
                elements[2] = CEIL_DIV(elements[1], 512);
                elements[1] = 512;
            }
        } وإلا {
            GGML_ABORT("invalid push constant type for CONV_3D");
        }
        اكسر;
    حالة GGML_OP_جوڑو:
    حالة GGML_OP_SUB:
    حالة GGML_OP_DIV:
    حالة GGML_OP_MUL:
    حالة GGML_OP_ADD1:
    حالة GGML_OP_باہرلا_PROD:
    حالة GGML_OP_ARANGE:
    حالة GGML_OP_FILL:
    حالة GGML_OP_SCALE:
    حالة GGML_OP_SQR:
    حالة GGML_OP_SQRT:
    حالة GGML_OP_SIN:
    حالة GGML_OP_COS:
    حالة GGML_OP_LOG:
    حالة GGML_OP_TRI:
    حالة GGML_OP_DIAG:
    حالة GGML_OP_CLAMP:
    حالة GGML_OP_LEAKY_RELU:
    حالة GGML_OP_PAD:
    حالة GGML_OP_PAD_REFLECT_1D:
    حالة GGML_OP_ROLL:
    حالة GGML_OP_REPEAT:
    حالة GGML_OP_REPEAT_پٹھ:
    حالة GGML_OP_CPY:
    حالة GGML_OP_CONCAT:
    حالة GGML_OP_UPSCALE:
    حالة GGML_OP_UNARY:
    حالة GGML_OP_GLU:
    حالة GGML_OP_CONV_2D_DW:
        {
            uint32_t ne = ggml_nelements(dst);
            إذا (op == GGML_OP_CPY && ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                // Convert from number of logical elements to 2- or 4-byte units.
                ne /= ggml_blck_ناپ(src0->type);
                إذا ((ggml_type_ناپ(src0->type) % 4) == 0) {
                    ne *= ggml_type_ناپ(src0->type) / 4;
                } وإلا {
                    ne *= ggml_type_ناپ(src0->type) / 2;
                }
            }
            إذا (op == GGML_OP_CONCAT && ggml_is_quantized(dst->type)) {
                ne = ne / ggml_blck_ناپ(dst->type) * ggml_type_ناپ(dst->type) / ggml_vk_concat_unit_ناپ(dst->type);
            }
            // copy_to_quant has block size of 32, and each thread does QUANT_K elements.
            // Splitting into 512x512xZ wouldn't work well since each workgroup does 1024 elements.
            // So divide by block size here before splitting into 512x512 groups.
            إذا (op == GGML_OP_CPY && !ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                ne = CEIL_DIV(ne, ggml_blck_ناپ(dst->type));
            }
            إذا (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } وإلا إذا (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } وإلا {
                elements = { ne, 1, 1 };
            }

            إذا (pipeline == ctx->device->pipeline_cpy_transpose_02_32 ||
                pipeline == ctx->device->pipeline_cpy_transpose_02_16) {
                // 32x32 tiles over dims 0 and 2; dim1 and dim3 are the batch
                elements[0] = (uint32_t)CEIL_DIV(dst->ne[0], 32);
                elements[1] = (uint32_t)CEIL_DIV(dst->ne[2], 32);
                elements[2] = (uint32_t)(dst->ne[1]*dst->ne[3]);
                elements[0] = std::min(elements[0], ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[0]);
                elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[1]);
                elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[2]);
            } وإلا إذا (pipeline == ctx->device->pipeline_cpy_transpose_32 ||
                pipeline == ctx->device->pipeline_cpy_transpose_16) {
                // 32x32 tiles
                elements[0] = (uint32_t)CEIL_DIV(dst->ne[0], 32);
                elements[1] = (uint32_t)CEIL_DIV(dst->ne[1], 32);
                elements[2] = (uint32_t)(dst->ne[2]*dst->ne[3]);
                elements[0] = std::min(elements[0], ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[0]);
                elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[1]);
                elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[2]);
            }
        } اكسر;
    حالة GGML_OP_جوڑو_ID:
        {
            elements = { (uint32_t)ne01, (uint32_t)ne02, 1 };
        } اكسر;
    حالة GGML_OP_سیٹ_صفاں:
        {
            uint32_t ne = ggml_nelements(src0);
            إذا (ggml_is_quantized(dst->type)) {
                // quants run 32 threads each doing QUANT_K elements
                ne = CEIL_DIV(ne, 32 * ggml_blck_ناپ(dst->type));
            } وإلا {
                // scalar types do one element per thread, running 512 threads
                ne = CEIL_DIV(ne, 512);
            }
            إذا (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } وإلا إذا (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } وإلا {
                elements = { ne, 1, 1 };
            }
        }
        اكسر;
    حالة GGML_OP_SSM_CONV:
        {
            ثابت uint32_t nr  = src0->ne[1];
            ثابت uint32_t n_t = dst->ne[1];
            ثابت uint32_t n_s = dst->ne[2];
            elements = { nr, n_t, n_s };
        }
        اكسر;
    افتراضي:
        elements = { (uint32_t)ggml_nelements(src0), 1, 1 };
        اكسر;
    }

    إذا (op == GGML_OP_جوڑو || op == GGML_OP_RMS_NORM) {
        vk_subbuffer a_buf = src0_buf;
        إذا (ctx->do_جوڑو_rms_partials) {
            a_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_جوڑو_rms_partials, ctx->prealloc_ناپ_جوڑو_rms_partials_وتھ);
        }
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            { src0_buf, src1_buf, dst_buf, a_buf }, pc, elements);
    } وإلا إذا (op == GGML_OP_GLU) {
        // Empty src1 is possible in glu, but the shader needs a buffer
        vk_subbuffer subbuf1 = use_src1 ? src1_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc, elements);
    } وإلا إذا (op == GGML_OP_SOFT_MAX) {
        // Empty src1 and src2 is possible in soft_max, but the shader needs a buffer
        vk_subbuffer subbuf1 = use_src1 ? src1_buf : src0_buf;
        vk_subbuffer subbuf2 = use_src2 ? src2_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, subbuf2, dst_buf }, pc, elements);
    } وإلا إذا (op == GGML_OP_ROPE || op == GGML_OP_ROPE_پٹھ) {
        // Empty src2 and src3 is possible in rope, but the shader needs a buffer
        vk_subbuffer subbuf2 = use_src2 ? src2_buf : src0_buf;
        vk_subbuffer subbuf3 = use_src3 ? src3_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, subbuf2, dst_buf, subbuf3 }, pc, elements);
    } وإلا إذا (op == GGML_OP_IM2_خانہ || op == GGML_OP_IM2_خانہ_3_D) {
        إذا (ctx->device->shader_int64 && ctx->device->بفر_device_پتہ) {
            // buffer device address path doesn't use dst buffer
            dst_buf.size = 1;
        }
        // im2col uses only src1 and dst buffers
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src1_buf, dst_buf }, pc, elements);
    } وإلا إذا (op == GGML_OP_گنتی_EQUAL) {
        // count_equal assumes that destination buffer is initialized with zeroes
        ggml_vk_بفر_memset_async(subctx, dst_buf.بفر, dst_buf.وتھ, 0, dst_buf.size);
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, dst_buf }, pc, elements);
    } وإلا إذا (op == GGML_OP_OPT_قدم_SGD) {
        // OPT_STEP_SGD works on src0, it does not need dst
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf }, pc, elements);
    } وإلا إذا (use_src3) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf, src3_buf, dst_buf }, pc, elements);
    } وإلا إذا (use_src2) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf, dst_buf }, pc, elements);
    } وإلا إذا (use_src1) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, dst_buf }, pc, elements);
    } وإلا {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, dst_buf }, pc, elements);
    }
}

فراغ ggml_vk_get_صفاں(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_ناپ = ggml_type_ناپ(src0->type);
    ثابت uint32_t src1_type_ناپ = ggml_type_ناپ(src1->type);
    ثابت uint32_t dst_type_ناپ = ggml_type_ناپ(dst->type);

    ggml_vk_op_f32<vk_op_ثنائی_ڈالو_constants>(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_GET_صفاں, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ناپ, (uint32_t)src0->nb[1] / src0_type_ناپ, (uint32_t)src0->nb[2] / src0_type_ناپ, (uint32_t)src0->nb[3] / src0_type_ناپ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ناپ, (uint32_t)src1->nb[1] / src1_type_ناپ, (uint32_t)src1->nb[2] / src1_type_ناپ, (uint32_t)src1->nb[3] / src1_type_ناپ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ناپ, (uint32_t) dst->nb[1] /  dst_type_ناپ, (uint32_t) dst->nb[2] /  dst_type_ناپ, (uint32_t) dst->nb[3] /  dst_type_ناپ,
        0,
        0.0f, 0.0f, 0,
    });
}

فراغ ggml_vk_get_صفاں_پٹھ(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_ناپ = ggml_type_ناپ(src0->type);
    ثابت uint32_t src1_type_ناپ = ggml_type_ناپ(src1->type);
    ثابت uint32_t dst_type_ناپ = ggml_type_ناپ(dst->type);

    ggml_vk_op_f32<vk_op_ثنائی_ڈالو_constants>(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_GET_صفاں_پٹھ, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2], (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ناپ, (uint32_t)src0->nb[1] / src0_type_ناپ, (uint32_t)src0->nb[2] / src0_type_ناپ, (uint32_t)src0->nb[3] / src0_type_ناپ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2], (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ناپ, (uint32_t)src1->nb[1] / src1_type_ناپ, (uint32_t)src1->nb[2] / src1_type_ناپ, (uint32_t)src1->nb[3] / src1_type_ناپ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], (uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ناپ, (uint32_t) dst->nb[1] /  dst_type_ناپ, (uint32_t) dst->nb[2] /  dst_type_ناپ, (uint32_t) dst->nb[3] /  dst_type_ناپ,
        0,
        0.0f, 0.0f, 0,
    });
}

فراغ ggml_vk_acc(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_ناپ = ggml_type_ناپ(src0->type);
    ثابت uint32_t src1_type_ناپ = ggml_type_ناپ(src1->type);
    ثابت uint32_t dst_type_ناپ = ggml_type_ناپ(dst->type);

    صحيح nb1 = dst->op_params[0] / src0_type_ناپ; // 4 bytes of float32
    صحيح nb2 = dst->op_params[1] / src0_type_ناپ; // 4 bytes of float32
    صحيح nb3 = dst->op_params[2] / src0_type_ناپ; // 4 bytes of float32
    صحيح وتھ = dst->op_params[3] / src0_type_ناپ; // offset in bytes

    ggml_vk_op_f32<vk_op_ثنائی_ڈالو_constants>(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, dst->op, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ناپ, (uint32_t)nb1, (uint32_t)nb2, (uint32_t)nb3,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ناپ, (uint32_t)src1->nb[1] / src1_type_ناپ, (uint32_t)src1->nb[2] / src1_type_ناپ, (uint32_t)src1->nb[3] / src1_type_ناپ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ناپ, (uint32_t)nb1, (uint32_t)nb2, (uint32_t)nb3,
        0,
        0.0f, 0.0f, وتھ,
    });
}

فراغ ggml_vk_multi_جوڑو(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, صحيح گرہ_idx) {
    ثابت ggml_tensor *پہلا_گرہ = cgraph->nodes[گرہ_idx];
    ثابت ggml_tensor *dst = cgraph->nodes[گرہ_idx + ctx->عدد_additional_fused_ops];

    // Make a list of all the tensors used by the op.
    // Last element of the list is the dest tensor.
    ثابت ggml_tensor *tensors[MAX_PARAMETER_گنتی];
    uint32_t عدد_srcs = ctx->عدد_additional_fused_ops + 2;
    uint32_t عدد_tensors = عدد_srcs + 1;
    GGML_ASSERT(عدد_tensors + ctx->do_جوڑو_rms_partials <= MAX_PARAMETER_گنتی);

    tensors[0] = پہلا_گرہ->src[0];
    tensors[1] = پہلا_گرہ->src[1];
    لكل (int32_t i = 0; i < ctx->عدد_additional_fused_ops; ++i) {
        // check whether the previous result is src[0] or src[1]
        إذا (cgraph->nodes[گرہ_idx + i] == cgraph->nodes[گرہ_idx + i + 1]->src[0]) {
            tensors[i+2] = cgraph->nodes[گرہ_idx + i + 1]->src[1];
        } وإلا {
            tensors[i+2] = cgraph->nodes[گرہ_idx + i + 1]->src[0];
        }
    }
    tensors[عدد_srcs] = dst;

    vk_op_multi_جوڑو_ڈالو_constants pc;
    pc.ne20 = (uint32_t)dst->ne[0];
    pc.ne21 = (uint32_t)dst->ne[1];
    pc.ne22 = (uint32_t)dst->ne[2];
    pc.ne23 = (uint32_t)dst->ne[3];

    لكل (uint32_t i = 0; i < عدد_tensors; ++i) {
        ثابت ggml_tensor *t = tensors[i];
        pc.nb[i][0] = (uint32_t)t->nb[0] / حجم(عائم);
        pc.nb[i][1] = (uint32_t)t->nb[1] / حجم(عائم);
        pc.nb[i][2] = (uint32_t)t->nb[2] / حجم(عائم);
        pc.nb[i][3] = (uint32_t)t->nb[3] / حجم(عائم);
    }
    pc.rms_partials = ctx->do_جوڑو_rms_partials;

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, tensors[0], tensors[1], خالی_سوچک, dst, dst->op);

    إذا (pipeline == خالی_سوچک) {
        std::cerr << "ggml_vulkan: Error: Missing multi_add";
        GGML_ABORT("fatal error");
    }

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ggml_backend_vk_بفر_context * buf_ctx[MAX_PARAMETER_گنتی];
    vk_بفر buf[MAX_PARAMETER_گنتی];
    size_t وتھ[MAX_PARAMETER_گنتی];
    منطقي uma[MAX_PARAMETER_گنتی];

    لكل (uint32_t i = 0; i < عدد_tensors; ++i) {
        buf_ctx[i] = (ggml_backend_vk_بفر_context *)tensors[i]->بفر->context;
        buf[i] = خالی_سوچک;
        وتھ[i] = 0;
        uma[i] = خطأ;

        إذا (ctx->device->uma) {
            ggml_vk_host_get(ctx->device, tensors[i]->انکڑے, buf[i], وتھ[i]);
            uma[i] = buf[i] != خالی_سوچک;
        }
        إذا (!uma[i]) {
            buf[i] = buf_ctx[i]->dev_بفر;
            وتھ[i] = vk_tensor_وتھ(tensors[i]) + tensors[i]->view_offs;
        }
        GGML_ASSERT(buf[i] != خالی_سوچک);
    }
    // If any remaining descriptors are unused, just point them at src[0]
    لكل (uint32_t i = عدد_tensors; i < MAX_PARAMETER_گنتی; ++i) {
        buf[i] = buf[0];
        وتھ[i] = 0;
    }
    إذا (ctx->do_جوڑو_rms_partials) {
        buf[عدد_tensors] = ctx->prealloc_جوڑو_rms_partials;
        وتھ[عدد_tensors] = ctx->prealloc_ناپ_جوڑو_rms_partials_وتھ;
    }

    std::array<uint32_t, 3> elements;

    uint32_t ne = ggml_nelements(dst);
    إذا (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } وإلا إذا (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } وإلا {
        elements = { ne, 1, 1 };
    }

    static_assert(MAX_PARAMETER_گنتی == 12);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            ggml_vk_subbuffer(ctx, buf[0], وتھ[0]),
            ggml_vk_subbuffer(ctx, buf[1], وتھ[1]),
            ggml_vk_subbuffer(ctx, buf[2], وتھ[2]),
            ggml_vk_subbuffer(ctx, buf[3], وتھ[3]),
            ggml_vk_subbuffer(ctx, buf[4], وتھ[4]),
            ggml_vk_subbuffer(ctx, buf[5], وتھ[5]),
            ggml_vk_subbuffer(ctx, buf[6], وتھ[6]),
            ggml_vk_subbuffer(ctx, buf[7], وتھ[7]),
            ggml_vk_subbuffer(ctx, buf[8], وتھ[8]),
            ggml_vk_subbuffer(ctx, buf[9], وتھ[9]),
            ggml_vk_subbuffer(ctx, buf[10], وتھ[10]),
            ggml_vk_subbuffer(ctx, buf[11], وتھ[11]),
        }, pc, elements);
}

فراغ ggml_vk_جوڑو(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_ناپ = ggml_type_ناپ(src0->type);
    ثابت uint32_t src1_type_ناپ = ggml_type_ناپ(src1->type);
    ثابت uint32_t dst_type_ناپ = ggml_type_ناپ(dst->type);

    ggml_vk_op_f32<vk_op_ثنائی_ڈالو_constants>(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_جوڑو, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ناپ, (uint32_t)src0->nb[1] / src0_type_ناپ, (uint32_t)src0->nb[2] / src0_type_ناپ, (uint32_t)src0->nb[3] / src0_type_ناپ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ناپ, (uint32_t)src1->nb[1] / src1_type_ناپ, (uint32_t)src1->nb[2] / src1_type_ناپ, (uint32_t)src1->nb[3] / src1_type_ناپ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ناپ, (uint32_t) dst->nb[1] /  dst_type_ناپ, (uint32_t) dst->nb[2] /  dst_type_ناپ, (uint32_t) dst->nb[3] /  dst_type_ناپ,
        0,
        0.0f, 0.0f, ctx->do_جوڑو_rms_partials,
    });
}

فراغ ggml_vk_باہرلا_prod(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_ناپ = ggml_type_ناپ(src0->type);
    ثابت uint32_t src1_type_ناپ = ggml_type_ناپ(src1->type);
    ثابت uint32_t dst_type_ناپ = ggml_type_ناپ(dst->type);

    ggml_vk_op_f32<vk_op_ثنائی_ڈالو_constants>(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_باہرلا_PROD, {
        (uint32_t)ggml_nelements(dst),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3],
        (uint32_t)src0->nb[0] / src0_type_ناپ, (uint32_t)src0->nb[1] / src0_type_ناپ, (uint32_t)src0->nb[2] / src0_type_ناپ, (uint32_t)src0->nb[3] / src0_type_ناپ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3],
        (uint32_t)src1->nb[0] / src1_type_ناپ, (uint32_t)src1->nb[1] / src1_type_ناپ, (uint32_t)src1->nb[2] / src1_type_ناپ, (uint32_t)src1->nb[3] / src1_type_ناپ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3],
        (uint32_t) dst->nb[0] /  dst_type_ناپ, (uint32_t) dst->nb[1] /  dst_type_ناپ, (uint32_t) dst->nb[2] /  dst_type_ناپ, (uint32_t) dst->nb[3] /  dst_type_ناپ,
        0,
        0.0f, 0.0f, 0,
    });
}

فراغ ggml_vk_sub(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_ناپ = ggml_type_ناپ(src0->type);
    ثابت uint32_t src1_type_ناپ = ggml_type_ناپ(src1->type);
    ثابت uint32_t dst_type_ناپ = ggml_type_ناپ(dst->type);

    ggml_vk_op_f32<vk_op_ثنائی_ڈالو_constants>(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_SUB, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ناپ, (uint32_t)src0->nb[1] / src0_type_ناپ, (uint32_t)src0->nb[2] / src0_type_ناپ, (uint32_t)src0->nb[3] / src0_type_ناپ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ناپ, (uint32_t)src1->nb[1] / src1_type_ناپ, (uint32_t)src1->nb[2] / src1_type_ناپ, (uint32_t)src1->nb[3] / src1_type_ناپ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ناپ, (uint32_t) dst->nb[1] /  dst_type_ناپ, (uint32_t) dst->nb[2] /  dst_type_ناپ, (uint32_t) dst->nb[3] /  dst_type_ناپ,
        0,
        0.0f, 0.0f, 0,
    });
}

فراغ ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_ناپ = ggml_type_ناپ(src0->type);
    ثابت uint32_t src1_type_ناپ = ggml_type_ناپ(src1->type);
    ثابت uint32_t dst_type_ناپ = ggml_type_ناپ(dst->type);

    ggml_vk_op_f32<vk_op_ثنائی_ڈالو_constants>(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_MUL, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ناپ, (uint32_t)src0->nb[1] / src0_type_ناپ, (uint32_t)src0->nb[2] / src0_type_ناپ, (uint32_t)src0->nb[3] / src0_type_ناپ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ناپ, (uint32_t)src1->nb[1] / src1_type_ناپ, (uint32_t)src1->nb[2] / src1_type_ناپ, (uint32_t)src1->nb[3] / src1_type_ناپ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ناپ, (uint32_t) dst->nb[1] /  dst_type_ناپ, (uint32_t) dst->nb[2] /  dst_type_ناپ, (uint32_t) dst->nb[3] /  dst_type_ناپ,
        0,
        0.0f, 0.0f, 0,
    });
}

صحيح ggml_vk_unary_mul_op_اشاریہ(ggml_unary_op op) {
    بدل (op) {
        حالة GGML_UNARY_OP_GELU:     أرجع 0;
        حالة GGML_UNARY_OP_SIGMOID:  أرجع 1;
        حالة GGML_UNARY_OP_SILU:     أرجع 2;
        حالة GGML_UNARY_OP_SOFTPLUS: أرجع 3;
        افتراضي:                     أرجع -1;
    }
}

فراغ ggml_vk_unary_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گرہ_idx) {
    ثابت ggml_tensor * unary = cgraph->nodes[گرہ_idx];
    ggml_tensor * mul = cgraph->nodes[گرہ_idx + 1];

    // unary on src1 that tiles into src0
    ثابت منطقي op_on_b = mul->src[1] == unary &&
                         !ggml_are_same_shape(unary->src[0], mul->src[0]) &&
                         ggml_can_repeat(unary, mul->src[0]);

    ثابت ggml_tensor * src0 = op_on_b ? mul->src[0] : unary->src[0];
    ثابت ggml_tensor * src1 = op_on_b ? unary->src[0] :
        ((mul->src[0] == unary) ? mul->src[1] : mul->src[0]);

    ثابت منطقي f16 = src0->type == GGML_TYPE_F16;
    ثابت منطقي norepeat = ggml_are_same_shape(src0, src1);
    ثابت صحيح oi = ggml_vk_unary_mul_op_اشاریہ(ggml_get_unary_op(unary));
    إذا (oi < 0) {
        GGML_ABORT("fatal error");
    }
    vk_pipeline pipeline = ctx->device->pipeline_unary_mul[oi][f16][norepeat][op_on_b];

    ثابت uint32_t src0_type_ناپ = ggml_type_ناپ(src0->type);
    ثابت uint32_t src1_type_ناپ = ggml_type_ناپ(src1->type);
    ثابت uint32_t dst_type_ناپ = ggml_type_ناپ(mul->type);

    ggml_vk_op_f32<vk_op_ثنائی_ڈالو_constants>(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, mul, GGML_OP_UNARY, {
        (uint32_t)ggml_nelements(op_on_b ? mul : src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ناپ, (uint32_t)src0->nb[1] / src0_type_ناپ, (uint32_t)src0->nb[2] / src0_type_ناپ, (uint32_t)src0->nb[3] / src0_type_ناپ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ناپ, (uint32_t)src1->nb[1] / src1_type_ناپ, (uint32_t)src1->nb[2] / src1_type_ناپ, (uint32_t)src1->nb[3] / src1_type_ناپ,
        (uint32_t) mul->ne[0], (uint32_t) mul->ne[1], (uint32_t) mul->ne[2],(uint32_t) mul->ne[3], (uint32_t) mul->nb[0] /  dst_type_ناپ, (uint32_t) mul->nb[1] /  dst_type_ناپ, (uint32_t) mul->nb[2] /  dst_type_ناپ, (uint32_t) mul->nb[3] /  dst_type_ناپ,
        0,
        0.0f, 0.0f, 0,
    }, pipeline);
}

فراغ ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_ناپ = ggml_type_ناپ(src0->type);
    ثابت uint32_t src1_type_ناپ = ggml_type_ناپ(src1->type);
    ثابت uint32_t dst_type_ناپ = ggml_type_ناپ(dst->type);

    ggml_vk_op_f32<vk_op_ثنائی_ڈالو_constants>(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_DIV, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ناپ, (uint32_t)src0->nb[1] / src0_type_ناپ, (uint32_t)src0->nb[2] / src0_type_ناپ, (uint32_t)src0->nb[3] / src0_type_ناپ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ناپ, (uint32_t)src1->nb[1] / src1_type_ناپ, (uint32_t)src1->nb[2] / src1_type_ناپ, (uint32_t)src1->nb[3] / src1_type_ناپ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ناپ, (uint32_t) dst->nb[1] /  dst_type_ناپ, (uint32_t) dst->nb[2] /  dst_type_ناپ, (uint32_t) dst->nb[3] /  dst_type_ناپ,
        0,
        0.0f, 0.0f, 0,
    });
}

فراغ ggml_vk_جوڑو_id(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ggml_tensor * dst) {
    ثابت uint32_t src0_type_ناپ = ggml_type_ناپ(src0->type);
    ثابت uint32_t src1_type_ناپ = ggml_type_ناپ(src1->type);
    ثابت uint32_t src2_type_ناپ = ggml_type_ناپ(src2->type);

    ggml_vk_op_f32<vk_op_جوڑو_id_ڈالو_constants>(ctx, subctx, src0, src1, src2, خالی_سوچک, dst, GGML_OP_جوڑو_ID, {
        (uint32_t)dst->ne[0],
        (uint32_t)dst->ne[1],
        (uint32_t)src0->nb[1] / src0_type_ناپ,
        (uint32_t)src0->nb[2] / src0_type_ناپ,
        (uint32_t)src1->nb[1] / src1_type_ناپ,
        (uint32_t)src2->nb[1] / src2_type_ناپ,
    });
}

ساكن فراغ ggml_vk_op_f32_wkv(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, ثابت vk_op_rwkv_wkv6_ڈالو_constants&& pc, صحيح نسخہ_2) {
    GGML_ASSERT(نسخہ_2 == 6 || نسخہ_2 == 7);
    صحيح عدد_srcs = نسخہ_2 == 6 ? 6 : 7;

    لكل (صحيح i = 0; i < عدد_srcs; i++) {
        GGML_ASSERT(!ggml_is_quantized(dst->src[i]->type));
    }

    GGML_ASSERT(dst->بفر != خالی_سوچک);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != خالی_سوچک);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[7] = {};
    لكل (صحيح i = 0; i < عدد_srcs; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    std::array<uint32_t, 3> elements = {
        (uint32_t)(pc.B * pc.H),
        1,
        1
    };

    إذا (نسخہ_2 == 6) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], dst_buf},
            pc, elements);
    } وإلا إذا (نسخہ_2 == 7) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], src_buf[6], dst_buf},
            pc, elements);
    } وإلا {
        // shouldn't happen
        GGML_ASSERT(خطأ);
    }
}

فراغ ggml_vk_rwkv_wkv6(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ثابت size_t seq_لمائی = dst->src[0]->ne[2];
    ثابت size_t n_embed = dst->ne[0];
    ثابت size_t n_heads = dst->src[0]->ne[1];
    ثابت size_t n_seqs = dst->src[5]->ne[1];

    ggml_vk_op_f32_wkv(
        ctx, subctx, dst,
        {
            (uint32_t)n_seqs,
            (uint32_t)seq_لمائی,
            (uint32_t)n_embed,
            (uint32_t)n_heads,
        },
        6
    );
}

فراغ ggml_vk_rwkv_wkv7(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ثابت size_t seq_لمائی = dst->src[0]->ne[2];
    ثابت size_t n_embed = dst->ne[0];
    ثابت size_t n_heads = dst->src[0]->ne[1];
    ثابت size_t n_seqs = dst->src[6]->ne[1];

    ggml_vk_op_f32_wkv(
        ctx, subctx, dst,
        {
            (uint32_t)n_seqs,
            (uint32_t)seq_لمائی,
            (uint32_t)n_embed,
            (uint32_t)n_heads,
        },
        7
    );
}

فراغ ggml_vk_gated_خطی_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ثابت size_t seq_لمائی = dst->src[0]->ne[2];
    ثابت size_t n_embed    = dst->ne[0];
    ثابت size_t n_heads    = dst->src[0]->ne[1];
    ثابت size_t n_seqs     = dst->src[4]->ne[1];

    عائم scale;
    حافظہ_نقل(&scale, dst->op_params, حجم(عائم));

    GGML_ASSERT(dst->بفر != خالی_سوچک);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != خالی_سوچک);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[5] = {};
    لكل (صحيح i = 0; i < 5; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    ثابت vk_op_gated_خطی_attn_ڈالو_constants pc = {
        (uint32_t)n_seqs,
        (uint32_t)seq_لمائی,
        (uint32_t)n_embed,
        (uint32_t)n_heads,
        scale,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], dst_buf},
        pc, { (uint32_t)(n_seqs * n_heads), 1, 1 });
}

فراغ ggml_vk_lightning_indexer(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ثابت ggml_tensor * q = dst->src[0];
    ثابت ggml_tensor * k = dst->src[1];
    ثابت ggml_tensor * w = dst->src[2];
    ثابت ggml_tensor * m = dst->src[3];

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, q, k, w, dst, dst->op);
    GGML_ASSERT(pipeline != خالی_سوچک);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ثابت uint32_t n_kv      = k->ne[2];
    ثابت uint32_t n_heads   = q->ne[1];
    ثابت uint32_t n_ٹوکن  = q->ne[2];
    ثابت uint32_t n_streams = q->ne[3];
    ثابت uint32_t n_masks   = m->ne[3];

    // one workgroup per tile of 64 keys and 8 tokens, see lightning_indexer.comp
    ثابت uint32_t n_tiles_kv = CEIL_DIV(n_kv, 64);
    ثابت uint32_t n_tiles_t  = CEIL_DIV(n_ٹوکن, 8);

    // q, w and dst are f32 and m is f16, so their strides are passed in elements;
    // k may be quantized, so its strides stay in bytes
    ثابت uint32_t q_nb1 = q->nb[1] / حجم(عائم);
    ثابت uint32_t q_nb2 = q->nb[2] / حجم(عائم);
    ثابت uint32_t q_nb3 = q->nb[3] / حجم(عائم);
    ثابت uint32_t k_nb2 = k->nb[2];
    ثابت uint32_t k_nb3 = k->nb[3];
    ثابت uint32_t w_nb1 = w->nb[1] / حجم(عائم);
    ثابت uint32_t w_nb3 = w->nb[3] / حجم(عائم);
    ثابت uint32_t m_nb1 = m->nb[1] / حجم(ggml_fp16_t);
    ثابت uint32_t m_nb3 = m->nb[3] / حجم(ggml_fp16_t);
    ثابت uint32_t d_nb1 = dst->nb[1] / حجم(عائم);
    ثابت uint32_t d_nb3 = dst->nb[3] / حجم(عائم);

    ثابت vk_op_lightning_indexer_ڈالو_constants pc = {
        n_kv, n_heads, n_ٹوکن, n_masks,
        q_nb1, q_nb2, q_nb3,
        k_nb2, k_nb3,
        w_nb1, w_nb3,
        m_nb1, m_nb3,
        d_nb1, d_nb3,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {ggml_vk_tensor_subbuffer(ctx, q), ggml_vk_tensor_subbuffer(ctx, k), ggml_vk_tensor_subbuffer(ctx, w), ggml_vk_tensor_subbuffer(ctx, m), ggml_vk_tensor_subbuffer(ctx, dst)},
        pc, {n_tiles_kv, n_tiles_t, n_streams});
}

فراغ ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src_q     = dst->src[0];
    ثابت ggml_tensor * src_v     = dst->src[2];
    ثابت ggml_tensor * src_beta  = dst->src[4];

    GGML_ASSERT(dst->بفر != خالی_سوچک);

    ثابت uint32_t S_v      = (uint32_t)src_v->ne[0];
    ثابت uint32_t H        = (uint32_t)src_v->ne[1];
    ثابت uint32_t n_ٹوکن = (uint32_t)src_v->ne[2];
    ثابت uint32_t n_seqs   = (uint32_t)src_v->ne[3];

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ثابت uint32_t K = (uint32_t)ggml_get_op_params_i32(dst, 0);

    ثابت uint32_t s_off = S_v * H * n_ٹوکن * n_seqs;

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != خالی_سوچک);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[6] = {};
    لكل (صحيح i = 0; i < 6; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    ثابت uint32_t sq1 = (uint32_t)(src_q->nb[1] / حجم(عائم));
    ثابت uint32_t sq2 = (uint32_t)(src_q->nb[2] / حجم(عائم));
    ثابت uint32_t sq3 = (uint32_t)(src_q->nb[3] / حجم(عائم));
    ثابت uint32_t sv1 = (uint32_t)(src_v->nb[1] / حجم(عائم));
    ثابت uint32_t sv2 = (uint32_t)(src_v->nb[2] / حجم(عائم));
    ثابت uint32_t sv3 = (uint32_t)(src_v->nb[3] / حجم(عائم));
    ثابت uint32_t sb1 = (uint32_t)(src_beta->nb[1] / حجم(عائم));
    ثابت uint32_t sb2 = (uint32_t)(src_beta->nb[2] / حجم(عائم));
    ثابت uint32_t sb3 = (uint32_t)(src_beta->nb[3] / حجم(عائم));

    ثابت uint32_t neq1 = (uint32_t)src_q->ne[1];
    ثابت uint32_t rq3  = (uint32_t)(src_v->ne[3] / src_q->ne[3]);

    ثابت عائم scale = 1.0f / sqrtf((عائم)S_v);
    ثابت vk_op_gated_delta_net_ڈالو_constants pc = {
        H, n_ٹوکن, n_seqs, s_off,
        sq1, sq2, sq3,
        sv1, sv2, sv3,
        sb1, sb2, sb3,
        neq1, rq3,
        scale,
        K
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], dst_buf},
        pc, { H, n_seqs, S_v });
}

فراغ ggml_vk_ssm_جانچ(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ثابت ggml_tensor * src2 = dst->src[2];
    ثابت ggml_tensor * src3 = dst->src[3];
    ثابت ggml_tensor * src4 = dst->src[4];
    ثابت ggml_tensor * src5 = dst->src[5];

    GGML_ASSERT(dst->بفر != خالی_سوچک);

    ثابت uint32_t سر_dim = src0->ne[1];
    ثابت uint32_t n_سر = src1->ne[1];
    ثابت uint32_t n_گروہ = src4->ne[1];
    ثابت uint32_t n_tok = src1->ne[2];
    ثابت uint32_t n_seq = src1->ne[3];

    منطقي is_mamba2 = (src3->nb[1] == حجم(عائم));
    GGML_ASSERT(is_mamba2);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, dst->op);
    GGML_ASSERT(pipeline != خالی_سوچک);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ثابت int64_t s_off = ggml_nelements(src1) * حجم(عائم);

    ثابت vk_op_ssm_جانچ_ڈالو_constants pc = {
        (uint32_t)src0->nb[2], (uint32_t)src0->nb[3],
        (uint32_t)src1->nb[2], (uint32_t)src1->nb[3],
        (uint32_t)src2->nb[1], (uint32_t)src2->nb[2],
        (uint32_t)src3->nb[1],
        (uint32_t)src4->nb[2], (uint32_t)src4->nb[3],
        (uint32_t)src5->nb[2], (uint32_t)src5->nb[3],
        (uint32_t)s_off,
        n_سر, سر_dim, n_گروہ, n_tok,
        n_seq, (uint32_t) ggml_get_op_params_i32(dst, 0)
    };

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[7] = {};
    لكل (صحيح i = 0; i < 7 && dst->src[i] != خالی_سوچک; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    std::array<uint32_t, 3> elements;

    ثابت uint32_t d_state = src0->ne[0];
    uint32_t عدد_subgroups = d_state / ctx->device->subgroup_ناپ;
    ثابت uint32_t عدد_workgroups_x = CEIL_DIV(n_سر * سر_dim, عدد_subgroups);
    ثابت uint32_t عدد_workgroups_y = n_seq;
    elements = { عدد_workgroups_x, عدد_workgroups_y, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], src_buf[6], dst_buf},
        pc, elements);
}

فراغ ggml_vk_ssm_conv(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گرہ_idx) {
    ggml_tensor * conv = cgraph->nodes[گرہ_idx];
    ثابت ggml_tensor * src0 = conv->src[0];
    ثابت ggml_tensor * src1 = conv->src[1];

    // Pick the destination tensor (last node in the fused chain) and the optional bias.
    // Fusion modes: 0 = ssm_conv, 1 = ssm_conv+silu, 2 = ssm_conv+add(bias)+silu.
    ggml_tensor * dst = conv;
    ثابت ggml_tensor * bias = خالی_سوچک;

    إذا (ctx->عدد_additional_fused_ops == 1) {
        dst = cgraph->nodes[گرہ_idx + 1]; // silu
    } وإلا إذا (ctx->عدد_additional_fused_ops == 2) {
        ggml_tensor * جوڑو_2 = cgraph->nodes[گرہ_idx + 1];
        bias = (جوڑو_2->src[0] == conv) ? جوڑو_2->src[1] : جوڑو_2->src[0];
        dst = cgraph->nodes[گرہ_idx + 2]; // silu
    }

    // The shader always declares 4 bindings; bind src0 as a dummy when bias isn't fused.
    ثابت ggml_tensor * src2 = bias ? bias : src0;

    ggml_vk_op_f32<vk_op_ssm_conv_ڈالو_constants>(ctx, subctx, src0, src1, src2, خالی_سوچک, dst, GGML_OP_SSM_CONV, {
        (uint32_t)src0->nb[1], (uint32_t)src0->nb[2],
        (uint32_t)src1->nb[1],
        (uint32_t)dst->nb[0], (uint32_t)dst->nb[1], (uint32_t)dst->nb[2],
        (uint32_t)src1->ne[0],
        (uint32_t)src0->ne[0],
        (uint32_t)src0->ne[1],
        (uint32_t)dst->ne[1],
        (uint32_t)dst->ne[2],
    });
}

ساكن فراغ ggml_vk_op_f32_opt_قدم_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, ثابت vk_op_ڈالو_constants&& pc) {
    ثابت ggml_tensor * x = dst->src[0];
    ثابت ggml_tensor * g = dst->src[1];
    ثابت ggml_tensor * gm = dst->src[2];
    ثابت ggml_tensor * gv = dst->src[3];
    ثابت ggml_tensor * p = dst->src[4];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(g->type == GGML_TYPE_F32);
    GGML_ASSERT(gm->type == GGML_TYPE_F32);
    GGML_ASSERT(gv->type == GGML_TYPE_F32);
    GGML_ASSERT(p->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->بفر != خالی_سوچک);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(gm));
    GGML_ASSERT(ggml_is_contiguous(gv));
    GGML_ASSERT(ggml_is_contiguous(p));
    GGML_ASSERT(ggml_are_same_shape(x, g));
    GGML_ASSERT(ggml_are_same_shape(x, gm));
    GGML_ASSERT(ggml_are_same_shape(x, gv));
    GGML_ASSERT(ggml_nelements(p) == 7);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, g, gm, gv, dst, GGML_OP_OPT_قدم_ADAMW);
    GGML_ASSERT(pipeline != خالی_سوچک);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x);
    vk_subbuffer g_buf = ggml_vk_tensor_subbuffer(ctx, g);
    vk_subbuffer gm_buf = ggml_vk_tensor_subbuffer(ctx, gm);
    vk_subbuffer gv_buf = ggml_vk_tensor_subbuffer(ctx, gv);
    vk_subbuffer p_buf = ggml_vk_tensor_subbuffer(ctx, p);

    std::array<uint32_t, 3> elements = { (uint32_t)ggml_nelements(x), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {x_buf, g_buf, gm_buf, gv_buf, p_buf},
        pc, elements);
}

فراغ ggml_vk_opt_قدم_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ثابت size_t n = ggml_nelements(dst->src[0]);

    ggml_vk_op_f32_opt_قدم_adamw(
        ctx, subctx, dst,
        { (uint32_t)n, 0, 0.0f, 0.0f, 0.0f, 0.0f }
    );
}

فراغ ggml_vk_opt_قدم_sgd(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ggml_tensor * dst) {
    ثابت size_t n = ggml_nelements(dst->src[0]);

    ggml_vk_op_f32<vk_op_ڈالو_constants>(ctx, subctx, src0, src1, src2, خالی_سوچک, dst, GGML_OP_OPT_قدم_SGD, { (uint32_t)n, 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

فراغ ggml_vk_concat(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    صحيح * op_params = (صحيح *)dst->op_params;

    ثابت uint32_t unit_ناپ = ggml_vk_concat_unit_ناپ(dst->type);
    ثابت uint32_t units_فی_بلاک = ggml_type_ناپ(dst->type) / unit_ناپ;
    ثابت uint32_t بلاک_ناپ_3 = ggml_blck_ناپ(dst->type);
    ثابت منطقي quantized = ggml_is_quantized(dst->type);

    // Address dimension 0 in packed storage units; higher strides may be noncontiguous.
    ثابت uint32_t ne00 = src0->ne[0] / بلاک_ناپ_3 * units_فی_بلاک;
    ثابت uint32_t ne10 = src1->ne[0] / بلاک_ناپ_3 * units_فی_بلاک;
    ثابت uint32_t ne20 =  dst->ne[0] / بلاک_ناپ_3 * units_فی_بلاک;
    ثابت uint32_t nb00 = quantized ? 1 : src0->nb[0] / unit_ناپ;
    ثابت uint32_t nb10 = quantized ? 1 : src1->nb[0] / unit_ناپ;
    ثابت uint32_t nb20 = quantized ? 1 :  dst->nb[0] / unit_ناپ;

    vk_op_concat_ڈالو_constants pc {{
        ne20 * (uint32_t)dst->ne[1] * (uint32_t)dst->ne[2] * (uint32_t)dst->ne[3],
        ne00, (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], nb00, (uint32_t)src0->nb[1] / unit_ناپ, (uint32_t)src0->nb[2] / unit_ناپ, (uint32_t)src0->nb[3] / unit_ناپ,
        ne10, (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], nb10, (uint32_t)src1->nb[1] / unit_ناپ, (uint32_t)src1->nb[2] / unit_ناپ, (uint32_t)src1->nb[3] / unit_ناپ,
        ne20, (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], nb20, (uint32_t) dst->nb[1] / unit_ناپ, (uint32_t) dst->nb[2] / unit_ناپ, (uint32_t) dst->nb[3] / unit_ناپ,
        0,
        0.0f, 0.0f, op_params[0],
    }};
    ggml_vk_op_f32<vk_op_concat_ڈالو_constants>(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_CONCAT, std::move(pc));
}

فراغ ggml_vk_upscale(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ثابت uint32_t src0_type_ناپ = ggml_type_ناپ(src0->type);
    ثابت uint32_t mode = (uint32_t)ggml_get_op_params_i32(dst, 0);

    GGML_TENSOR_UNARY_OP_LOCALS

    عائم sf0 = (عائم)ne0 / ne00;
    عائم sf1 = (عائم)ne1 / ne01;
    عائم sf2 = (عائم)ne2 / ne02;
    عائم sf3 = (عائم)ne3 / ne03;
    عائم pixel_وتھ = 0.5f;

    إذا (mode & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = ne0 > 1 && ne00 > 1 ? (عائم)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (عائم)(ne1 - 1) / (ne01 - 1) : sf1;
        pixel_وتھ = 0.0f;
    }

    ggml_vk_op_f32<vk_op_upscale_ڈالو_constants>(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_UPSCALE, {
        (uint32_t)ggml_nelements(dst), 0, 0,
        (uint32_t)ne00, (uint32_t)ne01,
        (uint32_t)nb00 / src0_type_ناپ, (uint32_t)nb01 / src0_type_ناپ, (uint32_t)nb02 / src0_type_ناپ, (uint32_t)nb03 / src0_type_ناپ,
        (uint32_t)ne0, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
        sf0, sf1, sf2, sf3, pixel_وتھ
    });
}

فراغ ggml_vk_scale(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ڈالو_constants p = vk_op_unary_ڈالو_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);
    p.param2 = ggml_get_op_params_f32(dst, 1);

    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_SCALE, std::move(p));
}

فراغ ggml_vk_sqr(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_SQR, vk_op_unary_ڈالو_constants_init(src0, dst));
}

فراغ ggml_vk_sqrt(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_SQRT, vk_op_unary_ڈالو_constants_init(src0, dst));
}

فراغ ggml_vk_add1(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_ناپ = ggml_type_ناپ(src0->type);
    ثابت uint32_t src1_type_ناپ = ggml_type_ناپ(src1->type);
    ثابت uint32_t dst_type_ناپ = ggml_type_ناپ(dst->type);

    ggml_vk_op_f32<vk_op_ثنائی_ڈالو_constants>(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_ADD1, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ناپ, (uint32_t)src0->nb[1] / src0_type_ناپ, (uint32_t)src0->nb[2] / src0_type_ناپ, (uint32_t)src0->nb[3] / src0_type_ناپ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ناپ, (uint32_t)src1->nb[1] / src1_type_ناپ, (uint32_t)src1->nb[2] / src1_type_ناپ, (uint32_t)src1->nb[3] / src1_type_ناپ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ناپ, (uint32_t) dst->nb[1] /  dst_type_ناپ, (uint32_t) dst->nb[2] /  dst_type_ناپ, (uint32_t) dst->nb[3] /  dst_type_ناپ,
        0,
        0.0f, 0.0f, 0,
    });
}

فراغ ggml_vk_arange(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_arange(dst=" << dst << ", ne=" << ggml_nelements(dst) << ")");

    vk_op_ڈالو_constants pc = {
        (uint32_t)ggml_nelements(dst),
        1,
        ggml_get_op_params_f32(dst, 0),
        ggml_get_op_params_f32(dst, 2),
        0.0f, 0.0f,
    };

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_ARANGE);
    GGML_ASSERT(pipeline != خالی_سوچک);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, خطأ);

    std::array<uint32_t, 3> elements = { (uint32_t)ggml_nelements(dst), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { dst_buf }, pc, elements);
}

فراغ ggml_vk_fill(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_fill(dst=" << dst << ", ne=" << ggml_nelements(dst) << ")");
    ثابت uint64_t n = ggml_nelements(dst);
    GGML_ASSERT(n > 0);

    vk_op_ڈالو_constants pc = {
        (uint32_t)n,
        1,
        ggml_get_op_params_f32(dst, 0),
        0.0f,
        0.0f, 0.0f,
    };

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_FILL);
    GGML_ASSERT(pipeline != خالی_سوچک);

    // Split the task distribution to 2D to avoid exceeding maxComputeWorkGroupCount
    ثابت uint32_t کل_wg = CEIL_DIV(n, pipeline->wg_denoms[0]);
    ثابت uint32_t wg_x = std::min(کل_wg, ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[0]);
    ثابت uint32_t wg_y = CEIL_DIV(کل_wg, wg_x);
    GGML_ASSERT(wg_y <= ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[1]);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, خطأ);

    std::array<uint32_t, 3> elements = { wg_x * pipeline->wg_denoms[0], wg_y * pipeline->wg_denoms[1], 1 };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { dst_buf }, pc, elements);
}

فراغ ggml_vk_sin(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_SIN, vk_op_unary_ڈالو_constants_init(src0, dst));
}

فراغ ggml_vk_cos(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_COS, vk_op_unary_ڈالو_constants_init(src0, dst));
}

فراغ ggml_vk_log(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_LOG, vk_op_unary_ڈالو_constants_init(src0, dst));
}

فراغ ggml_vk_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ڈالو_constants p = vk_op_unary_ڈالو_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);

    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_TRI, std::move(p));
}

فراغ ggml_vk_diag(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ڈالو_constants p = vk_op_unary_ڈالو_constants_init(src0, dst, ggml_nelements(dst));

    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_DIAG, std::move(p));
}

فراغ ggml_vk_clamp(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ڈالو_constants p = vk_op_unary_ڈالو_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);
    p.param2 = ggml_get_op_params_f32(dst, 1);

    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_CLAMP, std::move(p));
}

فراغ ggml_vk_pad(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_pad_ڈالو_constants p = vk_op_pad_ڈالو_constants_init(src0, dst);
    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_PAD, std::move(p));
}

فراغ ggml_vk_pad_reflect_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ثابت uint32_t p0 = (uint32_t)dst->op_params[0];
    ثابت uint32_t p1 = (uint32_t)dst->op_params[1];

    vk_op_unary_ڈالو_constants p = vk_op_unary_ڈالو_constants_init(src0, dst, ggml_nelements(dst));
    حافظہ_نقل(&p.param1, &p0, حجم(عائم));
    حافظہ_نقل(&p.param2, &p1, حجم(عائم));

    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_PAD_REFLECT_1D, std::move(p));
}

فراغ ggml_vk_roll(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ثابت int32_t s0 = ggml_get_op_params_i32(dst, 0);
    ثابت int32_t s1 = ggml_get_op_params_i32(dst, 1);
    ثابت int32_t s2 = ggml_get_op_params_i32(dst, 2);
    ثابت int32_t s3 = ggml_get_op_params_i32(dst, 3);
    ثابت uint32_t s01_packed = ((s0 + 0x8000) << 16) | (s1 + 0x8000);
    ثابت uint32_t s23_packed = ((s2 + 0x8000) << 16) | (s3 + 0x8000);

    vk_op_unary_ڈالو_constants p = vk_op_unary_ڈالو_constants_init(src0, dst);
    حافظہ_نقل(&p.param1, &s01_packed, حجم(عائم));
    حافظہ_نقل(&p.param2, &s23_packed, حجم(عائم));

    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_ROLL, std::move(p));
}

فراغ ggml_vk_repeat(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ڈالو_constants p = vk_op_unary_ڈالو_constants_init(src0, dst, ggml_nelements(dst));
    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_REPEAT, std::move(p));
}

فراغ ggml_vk_repeat_پٹھ(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ڈالو_constants p = vk_op_unary_ڈالو_constants_init(src0, dst, ggml_nelements(dst));
    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_REPEAT_پٹھ, std::move(p));
}

فراغ ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ne = (uint32_t)ggml_nelements(src0);
    إذا (ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
        // Convert from number of logical elements to 2- or 4-byte units.
        ne /= ggml_blck_ناپ(src0->type);
        إذا ((ggml_type_ناپ(src0->type) % 4) == 0) {
            ne *= ggml_type_ناپ(src0->type) / 4;
        } وإلا {
            ne *= ggml_type_ناپ(src0->type) / 2;
        }
    }

    vk_op_unary_ڈالو_constants p = vk_op_unary_ڈالو_constants_init(src0, dst, ne);
    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_CPY, std::move(p));
}

فراغ ggml_vk_سیٹ_صفاں(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_ناپ = ggml_type_ناپ(src0->type);
    ثابت uint32_t src1_type_ناپ = ggml_type_ناپ(src1->type);
    ثابت uint32_t dst_type_ناپ = ggml_type_ناپ(dst->type);

    // Skip empty skip_rows operations. For most ops the empty check at the start
    // of ggml_vk_build_graph is sufficient, but set_rows can have a nonempty dst
    // with empty srcs.
    إذا (ggml_is_خالی_تھاواں(src0) || ggml_is_خالی_تھاواں(src1)) {
        أرجع;
    }

    ggml_vk_op_f32<vk_op_ثنائی_ڈالو_constants>(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_سیٹ_صفاں, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ناپ, (uint32_t)src0->nb[1] / src0_type_ناپ, (uint32_t)src0->nb[2] / src0_type_ناپ, (uint32_t)src0->nb[3] / src0_type_ناپ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ناپ, (uint32_t)src1->nb[1] / src1_type_ناپ, (uint32_t)src1->nb[2] / src1_type_ناپ, (uint32_t)src1->nb[3] / src1_type_ناپ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ناپ, (uint32_t) dst->nb[1] /  dst_type_ناپ, (uint32_t) dst->nb[2] /  dst_type_ناپ, (uint32_t) dst->nb[3] /  dst_type_ناپ,
        0,
        0.0f, 0.0f, 0,
    });
}

فراغ ggml_vk_silu_پٹھ(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_ڈالو_constants>(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_SILU_پٹھ, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

فراغ ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    عائم * op_params = (عائم *)dst->op_params;
    vk_op_unary_ڈالو_constants p = vk_op_unary_ڈالو_constants_init(src0, dst);
    p.param1 = op_params[0];

    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_NORM, std::move(p));
}

فراغ ggml_vk_گروہ_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ثابت صحيح * int_op_params = (ثابت صحيح *)dst->op_params;
    ثابت عائم * float_op_params = (ثابت عائم *)dst->op_params;

    ثابت uint32_t عدد_groups = int_op_params[0];
    ثابت عائم eps = float_op_params[1];
    ثابت uint32_t گروہ_ناپ = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + عدد_groups - 1) / عدد_groups);

    ggml_vk_op_f32<vk_op_ڈالو_constants>(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_گروہ_NORM, { گروہ_ناپ, 0, eps, 0.0f, 0.0f, 0.0f });
}

ساكن uint32_t ggml_vk_rms_عدد_partials(ggml_backend_vk_context * ctx, ثابت ggml_tensor *گرہ) {
    ثابت uint32_t ne = (uint32_t)گرہ->ne[0];
    ثابت uint32_t denom = ctx->device->pipeline_جوڑو_rms[0][0][0]->wg_denoms[0];
    ثابت uint32_t عدد_partials = CEIL_DIV(ne, denom);
    أرجع عدد_partials;
}

uint32_t ggml_vk_rms_partials_ناپ(ggml_backend_vk_context * ctx, ثابت ggml_tensor *گرہ) {
    ثابت uint32_t عدد_partials = ggml_vk_rms_عدد_partials(ctx, گرہ);
    ثابت uint32_t عدد_بائٹاں = ROUNDUP_POW2(عدد_partials * حجم(uint32_t), ctx->device->partials_binding_alignment);
    أرجع عدد_بائٹاں;
}

ساكن vk_op_rope_ڈالو_constants ggml_vk_make_rope_constants(ثابت ggml_tensor *dst, ثابت ggml_tensor *src0, ثابت منطقي has_ff, منطقي backprop, ثابت uint32_t سیٹ_صفاں_stride) {
    ثابت صحيح n_dims        = ((ثابت int32_t *) dst->op_params)[1];
    ثابت صحيح mode          = ((ثابت int32_t *) dst->op_params)[2];
    ثابت صحيح n_offs        = ((ثابت int32_t *) dst->op_params)[15];
    // const int n_ctx         = ((const int32_t *) dst->op_params)[3];
    ثابت صحيح n_ctx_orig    = ((ثابت int32_t *) dst->op_params)[4];
    ثابت عائم freq_ادھار   = ((ثابت عائم *)   dst->op_params)[5];
    ثابت عائم freq_scale  = ((ثابت عائم *)   dst->op_params)[6];
    ثابت عائم ext_ضارب  = ((ثابت عائم *)   dst->op_params)[7];
    ثابت عائم attn_ضارب = ((ثابت عائم *)   dst->op_params)[8];
    ثابت عائم beta_fast   = ((ثابت عائم *)   dst->op_params)[9];
    ثابت عائم beta_slow   = ((ثابت عائم *)   dst->op_params)[10];
    صحيح sections[4] {};
    إذا (mode & GGML_ROPE_TYPE_MROPE) {
        حافظہ_نقل(sections, (ثابت int32_t *) dst->op_params + 11, حجم(صحيح)*4);
    }

    ثابت منطقي is_imrope = mode == GGML_ROPE_TYPE_IMROPE;

    عائم corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ادھار, beta_fast, beta_slow, corr_dims);

    ثابت عائم theta_scale = powf(freq_ادھار, -2.0f/n_dims);

    uint32_t nb01 = src0->nb[1] / ggml_type_ناپ(src0->type);
    uint32_t nb02 = src0->nb[2] / ggml_type_ناپ(src0->type);
    uint32_t nb03 = src0->nb[3] / ggml_type_ناپ(src0->type);

    uint32_t nb11 = dst->nb[1] / ggml_type_ناپ(dst->type);
    uint32_t nb12 = dst->nb[2] / ggml_type_ناپ(dst->type);
    uint32_t nb13 = dst->nb[3] / ggml_type_ناپ(dst->type);

    vk_op_rope_ڈالو_constants rope {
        (uint32_t)mode, (uint32_t)ggml_nrows(src0), (uint32_t)n_dims, (uint32_t)n_offs, freq_scale,
        freq_ادھار, ext_ضارب, attn_ضارب, {corr_dims[0], corr_dims[1]}, theta_scale, has_ff,
        { sections[0], sections[1], sections[2], sections[3] }, is_imrope, backprop, سیٹ_صفاں_stride,

        (uint32_t)src0->ne[0],
        (uint32_t)src0->ne[1],
        (uint32_t)src0->ne[2],
        nb01, nb02, nb03,
        nb11, nb12, nb13,
        0, 0, // a_offset, d_offset filled in by init_pushconst_tensor_offsets
    };

    أرجع rope;
}

ساكن فراغ ggml_vk_rms_norm_انت(ggml_backend_vk_context * ctx, ثابت ggml_tensor * src0) {
    إذا (ctx->do_جوڑو_rms_partials_وتھ_calculation) {
        ctx->prealloc_ناپ_جوڑو_rms_partials_وتھ += ggml_vk_rms_partials_ناپ(ctx, src0);
        ctx->do_جوڑو_rms_partials = خطأ;
        ctx->do_جوڑو_rms_partials_وتھ_calculation = خطأ;
    }
}

فراغ ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گرہ_idx, عائم * op_params) {
    ggml_tensor * rms = cgraph->nodes[گرہ_idx];
    ثابت ggml_tensor * src0 = rms->src[0];

    إذا (ctx->fused_rms_norm_منوال == RMS_NORM_VIEW_سیٹ_صفاں) {
        GGML_ASSERT(ctx->عدد_additional_fused_ops == 2);
        ggml_tensor * سیٹ_صفاں = cgraph->nodes[گرہ_idx + 2];
        ثابت ggml_tensor * indices = سیٹ_صفاں->src[1];
        vk_op_ثنائی_ڈالو_constants pc = ggml_vk_rms_norm_ڈالو_constants(src0, src0, سیٹ_صفاں, op_params[0], 0);
        init_pushconst_tensor_offsets(ctx, pc, src0, src0, خالی_سوچک, خالی_سوچک, سیٹ_صفاں);

        vk_pipeline pipeline = سیٹ_صفاں->type == GGML_TYPE_F16 ?
            ctx->device->pipeline_rms_norm_سیٹ_صفاں_f32_f16 : ctx->device->pipeline_rms_norm_سیٹ_صفاں_f32_f32;
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {
                ggml_vk_tensor_subbuffer(ctx, src0, سچ),
                ggml_vk_tensor_subbuffer(ctx, src0, سچ),
                ggml_vk_tensor_subbuffer(ctx, سیٹ_صفاں, سچ),
                ggml_vk_tensor_subbuffer(ctx, indices),
            }, pc, {
                (uint32_t)src0->ne[1],
                std::min((uint32_t)src0->ne[2], ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[1]),
                std::min((uint32_t)src0->ne[3], ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[2]),
            });
        ggml_vk_rms_norm_انت(ctx, src0);
        أرجع;
    }

    إذا (ctx->fused_rms_norm_منوال == RMS_NORM_MUL_جوڑو || ctx->fused_rms_norm_منوال == RMS_NORM_MUL_جوڑو_MUL) {
        ggml_tensor * mul = cgraph->nodes[گرہ_idx + 1];
        ggml_tensor * جوڑو_2 = cgraph->nodes[گرہ_idx + 2];
        ثابت ggml_tensor * بھار = mul->src[0] == rms ? mul->src[1] : mul->src[0];
        ثابت ggml_tensor * residual = جوڑو_2->src[0] == mul ? جوڑو_2->src[1] : جوڑو_2->src[0];
        ثابت منطقي do_post_multiply = ctx->fused_rms_norm_منوال == RMS_NORM_MUL_جوڑو_MUL;
        GGML_ASSERT(ctx->عدد_additional_fused_ops == (do_post_multiply ? 3 : 2));
        ggml_tensor * dst = do_post_multiply ? cgraph->nodes[گرہ_idx + 3] : جوڑو_2;
        ثابت ggml_tensor * post_scale = do_post_multiply ?
            (dst->src[0] == جوڑو_2 ? dst->src[1] : dst->src[0]) : src0;

        ثابت uint32_t عدد_partials = ctx->do_جوڑو_rms_partials ? ggml_vk_rms_عدد_partials(ctx, dst) : 0;
        vk_op_ثنائی_ڈالو_constants pc = ggml_vk_rms_norm_ڈالو_constants(src0, بھار, dst, op_params[0], عدد_partials);
        init_pushconst_tensor_offsets(ctx, pc, src0, بھار, residual, post_scale, dst);

        vk_pipeline pipeline;
        إذا (ctx->do_جوڑو_rms_partials) {
            pipeline = do_post_multiply ?
                ctx->device->pipeline_rms_norm_mul_جوڑو_mul_partials_f32 : ctx->device->pipeline_rms_norm_mul_جوڑو_partials_f32;
        } وإلا {
            pipeline = do_post_multiply ?
                ctx->device->pipeline_rms_norm_mul_جوڑو_mul_f32 : ctx->device->pipeline_rms_norm_mul_جوڑو_f32;
        }
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        إذا (ctx->do_جوڑو_rms_partials) {
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                {
                    ggml_vk_tensor_subbuffer(ctx, src0, سچ),
                    ggml_vk_tensor_subbuffer(ctx, بھار, سچ),
                    ggml_vk_tensor_subbuffer(ctx, dst, سچ),
                    ggml_vk_subbuffer(ctx, ctx->prealloc_جوڑو_rms_partials, ctx->prealloc_ناپ_جوڑو_rms_partials_وتھ),
                    ggml_vk_tensor_subbuffer(ctx, residual),
                    ggml_vk_tensor_subbuffer(ctx, post_scale),
                }, pc, { (uint32_t)CEIL_DIV(src0->ne[0], 128), 1, 1 });
        } وإلا {
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                {
                    ggml_vk_tensor_subbuffer(ctx, src0, سچ),
                    ggml_vk_tensor_subbuffer(ctx, بھار, سچ),
                    ggml_vk_tensor_subbuffer(ctx, dst, سچ),
                    ggml_vk_tensor_subbuffer(ctx, residual),
                    ggml_vk_tensor_subbuffer(ctx, post_scale),
                }, pc, {
                    (uint32_t)src0->ne[1],
                    std::min((uint32_t)src0->ne[2], ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[1]),
                    std::min((uint32_t)src0->ne[3], ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[2]),
                });
        }
        ggml_vk_rms_norm_انت(ctx, src0);
        أرجع;
    }

    ggml_tensor * dst;
    ثابت ggml_tensor * src1;

    إذا (ctx->fused_rms_norm_منوال != RMS_NORM_گنتی) {
        ggml_tensor * mul = cgraph->nodes[گرہ_idx + 1];
        dst = mul;
        src1 = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    } وإلا {
        dst = rms;
        src1 = src0;
    }

    ثابت uint32_t عدد_partials = ctx->do_جوڑو_rms_partials ? ggml_vk_rms_عدد_partials(ctx, dst) : 0;
    vk_op_ثنائی_ڈالو_constants bin = ggml_vk_rms_norm_ڈالو_constants(src0, src1, dst, op_params[0], عدد_partials);

    إذا (ctx->fused_rms_norm_منوال == RMS_NORM_MUL_ROPE ||
        ctx->fused_rms_norm_منوال == RMS_NORM_MUL_ROPE_VIEW_سیٹ_صفاں) {
        ساكن constexpr uint32_t max_tensors = 7;
        ثابت ggml_tensor *tensors[max_tensors] {};

        ggml_tensor *rms = cgraph->nodes[گرہ_idx + 0];
        ggml_tensor *mul = cgraph->nodes[گرہ_idx + 1];
        ggml_tensor *rope = cgraph->nodes[گرہ_idx + 2];

        ggml_tensor *other_src = mul->src[0] == rms ? mul->src[1] : mul->src[0];

        منطقي do_سیٹ_صفاں = ctx->fused_rms_norm_منوال == RMS_NORM_MUL_ROPE_VIEW_سیٹ_صفاں;
        GGML_ASSERT(ctx->عدد_additional_fused_ops == (do_سیٹ_صفاں ? 4 : 2));

        tensors[0] = rms->src[0];
        tensors[1] = other_src;
        tensors[2] = mul;
        tensors[3] = rope->src[1]; // pos
        tensors[4] = rope->src[2]; // ff
        tensors[5] = cgraph->nodes[گرہ_idx + ctx->عدد_additional_fused_ops]; // dst
        tensors[6] = do_سیٹ_صفاں ? tensors[5]->src[1] : خالی_سوچک;
        ثابت uint32_t سیٹ_صفاں_stride = do_سیٹ_صفاں ? tensors[5]->nb[1] / ggml_type_ناپ(tensors[5]->type) : 0;

        vk_op_rms_norm_mul_rope_ڈالو_constants pc;
        pc.bin = bin;
        pc.rope = ggml_vk_make_rope_constants(rope, rope->src[0], tensors[4] != خالی_سوچک, خطأ, سیٹ_صفاں_stride);

        vk_pipeline pipeline = tensors[5]->type == GGML_TYPE_F16 ? ctx->device->pipeline_rms_norm_mul_rope_f32_f16 : ctx->device->pipeline_rms_norm_mul_rope_f32_f32;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

        ggml_backend_vk_بفر_context * buf_ctx[max_tensors];
        vk_بفر buf[max_tensors];
        size_t وتھ[max_tensors];
        منطقي uma[max_tensors];

        لكل (uint32_t i = 0; i < max_tensors; ++i) {
            إذا (!tensors[i]) {
                // If any remaining descriptors are unused, just point them at src[0]
                buf[i] = buf[0];
                وتھ[i] = 0;
                استمر;
            }
            buf_ctx[i] = (ggml_backend_vk_بفر_context *)tensors[i]->بفر->context;
            buf[i] = خالی_سوچک;
            وتھ[i] = 0;
            uma[i] = خطأ;

            إذا (ctx->device->uma) {
                ggml_vk_host_get(ctx->device, tensors[i]->انکڑے, buf[i], وتھ[i]);
                uma[i] = buf[i] != خالی_سوچک;
            }
            إذا (!uma[i]) {
                buf[i] = buf_ctx[i]->dev_بفر;
                وتھ[i] = vk_tensor_وتھ(tensors[i]) + tensors[i]->view_offs;
            }
            GGML_ASSERT(buf[i] != خالی_سوچک);
        }

        // a_offset is unused (the fused path reads from shared memory), but the rope/set_rows dst can be misaligned.
        // Round the binding offset down to the storage buffer alignment; the in-element shift goes in pc.rope.d_offset.
        pc.rope.d_وتھ = get_misalign_بائٹاں(ctx, tensors[5]) / ggml_type_ناپ(tensors[5]->type);
        وتھ[5] &= ~(size_t(ctx->device->properties.limits.min_Storage_بفر_وتھ_Alignment) - 1);

        std::array<uint32_t, 3> elements;
        elements = { (uint32_t)rms->src[0]->ne[1], (uint32_t)rms->src[0]->ne[2], (uint32_t)rms->src[0]->ne[3] };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[1]);
        elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[2]);

        static_assert(max_tensors == 7);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {
                ggml_vk_subbuffer(ctx, buf[0], وتھ[0]),
                ggml_vk_subbuffer(ctx, buf[1], وتھ[1]),
                ggml_vk_subbuffer(ctx, buf[2], وتھ[2]),
                ggml_vk_subbuffer(ctx, buf[3], وتھ[3]),
                ggml_vk_subbuffer(ctx, buf[4], وتھ[4]),
                ggml_vk_subbuffer(ctx, buf[5], وتھ[5]),
                ggml_vk_subbuffer(ctx, buf[6], وتھ[6]),
            }, pc, elements);
    } وإلا {
        GGML_ASSERT(ctx->fused_rms_norm_منوال == RMS_NORM_MUL || ctx->fused_rms_norm_منوال == RMS_NORM_گنتی);
        ggml_vk_op_f32<vk_op_ثنائی_ڈالو_constants>(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_RMS_NORM, std::move(bin));
    }

    ggml_vk_rms_norm_انت(ctx, src0);
}

فراغ ggml_vk_rms_norm_پٹھ(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    عائم * op_params = (عائم *)dst->op_params;
    ggml_vk_op_f32<vk_op_ڈالو_constants>(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_RMS_NORM_پٹھ, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], 0.0f, 0.0f, 0.0f });
}

فراغ ggml_vk_l2_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ثابت عائم * op_params = (ثابت عائم *)dst->op_params;
    vk_op_unary_ڈالو_constants p = vk_op_unary_ڈالو_constants_init(src0, dst);
    p.param1 = op_params[0];
    ggml_vk_op_f32<vk_op_unary_ڈالو_constants>(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_L2_NORM, std::move(p));
}

فراغ ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_UNARY, vk_op_unary_ڈالو_constants_init(src0, dst));
}

فراغ ggml_vk_xielu(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    عائم * op_params = (عائم *)dst->op_params;
    vk_op_unary_ڈالو_constants p = vk_op_unary_ڈالو_constants_init(src0, dst);
    p.param1 = op_params[1];
    p.param2 = op_params[2];
    p.param3 = op_params[3];
    p.param4 = op_params[4];
    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_UNARY, std::move(p));
}

فراغ ggml_vk_glu(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت عائم * op_params_f = (ثابت عائم *)dst->op_params;

    ثابت منطقي swapped = (منطقي)dst->op_params[1];
    ثابت منطقي ونڈو = src1 != خالی_سوچک;
    ثابت عائم alpha = op_params_f[2];
    ثابت عائم حد_بندی = op_params_f[3];

    إذا (!ونڈو) {
        GGML_ASSERT(src0->ne[0] / 2 == dst->ne[0]);
    } وإلا {
        GGML_ASSERT(src0->ne[0] == src1->ne[0]);
        GGML_ASSERT(src0->ne[0] == dst->ne[0]);
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت uint32_t mode = ونڈو ? 2 : (swapped ? 1 : 0);
    ثابت uint32_t src0_type_ناپ = ggml_type_ناپ(src0->type);
    ثابت uint32_t src1_type_ناپ = ونڈو ? ggml_type_ناپ(src1->type) : src0_type_ناپ;
    ثابت uint32_t dst_type_ناپ  = ggml_type_ناپ(dst->type);

    ggml_vk_op_f32<vk_op_glu_ڈالو_constants>(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_GLU,
        {
            (uint32_t)ggml_nelements(dst),
            (uint32_t)src0->ne[0],
            (uint32_t)dst->ne[0],
            mode,
            alpha,
            حد_بندی,
            (uint32_t)(src0->nb[0] / src0_type_ناپ),
            (uint32_t)(src0->nb[1] / src0_type_ناپ),
            (uint32_t)(src0->nb[2] / src0_type_ناپ),
            (uint32_t)(src0->nb[3] / src0_type_ناپ),
            (uint32_t)((ونڈو ? src1->nb[0] : src0->nb[0]) / src1_type_ناپ),
            (uint32_t)((ونڈو ? src1->nb[1] : src0->nb[1]) / src1_type_ناپ),
            (uint32_t)((ونڈو ? src1->nb[2] : src0->nb[2]) / src1_type_ناپ),
            (uint32_t)((ونڈو ? src1->nb[3] : src0->nb[3]) / src1_type_ناپ),
            (uint32_t)(dst->nb[0] / dst_type_ناپ),
            (uint32_t)(dst->nb[1] / dst_type_ناپ),
            (uint32_t)(dst->nb[2] / dst_type_ناپ),
            (uint32_t)(dst->nb[3] / dst_type_ناپ),
            (uint32_t)dst->ne[1],
            (uint32_t)dst->ne[2],
            0,
            0, 0, 0, 0, 0, 0,
        });
}

فراغ ggml_vk_diag_نقاب_inf(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    int32_t * op_params = (int32_t *)dst->op_params;
    ggml_vk_op_f32<vk_op_diag_نقاب_ڈالو_constants>(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_DIAG_نقاب_INF, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0] });
}

فراغ ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ggml_tensor * dst) {
    عائم * op_params = (عائم *)dst->op_params;

    عائم scale = op_params[0];
    عائم max_bias = op_params[1];

    ثابت uint32_t ncols =   (uint32_t)src0->ne[0];
    ثابت uint32_t nrows_x = (uint32_t)ggml_nrows(src0);
    ثابت uint32_t nrows_y = (uint32_t)src0->ne[1];

    ثابت uint32_t ne12 = src1 ? (uint32_t)(src1->ne[2]) : 0u;
    ثابت uint32_t ne13 = src1 ? (uint32_t)(src1->ne[3]) : 0u;
    ثابت uint32_t nb11 = src1 ? (uint32_t)(src1->nb[1] / src1->nb[0]) : 0u;
    ثابت uint32_t nb12 = src1 ? (uint32_t)(src1->nb[2] / src1->nb[0]) : 0u;
    ثابت uint32_t nb13 = src1 ? (uint32_t)(src1->nb[3] / src1->nb[0]) : 0u;

    ثابت uint32_t n_سر_kv   = src0->ne[2];
    ثابت uint32_t n_سر_log2 = 1u << (uint32_t) floorf(log2f((عائم) n_سر_kv));

    ثابت عائم m0 = powf(2.0f, -(max_bias       ) / n_سر_log2);
    ثابت عائم m1 = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);

    vk_op_soft_max_ڈالو_constants pc {
        ncols,
        src1 != خالی_سوچک ? nrows_y : (uint32_t)0,
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
        ne12, ne13,
        nb11, nb12, nb13,
        scale, max_bias,
        m0, m1,
        n_سر_log2,
        nrows_x,
        src2 != خالی_سوچک
    };

    إذا (ncols <= 16384) {
        ggml_vk_op_f32<vk_op_soft_max_ڈالو_constants>(ctx, subctx, src0, src1, src2, خالی_سوچک, dst, GGML_OP_SOFT_MAX, std::move(pc));
    } وإلا {

        vk_subbuffer buf_a = ggml_vk_tensor_subbuffer(ctx, src0);
        vk_subbuffer buf_b = src1 ? ggml_vk_tensor_subbuffer(ctx, src1) : buf_a;
        vk_subbuffer buf_c = src2 ? ggml_vk_tensor_subbuffer(ctx, src2) : buf_a;
        vk_subbuffer buf_d = ggml_vk_tensor_subbuffer(ctx, dst);

        uint32_t elems_فی_wg = 128 * 4;
        uint32_t عدد_wgs = CEIL_DIV(ncols, elems_فی_wg);
        size_t tmp_ناپ = عدد_wgs * nrows_x * حجم(عائم);

        إذا (ctx->prealloc_ناپ_x < tmp_ناپ) {
            ctx->prealloc_ناپ_x = tmp_ناپ;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا (ctx->prealloc_ناپ_y < tmp_ناپ) {
            ctx->prealloc_ناپ_y = tmp_ناپ;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا (ctx->prealloc_x_need_sync || ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        vk_subbuffer buf_x = { ctx->prealloc_x, 0, tmp_ناپ };
        vk_subbuffer buf_y = { ctx->prealloc_y, 0, tmp_ناپ };

        // the partial results overwrite a matmul input converted into prealloc_y
        ctx->prealloc_y_آخری_pipeline_ورتیا = خالی_سوچک;
        ctx->prealloc_y_آخری_tensor_ورتیا = خالی_سوچک;

        std::array<uint32_t, 3> elements = { عدد_wgs, nrows_x, 1 };

        vk_pipeline pipeline1 = src1 && src1->type == GGML_TYPE_F16 ? ctx->device->pipeline_soft_max_large1_f32_f16 : ctx->device->pipeline_soft_max_large1_f32;
        vk_pipeline pipeline2 = src1 && src1->type == GGML_TYPE_F16 ? ctx->device->pipeline_soft_max_large2_f32_f16 : ctx->device->pipeline_soft_max_large2_f32;
        vk_pipeline pipeline3 = src1 && src1->type == GGML_TYPE_F16 ? ctx->device->pipeline_soft_max_large3_f32_f16 : ctx->device->pipeline_soft_max_large3_f32;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline1, 1);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline2, 1);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline3, 1);

        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline1, { buf_a, buf_b, buf_c, buf_d, buf_x, buf_y }, pc, elements);
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline2, { buf_a, buf_b, buf_c, buf_d, buf_x, buf_y }, pc, elements);
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline3, { buf_a, buf_b, buf_c, buf_d, buf_x, buf_y }, pc, elements);

        ctx->prealloc_x_need_sync = سچ;
        ctx->prealloc_y_need_sync = سچ;
    }
}

فراغ ggml_vk_soft_max_پٹھ(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    عائم * op_params = (عائم *)dst->op_params;
    ggml_vk_op_f32<vk_op_ڈالو_constants>(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_SOFT_MAX_پٹھ, { (uint32_t)src0->ne[0], (uint32_t)ggml_nrows(src0), op_params[0], op_params[1], 0.0f, 0.0f });
}

فراغ ggml_vk_topk_moe(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, صحيح گرہ_idx) {
    topk_moe_منوال mode = ctx->fused_topk_moe_منوال;
    ثابت منطقي has_bias = mode == TOPK_MOE_SIGMOID_NORM_BIAS || mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS;
    ggml_tensor * logits = cgraph->nodes[گرہ_idx + 0]->src[0];
    ggml_tensor * bias = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? cgraph->nodes[گرہ_idx + 2]->src[1] :
                         mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? cgraph->nodes[گرہ_idx + 3]->src[1] :
                                                                   logits;
    ggml_tensor * weights = cgraph->nodes[گرہ_idx + ctx->عدد_additional_fused_ops];
    ggml_tensor * ids = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? cgraph->nodes[گرہ_idx + 4] :
                        mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? cgraph->nodes[گرہ_idx + 5] :
                        mode == TOPK_MOE_LATE_SOFTMAX             ? cgraph->nodes[گرہ_idx + 1] :
                                                                   cgraph->nodes[گرہ_idx + 3];

    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(bias->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ثابت صحيح n_experts = logits->ne[0];
    ثابت صحيح n_صفاں    = logits->ne[1];
    ثابت صحيح n_expert_ورتیا = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_ناپ(ids->type) == (size_t) n_experts);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, خالی_سوچک, خالی_سوچک, خالی_سوچک, cgraph->nodes[گرہ_idx], GGML_OP_SOFT_MAX);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer logits_buf = ggml_vk_tensor_subbuffer(ctx, logits);
    vk_subbuffer bias_buf = ggml_vk_tensor_subbuffer(ctx, bias);
    vk_subbuffer weights_buf = ggml_vk_tensor_subbuffer(ctx, weights);
    vk_subbuffer ids_buf = ggml_vk_tensor_subbuffer(ctx, ids);

    vk_op_topk_moe_ڈالو_constants pc {};
    pc.n_صفاں = n_صفاں;
    pc.n_experts_ڈالو = n_experts;
    pc.n_expert_ورتیا = n_expert_ورتیا;
    pc.clamp_min = -std::numeric_limits<عائم>::infinity();
    pc.clamp_max = std::numeric_limits<عائم>::infinity();
    إذا (mode == TOPK_MOE_EARLY_SOFTMAX_NORM) {
        ggml_tensor * clamp = cgraph->nodes[گرہ_idx + 7];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }
    إذا (mode == TOPK_MOE_SIGMOID_NORM_BIAS) {
        ggml_tensor * clamp = cgraph->nodes[گرہ_idx + 8];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }
    إذا (mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS) {
        ggml_tensor * clamp = cgraph->nodes[گرہ_idx + 9];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }

#تعریف_کرو GATING_FUNC_SOFTMAX 0
#تعریف_کرو GATING_FUNC_SIGMOID 1
#تعریف_کرو GATING_FUNC_SOFTMAX_بھار 2
#تعریف_کرو GATING_FUNC_SQRT_SOFTPLUS 3

    pc.gating_func = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? GATING_FUNC_SIGMOID :
                     mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? GATING_FUNC_SQRT_SOFTPLUS :
                     mode == TOPK_MOE_LATE_SOFTMAX             ? GATING_FUNC_SOFTMAX_بھار :
                                                                 GATING_FUNC_SOFTMAX;
    pc.has_bias = has_bias;
    pc.with_norm = mode == TOPK_MOE_EARLY_SOFTMAX_NORM || has_bias;
    إذا (ctx->fused_topk_moe_scale) {
        GGML_ASSERT(weights->op == GGML_OP_SCALE);
        pc.output_scale = ggml_get_op_params_f32(weights, 0);
        pc.output_bias = ggml_get_op_params_f32(weights, 1);
    } وإلا {
        pc.output_scale = 1.0f;
        pc.output_bias = 0.0f;
    }

    GGML_ASSERT(n_expert_ورتیا <= n_experts);

    ثابت uint32_t صفاں_فی_بلاک = 4;
    std::array<uint32_t, 3> elements = { CEIL_DIV(n_صفاں, صفاں_فی_بلاک), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, {logits_buf, bias_buf, weights_buf, ids_buf}, pc, elements);
}

فراغ ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_cgraph * cgraph, صحيح گرہ_idx, منطقي backprop) {
    ggml_tensor * dst = cgraph->nodes[گرہ_idx];
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ثابت ggml_tensor * src2 = dst->src[2];
    ثابت ggml_tensor * src3 = خالی_سوچک;
    ثابت صحيح n_dims        = ((int32_t *) dst->op_params)[1];
    ثابت صحيح mode          = ((int32_t *) dst->op_params)[2];
    // const int n_ctx         = ((int32_t *) dst->op_params)[3];
    ثابت صحيح n_ctx_orig    = ((int32_t *) dst->op_params)[4];
    ثابت عائم freq_ادھار   = ((عائم *)   dst->op_params)[5];
    ثابت عائم beta_fast   = ((عائم *)   dst->op_params)[9];
    ثابت عائم beta_slow   = ((عائم *)   dst->op_params)[10];
    صحيح sections[4] {};
    إذا (mode & GGML_ROPE_TYPE_MROPE) {
        حافظہ_نقل(sections, (int32_t *) dst->op_params + 11, حجم(صحيح)*4);
    }

    عائم corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ادھار, beta_fast, beta_slow, corr_dims);

    uint32_t سیٹ_صفاں_stride = 0;
    // Fused rope + view + set_rows passes the set_rows destination stride in set_rows_stride
    // and overrides the dst and sets src3=row_indices
    إذا (ctx->عدد_additional_fused_ops > 0) {
        سیٹ_صفاں_stride = cgraph->nodes[گرہ_idx + 2]->nb[1] / ggml_type_ناپ(cgraph->nodes[گرہ_idx + 2]->type);
        src3 = cgraph->nodes[گرہ_idx + 2]->src[1];
        dst = cgraph->nodes[گرہ_idx + 2];
    }

    ggml_vk_op_f32<vk_op_rope_ڈالو_constants>(ctx, subctx, src0, src1, src2, src3, dst, GGML_OP_ROPE,
        ggml_vk_make_rope_constants(cgraph->nodes[گرہ_idx], src0, src2 != خالی_سوچک, backprop, سیٹ_صفاں_stride));
}

فراغ ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ثابت uint32_t * op_params = (ثابت uint32_t *)dst->op_params;

    uint32_t ncols = src0->ne[0];
    uint32_t nrows = ggml_nrows(src0);

    uint32_t ncols_pad_log2 = (uint32_t)ceilf(log2f(عائم(ncols)));
    uint32_t ncolsp2 = 1 << ncols_pad_log2;

    vk_op_argsort_ڈالو_constants pc { ncols, ncolsp2, ncols_pad_log2, nrows, op_params[0], 0, 0, 0, 0, };

    // Pick the largest workgroup size <= ncolsp2
    uint32_t pipeline_idx = std::min(ncols_pad_log2, عدد_argsort_pipelines - 1);

    uint32_t max_wg_log2 = std::min(ctx->device->max_workgroup_ناپ_log2, عدد_argsort_pipelines - 1);

    // Use the "small" argsort shader if the whole sort can be done by a single workgroup.
    منطقي use_small = ncols_pad_log2 <= max_wg_log2 &&
                     ctx->device->pipeline_argsort_f32[pipeline_idx] != خالی_سوچک;

    vk_pipeline pipeline = use_small ? ctx->device->pipeline_argsort_f32[pipeline_idx]
                                     : ctx->device->pipeline_argsort_large_f32[pipeline_idx];

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer subbuf1 = dst_buf;

    // Reserve space for ivec2 per element, with rows padded to a power of two
    إذا (!use_small) {
        ثابت size_t x_sz = size_t{ncolsp2} * nrows * 2 * حجم(صحيح);

        إذا (ctx->prealloc_ناپ_x < x_sz) {
            ctx->prealloc_ناپ_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
        subbuf1 = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    }

    std::array<uint32_t, 3> elements;

    elements[0] = ncolsp2;
    elements[1] = std::min((uint32_t)ggml_nrows(src0), ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[1]);
    elements[2] = 1;

    // First dispatch initializes tmp_idx and does the first N passes where
    // there is only communication between threads in the same workgroup.
    {
        vk_op_argsort_ڈالو_constants pc2 = pc;
        pc2.outer_شروع = 0;
        pc2.outer_سرا = std::min(ncols_pad_log2, max_wg_log2);
        pc2.inner_شروع = 0;
        pc2.inner_سرا = 100;
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc2, elements);
    }
    إذا (!use_small) {
        ggml_vk_sync_buffers(ctx, subctx);
        // Loop over outer/inner passes, synchronizing between each pass.
        لكل (uint32_t outer = max_wg_log2; outer < ncols_pad_log2; ++outer) {
            لكل (uint32_t inner = 0; inner < outer + 1; ++inner) {
                vk_op_argsort_ڈالو_constants pc2 = pc;
                pc2.outer_شروع = outer;
                pc2.outer_سرا = outer + 1;
                pc2.inner_شروع = inner;
                pc2.inner_سرا = inner + 1;
                // When the inner idx is large enough, there's only communication
                // within a workgroup. So the remaining inner iterations can all
                // run in the same dispatch.
                إذا (outer - inner < pipeline_idx) {
                    pc2.inner_سرا = 100;
                    inner = outer;
                    pipeline = ctx->device->pipeline_argsort_large_f32[pipeline_idx];
                } وإلا {
                    // Smaller workgroup empirically seems to perform better
                    pipeline = ctx->device->pipeline_argsort_large_f32[pipeline_idx - 2];
                }
                ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
                ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc2, elements);
                ggml_vk_sync_buffers(ctx, subctx);
            }
        }
        ctx->prealloc_x_need_sync = سچ;
    }
}

فراغ ggml_vk_topk(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ncols = src0->ne[0];
    uint32_t nrows = ggml_nrows(src0);
    uint32_t k = dst->ne[0];

    // tournament path is faster where it fits; use radix-select only past its k limit
    ثابت uint32_t k_min_pipeline = std::max((uint32_t) log2f(عائم(k)) + 1, ctx->device->subgroup_ناپ_log2);
    إذا (k_min_pipeline >= عدد_topk_pipelines || ctx->device->pipeline_topk_f32[k_min_pipeline] == خالی_سوچک) {
        vk_pipeline pipeline = ctx->device->pipeline_topk_radix_f32;
        GGML_ASSERT(pipeline != خالی_سوچک);

        إذا (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        vk_op_topk_radix_ڈالو_constants pc { ncols, k, nrows, 0, 0, 0 };
        std::array<uint32_t, 3> elements {
            pipeline->wg_denoms[0],
            std::min(nrows, ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[1]),
            1,
        };
        // the non-QSA path only uses bindings 0/1; bind valid buffers for the unused QSA slots
        vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
        vk_subbuffer dst_buf  = ggml_vk_tensor_subbuffer(ctx, dst);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            { src0_buf, dst_buf, src0_buf, src0_buf, src0_buf }, pc, elements);
        أرجع;
    }

    vk_op_topk_ڈالو_constants pc { ncols, ncols, ncols, k, nrows, 0, 0 };

    إذا (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    std::array<uint32_t, 3> elements;
    elements[1] = std::min(nrows, ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[1]);
    elements[2] = 1;

    uint32_t عدد_elements = ncols;

    // Each iteration reduces a workgroup's worth of elements down to the K
    // largest elements. Repeat until we have the top K elements.
    // Need to do at least one iteration to write out the results.
    منطقي پورا_one_iter = خطأ;
    uint32_t dbl_buf_اشاریہ = 0;
    size_t dbl_buf_ناپ;
    طالما (عدد_elements > k || !پورا_one_iter) {

        // Prefer going as small as num_topk_pipelines - 3 for perf reasons.
        // But if K is larger, then we need a larger workgroup
        uint32_t max_pipeline = عدد_topk_pipelines - 1;
        uint32_t preferred_pipeline = std::max(عدد_topk_pipelines - 3, (uint32_t)log2f(عائم(k)) + 2);
        max_pipeline = std::min(preferred_pipeline, max_pipeline);
        uint32_t min_pipeline = (uint32_t)log2f(عائم(k)) + 1;
        // require full subgroup
        min_pipeline = std::max(min_pipeline, ctx->device->subgroup_ناپ_log2);

        uint32_t pipeline_idx = (uint32_t)ceilf(log2f(عائم(عدد_elements)));
        pipeline_idx = std::min(pipeline_idx, max_pipeline);
        pipeline_idx = std::max(pipeline_idx, min_pipeline);

        إذا (عدد_elements > (1u << pipeline_idx)) {
            // If we could finish on this loop iteration (i.e. a single workgroup)
            // then do so. It's better than the overhead of another pass.
            لكل (uint32_t i = pipeline_idx; i < عدد_topk_pipelines; ++i) {
                إذا (عدد_elements <= (1u << i)) {
                    pipeline_idx = i;
                    اكسر;
                }
            }
        }

        vk_pipeline pipeline = ctx->device->pipeline_topk_f32[pipeline_idx];
        // If the device doesn't support a pipeline this large, use smaller
        طالما (!pipeline) {
            pipeline_idx--;
            GGML_ASSERT(pipeline_idx >= min_pipeline);
            pipeline = ctx->device->pipeline_topk_f32[pipeline_idx];
        }

        vk_op_topk_ڈالو_constants pc2 = pc;
        pc2.ncols_داخلہ = عدد_elements;

        // Number of elements remaining after this pass
        uint32_t عدد_dst_elements = (عدد_elements / pipeline->wg_denoms[0]) * k + std::min(k, عدد_elements % pipeline->wg_denoms[0]);

        pc2.ncols_output = عدد_dst_elements;

        إذا (!پورا_one_iter) {
            // Reserve space for ivec2 per element, double buffered
            // K per workgroup per row
            dbl_buf_ناپ = عدد_dst_elements * nrows * 2 * حجم(صحيح);
            dbl_buf_ناپ = ROUNDUP_POW2(dbl_buf_ناپ, ctx->device->properties.limits.min_Storage_بفر_وتھ_Alignment);
            ثابت size_t x_sz = dbl_buf_ناپ * 2;

            إذا (ctx->prealloc_ناپ_x < x_sz) {
                ctx->prealloc_ناپ_x = x_sz;
                ggml_vk_preallocate_buffers(ctx, subctx);
            }
        }

        vk_subbuffer src_buf;
        vk_subbuffer dst_buf;

        إذا (عدد_elements == ncols) {
            pc2.پہلا_pass = 1;
            src_buf = ggml_vk_tensor_subbuffer(ctx, src0);
        } وإلا {
            src_buf = { ctx->prealloc_x, dbl_buf_اشاریہ * dbl_buf_ناپ, dbl_buf_ناپ };
        }
        إذا (عدد_dst_elements == k) {
            pc2.آخری_pass = 1;
            dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
        } وإلا {
            dst_buf = { ctx->prealloc_x, (dbl_buf_اشاریہ ^ 1) * dbl_buf_ناپ, dbl_buf_ناپ };
        }

        elements[0] = عدد_elements;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf }, pc2, elements);
        عدد_elements = عدد_dst_elements;
        dbl_buf_اشاریہ ^= 1;
        إذا (عدد_elements > k) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
        پورا_one_iter = سچ;
    }
    ctx->prealloc_x_need_sync = سچ;
}

فراغ ggml_vk_topk_qsa(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_cgraph * cgraph, صحيح گرہ_idx) {
    ثابت ggml_tensor * get_صفاں = cgraph->nodes[گرہ_idx + 0];
    ثابت ggml_tensor * جوڑو_2      = cgraph->nodes[گرہ_idx + ctx->عدد_additional_fused_ops - 1];
    ggml_tensor *       چوٹی_k    = cgraph->nodes[گرہ_idx + ctx->عدد_additional_fused_ops];

    ثابت ggml_tensor * scores   = get_صفاں->src[0]; // [n_tps, n_blocks, n_stream]
    ثابت ggml_tensor * cell_blk = get_صفاں->src[1]; // [n_kv, n_stream]

    // raw f16 mask: follow the reshape/cpy chain back to the materialized input
    ثابت ggml_tensor * نقاب = جوڑو_2->src[1];
    طالما (نقاب->op == GGML_OP_RESHAPE || نقاب->op == GGML_OP_CPY) {
        نقاب = نقاب->src[0];
    }

    ثابت uint32_t n_tps    = scores->ne[0];
    ثابت uint32_t n_بلاکاں = scores->ne[1];
    ثابت uint32_t n_دھارا = scores->ne[2];
    ثابت uint32_t n_kv     = cell_blk->ne[0];
    ثابت uint32_t چوڑائی    = چوٹی_k->ne[0];
    ثابت uint32_t nrows    = n_tps * n_دھارا;

    vk_pipeline pipeline = ctx->device->pipeline_topk_radix_qsa;
    GGML_ASSERT(pipeline != خالی_سوچک);

    // scratch holds the gathered+masked input, materialized once and reused across passes
    ثابت size_t scratch_ناپ = size_t{ n_kv } * nrows * حجم(عائم);
    إذا (ctx->prealloc_ناپ_x < scratch_ناپ) {
        ctx->prealloc_ناپ_x = scratch_ناپ;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    إذا (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_op_topk_radix_ڈالو_constants pc { n_kv, چوڑائی, nrows, n_tps, n_بلاکاں, n_دھارا };
    std::array<uint32_t, 3> elements {
        pipeline->wg_denoms[0],
        std::min(nrows, ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[1]),
        1,
    };
    vk_subbuffer scratch_buf { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        { ggml_vk_tensor_subbuffer(ctx, scores), ggml_vk_tensor_subbuffer(ctx, چوٹی_k),
          ggml_vk_tensor_subbuffer(ctx, cell_blk), ggml_vk_tensor_subbuffer(ctx, نقاب),
          scratch_buf }, pc, elements);
    ctx->prealloc_x_need_sync = سچ;
}

فراغ ggml_vk_جوڑ(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_جوڑ_صفاں_ڈالو_constants p = vk_op_جوڑ_صفاں_ڈالو_constants_init(src0, dst, ggml_nelements(src0));
    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_جوڑ, p);
}

فراغ ggml_vk_جوڑ_صفاں(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_جوڑ_صفاں_ڈالو_constants p = vk_op_جوڑ_صفاں_ڈالو_constants_init(src0, dst, src0->ne[0]);
    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_جوڑ_صفاں, p);
}

فراغ ggml_vk_اوسط(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_جوڑ_صفاں_ڈالو_constants p = vk_op_جوڑ_صفاں_ڈالو_constants_init(src0, dst, src0->ne[0]);
    p.بھار = 1.0f / (عائم)src0->ne[0];
    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_اوسط, p);
}

فراغ ggml_vk_cumsum(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_جوڑ_صفاں_ڈالو_constants pc = vk_op_جوڑ_صفاں_ڈالو_constants_init(src0, dst, src0->ne[0]);
    // Use the single pass shader when the rows are small or there are enough rows to fill the GPU.
    // For fewer, larger rows, use the multipass shader to spread each row across SMs.
    إذا (dst->ne[0] <= 4096 || ggml_nrows(dst) >= ctx->device->shader_core_گنتی) {
        ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_CUMSUM, pc);
        أرجع;
    }

    // First pass computes partial sums within a block, and stores the last partial
    // to the temp buffer. Second pass sums the block partials from the temp buffer
    // and adds that to the result of the first pass.
    vk_pipeline pipeline1 = ctx->device->pipeline_cumsum_multipass1_f32;
    vk_pipeline pipeline2 = ctx->device->pipeline_cumsum_multipass2_f32;
    GGML_ASSERT(pipeline1 != خالی_سوچک && pipeline2 != خالی_سوچک);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline1, 1);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline2, 1);

    std::array<uint32_t, 3> elements;

    elements[0] = dst->ne[0];
    elements[1] = (uint32_t)ggml_nrows(dst);
    elements[2] = 1;

    size_t عارضی_ناپ = حجم(عائم) * elements[0] * ggml_nrows(dst);

    إذا (ctx->prealloc_ناپ_ونڈو_k < عارضی_ناپ) {
        ctx->prealloc_ناپ_ونڈو_k = عارضی_ناپ;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }

    vk_subbuffer src_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer عارضی_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_ونڈو_k, 0);

    إذا (ctx->prealloc_ونڈو_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline1, {src_buf, dst_buf, عارضی_buf}, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline2, {src_buf, dst_buf, عارضی_buf}, pc, elements);

    ctx->prealloc_ونڈو_k_need_sync = سچ;
}

ساكن std::array<uint32_t, 3> ggml_vk_nrows_elements(uint32_t nr) {
    إذا (nr > 262144) {
        أرجع { 512, 512, CEIL_DIV(nr, 262144) };
    }
    إذا (nr > 512) {
        أرجع { 512, CEIL_DIV(nr, 512), 1 };
    }
    أرجع { nr, 1, 1 };
}

فراغ ggml_vk_cross_entropy_loss(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    ثابت uint32_t nclasses = (uint32_t)src0->ne[0];
    ثابت uint32_t nrows    = (uint32_t)ggml_nrows(src0);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, خالی_سوچک, dst, GGML_OP_CROSS_ENTROPY_LOSS);
    GGML_ASSERT(pipeline != خالی_سوچک);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_جوڑ_صفاں_f32, 1);

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer src1_buf = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer dst_buf  = ggml_vk_tensor_subbuffer(ctx, dst, سچ);

    ثابت vk_op_ڈالو_constants pc = { nclasses, nrows, 0.0f, 0.0f, 0.0f, 0.0f };

    ثابت size_t tmp_ناپ = (size_t)nrows * حجم(عائم);
    إذا (ctx->prealloc_ناپ_x < tmp_ناپ) {
        ctx->prealloc_ناپ_x = tmp_ناپ;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    إذا (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_subbuffer tmp_buf = { ctx->prealloc_x, 0, tmp_ناپ };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, tmp_buf }, pc, ggml_vk_nrows_elements(nrows));
    ggml_vk_sync_buffers(ctx, subctx);

    vk_op_جوڑ_صفاں_ڈالو_constants sp = {};
    sp.n_خانے_گنتی = nrows;
    sp.ne01 = 1;
    sp.ne02 = 1;
    sp.بھار = 1.0f;
    init_pushconst_fastdiv(sp);
    sp.misalign_offsets = get_misalign_بائٹاں(ctx, dst) / ggml_type_ناپ(dst->type);

    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_جوڑ_صفاں_f32, { tmp_buf, dst_buf }, sp, { 1, 1, 1 });
    ctx->prealloc_x_need_sync = سچ;
}

فراغ ggml_vk_cross_entropy_loss_پٹھ(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ثابت ggml_tensor * grad   = dst->src[0];
    ثابت ggml_tensor * logits = dst->src[1];
    ثابت ggml_tensor * labels = dst->src[2];

    GGML_ASSERT(grad->type   == GGML_TYPE_F32);
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(labels->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_scalar(grad));
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_is_contiguous(logits));
    GGML_ASSERT(ggml_is_contiguous(labels));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(logits, labels));
    GGML_ASSERT(ggml_are_same_shape(logits, dst));

    ثابت uint32_t nclasses = (uint32_t)logits->ne[0];
    ثابت uint32_t nrows    = (uint32_t)ggml_nrows(logits);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, grad, logits, labels, dst, GGML_OP_CROSS_ENTROPY_LOSS_پٹھ);
    GGML_ASSERT(pipeline != خالی_سوچک);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer grad_buf   = ggml_vk_tensor_subbuffer(ctx, grad);
    vk_subbuffer logits_buf = ggml_vk_tensor_subbuffer(ctx, logits);
    vk_subbuffer labels_buf = ggml_vk_tensor_subbuffer(ctx, labels);
    vk_subbuffer dst_buf    = ggml_vk_tensor_subbuffer(ctx, dst);

    ثابت vk_op_ڈالو_constants pc = { nclasses, nrows, 0.0f, 0.0f, 0.0f, 0.0f };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { grad_buf, logits_buf, labels_buf, dst_buf }, pc, ggml_vk_nrows_elements(nrows));
}

فراغ ggml_vk_argmax(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_ڈالو_constants>(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_ARGMAX, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], 0.0f, 0.0f, 0.0f, 0.0f });
}

فراغ ggml_vk_گنتی_equal(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_ڈالو_constants>(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_گنتی_EQUAL, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

فراغ ggml_vk_solve_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_ناپ = ggml_type_ناپ(src0->type);
    ثابت uint32_t src1_type_ناپ = ggml_type_ناپ(src1->type);
    ثابت uint32_t dst_type_ناپ = ggml_type_ناپ(dst->type);

    ggml_vk_op_f32<vk_op_ثنائی_ڈالو_constants>(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_SOLVE_TRI, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ناپ, (uint32_t)src0->nb[1] / src0_type_ناپ, (uint32_t)src0->nb[2] / src0_type_ناپ, (uint32_t)src0->nb[3] / src0_type_ناپ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ناپ, (uint32_t)src1->nb[1] / src1_type_ناپ, (uint32_t)src1->nb[2] / src1_type_ناپ, (uint32_t)src1->nb[3] / src1_type_ناپ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ناپ, (uint32_t) dst->nb[1] /  dst_type_ناپ, (uint32_t) dst->nb[2] /  dst_type_ناپ, (uint32_t) dst->nb[3] /  dst_type_ناپ,
        0,
        0.0f, 0.0f, 0,
    });
}

فراغ ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت int32_t s0 = dst->op_params[0];
    ثابت int32_t s1 = dst->op_params[1];
    ثابت int32_t p0 = dst->op_params[2];
    ثابت int32_t p1 = dst->op_params[3];
    ثابت int32_t d0 = dst->op_params[4];
    ثابت int32_t d1 = dst->op_params[5];

    ثابت منطقي is_2D = dst->op_params[6] == 1;

    ثابت uint32_t IC = src1->ne[is_2D ? 2 : 1];
    ثابت uint32_t IH = is_2D ? src1->ne[1] : 1;
    ثابت uint32_t IW =         src1->ne[0];

    ثابت uint32_t KH = is_2D ? src0->ne[1] : 1;
    ثابت uint32_t KW =         src0->ne[0];

    ثابت uint32_t OH = is_2D ? dst->ne[2] : 1;
    ثابت uint32_t OW =         dst->ne[1];

    ثابت uint32_t وتھ_delta = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    ثابت uint32_t batch_وتھ = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    ثابت uint32_t batch = src1->ne[is_2D ? 3 : 2];

    ثابت ggml_backend_vk_بفر_context * d_buf_ctx = (ggml_backend_vk_بفر_context *)dst->بفر->context;
    ثابت vk_بفر d_buf = d_buf_ctx->dev_بفر;

    ثابت vk::Device_پتہ dst_addr = d_buf->bda_addr + vk_tensor_وتھ(dst) + dst->view_offs;

    ggml_vk_op_f32<vk_op_im2col_ڈالو_constants>(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_IM2_خانہ, {
        dst_addr,
        batch_وتھ, وتھ_delta,
        IC, IW, IH, OW, OH, KW, KH,
        OH * batch,
        IC * KH * KW,
        s0, s1, p0, p1, d0, d1, batch * IC
    });
}

فراغ ggml_vk_im2col_3d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت int32_t s0 = ((ثابت int32_t *)(dst->op_params))[0];
    ثابت int32_t s1 = ((ثابت int32_t *)(dst->op_params))[1];
    ثابت int32_t s2 = ((ثابت int32_t *)(dst->op_params))[2];
    ثابت int32_t p0 = ((ثابت int32_t *)(dst->op_params))[3];
    ثابت int32_t p1 = ((ثابت int32_t *)(dst->op_params))[4];
    ثابت int32_t p2 = ((ثابت int32_t *)(dst->op_params))[5];
    ثابت int32_t d0 = ((ثابت int32_t *)(dst->op_params))[6];
    ثابت int32_t d1 = ((ثابت int32_t *)(dst->op_params))[7];
    ثابت int32_t d2 = ((ثابت int32_t *)(dst->op_params))[8];
    ثابت int32_t IC = ((ثابت int32_t *)(dst->op_params))[9];

    ثابت int64_t N  = ne13 / IC;
    ثابت int64_t ID = ne12;
    ثابت int64_t IH = ne11;
    ثابت int64_t IW = ne10;

    ثابت int64_t KD = ne02;
    ثابت int64_t KH = ne01;
    ثابت int64_t KW = ne00;

    ثابت int64_t OD = ne3 / N;
    ثابت int64_t OH = ne2;
    ثابت int64_t OW = ne1;

    ثابت ggml_backend_vk_بفر_context * d_buf_ctx = (ggml_backend_vk_بفر_context *)dst->بفر->context;
    ثابت vk_بفر d_buf = d_buf_ctx->dev_بفر;

    ثابت vk::Device_پتہ dst_addr = d_buf->bda_addr + vk_tensor_وتھ(dst) + dst->view_offs;

    vk_op_im2col_3d_ڈالو_constants pc {};

    pc.dst_addr = dst_addr;
    pc.nb10 = nb10 / ggml_type_ناپ(src1->type);
    pc.nb11 = nb11 / ggml_type_ناپ(src1->type);
    pc.nb12 = nb12 / ggml_type_ناپ(src1->type);
    pc.nb13 = nb13 / ggml_type_ناپ(src1->type);
    pc.s0 = s0;
    pc.s1 = s1;
    pc.s2 = s2;
    pc.p0 = p0;
    pc.p1 = p1;
    pc.p2 = p2;
    pc.d0 = d0;
    pc.d1 = d1;
    pc.d2 = d2;
    pc.IW = IW;
    pc.IH = IH;
    pc.ID = ID;
    pc.IC = IC;
    pc.KW = KW;
    pc.OH = OH;
    pc.KD_KH_KW = KD*KH*KW;
    pc.KH_KW = KH*KW;
    pc.IC_KD_KH_KW = IC*KD*KH*KW;
    pc.N_OD_OH = N*OD*OH;
    pc.OD_OH = OD*OH;
    pc.OD_OH_OW_IC_KD_KH_KW = OD*OH*OW*IC*KD*KH*KW;
    pc.OH_OW_IC_KD_KH_KW = OH*OW*IC*KD*KH*KW;
    pc.OW_IC_KD_KH_KW = OW*IC*KD*KH*KW;

    ggml_vk_op_f32<vk_op_im2col_3d_ڈالو_constants>(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_IM2_خانہ_3_D, std::move(pc));
}

فراغ ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ثابت uint32_t dim = dst->op_params[0];
    ثابت uint32_t max_period = dst->op_params[1];
    ثابت uint32_t nb1 = dst->nb[1] / ggml_type_ناپ(dst->type);

    ggml_vk_op_f32<vk_op_timestep_embedding_ڈالو_constants>(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_TIMESTEP_EMBEDDING, {
        nb1, dim, max_period,
    });
}

فراغ ggml_vk_conv_transpose_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    // src0: (K, Cout, Cin, 1) -- kernel
    // src1: (L, Cin, 1, 1) -- input
    // dst: (*, Cout, 1, 1)

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ثنائی_OP_LOCALS

    GGML_ASSERT(nb00 == حجم(عائم));
    GGML_ASSERT(nb10 == حجم(عائم));

    ثابت int32_t s0 = dst->op_params[0];

    vk_op_conv_transpose_1d_ڈالو_constants p{};
    p.Cout = static_cast<uint32_t>(ne01);
    p.Cin = static_cast<uint32_t>(ne02);
    p.K = static_cast<uint32_t>(ne00);
    p.L = static_cast<uint32_t>(ne10);
    p.KL = static_cast<uint32_t>(ne0);
    p.nb01 = static_cast<uint32_t>(nb01 / nb00);
    p.nb02 = static_cast<uint32_t>(nb02 / nb00);
    p.nb11 = static_cast<uint32_t>(nb11 / nb10);
    p.nb1 = static_cast<uint32_t>(nb1 / nb0);
    p.s0 = static_cast<uint32_t>(s0);

    ggml_vk_op_f32(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_CONV_TRANSPOSE_1D, std::move(p));
}

فراغ ggml_vk_col2im_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    // src0: [K_OC, T_in] columns from matmul
    // dst:  [T_out, OC]

    ثابت int32_t stride = dst->op_params[0];
    ثابت int32_t oc     = dst->op_params[1];
    ثابت int32_t p0     = dst->op_params[2];

    ثابت uint32_t K_OC  = static_cast<uint32_t>(src0->ne[0]);
    ثابت uint32_t T_وچ  = static_cast<uint32_t>(src0->ne[1]);
    ثابت uint32_t T_باہرلا = static_cast<uint32_t>(dst->ne[0]);
    ثابت uint32_t OC    = static_cast<uint32_t>(oc);
    ثابت uint32_t K     = K_OC / OC;

    vk_op_col2im_1d_ڈالو_constants p{};
    p.T_باہرلا  = T_باہرلا;
    p.OC     = OC;
    p.K_OC   = K_OC;
    p.T_وچ   = T_وچ;
    p.K      = K;
    p.stride = stride;
    p.p0     = p0;

    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_COL2IM_1D, std::move(p));
}

فراغ ggml_vk_snake_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, صحيح گرہ_idx) {
    ثابت ggml_tensor * mul0 = cgraph->nodes[گرہ_idx + 0];
    ثابت ggml_tensor * sqr  = cgraph->nodes[گرہ_idx + 2];
    ثابت ggml_tensor * mul1 = cgraph->nodes[گرہ_idx + 3];
    ggml_tensor *       جوڑو_2  = cgraph->nodes[گرہ_idx + 4];

    // x carries the full activation shape, a is the broadcast operand
    ثابت ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ثابت ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    ثابت ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    vk_pipeline pipeline = خالی_سوچک;
    بدل (x->type) {
        حالة GGML_TYPE_F32:  pipeline = ctx->device->pipeline_snake_f32;  اكسر;
        حالة GGML_TYPE_F16:  pipeline = ctx->device->pipeline_snake_f16;  اكسر;
        حالة GGML_TYPE_BF16: pipeline = ctx->device->pipeline_snake_bf16; اكسر;
        افتراضي:             GGML_ABORT("unsupported type");
    }
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer x_buf     = ggml_vk_tensor_subbuffer(ctx, x);
    vk_subbuffer a_buf     = ggml_vk_tensor_subbuffer(ctx, a);
    vk_subbuffer inv_b_buf = ggml_vk_tensor_subbuffer(ctx, inv_b);
    vk_subbuffer dst_buf   = ggml_vk_tensor_subbuffer(ctx, جوڑو_2);

    vk_op_snake_ڈالو_constants pc{};
    pc.ne0 = static_cast<uint32_t>(x->ne[0]);
    pc.ne1 = static_cast<uint32_t>(x->ne[1]);

    std::array<uint32_t, 3> elements = { pc.ne0, pc.ne1, 1 };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, a_buf, inv_b_buf, dst_buf }, pc, elements);
}

فراغ ggml_vk_pool_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t op = static_cast<uint32_t>(dst->op_params[0]);
    ثابت int32_t k0 = dst->op_params[1];
    ثابت int32_t s0 = dst->op_params[2];
    ثابت int32_t p0 = dst->op_params[3];

    ثابت uint32_t IL = src0->ne[0];

    ثابت uint32_t N = dst->ne[3] * dst->ne[2];

    ثابت uint32_t OC = dst->ne[1];
    ثابت uint32_t OL = dst->ne[0];

    ثابت uint32_t parallel_elements = N * OC * OL;

    ggml_vk_op_f32<vk_op_pool1d_ڈالو_constants>(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_POOL_1D, {
        IL, OL, OC,
        parallel_elements,
        op,
        k0, s0, p0,
    });
}

فراغ ggml_vk_pool_2d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t op = static_cast<uint32_t>(dst->op_params[0]);
    ثابت int32_t k1 = dst->op_params[1];
    ثابت int32_t k0 = dst->op_params[2];
    ثابت int32_t s1 = dst->op_params[3];
    ثابت int32_t s0 = dst->op_params[4];
    ثابت int32_t p1 = dst->op_params[5];
    ثابت int32_t p0 = dst->op_params[6];

    ثابت uint32_t IH = src0->ne[1];
    ثابت uint32_t IW = src0->ne[0];

    ثابت uint32_t N = dst->ne[3];

    ثابت uint32_t OC = dst->ne[2];
    ثابت uint32_t OH = dst->ne[1];
    ثابت uint32_t OW = dst->ne[0];

    ثابت uint32_t parallel_elements = N * OC * OH * OW;

    ggml_vk_op_f32<vk_op_pool2d_ڈالو_constants>(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_POOL_2D, {
        IW, IH, OW, OH, OC,
        parallel_elements,
        op,
        k0, k1, s0, s1, p0, p1,
    });
}

فراغ ggml_vk_conv_2d(ggml_backend_vk_context * ctx, vk_context & subctx, ثابت ggml_tensor * src0,
                            ثابت ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ثنائی_OP_LOCALS
    GGML_ASSERT(nb00 == حجم(عائم) || nb00 == حجم(ggml_fp16_t));
    GGML_ASSERT(nb10 == حجم(عائم));
    GGML_ASSERT(nb0 == حجم(عائم));

    منطقي transpose = dst->op == GGML_OP_CONV_TRANSPOSE_2D;

    vk_op_conv2d_ڈالو_constants p{};
    p.Cout = static_cast<uint32_t>(!transpose ? ne03 : ne02);
    p.Cin  = static_cast<uint32_t>(!transpose ? ne02 : ne03);
    p.N    = static_cast<uint32_t>(ne13);
    GGML_ASSERT(p.Cout == ne2);
    GGML_ASSERT(p.Cin == ne12);

    p.W  = static_cast<uint32_t>(ne10);
    p.H  = static_cast<uint32_t>(ne11);
    p.OW = static_cast<uint32_t>(ne0);
    p.OH = static_cast<uint32_t>(ne1);

    p.nb01 = static_cast<uint32_t>(nb01 / nb00);
    p.nb02 = static_cast<uint32_t>(nb02 / nb00);
    p.nb03 = static_cast<uint32_t>(nb03 / nb00);

    p.nb11 = static_cast<uint32_t>(nb11 / nb10);
    p.nb12 = static_cast<uint32_t>(nb12 / nb10);
    p.nb13 = static_cast<uint32_t>(nb13 / nb10);

    p.nb1 = static_cast<uint32_t>(nb1 / nb0);
    p.nb2 = static_cast<uint32_t>(nb2 / nb0);
    p.nb3 = static_cast<uint32_t>(nb3 / nb0);

    ggml_vk_op_f32(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, dst->op, std::move(p));
}

فراغ ggml_vk_conv_3d(ggml_backend_vk_context * ctx, vk_context & subctx, ثابت ggml_tensor * src0,
                            ثابت ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ثنائی_OP_LOCALS
    GGML_ASSERT(nb00 == حجم(عائم) || nb00 == حجم(ggml_fp16_t));
    GGML_ASSERT(nb10 == حجم(عائم));
    GGML_ASSERT(nb0 == حجم(عائم));

    vk_op_conv3d_ڈالو_constants p{};
    p.IC = static_cast<uint32_t>(ggml_get_op_params_i32(dst, 9));
    p.N  = static_cast<uint32_t>(ggml_get_op_params_i32(dst, 10));
    p.OC = static_cast<uint32_t>(ggml_get_op_params_i32(dst, 11));
    GGML_ASSERT(src0->ne[3] == (int64_t)p.IC * p.OC);
    GGML_ASSERT(src1->ne[3] == (int64_t)p.IC * p.N);
    GGML_ASSERT(dst->ne[3] == (int64_t)p.OC * p.N);

    p.IW = static_cast<uint32_t>(ne10);
    p.IH = static_cast<uint32_t>(ne11);
    p.ID = static_cast<uint32_t>(ne12);
    p.OW = static_cast<uint32_t>(ne0);
    p.OH = static_cast<uint32_t>(ne1);
    p.OD = static_cast<uint32_t>(ne2);

    // the shader clamps src addresses to p.IC * p.N * p.IW * p.IH * p.ID - 1 in uint32, so the
    // total input element count must fit in a uint32.
    GGML_ASSERT((uint64_t)p.IC * p.N * p.IW * p.IH * p.ID <= 0xFFFFFFFFull);

    p.nb01 = static_cast<uint32_t>(nb01 / nb00);
    p.nb02 = static_cast<uint32_t>(nb02 / nb00);
    p.nb03 = static_cast<uint32_t>(nb03 / nb00);

    p.nb11 = static_cast<uint32_t>(nb11 / nb10);
    p.nb12 = static_cast<uint32_t>(nb12 / nb10);
    p.nb13 = static_cast<uint32_t>(nb13 / nb10);

    p.nb1 = static_cast<uint32_t>(nb1 / nb0);
    p.nb2 = static_cast<uint32_t>(nb2 / nb0);
    p.nb3 = static_cast<uint32_t>(nb3 / nb0);

    ggml_vk_op_f32(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_CONV_3D, std::move(p));
}

فراغ ggml_vk_conv_2d_dw(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    vk_op_conv2d_dw_ڈالو_constants p{};
    p.ne = ggml_nelements(dst);
    p.channels = dst->ne[2];
    p.batches = dst->ne[3];
    p.dst_w = dst->ne[0];
    p.dst_h = dst->ne[1];
    p.src_w = src1->ne[0];
    p.src_h = src1->ne[1];
    p.knl_w = src0->ne[0];
    p.knl_h = src0->ne[1];
    p.stride_x = dst->op_params[0];
    p.stride_y = dst->op_params[1];
    p.pad_x = dst->op_params[2];
    p.pad_y = dst->op_params[3];
    p.dilation_x = dst->op_params[4];
    p.dilation_y = dst->op_params[5];

    GGML_ASSERT(src0->ne[3] == p.channels);
    GGML_ASSERT(src1->ne[3] == p.batches);

    ggml_vk_op_f32(ctx, subctx, src0, src1, خالی_سوچک, خالی_سوچک, dst, GGML_OP_CONV_2D_DW, std::move(p));
}

فراغ ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ثابت عائم * op_params = (ثابت عائم *)dst->op_params;
    vk_op_unary_ڈالو_constants p = vk_op_unary_ڈالو_constants_init(src0, dst);
    p.param1 = op_params[0];

    ggml_vk_op_f32(ctx, subctx, src0, خالی_سوچک, خالی_سوچک, خالی_سوچک, dst, GGML_OP_LEAKY_RELU, std::move(p));
}

فراغ ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx) {
#إذا defined(GGML_VULKAN_لگاتار_TESTS)
    ثابت std::vector<size_t> vals {
        512, 512, 128,
        128, 512, 512,
        4096, 512, 4096,
        11008, 512, 4096,
        4096, 512, 11008,
        32000, 512, 4096,
        8, 8, 8,
        100, 46, 576,
        623, 111, 128,
        100, 46, 558,
        512, 1, 256,
        128, 110, 622,
        511, 511, 127,
        511, 511, 7,
        511, 511, 17,
        49, 49, 128,
        128, 49, 49,
        4096, 49, 4096,
    };
    ثابت size_t عدد_it = 100;

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, عدد_it, 1, 0, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, عدد_it, 1, 1, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, عدد_it, 1, 2, GGML_TYPE_Q4_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, عدد_it, 1, 0, GGML_TYPE_Q4_0, سچ);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, عدد_it, 1, 1, GGML_TYPE_Q4_0, سچ);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, عدد_it, 1, 2, GGML_TYPE_Q4_0, سچ);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, عدد_it, 1, 0, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, عدد_it, 1, 1, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, عدد_it, 1, 2, GGML_TYPE_Q8_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, عدد_it, 1, 0, GGML_TYPE_Q8_0, سچ);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, عدد_it, 1, 1, GGML_TYPE_Q8_0, سچ);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, عدد_it, 1, 2, GGML_TYPE_Q8_0, سچ);

    abort();

    لكل (size_t i = 0; i < vals.size(); i += 3) {
        ggml_vk_test_matmul<ggml_fp16_t, عائم>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 1, 0);
        ggml_vk_test_matmul<ggml_fp16_t, عائم>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 1, 1);
        ggml_vk_test_matmul<ggml_fp16_t, عائم>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 1, 2);
        std::cerr << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, عائم>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 2, 0);
        ggml_vk_test_matmul<ggml_fp16_t, عائم>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 2, 1);
        ggml_vk_test_matmul<ggml_fp16_t, عائم>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 2, 2);
        std::cerr << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, عائم>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 4, 0);
        ggml_vk_test_matmul<ggml_fp16_t, عائم>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 4, 1);
        ggml_vk_test_matmul<ggml_fp16_t, عائم>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 4, 2);
        std::cerr << '\n' << std::سطر_انت;

        إذا (vals[i + 2] % 32 == 0) {
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 1, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 1, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 1, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 2, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 2, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 2, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 4, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 4, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 4, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n' << std::سطر_انت;
        }

        إذا (vals[i + 2] % 256 == 0) {
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 1, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 1, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 1, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 2, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 2, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 2, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 4, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 4, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, عدد_it, 4, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n' << std::سطر_انت;
        }
    }

    GGML_ABORT("fatal error");
#جے_انت

    إذا (subctx) {
        // Submit and wait for any pending work before reallocating the buffers
        ggml_vk_ctx_سرا(subctx);
        ggml_vk_submit(subctx, {});
        ctx->submit_pending = سچ;
        ggml_vk_synchronize(ctx);
        GGML_ASSERT(ctx->compute_ctx.expired());
        ggml_vk_ctx_begin(ctx->device, subctx);
        ctx->compute_ctx = subctx;
    }

    إذا (ctx->prealloc_x == خالی_سوچک || (ctx->prealloc_ناپ_x > 0 && ctx->prealloc_x->size < ctx->prealloc_ناپ_x)) {
        VK_LOG_حافظہ("ggml_vk_preallocate_buffers(x_size: " << ctx->prealloc_ناپ_x << ")");
        // Resize buffer
        إذا (ctx->prealloc_x != خالی_سوچک) {
            ggml_vk_destroy_بفر(ctx->prealloc_x);
        }
        ctx->prealloc_x = ggml_vk_create_بفر_device(ctx->device, ctx->prealloc_ناپ_x);
    }
    إذا (ctx->prealloc_y == خالی_سوچک || (ctx->prealloc_ناپ_y > 0 && ctx->prealloc_y->size < ctx->prealloc_ناپ_y)) {
        VK_LOG_حافظہ("ggml_vk_preallocate_buffers(y_size: " << ctx->prealloc_ناپ_y << ")");
        // Resize buffer
        إذا (ctx->prealloc_y != خالی_سوچک) {
            ggml_vk_destroy_بفر(ctx->prealloc_y);
        }
        ctx->prealloc_y = ggml_vk_create_بفر_device(ctx->device, ctx->prealloc_ناپ_y);
        ctx->prealloc_y_آخری_pipeline_ورتیا = خالی_سوچک;
        ctx->prealloc_y_آخری_tensor_ورتیا = خالی_سوچک;
        ctx->prealloc_y_آخری_k_padded = خطأ;
    }
    إذا (ctx->prealloc_ونڈو_k == خالی_سوچک || (ctx->prealloc_ناپ_ونڈو_k > 0 && ctx->prealloc_ونڈو_k->size < ctx->prealloc_ناپ_ونڈو_k)) {
        VK_LOG_حافظہ("ggml_vk_preallocate_buffers(split_k_size: " << ctx->prealloc_ناپ_ونڈو_k << ")");
        // Resize buffer
        إذا (ctx->prealloc_ونڈو_k != خالی_سوچک) {
            ggml_vk_destroy_بفر(ctx->prealloc_ونڈو_k);
        }
        ctx->prealloc_ونڈو_k = ggml_vk_create_بفر_device(ctx->device, ctx->prealloc_ناپ_ونڈو_k);
    }
    إذا (ctx->prealloc_جوڑو_rms_partials == خالی_سوچک || (ctx->prealloc_ناپ_جوڑو_rms_partials > 0 && ctx->prealloc_جوڑو_rms_partials->size < ctx->prealloc_ناپ_جوڑو_rms_partials)) {
        VK_LOG_حافظہ("ggml_vk_preallocate_buffers(add_partials_size: " << ctx->prealloc_جوڑو_rms_partials << ")");
        // Resize buffer
        إذا (ctx->prealloc_جوڑو_rms_partials != خالی_سوچک) {
            ggml_vk_destroy_بفر(ctx->prealloc_جوڑو_rms_partials);
        }
        ctx->prealloc_جوڑو_rms_partials = ggml_vk_create_بفر_device(ctx->device, ctx->prealloc_ناپ_جوڑو_rms_partials);
    }
}

منطقي ggml_vk_build_گراف(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, صحيح گرہ_idx, ggml_tensor *گرہ_begin, صحيح گرہ_idx_begin, منطقي آخری_گرہ, منطقي almost_ready, منطقي submit){
    ggml_tensor * گرہ = cgraph->nodes[گرہ_idx];
    إذا (ggml_is_خالی_تھاواں(گرہ) || ggml_op_is_خالی_تھاواں(گرہ->op) || !گرہ->بفر) {
        أرجع خطأ;
    }
    إذا ((گرہ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        أرجع خطأ;
    }

    VK_LOG_DEBUG("ggml_vk_build_graph(" << گرہ << ", " << ggml_op_name(گرہ->op) << ")");
    ctx->semaphore_idx = 0;

    ggml_tensor * src0 = گرہ->src[0];
    ggml_tensor * src1 = گرہ->src[1];
    ggml_tensor * src2 = گرہ->src[2];
    ggml_tensor * src3 = گرہ->src[3];

    إذا (گرہ->op == GGML_OP_جوڑو) {
        صحيح اگلا_گرہ_idx = گرہ_idx + 1 + ctx->عدد_additional_fused_ops;
        إذا (اگلا_گرہ_idx < cgraph->n_nodes &&
            cgraph->nodes[اگلا_گرہ_idx]->op == GGML_OP_RMS_NORM &&
            cgraph->nodes[اگلا_گرہ_idx]->src[0] == cgraph->nodes[اگلا_گرہ_idx - 1] &&
            ggml_nrows(cgraph->nodes[اگلا_گرہ_idx]) == 1 &&
            ctx->device->جوڑو_rms_fusion) {
            uint32_t size = ggml_vk_rms_partials_ناپ(ctx, cgraph->nodes[گرہ_idx]);
            ctx->do_جوڑو_rms_partials_وتھ_calculation = سچ;
            إذا (ctx->prealloc_ناپ_جوڑو_rms_partials_وتھ + size <= ctx->prealloc_ناپ_جوڑو_rms_partials) {
                ctx->do_جوڑو_rms_partials = سچ;
            }
        }
    }

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    {
        // This logic detects dependencies between modes in the graph and calls ggml_vk_sync_buffers
        // to synchronize them. This handles most "normal" synchronization when computing the graph, and when
        // there is no auxiliary memory use, it shouldn't be necessary to call ggml_vk_sync_buffers
        // outside of this logic. When a node uses one of the prealloc buffers for something like
        // dequantization or split_k, additional synchronization is needed between those passes.
        منطقي need_sync = خطأ;

        // Check whether "node" requires synchronization. The node requires synchronization if it
        // overlaps in memory with another unsynchronized node and at least one of them is a write.
        // Destination nodes are checked against both the written/read lists. Source nodes are only
        // checked against the written list. Two nodes overlap in memory if they come from the same
        // buffer and the tensor or view ranges overlap.
        خودکار ثابت &overlaps_unsynced = [&](ثابت ggml_tensor *گرہ, ثابت std::vector<ثابت ggml_tensor *> &unsynced_nodes) -> منطقي {
            إذا (unsynced_nodes.size() == 0) {
                أرجع خطأ;
            }
            خودکار n_ادھار = vk_tensor_وتھ(گرہ) + گرہ->view_offs;
            خودکار n_ناپ = ggml_nbytes(گرہ);
            ggml_backend_vk_بفر_context * a_buf_ctx = (ggml_backend_vk_بفر_context *)گرہ->بفر->context;
            vk_بفر a_buf = a_buf_ctx->dev_بفر;
            لكل (خودکار &other : unsynced_nodes) {
                ggml_backend_vk_بفر_context * o_buf_ctx = (ggml_backend_vk_بفر_context *)other->بفر->context;
                vk_بفر o_buf = o_buf_ctx->dev_بفر;
                إذا (a_buf == o_buf) {
                    خودکار o_ادھار = vk_tensor_وتھ(other) + other->view_offs;
                    خودکار o_ناپ = ggml_nbytes(other);

                    إذا ((o_ادھار <= n_ادھار && n_ادھار < o_ادھار + o_ناپ) ||
                        (n_ادھار <= o_ادھار && o_ادھار < n_ادھار + n_ناپ)) {
                        أرجع سچ;
                    }
                }
            }
            أرجع خطأ;
        };

        // For all fused ops, check if the destination node or any of the source
        // nodes require synchronization.
        لكل (int32_t i = 0; i < ctx->عدد_additional_fused_ops + 1 && !need_sync; ++i) {
            ثابت ggml_tensor *cur_گرہ = cgraph->nodes[گرہ_idx + i];
            // If the node actually writes to memory, then check if it needs to sync
            إذا (ctx->fused_ops_write_نقاب & (1 << i)) {
                إذا (overlaps_unsynced(cur_گرہ, ctx->unsynced_nodes_read) || overlaps_unsynced(cur_گرہ, ctx->unsynced_nodes_written)) {
                    need_sync = سچ;
                    اكسر;
                }
            }
            لكل (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                إذا (!cur_گرہ->src[j]) {
                    استمر;
                }
                إذا (overlaps_unsynced(cur_گرہ->src[j], ctx->unsynced_nodes_written)) {
                    need_sync = سچ;
                    اكسر;
                }
            }
        }

        إذا (need_sync) {
            إذا (vk_enable_sync_logger) {
                std::cerr <<  "sync" << std::سطر_انت;
            }
            ctx->unsynced_nodes_written.clear();
            ctx->unsynced_nodes_read.clear();
            ggml_vk_sync_buffers(ctx, compute_ctx);

            إذا (vk_perf_logger_enabled && vk_perf_logger_concurrent) {
                ctx->query_گرہ_idx[ctx->query_idx] = گرہ_idx;
                compute_ctx->s->بفر->buf.writeTimestamp(vk::Pipeline_Stage_Flag_بٹ::eAllCommands, ctx->query_pool, ctx->query_idx++);
                ggml_vk_sync_buffers(ctx, compute_ctx);
            }
        }
        // Add all fused nodes to the unsynchronized lists.
        لكل (int32_t i = 0; i < ctx->عدد_additional_fused_ops + 1; ++i) {
            ثابت ggml_tensor *cur_گرہ = cgraph->nodes[گرہ_idx + i];
            // Multiple outputs could be written, e.g. in topk_moe. Add them all to the list.
            إذا (ctx->fused_ops_write_نقاب & (1 << i)) {
                ctx->unsynced_nodes_written.push_back(cur_گرہ);
            }
            لكل (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                إذا (!cur_گرہ->src[j]) {
                    استمر;
                }
                ctx->unsynced_nodes_read.push_back(cur_گرہ->src[j]);
            }
        }
    }
    إذا (vk_enable_sync_logger) {
        لكل (صحيح i = 0; i < ctx->عدد_additional_fused_ops + 1; ++i) {
            خودکار *n = cgraph->nodes[گرہ_idx + i];
            std::cerr << گرہ_idx + i << " " << ggml_op_name(n->op) << " " <<  n->name;
            إذا (n->op == GGML_OP_GLU) {
                std::cerr << " " << ggml_glu_op_name(ggml_get_glu_op(n)) << " " << (n->src[1] ? "split" : "single") << " ";
            }
            إذا (n->op == GGML_OP_ROPE) {
                ثابت صحيح mode = ((ثابت int32_t *) n->op_params)[2];
                std::cerr << " rope mode: " << mode;
            }
            std::cerr << std::سطر_انت;
        }
    }

    // closed explicitly below, and by the destructor on the early returns
    ggml_vk_debug_label dbg(compute_ctx, cgraph, گرہ_idx, ctx->عدد_additional_fused_ops);

    بدل (گرہ->op) {
    حالة GGML_OP_REPEAT:
        ggml_vk_repeat(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_REPEAT_پٹھ:
        ggml_vk_repeat_پٹھ(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_ACC:
    حالة GGML_OP_سیٹ:
        ggml_vk_acc(ctx, compute_ctx, src0, src1, گرہ);

        اكسر;
    حالة GGML_OP_GET_صفاں:
        إذا (ctx->fused_topk_qsa) {
            ggml_vk_topk_qsa(ctx, compute_ctx, cgraph, گرہ_idx);
        } وإلا {
            ggml_vk_get_صفاں(ctx, compute_ctx, src0, src1, گرہ);
        }

        اكسر;
    حالة GGML_OP_GET_صفاں_پٹھ:
        ggml_vk_get_صفاں_پٹھ(ctx, compute_ctx, src0, src1, گرہ);

        اكسر;
    حالة GGML_OP_جوڑو:
        إذا (ctx->عدد_additional_fused_ops) {
            ggml_vk_multi_جوڑو(ctx, compute_ctx, cgraph, گرہ_idx);
        } وإلا {
            ggml_vk_جوڑو(ctx, compute_ctx, src0, src1, گرہ);
        }
        اكسر;
    حالة GGML_OP_باہرلا_PROD:
        ggml_vk_باہرلا_prod(ctx, compute_ctx, src0, src1, گرہ);
        اكسر;
    حالة GGML_OP_SUB:
        ggml_vk_sub(ctx, compute_ctx, src0, src1, گرہ);

        اكسر;
    حالة GGML_OP_MUL:
        إذا (ctx->عدد_additional_fused_ops) {
            ggml_vk_snake_dispatch_fused(ctx, compute_ctx, cgraph, گرہ_idx);
        } وإلا {
            ggml_vk_mul(ctx, compute_ctx, src0, src1, گرہ);
        }

        اكسر;
    حالة GGML_OP_DIV:
        ggml_vk_div(ctx, compute_ctx, src0, src1, گرہ);

        اكسر;
    حالة GGML_OP_جوڑو_ID:
        ggml_vk_جوڑو_id(ctx, compute_ctx, src0, src1, src2, گرہ);

        اكسر;
    حالة GGML_OP_CONCAT:
        ggml_vk_concat(ctx, compute_ctx, src0, src1, گرہ);

        اكسر;
    حالة GGML_OP_UPSCALE:
        ggml_vk_upscale(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_ADD1:
        ggml_vk_add1(ctx, compute_ctx, src0, src1, گرہ);

        اكسر;
    حالة GGML_OP_ARANGE:
        ggml_vk_arange(ctx, compute_ctx, گرہ);

        اكسر;
    حالة GGML_OP_FILL:
        ggml_vk_fill(ctx, compute_ctx, گرہ);

        اكسر;
    حالة GGML_OP_SCALE:
        إذا (ctx->fused_hc_post_gate) {
            ggml_tensor * hc_post = cgraph->nodes[گرہ_idx + ctx->عدد_additional_fused_ops];
            ggml_vk_dsv4_hc_post(ctx, compute_ctx, hc_post->src[0], hc_post->src[1], hc_post->src[2], hc_post->src[3], hc_post, گرہ);
        } وإلا {
            ggml_vk_scale(ctx, compute_ctx, src0, گرہ);
        }

        اكسر;
    حالة GGML_OP_SQR:
        ggml_vk_sqr(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_SQRT:
        ggml_vk_sqrt(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_SIN:
        ggml_vk_sin(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_COS:
        ggml_vk_cos(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_LOG:
        ggml_vk_log(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_TRI:
        ggml_vk_tri(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_DIAG:
        ggml_vk_diag(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_CLAMP:
        ggml_vk_clamp(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_PAD:
        ggml_vk_pad(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_PAD_REFLECT_1D:
        ggml_vk_pad_reflect_1d(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_ROLL:
        ggml_vk_roll(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_CPY:
    حالة GGML_OP_CONT:
    حالة GGML_OP_DUP:
        ggml_vk_cpy(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_سیٹ_صفاں:
        ggml_vk_سیٹ_صفاں(ctx, compute_ctx, src0, src1, گرہ);

        اكسر;
    حالة GGML_OP_SILU_پٹھ:
        ggml_vk_silu_پٹھ(ctx, compute_ctx, src0, src1, گرہ);

        اكسر;
    حالة GGML_OP_NORM:
        ggml_vk_norm(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_گروہ_NORM:
        ggml_vk_گروہ_norm(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_RMS_NORM:
        ggml_vk_rms_norm(ctx, compute_ctx, cgraph, گرہ_idx, (عائم *)گرہ->op_params);
        اكسر;
    حالة GGML_OP_RMS_NORM_پٹھ:
        ggml_vk_rms_norm_پٹھ(ctx, compute_ctx, src0, src1, گرہ);

        اكسر;
    حالة GGML_OP_L2_NORM:
        ggml_vk_l2_norm(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_UNARY:
        إذا (ctx->fused_topk_moe_منوال != TOPK_MOE_گنتی) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, گرہ_idx);
            اكسر;
        }
        إذا (ctx->عدد_additional_fused_ops) {
            ggml_vk_unary_mul(ctx, compute_ctx, cgraph, گرہ_idx);
            اكسر;
        }

        بدل (ggml_get_unary_op(گرہ)) {
        حالة GGML_UNARY_OP_ELU:
        حالة GGML_UNARY_OP_EXP:
        حالة GGML_UNARY_OP_EXPM1:
        حالة GGML_UNARY_OP_SILU:
        حالة GGML_UNARY_OP_GELU:
        حالة GGML_UNARY_OP_GELU_ERF:
        حالة GGML_UNARY_OP_GELU_تیز:
        حالة GGML_UNARY_OP_RELU:
        حالة GGML_UNARY_OP_NEG:
        حالة GGML_UNARY_OP_TANH:
        حالة GGML_UNARY_OP_SIGMOID:
        حالة GGML_UNARY_OP_HARDSIGMOID:
        حالة GGML_UNARY_OP_HARDSWISH:
        حالة GGML_UNARY_OP_ABS:
        حالة GGML_UNARY_OP_SOFTPLUS:
        حالة GGML_UNARY_OP_قدم:
        حالة GGML_UNARY_OP_گیڑا:
        حالة GGML_UNARY_OP_CEIL:
        حالة GGML_UNARY_OP_FLOOR:
        حالة GGML_UNARY_OP_TRUNC:
        حالة GGML_UNARY_OP_SGN:
            ggml_vk_unary(ctx, compute_ctx, src0, گرہ);
            اكسر;
        حالة GGML_UNARY_OP_XIELU:
            ggml_vk_xielu(ctx, compute_ctx, src0, گرہ);
            اكسر;
        افتراضي:
            أرجع خطأ;
        }
        اكسر;
    حالة GGML_OP_GLU:
        بدل (ggml_get_glu_op(گرہ)) {
        حالة GGML_GLU_OP_GEGLU:
        حالة GGML_GLU_OP_REGLU:
        حالة GGML_GLU_OP_SWIGLU:
        حالة GGML_GLU_OP_SWIGLU_OAI:
        حالة GGML_GLU_OP_GEGLU_ERF:
        حالة GGML_GLU_OP_GEGLU_تیز:
        حالة GGML_GLU_OP_SWIGLU_CLAMP:
            ggml_vk_glu(ctx, compute_ctx, src0, src1, گرہ);
            اكسر;
        افتراضي:
            أرجع خطأ;
        }
        اكسر;
    حالة GGML_OP_DIAG_نقاب_INF:
        ggml_vk_diag_نقاب_inf(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_SOFT_MAX:
        إذا (ctx->fused_topk_moe_منوال != TOPK_MOE_گنتی) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, گرہ_idx);
        } وإلا {
            ggml_vk_soft_max(ctx, compute_ctx, src0, src1, src2, گرہ);
        }

        اكسر;
    حالة GGML_OP_SOFT_MAX_پٹھ:
        ggml_vk_soft_max_پٹھ(ctx, compute_ctx, src0, src1, گرہ);

        اكسر;
    حالة GGML_OP_ROPE:
        ggml_vk_rope(ctx, compute_ctx, cgraph, گرہ_idx, خطأ);

        اكسر;
    حالة GGML_OP_ROPE_پٹھ:
        ggml_vk_rope(ctx, compute_ctx, cgraph, گرہ_idx, سچ);

        اكسر;
    حالة GGML_OP_ARGSORT:
        إذا (ctx->fused_topk_moe_منوال != TOPK_MOE_گنتی) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, گرہ_idx);
        } وإلا {
            ggml_vk_argsort(ctx, compute_ctx, src0, گرہ);
        }

        اكسر;
    حالة GGML_OP_چوٹی_K:
        ggml_vk_topk(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_جوڑ:
        ggml_vk_جوڑ(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_جوڑ_صفاں:
        ggml_vk_جوڑ_صفاں(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_CUMSUM:
        ggml_vk_cumsum(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_DSV4_HC_COMB:
        ggml_vk_dsv4_hc_comb(ctx, compute_ctx, src0, src1, src2, گرہ);

        اكسر;
    حالة GGML_OP_DSV4_HC_PRE:
        ggml_vk_dsv4_hc_pre(ctx, compute_ctx, src0, src1, گرہ);

        اكسر;
    حالة GGML_OP_DSV4_HC_POST:
        ggml_vk_dsv4_hc_post(ctx, compute_ctx, src0, src1, src2, src3, گرہ);

        اكسر;
    حالة GGML_OP_اوسط:
        ggml_vk_اوسط(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_ARGMAX:
        ggml_vk_argmax(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_CROSS_ENTROPY_LOSS:
        ggml_vk_cross_entropy_loss(ctx, compute_ctx, گرہ);

        اكسر;
    حالة GGML_OP_CROSS_ENTROPY_LOSS_پٹھ:
        ggml_vk_cross_entropy_loss_پٹھ(ctx, compute_ctx, گرہ);

        اكسر;
    حالة GGML_OP_گنتی_EQUAL:
        ggml_vk_گنتی_equal(ctx, compute_ctx, src0, src1, گرہ);

        اكسر;
    حالة GGML_OP_SOLVE_TRI:
        ggml_vk_solve_tri(ctx, compute_ctx, src0, src1, گرہ);

        اكسر;
    حالة GGML_OP_IM2_خانہ:
        ggml_vk_im2col(ctx, compute_ctx, src0, src1, گرہ);

        اكسر;
    حالة GGML_OP_IM2_خانہ_3_D:
        ggml_vk_im2col_3d(ctx, compute_ctx, src0, src1, گرہ);

        اكسر;
    حالة GGML_OP_TIMESTEP_EMBEDDING:
        ggml_vk_timestep_embedding(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_COL2IM_1D:
        ggml_vk_col2im_1d(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_CONV_TRANSPOSE_1D:
        ggml_vk_conv_transpose_1d(ctx, compute_ctx, src0, src1, گرہ);

        اكسر;
    حالة GGML_OP_POOL_1D:
        ggml_vk_pool_1d(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_POOL_2D:
        ggml_vk_pool_2d(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_CONV_2D:
    حالة GGML_OP_CONV_TRANSPOSE_2D:
        ggml_vk_conv_2d(ctx, compute_ctx, src0, src1, گرہ);

        اكسر;
    حالة GGML_OP_CONV_3D:
        ggml_vk_conv_3d(ctx, compute_ctx, src0, src1, گرہ);

        اكسر;
    حالة GGML_OP_CONV_2D_DW:
        ggml_vk_conv_2d_dw(ctx, compute_ctx, src0, src1, گرہ);

        اكسر;
    حالة GGML_OP_LEAKY_RELU:
        ggml_vk_leaky_relu(ctx, compute_ctx, src0, گرہ);

        اكسر;
    حالة GGML_OP_MUL_MAT:
        ggml_vk_mul_mat(ctx, compute_ctx, cgraph, گرہ_idx);

        اكسر;
    حالة GGML_OP_MUL_MAT_ID:
        ggml_vk_mul_mat_id(ctx, compute_ctx, cgraph, گرہ_idx);

        اكسر;

    حالة GGML_OP_FLASH_ATTN_EXT:
        ggml_vk_flash_attn(ctx, compute_ctx, src0, src1, src2, src3, گرہ->src[4], گرہ);

        اكسر;

    حالة GGML_OP_RWKV_WKV6:
        ggml_vk_rwkv_wkv6(ctx, compute_ctx, گرہ);

        اكسر;

    حالة GGML_OP_RWKV_WKV7:
        ggml_vk_rwkv_wkv7(ctx, compute_ctx, گرہ);

        اكسر;

    حالة GGML_OP_GATED_خطی_ATTN:
        ggml_vk_gated_خطی_attn(ctx, compute_ctx, گرہ);

        اكسر;

    حالة GGML_OP_LIGHTNING_INDEXER:
        ggml_vk_lightning_indexer(ctx, compute_ctx, گرہ);

        اكسر;

    حالة GGML_OP_GATED_DELTA_NET:
        ggml_vk_gated_delta_net(ctx, compute_ctx, گرہ);

        اكسر;

    حالة GGML_OP_SSM_جانچ:
        ggml_vk_ssm_جانچ(ctx, compute_ctx, گرہ);

        اكسر;

    حالة GGML_OP_SSM_CONV:
        ggml_vk_ssm_conv(ctx, compute_ctx, cgraph, گرہ_idx);

        اكسر;

    حالة GGML_OP_OPT_قدم_ADAMW:
        ggml_vk_opt_قدم_adamw(ctx, compute_ctx, گرہ);

        اكسر;

    حالة GGML_OP_OPT_قدم_SGD:
        ggml_vk_opt_قدم_sgd(ctx, compute_ctx, src0, src1, src2, گرہ);

        اكسر;
    افتراضي:
        أرجع خطأ;
    }

    // the submit path below can end the command buffer, so close the region first
    dbg.close();

    ctx->tensor_ctxs[گرہ_idx] = compute_ctx;

#إذا defined(GGML_VULKAN_CHECK_RESULTS)
    // Force context reset on each node so that each tensor ends up in its own context
    // and can be run and compared to its CPU equivalent separately
    آخری_گرہ = سچ;
#جے_انت

    إذا (submit || آخری_گرہ) {
        ggml_vk_ctx_سرا(compute_ctx);

        // TODO probably it'd be better to pass a exit_node flag to ggml_vk_compute_forward
        إذا (آخری_گرہ) {
            compute_ctx->exit_tensor_idx = گرہ_idx_begin;
        }
        وإلا {
            compute_ctx->exit_tensor_idx = -1;
        }

        ctx->compute_ctx.reset();

        ggml_vk_compute_forward(ctx, cgraph, گرہ_begin, گرہ_idx_begin, almost_ready);
    }
    أرجع سچ;
}

فراغ ggml_vk_compute_forward(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ggml_tensor * tensor, صحيح tensor_idx, منطقي almost_ready) {
    GGML_بن_ورتیا(cgraph);
    GGML_بن_ورتیا(tensor);

    VK_LOG_DEBUG("ggml_vk_compute_forward(" << tensor << ", name=" << tensor->name << ", op=" << ggml_op_name(tensor->op) << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << ", view_src=" << tensor->view_src << ", view_offs=" << tensor->view_offs << ")");

    vk_context subctx = ctx->tensor_ctxs[tensor_idx].جندرا();

    // Only run if ctx hasn't been submitted yet
    إذا (!subctx->seqs.خالی_تھاواں()) {
#جے_تعریف_شدہ GGML_VULKAN_CHECK_RESULTS
        ggml_vk_check_results_0(ctx, cgraph, tensor_idx);
#جے_انت

        // Do staging buffer copies
        لكل (خودکار& cpy : subctx->وچ_memcpys) {
            حافظہ_نقل(cpy.dst, cpy.src, cpy.n);
        }

        لكل (خودکار& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        إذا (ctx->device->serialize_submissions) {
            ggml_vk_submit(subctx, ctx->fence);
        } وإلا إذا (almost_ready && !ctx->almost_ready_fence_pending) {
            ggml_vk_submit(subctx, ctx->almost_ready_fence);
            ctx->almost_ready_fence_pending = سچ;
        } وإلا {
            ggml_vk_submit(subctx, {});
        }
        ctx->submit_pending = سچ;

#جے_تعریف_شدہ GGML_VULKAN_CHECK_RESULTS
        ggml_vk_synchronize(ctx);
        ggml_vk_check_results_1(ctx, cgraph, tensor_idx);
#جے_انت
    }

    إذا (tensor_idx == subctx->exit_tensor_idx) {
        // Do staging buffer copies
        لكل (خودکار& cpy : subctx->باہرلا_memcpys) {
            حافظہ_نقل(cpy.dst, cpy.src, cpy.n);
        }
        subctx->وچ_memcpys.clear();
        subctx->باہرلا_memcpys.clear();
        subctx->memsets.clear();
    }
}

فراغ ggml_vk_گراف_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_graph_cleanup()");
    ctx->prealloc_y_آخری_pipeline_ورتیا = {};
    ctx->prealloc_y_آخری_tensor_ورتیا = خالی_سوچک;
    ctx->prealloc_y_آخری_k_padded = خطأ;

    ctx->unsynced_nodes_written.clear();
    ctx->unsynced_nodes_read.clear();
    ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_ونڈو_k_need_sync = خطأ;

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);
    إذا (ctx->device->async_use_transfer_قطار) {
        ggml_vk_command_pool_cleanup(ctx->device, ctx->transfer_cmd_pool);
    }

    لكل (size_t i = 0; i < ctx->gc.semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.semaphores[i].s });
    }
    ctx->gc.semaphores.clear();

    لكل (size_t i = 0; i < ctx->gc.tl_semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.tl_semaphores[i].s });
    }
    ctx->gc.tl_semaphores.clear();
    ctx->semaphore_idx = 0;

    ctx->event_idx = 0;

    لكل (خودکار& event : ctx->gc.events) {
        ctx->device->device.resetEvent(event);
    }

    ctx->tensor_ctxs.clear();
    ctx->gc.contexts.clear();
    ctx->pipeline_descriptor_سیٹ_requirements = 0;
    ctx->descriptor_سیٹ_idx = 0;
}

فراغ ggml_vk_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_cleanup(" << ctx->name << ")");
    // discard any unsubmitted command buffers
    ctx->compute_ctx.reset();
    // wait for any pending command buffers to finish
    ggml_vk_synchronize(ctx);

    ggml_vk_گراف_cleanup(ctx);

    ggml_vk_destroy_بفر(ctx->prealloc_x);
    ggml_vk_destroy_بفر(ctx->prealloc_y);
    ggml_vk_destroy_بفر(ctx->prealloc_ونڈو_k);
    ggml_vk_destroy_بفر(ctx->prealloc_جوڑو_rms_partials);
    ggml_vk_destroy_بفر(ctx->sync_staging);

    ctx->prealloc_y_آخری_pipeline_ورتیا = خالی_سوچک;
    ctx->prealloc_y_آخری_tensor_ورتیا = خالی_سوچک;
    ctx->prealloc_y_آخری_k_padded = خطأ;

    ctx->prealloc_ناپ_x = 0;
    ctx->prealloc_ناپ_y = 0;
    ctx->prealloc_ناپ_ونڈو_k = 0;

    لكل (خودکار& event : ctx->gc.events) {
        ctx->device->device.destroyEvent(event);
    }
    ctx->gc.events.clear();

    ctx->device->device.destroyFence(ctx->fence);
    ctx->device->device.destroyFence(ctx->almost_ready_fence);

    لكل (خودکار& pool : ctx->descriptor_pools) {
        ctx->device->device.destroyDescriptorPool(pool);
    }
    ctx->descriptor_pools.clear();
    ctx->descriptor_sets.clear();
    ctx->descriptor_سیٹ_bindings.clear();

    ctx->compute_cmd_pool.destroy(ctx->device->device);
    إذا (ctx->device->async_use_transfer_قطار) {
        ctx->device->device.destroySemaphore(ctx->transfer_semaphore.s);

        ctx->transfer_cmd_pool.destroy(ctx->device->device);
    }
    إذا (vk_perf_logger_enabled) {
        ctx->perf_logger->print_timings(سچ);
    }
}

صحيح ggml_vk_get_device_گنتی() {
    ggml_vk_instance_init();

    أرجع vk_instance.device_indices.size();
}

فراغ ggml_vk_get_device_description(صحيح device, حرف * description, size_t description_ناپ) {
    ggml_vk_instance_init();

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    vk::PhysicalDeviceProperties props;
    devices[device].getProperties(&props);

    snprintf(description, description_ناپ, "%s", props.deviceName.انکڑے());
}

منطقي ggml_backend_بفر_is_vk(ggml_backend_بفر_t بفر) {
    أرجع بفر->buft->iface.get_name == ggml_backend_vk_بفر_type_name;
}

فراغ ggml_backend_vk_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    VK_LOG_حافظہ("ggml_backend_vk_buffer_free_buffer()");
    ggml_backend_vk_بفر_context * ctx = (ggml_backend_vk_بفر_context *)بفر->context;
    ggml_vk_destroy_بفر(ctx->dev_بفر);
    مٹاؤ ctx;
}

فراغ * ggml_backend_vk_بفر_get_ادھار(ggml_backend_بفر_t بفر) {
    أرجع vk_ptr_ادھار;

    بن_ورتیا(بفر);
}

تعداد ggml_حالت ggml_backend_vk_بفر_init_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_init_tensor(" << بفر << " (" << بفر->context << "), " << tensor << ")");
    إذا (tensor->view_src != خالی_سوچک) {
        GGML_ASSERT(tensor->view_src->بفر->buft == بفر->buft);
    }
    أرجع GGML_حالت_SUCCESS;
}

فراغ ggml_backend_vk_بفر_memset_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, uint8_t value, size_t وتھ, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_memset_tensor(" << بفر << ", " << tensor << ", " << value << ", " << وتھ << ", " << size << ")");
    ggml_backend_vk_بفر_context * buf_ctx = (ggml_backend_vk_بفر_context *)بفر->context;
    vk_بفر buf = buf_ctx->dev_بفر;

    إذا (size == 0) {
        أرجع;
    }

    uint32_t val32 = (uint32_t)value * 0x01010101;
    ggml_vk_بفر_memset(buf, vk_tensor_وتھ(tensor) + tensor->view_offs + وتھ, val32, size);
}

فراغ ggml_backend_vk_بفر_سیٹ_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, ثابت فراغ * انکڑے, size_t وتھ, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor(" << بفر << ", " << tensor << ", " << انکڑے << ", " << وتھ << ", " << size << ")");
    ggml_backend_vk_بفر_context * buf_ctx = (ggml_backend_vk_بفر_context *)بفر->context;
    vk_بفر buf = buf_ctx->dev_بفر;

    إذا (size == 0) {
        أرجع;
    }

    ggml_vk_بفر_write(buf, vk_tensor_وتھ(tensor) + tensor->view_offs + وتھ, انکڑے, size);
}

فراغ ggml_backend_vk_بفر_سیٹ_tensor_2d(ggml_backend_بفر_t بفر, ggml_tensor * tensor, ثابت فراغ * انکڑے, size_t وتھ,
                                                 size_t size, size_t n_copies, size_t stride_tensor, size_t stride_انکڑے) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor_2d(" << بفر << ", " << tensor << ", " << انکڑے << ", " << وتھ << ", " << size << ", " <<
                 n_copies << ", " << stride_tensor << ", " << stride_انکڑے << ")");
    ggml_backend_vk_بفر_context * buf_ctx = (ggml_backend_vk_بفر_context *)بفر->context;
    vk_بفر buf = buf_ctx->dev_بفر;

    إذا (size == 0) {
        أرجع;
    }

    ggml_vk_بفر_write_2d(buf, vk_tensor_وتھ(tensor) + tensor->view_offs + وتھ, انکڑے, stride_انکڑے, stride_tensor, size, n_copies);
}

فراغ ggml_backend_vk_بفر_get_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * tensor, فراغ * انکڑے, size_t وتھ, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor(" << بفر << ", " << tensor << ", " << انکڑے << ", " << وتھ << ", " << size << ")");
    ggml_backend_vk_بفر_context * buf_ctx = (ggml_backend_vk_بفر_context *)بفر->context;

    إذا (size == 0) {
        أرجع;
    }

    vk_بفر buf = buf_ctx->dev_بفر;

    ggml_vk_بفر_read(buf, vk_tensor_وتھ(tensor) + tensor->view_offs + وتھ, انکڑے, size);
}

فراغ ggml_backend_vk_بفر_get_tensor_2d(ggml_backend_بفر_t بفر, ثابت ggml_tensor * tensor, فراغ * انکڑے, size_t وتھ,
                                                 size_t size, size_t n_copies, size_t stride_tensor, size_t stride_انکڑے) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor_2d(" << بفر << ", " << tensor << ", " << انکڑے << ", " << وتھ << ", " << size << ", " <<
                 n_copies << ", " << stride_tensor << ", " << stride_انکڑے << ")");
    ggml_backend_vk_بفر_context * buf_ctx = (ggml_backend_vk_بفر_context *)بفر->context;

    إذا (size == 0) {
        أرجع;
    }

    vk_بفر buf = buf_ctx->dev_بفر;

    ggml_vk_بفر_read_2d(buf, vk_tensor_وتھ(tensor) + tensor->view_offs + وتھ, انکڑے, stride_tensor, stride_انکڑے, size, n_copies);
}

منطقي ggml_backend_vk_بفر_cpy_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * src, ggml_tensor * dst) {
    إذا (ggml_nbytes(src) == 0) {
        أرجع سچ;
    }

    إذا (ggml_backend_بفر_is_vk(src->بفر)) {
        ggml_backend_vk_بفر_context * src_buf_ctx = (ggml_backend_vk_بفر_context *)src->بفر->context;
        ggml_backend_vk_بفر_context * dst_buf_ctx = (ggml_backend_vk_بفر_context *)dst->بفر->context;

        vk_بفر src_buf = src_buf_ctx->dev_بفر;
        vk_بفر dst_buf = dst_buf_ctx->dev_بفر;

        ggml_vk_بفر_copy(dst_buf, vk_tensor_وتھ(dst) + dst->view_offs, src_buf, vk_tensor_وتھ(src) + src->view_offs, ggml_nbytes(src));

        أرجع سچ;
    }
    أرجع خطأ;

    بن_ورتیا(بفر);
}

فراغ ggml_backend_vk_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    ggml_backend_vk_بفر_context * ctx = (ggml_backend_vk_بفر_context *)بفر->context;

    ggml_vk_بفر_memset(ctx->dev_بفر, 0, value, بفر->size);
}

ثابت حرف * ggml_backend_vk_بفر_type_name(ggml_backend_بفر_type_t buft) {
    ggml_backend_vk_بفر_type_context * ctx = (ggml_backend_vk_بفر_type_context *)buft->context;

    أرجع ctx->name.c_str();
}

ggml_backend_بفر_t ggml_backend_vk_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    VK_LOG_حافظہ("ggml_backend_vk_buffer_type_alloc_buffer(" << size << ")");
    ggml_backend_vk_بفر_type_context * ctx = (ggml_backend_vk_بفر_type_context *) buft->context;

    vk_بفر dev_بفر = خالی_سوچک;
    حاول {
        dev_بفر = ggml_vk_create_بفر_device(ctx->device, size);
    } امسك (ثابت vk::SystemError& e) {
        أرجع خالی_سوچک;
    }

    ggml_backend_vk_بفر_context * bufctx = جديد ggml_backend_vk_بفر_context(ctx->device, std::move(dev_بفر), ctx->name);

    أرجع ggml_backend_بفر_init(buft, ggml_backend_vk_بفر_interface, bufctx, size);
}

size_t ggml_backend_vk_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    ggml_backend_vk_بفر_type_context * ctx = (ggml_backend_vk_بفر_type_context *) buft->context;
    أرجع ctx->device->properties.limits.min_Storage_بفر_وتھ_Alignment;
}

size_t ggml_backend_vk_بفر_type_get_max_ناپ(ggml_backend_بفر_type_t buft) {
    ggml_backend_vk_بفر_type_context * ctx = (ggml_backend_vk_بفر_type_context *) buft->context;
    أرجع ctx->device->suballocation_بلاک_ناپ;
}

size_t ggml_backend_vk_بفر_type_get_alloc_ناپ(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    أرجع ggml_nbytes(tensor);

    بن_ورتیا(buft);
}

ggml_backend_بفر_type_t ggml_backend_vk_بفر_type(size_t dev_عدد) {
    ggml_vk_instance_init();

    VK_LOG_DEBUG("ggml_backend_vk_buffer_type(" << dev_عدد << ")");

    vk_device dev = ggml_vk_get_device(dev_عدد);

    أرجع &dev->بفر_type;
}

ساكن ثابت حرف * ggml_backend_vk_host_بفر_type_name(ggml_backend_بفر_type_t buft) {
    أرجع GGML_VK_NAME "_Host";

    بن_ورتیا(buft);
}

ساكن فراغ ggml_backend_vk_host_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    VK_LOG_حافظہ("ggml_backend_vk_host_buffer_free_buffer()");
    ggml_vk_host_free(بفر->context);
}

ساكن ggml_backend_بفر_t ggml_backend_vk_host_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    VK_LOG_حافظہ("ggml_backend_vk_host_buffer_type_alloc_buffer(" << size << ")");

    size += 32;  // Behave like the CPU buffer type
    فراغ * ptr = ggml_vk_host_malloc(size);
    إذا (ptr == خالی_سوچک) {
        GGML_LOG_WARN("ggml_vulkan: Failed to allocate pinned memory\n");
        // fallback to cpu buffer
        أرجع ggml_backend_buft_alloc_بفر(ggml_backend_cpu_بفر_type(), size);
    }

    ggml_backend_بفر_t بفر = ggml_backend_cpu_بفر_from_ptr(ptr, size);
    بفر->buft = buft;
    بفر->iface.free_بفر = ggml_backend_vk_host_بفر_free_بفر;

    أرجع بفر;

    بن_ورتیا(buft);
}

ساكن size_t ggml_backend_vk_host_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    أرجع vk_instance.devices[0]->properties.limits.min_حافظہ_Map_Alignment;

    بن_ورتیا(buft);
}

ساكن size_t ggml_backend_vk_host_بفر_type_get_max_ناپ(ggml_backend_بفر_type_t buft) {
    أرجع vk_instance.devices[0]->suballocation_بلاک_ناپ;

    بن_ورتیا(buft);
}

ggml_backend_بفر_type_t ggml_backend_vk_host_بفر_type() {
    ساكن بنية ggml_backend_بفر_type ggml_backend_vk_بفر_type_host = {
        /* .iface    = */ {
            /* .get_name            = */ ggml_backend_vk_host_بفر_type_name,
            /* .alloc_buffer        = */ ggml_backend_vk_host_بفر_type_alloc_بفر,
            /* .alloc_buffer_n      = */ خالی_سوچک,
            /* .get_alignment       = */ ggml_backend_vk_host_بفر_type_get_alignment,
            /* .get_max_size        = */ ggml_backend_vk_host_بفر_type_get_max_ناپ,
            /* .get_alloc_size      = */ ggml_backend_cpu_بفر_type()->iface.get_alloc_ناپ,
            /* .get_alloc_size_n    = */ عدم,
            /* .is_host             = */ ggml_backend_cpu_بفر_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), 0),
        /* .context  = */ خالی_سوچک,
    };

    // Make sure device 0 is initialized
    ggml_vk_instance_init();
    ggml_vk_get_device(0);

    أرجع &ggml_backend_vk_بفر_type_host;
}

ساكن ثابت حرف * ggml_backend_vk_name(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    أرجع ctx->name.c_str();
}

فراغ ggml_backend_vk_free(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    VK_LOG_DEBUG("ggml_backend_vk_free(" << ctx->name << ")");

    ggml_vk_cleanup(ctx);

    مٹاؤ ctx;
    مٹاؤ backend;
}

ساكن ggml_backend_بفر_type_t ggml_backend_vk_get_default_بفر_type(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    أرجع &ctx->device->بفر_type;
}

ساكن فراغ ggml_backend_vk_سیٹ_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, ثابت فراغ * انکڑے, size_t وتھ,
                                                size_t size, size_t n_copies, size_t stride_tensor, size_t stride_انکڑے) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->بفر->buft == ggml_backend_vk_get_default_بفر_type(backend) || tensor->بفر->buft == ggml_backend_vk_host_بفر_type()) && "unsupported buffer type");

    إذا (size == 0) {
        أرجع;
    }

    ggml_backend_vk_بفر_context * buf_ctx = (ggml_backend_vk_بفر_context *)tensor->بفر->context;

    vk_context cpy_ctx;

    إذا (ctx->device->async_use_transfer_قطار) {
        cpy_ctx = ggml_vk_get_transfer_ctx(ctx);
    } وإلا {
        cpy_ctx = ggml_vk_get_compute_ctx(ctx);
    }

    vk_بفر buf = buf_ctx->dev_بفر;

    خودکار dst_وتھ = vk_tensor_وتھ(tensor) + tensor->view_offs + وتھ;

    منطقي ret = ggml_vk_بفر_write_2d_async(cpy_ctx, buf, dst_وتھ, انکڑے, stride_انکڑے, stride_tensor, size, n_copies);

    إذا (!ret) {
        ثابت size_t staging_ناپ = size * n_copies;
        ggml_vk_ensure_sync_staging_بفر(ctx, staging_ناپ);
        ggml_vk_sync_buffers(خالی_سوچک, cpy_ctx);

        std::vector<vk::بفر_Copy> slices(1);
        إذا (size == stride_tensor) {
            slices[0].src_وتھ_2 = 0;
            slices[0].dst_وتھ_2 = dst_وتھ;
            slices[0].size = staging_ناپ;
        } وإلا {
            slices.resize(n_copies);
            لكل (size_t i = 0; i < n_copies; i++) {
                slices[i].src_وتھ_2 = i * size;
                slices[i].dst_وتھ_2 = dst_وتھ + i * stride_tensor;
                slices[i].size = size;
            }
        }

        cpy_ctx->s->بفر->buf.copy_بفر(ctx->sync_staging->بفر, buf->بفر, slices);

        إذا (size == stride_انکڑے) {
            deferred_memcpy(ctx->sync_staging->ptr, انکڑے, staging_ناپ, &cpy_ctx->وچ_memcpys);
        } وإلا {
            لكل (size_t i = 0; i < n_copies; i++) {
                deferred_memcpy((uint8_t *)ctx->sync_staging->ptr + i * size, (ثابت uint8_t *)انکڑے + i * stride_انکڑے, size, &cpy_ctx->وچ_memcpys);
            }
        }
        ggml_vk_synchronize(ctx);
    }
}

ساكن فراغ ggml_backend_vk_سیٹ_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ثابت فراغ * انکڑے, size_t وتھ, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_async(" << size << ")");
    ggml_backend_vk_سیٹ_tensor_2d_async(backend, tensor, انکڑے, وتھ, size, 1, size, size);
}

ساكن فراغ ggml_backend_vk_get_tensor_2d_async(ggml_backend_t backend, ثابت ggml_tensor * tensor, فراغ * انکڑے, size_t وتھ,
                                                size_t size, size_t n_copies, size_t stride_tensor, size_t stride_انکڑے) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->بفر->buft == ggml_backend_vk_get_default_بفر_type(backend) || tensor->بفر->buft == ggml_backend_vk_host_بفر_type()) && "unsupported buffer type");

    إذا (size == 0) {
        أرجع;
    }

    ggml_backend_vk_بفر_context * buf_ctx = (ggml_backend_vk_بفر_context *)tensor->بفر->context;

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    vk_بفر buf = buf_ctx->dev_بفر;

    خودکار src_وتھ = vk_tensor_وتھ(tensor) + tensor->view_offs + وتھ;
    منطقي ret = ggml_vk_بفر_read_2d_async(compute_ctx, buf, src_وتھ, انکڑے, stride_tensor, stride_انکڑے, size, n_copies);

    إذا (!ret) {
        ثابت size_t staging_ناپ = size * n_copies;
        ggml_vk_ensure_sync_staging_بفر(ctx, staging_ناپ);
        ggml_vk_sync_buffers(خالی_سوچک, compute_ctx);

        std::vector<vk::بفر_Copy> slices(1);
        إذا (size == stride_tensor) {
            slices[0].src_وتھ_2 = src_وتھ;
            slices[0].dst_وتھ_2 = 0;
            slices[0].size = staging_ناپ;
        } وإلا {
            slices.resize(n_copies);
            لكل (size_t i = 0; i < n_copies; i++) {
                slices[i].src_وتھ_2 = src_وتھ + i * stride_tensor;
                slices[i].dst_وتھ_2 = i * size;
                slices[i].size = size;
            }
        }

        compute_ctx->s->بفر->buf.copy_بفر(buf->بفر, ctx->sync_staging->بفر, slices);

        إذا (size == stride_انکڑے) {
            deferred_memcpy(انکڑے, ctx->sync_staging->ptr, staging_ناپ, &compute_ctx->باہرلا_memcpys);
        } وإلا {
            لكل (size_t i = 0; i < n_copies; i++) {
                deferred_memcpy((uint8_t *)انکڑے + i * stride_انکڑے, (ثابت uint8_t *)ctx->sync_staging->ptr + i * size, size, &compute_ctx->باہرلا_memcpys);
            }
        }
        ggml_vk_synchronize(ctx);
    }
}

ساكن فراغ ggml_backend_vk_get_tensor_async(ggml_backend_t backend, ثابت ggml_tensor * tensor, فراغ * انکڑے, size_t وتھ, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_async(" << size << ")");
    ggml_backend_vk_get_tensor_2d_async(backend, tensor, انکڑے, وتھ, size, 1, size, size);
}

ساكن منطقي ggml_backend_vk_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ثابت ggml_tensor * src, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_backend_vk_cpy_tensor_async(" << src << " -> " << dst << ", size=" << ggml_nbytes(src) << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend_dst->context;

    // Skip zero-size tensors
    إذا (ggml_nbytes(src) == 0) {
        أرجع سچ;
    }

    إذا (dst->بفر->buft != ggml_backend_vk_get_default_بفر_type(backend_dst)) {
        أرجع خطأ;
    }

    ggml_backend_vk_بفر_context * dst_buf_ctx = (ggml_backend_vk_بفر_context *)dst->بفر->context;
    vk_بفر dst_buf = dst_buf_ctx->dev_بفر;

    إذا (ggml_backend_بفر_is_vk(src->بفر)) {
        ggml_backend_vk_بفر_context * src_buf_ctx = (ggml_backend_vk_بفر_context *)src->بفر->context;

        // Async copy only works within the same device
        إذا (src_buf_ctx->dev_بفر->device != dst_buf->device) {
            أرجع خطأ;
        }

        vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

        ggml_vk_بفر_copy_async(compute_ctx, dst_buf, vk_tensor_وتھ(dst) + dst->view_offs,
                                   src_buf_ctx->dev_بفر, vk_tensor_وتھ(src) + src->view_offs,
                                   ggml_nbytes(src));
        أرجع سچ;
    }

    إذا (ggml_backend_بفر_is_host(src->بفر)) {
        vk_بفر pinned_buf = خالی_سوچک;
        size_t pinned_وتھ = 0;
        ggml_vk_host_get(ctx->device, src->انکڑے, pinned_buf, pinned_وتھ);
        إذا (pinned_buf == خالی_سوچک) {
            أرجع خطأ;
        }

        // If the backend is idle, use a CPU copy to avoid GPU synchronization overhead.
        ساكن constexpr size_t max_cpu_copy_ناپ = 128 * 1024;
        ثابت منطقي src_backend_synchronous = backend_src->iface.synchronize == خالی_سوچک;
        ثابت منطقي transfer_idle = !ctx->device->async_use_transfer_قطار ||
                                   ctx->transfer_semaphore_آخری_submitted == ctx->transfer_semaphore.value;
        ثابت منطقي backend_idle = ctx->compute_ctx.expired() && ctx->transfer_ctx.expired() &&
                                  !ctx->submit_pending && !ctx->almost_ready_fence_pending && transfer_idle;
        ثابت منطقي dst_host_coherent =
            (dst_buf->حافظہ_property_flags & (vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent)) ==
            (vk::حافظہ_Property_Flag_بٹ::eHostVisible | vk::حافظہ_Property_Flag_بٹ::eHostCoherent);

        إذا ((backend_src == backend_dst || src_backend_synchronous) && backend_idle && dst_host_coherent && ggml_nbytes(src) <= max_cpu_copy_ناپ) {
            ggml_vk_بفر_write(dst_buf, vk_tensor_وتھ(dst) + dst->view_offs, src->انکڑے, ggml_nbytes(src));
            أرجع سچ;
        }

        vk_context cpy_ctx;
        إذا (ctx->device->async_use_transfer_قطار) {
            cpy_ctx = ggml_vk_get_transfer_ctx(ctx);
        } وإلا {
            cpy_ctx = ggml_vk_get_compute_ctx(ctx);
        }

        أرجع ggml_vk_بفر_write_async(cpy_ctx, dst_buf,
                                          vk_tensor_وتھ(dst) + dst->view_offs,
                                          src->انکڑے, ggml_nbytes(src));
    }

    أرجع خطأ;
}

فراغ ggml_vk_synchronize(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_synchronize()");

    منطقي do_transfer = !ctx->compute_ctx.expired();

    إذا (ggml_vk_submit_transfer_ctx(ctx)) {
        ctx->submit_pending = سچ;
    }

    vk_context compute_ctx;
    vk_command_بفر* cmd_buf = خالی_سوچک;
    إذا (do_transfer) {
        compute_ctx = ctx->compute_ctx.جندرا();
        إذا (compute_ctx->s) {
            cmd_buf = compute_ctx->s->بفر;
        }

        ggml_vk_ctx_سرا(compute_ctx);

        لكل (خودکار& cpy : compute_ctx->وچ_memcpys) {
            حافظہ_نقل(cpy.dst, cpy.src, cpy.n);
        }

        إذا (ctx->device->serialize_submissions) {
            ggml_vk_submit(compute_ctx, ctx->fence);
            VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, سچ, UINT64_MAX), "synchronize waitForFences", ctx->device);
            ctx->device->device.resetFences({ ctx->fence });
        } وإلا {
            ggml_vk_submit(compute_ctx, {});
        }
        ctx->submit_pending = سچ;
    }

    إذا (ctx->submit_pending) {
        إذا (ctx->device->serialize_submissions) {
            ctx->submit_pending = خطأ;
        } وإلا إذا (ctx->device->async_use_transfer_قطار && ctx->transfer_semaphore_آخری_submitted < ctx->transfer_semaphore.value) {
            vk::TimelineSemaphoreSubmitInfo tl_info{
                1, &ctx->transfer_semaphore.value,
                0, خالی_سوچک,
            };
            vk::PipelineStageFlags stage = ctx->device->transfer_قطار->stage_flags;
            vk::SubmitInfo si{
                1, &ctx->transfer_semaphore.s, &stage,
                0, خالی_سوچک,
                0, خالی_سوچک,
            };
            si.سیٹ_PNext(&tl_info);
            ctx->device->compute_قطار->handle->submit({ si }, ctx->fence);
            ctx->transfer_semaphore_آخری_submitted = ctx->transfer_semaphore.value;
        } وإلا {
            ctx->device->compute_قطار->handle->submit({}, ctx->fence);
        }
        إذا (!ctx->device->serialize_submissions) {
            ggml_vk_wait_for_fence(ctx);
        }
        ctx->submit_pending = خطأ;
        إذا (cmd_buf) {
            cmd_buf->وچ_use = خطأ;
            cmd_buf->buf.reset();
        }
    }

    إذا (do_transfer) {
        لكل (خودکار& cpy : compute_ctx->باہرلا_memcpys) {
            حافظہ_نقل(cpy.dst, cpy.src, cpy.n);
        }
        ctx->compute_ctx.reset();
    }
}

ساكن فراغ ggml_backend_vk_synchronize(ggml_backend_t backend) {
    VK_LOG_DEBUG("ggml_backend_vk_synchronize()");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ggml_vk_synchronize(ctx);

    ggml_vk_گراف_cleanup(ctx);
}

منطقي ggml_vk_is_خالی_تھاواں(ggml_tensor * گرہ) {
    أرجع ggml_is_خالی_تھاواں(گرہ) || گرہ->op == GGML_OP_NONE || گرہ->op == GGML_OP_RESHAPE || گرہ->op == GGML_OP_TRANSPOSE || گرہ->op == GGML_OP_VIEW || گرہ->op == GGML_OP_ترتیب_بدلو;
}

ساكن منطقي ggml_vk_can_fuse_unary_mul(ثابت بنية ggml_cgraph * cgraph, صحيح unary_idx, صحيح mul_idx) {
    ثابت ggml_tensor * unary = cgraph->nodes[unary_idx];
    ثابت ggml_tensor * mul = cgraph->nodes[mul_idx];

    إذا (ggml_vk_unary_mul_op_اشاریہ(ggml_get_unary_op(unary)) < 0) {
        أرجع خطأ;
    }
    إذا (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
        أرجع خطأ;
    }
    إذا (unary->type != mul->type) {
        أرجع خطأ;
    }
    إذا (mul->src[0] != unary && mul->src[1] != unary) {
        أرجع خطأ;
    }
    ثابت ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
    إذا (other == خالی_سوچک || other->type != unary->type) {
        أرجع خطأ;
    }
    إذا (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0])) {
        أرجع خطأ;
    }
    // fastmod needs src to tile into dst
    إذا (mul->src[0] == unary) {
        أرجع ggml_can_repeat(other, unary);
    }
    أرجع ggml_can_repeat(unary, mul->src[0]);
}

ساكن منطقي ggml_vk_can_fuse_unary_mul_pair(ثابت بنية ggml_cgraph * cgraph, صحيح گرہ_idx) {
    ثابت تعداد ggml_op ops[]    = { GGML_OP_UNARY, GGML_OP_MUL };
    ثابت صحيح           outputs[] = { گرہ_idx + 1 };
    أرجع ggml_can_fuse_subgraph(cgraph, گرہ_idx, 2, ops, outputs, 1) &&
           ggml_vk_can_fuse_unary_mul(cgraph, گرہ_idx, گرہ_idx + 1);
}

ساكن منطقي ggml_vk_can_fuse_hc_post_gate(ثابت بنية ggml_cgraph * cgraph, صحيح گرہ_idx) {
    ثابت ggml_tensor * scale_وچ  = cgraph->nodes[گرہ_idx];
    ثابت ggml_tensor * sigmoid   = cgraph->nodes[گرہ_idx + 1];
    ثابت ggml_tensor * scale_باہرلا = cgraph->nodes[گرہ_idx + 2];

    // the shader folds scale -> sigmoid -> scale; a bias on either scale is not handled
    أرجع ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID &&
           ggml_get_op_params_f32(scale_وچ, 1) == 0.0f &&
           ggml_get_op_params_f32(scale_باہرلا, 1) == 0.0f &&
           scale_وچ->src[0]->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(scale_وچ->src[0], scale_باہرلا);
}

منطقي ggml_vk_can_fuse(ثابت ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح گرہ_idx, std::initializer_list<تعداد ggml_op> ops) {
    إذا (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL) {
        أرجع ggml_vk_can_fuse_unary_mul_pair(cgraph, گرہ_idx);
    }

    إذا (!ggml_can_fuse(cgraph, گرہ_idx, ops)) {
        أرجع خطأ;
    }

    إذا ((ops.size() == 2 || ops.size() == 3 || ops.size() == 4) &&
        ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        ثابت ggml_tensor *rms_norm = cgraph->nodes[گرہ_idx];
        ثابت ggml_tensor *mul = cgraph->nodes[گرہ_idx + 1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        // rms_norm only supports f32
        إذا (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }
        // if rms_norm is the B operand, then we don't handle broadcast
        إذا (rms_norm == mul->src[1] &&
            !ggml_are_same_shape(mul->src[0], rms_norm)) {
            أرجع خطأ;
        }
        // rms_norm shader assumes contiguous rows
        إذا (!ggml_is_contiguous_صفاں(mul->src[0]) || !ggml_is_contiguous_صفاں(mul->src[1])) {
            أرجع خطأ;
        }

        إذا (ops.size() >= 3 && ops.begin()[2] == GGML_OP_جوڑو) {
            ثابت ggml_tensor *جوڑو_2 = cgraph->nodes[گرہ_idx + 2];
            ثابت ggml_tensor *residual = جوڑو_2->src[0] == mul ? جوڑو_2->src[1] : جوڑو_2->src[0];
            إذا (جوڑو_2->src[0] != mul && جوڑو_2->src[1] != mul) {
                أرجع خطأ;
            }
            إذا (residual->type != GGML_TYPE_F32 || جوڑو_2->type != GGML_TYPE_F32 ||
                !ggml_are_same_shape(جوڑو_2, residual) || !ggml_is_contiguous(residual) ||
                !ggml_is_contiguous(جوڑو_2) || get_misalign_بائٹاں(ctx, residual) != 0) {
                أرجع خطأ;
            }

            ثابت ggml_tensor *dst = جوڑو_2;
            إذا (ops.size() == 4) {
                إذا (ops.begin()[3] != GGML_OP_MUL) {
                    أرجع خطأ;
                }

                ثابت ggml_tensor *post_mul = cgraph->nodes[گرہ_idx + 3];
                ثابت ggml_tensor *scale = post_mul->src[0] == جوڑو_2 ? post_mul->src[1] : post_mul->src[0];
                إذا (post_mul->src[0] != جوڑو_2 && post_mul->src[1] != جوڑو_2) {
                    أرجع خطأ;
                }
                // The shader reads data_e[0], so the final multiply must use a scalar.
                إذا (scale->type != GGML_TYPE_F32 || post_mul->type != GGML_TYPE_F32 ||
                    ggml_nelements(scale) != 1 || !ggml_is_contiguous(post_mul) ||
                    get_misalign_بائٹاں(ctx, scale) != 0) {
                    أرجع خطأ;
                }
                dst = post_mul;
            }

            إذا (get_misalign_بائٹاں(ctx, dst) != 0) {
                أرجع خطأ;
            }
        }
    }

    خودکار ثابت &mm_جوڑو_ok = [&](ثابت ggml_tensor *mul, ثابت ggml_tensor *جوڑو_2) {
        ثابت ggml_tensor *bias = جوڑو_2->src[0] == mul ? جوڑو_2->src[1] : جوڑو_2->src[0];

        // mat-vec only
        إذا (ggml_nrows(mul) != 1) {
            أرجع خطأ;
        }
        // shaders assume the types match
        إذا (mul->type != bias->type) {
            أرجع خطأ;
        }
        // shaders reuse the D shape for bias
        إذا (!ggml_are_same_shape(mul, bias) ||
            !ggml_are_same_stride(mul, bias)) {
            أرجع خطأ;
        }
        // unaligned bias isn't handled
        إذا (get_misalign_بائٹاں(ctx, bias) != 0) {
            أرجع خطأ;
        }
        أرجع سچ;
    };

    إذا ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_جوڑو) {
        // additional constraints specific to this fusion
        ثابت ggml_tensor *mul = cgraph->nodes[گرہ_idx];
        ثابت ggml_tensor *جوڑو_2 = cgraph->nodes[گرہ_idx + 1];

        إذا (!mm_جوڑو_ok(mul, جوڑو_2)) {
            أرجع خطأ;
        }
        إذا (ops.size() == 3) {
            إذا (ops.begin()[2] != GGML_OP_جوڑو) {
                أرجع خطأ;
            }
            إذا (!mm_جوڑو_ok(جوڑو_2, cgraph->nodes[گرہ_idx + 2])) {
                أرجع خطأ;
            }
        }
    }

    خودکار ثابت &mmid_mul_ok = [&](ثابت ggml_tensor *mmid, ثابت ggml_tensor *mul) {
        ثابت ggml_tensor *scale = mul->src[1];

        إذا (mmid != mul->src[0]) {
            أرجع خطأ;
        }
        // mat-vec only
        إذا (!ggml_vk_use_mul_mat_vec_id(cgraph, گرہ_idx)) {
            أرجع خطأ;
        }
        // shaders assume the types match
        إذا (mmid->type != scale->type) {
            أرجع خطأ;
        }
        // shaders assume the bias is contiguous
        إذا (!ggml_is_contiguous(scale)) {
            أرجع خطأ;
        }
        // unaligned bias isn't handled
        إذا (get_misalign_بائٹاں(ctx, scale) != 0) {
            أرجع خطأ;
        }
        // shader only indexes by expert index
        إذا (scale->ne[0] != 1 ||
            scale->ne[1] != mul->ne[1] ||
            scale->ne[2] != 1 ||
            scale->ne[3] != 1) {
            أرجع خطأ;
        }
        أرجع سچ;
    };

    إذا ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_جوڑو_ID) {
        // additional constraints specific to this fusion
        ثابت ggml_tensor *mul = cgraph->nodes[گرہ_idx];
        ثابت ggml_tensor *جوڑو_2 = cgraph->nodes[گرہ_idx + 1];
        ثابت ggml_tensor *bias = جوڑو_2->src[1];

        إذا (mul != جوڑو_2->src[0]) {
            أرجع خطأ;
        }
        // mat-vec only
        إذا (!ggml_vk_use_mul_mat_vec_id(cgraph, گرہ_idx)) {
            أرجع خطأ;
        }
        // shaders assume the types match
        إذا (mul->type != bias->type) {
            أرجع خطأ;
        }
        // shaders assume the bias is contiguous
        إذا (!ggml_is_contiguous(bias)) {
            أرجع خطأ;
        }
        // the ID tensor must be the same for mul_mat_id and add_id
        إذا (mul->src[2] != جوڑو_2->src[2]) {
            أرجع خطأ;
        }
        // unaligned bias isn't handled
        إذا (get_misalign_بائٹاں(ctx, bias) != 0) {
            أرجع خطأ;
        }

        إذا (ops.size() == 3) {
            إذا (ops.begin()[2] != GGML_OP_MUL) {
                أرجع خطأ;
            }
            ثابت ggml_tensor *mul = cgraph->nodes[گرہ_idx + 2];
            أرجع mmid_mul_ok(جوڑو_2, mul);
        }
    }

    إذا (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        ثابت ggml_tensor *mmid = cgraph->nodes[گرہ_idx];
        ثابت ggml_tensor *mul = cgraph->nodes[گرہ_idx + 1];

        إذا (!mmid_mul_ok(mmid, mul)) {
            أرجع خطأ;
        }
    }

    أرجع سچ;
}

منطقي ggml_vk_can_fuse_ssm_conv(ثابت ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph,
                                      صحيح گرہ_idx, صحيح عدد_extra) {
    ثابت ggml_tensor * conv = cgraph->nodes[گرہ_idx];
    إذا (conv->op != GGML_OP_SSM_CONV) {
        أرجع خطأ;
    }

    ثابت ggml_tensor * silu = خالی_سوچک;
    ثابت ggml_tensor * bias = خالی_سوچک;

    إذا (عدد_extra == 1) {
        إذا (!ggml_can_fuse(cgraph, گرہ_idx, { GGML_OP_SSM_CONV, GGML_OP_UNARY })) {
            أرجع خطأ;
        }
        silu = cgraph->nodes[گرہ_idx + 1];
    } وإلا إذا (عدد_extra == 2) {
        إذا (!ggml_can_fuse(cgraph, گرہ_idx, { GGML_OP_SSM_CONV, GGML_OP_جوڑو, GGML_OP_UNARY })) {
            أرجع خطأ;
        }
        ثابت ggml_tensor * جوڑو_2 = cgraph->nodes[گرہ_idx + 1];
        silu = cgraph->nodes[گرہ_idx + 2];
        bias = (جوڑو_2->src[0] == conv) ? جوڑو_2->src[1] : جوڑو_2->src[0];

        إذا (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            أرجع خطأ;
        }
        // bias must be channel-wise (one element per channel of the conv output)
        إذا (ggml_nelements(bias) != conv->ne[0] || bias->ne[0] != conv->ne[0]) {
            أرجع خطأ;
        }
        إذا (جوڑو_2->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }
        // The shader doesn't apply per-tensor offsets, so reject misaligned bias.
        إذا (get_misalign_بائٹاں(ctx, bias) != 0) {
            أرجع خطأ;
        }
    } وإلا {
        أرجع خطأ;
    }

    إذا (ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU) {
        أرجع خطأ;
    }
    إذا (conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }
    // The shader writes to the fused dst using its own strides, but the push constants don't
    // carry a per-tensor offset, so the binding must be naturally aligned.
    إذا (get_misalign_بائٹاں(ctx, silu) != 0) {
        أرجع خطأ;
    }
    أرجع سچ;
}

منطقي ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph,
                                      صحيح گرہ_idx, topk_moe_منوال mode) {

    ثابت ggml_tensor * softmax;
    ثابت ggml_tensor * weights;
    ثابت ggml_tensor * get_صفاں;
    ثابت ggml_tensor * argsort;

    بدل (mode) {
    حالة TOPK_MOE_EARLY_SOFTMAX_NORM:
        softmax = cgraph->nodes[گرہ_idx + 0];
        weights = cgraph->nodes[گرہ_idx + 9];
        get_صفاں = cgraph->nodes[گرہ_idx + 4];
        argsort = cgraph->nodes[گرہ_idx + 2];
        اكسر;
    حالة TOPK_MOE_SIGMOID_NORM_BIAS:
        softmax = cgraph->nodes[گرہ_idx + 0]; // really sigmoid
        weights = cgraph->nodes[گرہ_idx + 10];
        get_صفاں = cgraph->nodes[گرہ_idx + 5];
        argsort = cgraph->nodes[گرہ_idx + 3];
        إذا (ggml_get_unary_op(softmax) != GGML_UNARY_OP_SIGMOID) {
            أرجع خطأ;
        }
        // bias is expected to be 1D
        إذا (ggml_nrows(cgraph->nodes[گرہ_idx + 2]->src[1]) != 1 ||
            !ggml_is_contiguous(cgraph->nodes[گرہ_idx + 2]->src[1])) {
            أرجع خطأ;
        }
        // sigmoid fusion seems to generate infinities on moltenvk
        إذا (ctx->device->driver_id == vk::DriverId::eMoltenvk) {
            أرجع خطأ;
        }
        اكسر;
    حالة TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS:
        softmax = cgraph->nodes[گرہ_idx + 0]; // really softplus
        weights = cgraph->nodes[گرہ_idx + 11];
        get_صفاں = cgraph->nodes[گرہ_idx + 6];
        argsort = cgraph->nodes[گرہ_idx + 4];
        إذا (ggml_get_unary_op(softmax) != GGML_UNARY_OP_SOFTPLUS) {
            أرجع خطأ;
        }
        // bias is expected to be 1D
        إذا (ggml_nrows(cgraph->nodes[گرہ_idx + 3]->src[1]) != 1 ||
            !ggml_is_contiguous(cgraph->nodes[گرہ_idx + 3]->src[1])) {
            أرجع خطأ;
        }
        اكسر;
    حالة TOPK_MOE_EARLY_SOFTMAX:
        softmax = cgraph->nodes[گرہ_idx + 0];
        weights = cgraph->nodes[گرہ_idx + 4];
        get_صفاں = cgraph->nodes[گرہ_idx + 4];
        argsort = cgraph->nodes[گرہ_idx + 2];
        اكسر;
    حالة TOPK_MOE_LATE_SOFTMAX:
        softmax = cgraph->nodes[گرہ_idx + 4];
        weights = cgraph->nodes[گرہ_idx + 5];
        get_صفاں = cgraph->nodes[گرہ_idx + 2];
        argsort = cgraph->nodes[گرہ_idx + 0];
        اكسر;
    افتراضي:
        أرجع خطأ;
    }

    ggml_tensor * probs = get_صفاں->src[0];
    إذا (probs->op != GGML_OP_RESHAPE) {
        أرجع خطأ;
    }
    probs = probs->src[0];
    ggml_tensor * selection_probs = argsort->src[0];

    إذا (probs != selection_probs &&
        mode != TOPK_MOE_SIGMOID_NORM_BIAS &&
        mode != TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS) {
        أرجع خطأ;
    }

    إذا (!ggml_is_contiguous(softmax->src[0]) || !ggml_is_contiguous(weights)) {
        أرجع خطأ;
    }

    إذا (softmax->op == GGML_OP_SOFT_MAX) {
        ثابت عائم * op_params = (ثابت عائم *)softmax->op_params;

        عائم scale = op_params[0];
        عائم max_bias = op_params[1];

        إذا (scale != 1.0f || max_bias != 0.0f) {
            أرجع خطأ;
        }

        // don't fuse when masks or sinks are present
        إذا (softmax->src[1] || softmax->src[2]) {
            أرجع خطأ;
        }
    }

    ثابت صحيح n_expert = softmax->ne[0];
    إذا (n_expert > (1 << (عدد_topk_moe_pipelines-1))) {
        أرجع خطأ;
    }

    إذا (!ctx->device->subgroup_arithmetic ||
        !ctx->device->subgroup_shuffle ||
        !ctx->device->subgroup_require_بھریا_support ||
        ctx->device->disable_fusion) {
        أرجع خطأ;
    }

    أرجع سچ;
}

ساكن منطقي ggml_vk_میل_ops(ثابت بنية ggml_cgraph * cgraph, صحيح گرہ_idx,
                              ثابت std::initializer_list<ggml_op> & ops) {
    إذا (گرہ_idx + (صحيح) ops.size() > cgraph->n_nodes) {
        أرجع خطأ;
    }
    لكل (size_t j = 0; j < ops.size(); ++j) {
        ثابت ggml_tensor * گرہ = cgraph->nodes[گرہ_idx + j];
        إذا (گرہ->op != ops.begin()[j] ||
            (گرہ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
            (گرہ->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            أرجع خطأ;
        }
    }
    أرجع سچ;
}

منطقي ggml_vk_can_fuse_topk_qsa(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح گرہ_idx) {
    إذا (ctx->device->disable_fusion || !ctx->device->pipeline_topk_radix_qsa) {
        أرجع خطأ;
    }

    ثابت صحيح n_ops = topk_qsa_نمونہ.size();
    إذا (!ggml_vk_میل_ops(cgraph, گرہ_idx, topk_qsa_نمونہ) ||
        !ggml_check_کنارے(cgraph, گرہ_idx, topk_qsa_کنارے)) {
        أرجع خطأ;
    }

    // elided nodes must be single-use (cpy counts its own src[1] self-reference)
    لكل (صحيح j = 0; j < n_ops - 1; ++j) {
        ثابت ggml_tensor * گرہ = cgraph->nodes[گرہ_idx + j];
        ثابت int32_t want = گرہ->op == GGML_OP_CPY ? 2 : 1;
        إذا (ggml_گرہ_get_use_گنتی(cgraph, گرہ_idx + j) != want) {
            أرجع خطأ;
        }
    }

    ثابت ggml_tensor * get_صفاں = cgraph->nodes[گرہ_idx + 0];
    ثابت ggml_tensor * جوڑو_2      = cgraph->nodes[گرہ_idx + n_ops - 2];
    ثابت ggml_tensor * چوٹی_k    = cgraph->nodes[گرہ_idx + n_ops - 1];

    ثابت ggml_tensor * scores   = get_صفاں->src[0]; // [n_tps, n_blocks, n_stream]
    ثابت ggml_tensor * cell_blk = get_صفاں->src[1]; // [n_kv, n_stream]
    ثابت ggml_tensor * expanded = جوڑو_2->src[0];      // [n_kv, n_tps, n_stream]

    // raw mask: follow the reshape/cpy chain back to the materialized f16 input
    ثابت ggml_tensor * نقاب = جوڑو_2->src[1];
    طالما (نقاب && (نقاب->op == GGML_OP_RESHAPE || نقاب->op == GGML_OP_CPY)) {
        نقاب = نقاب->src[0];
    }
    إذا (!نقاب || نقاب->type != GGML_TYPE_F16) {
        أرجع خطأ;
    }

    إذا (scores->type != GGML_TYPE_F32 || cell_blk->type != GGML_TYPE_I32 || چوٹی_k->type != GGML_TYPE_I32) {
        أرجع خطأ;
    }
    إذا (!ggml_is_contiguous(scores) || !ggml_is_contiguous(cell_blk) || !ggml_is_contiguous(نقاب) ||
        !ggml_is_contiguous(expanded) || !ggml_is_contiguous(چوٹی_k)) {
        أرجع خطأ;
    }

    ثابت int64_t n_tps    = scores->ne[0];
    ثابت int64_t n_بلاکاں = scores->ne[1];
    ثابت int64_t n_دھارا = scores->ne[2];
    ثابت int64_t n_kv     = cell_blk->ne[0];
    ثابت int64_t چوڑائی    = چوٹی_k->ne[0];

    // pin the indexer layout the shader's addressing assumes
    إذا (scores->ne[3] != 1 || cell_blk->ne[1] != n_دھارا || ggml_nrows(cell_blk) != n_دھارا ||
        ggml_nelements(نقاب) != n_kv * n_tps * n_دھارا ||
        expanded->ne[0] != n_kv || expanded->ne[1] != n_tps || expanded->ne[2] != n_دھارا ||
        چوٹی_k->ne[1] != n_tps || چوٹی_k->ne[2] != n_دھارا || چوٹی_k->ne[3] != 1 ||
        n_بلاکاں <= 0 || n_kv <= 0 || چوڑائی <= 0 || چوڑائی > n_kv) {
        أرجع خطأ;
    }

    // only worth it in the radix regime; small k uses the faster tournament unfused
    ثابت uint32_t k_min_pipeline = std::max((uint32_t) log2f(عائم(چوڑائی)) + 1, ctx->device->subgroup_ناپ_log2);
    إذا (k_min_pipeline < عدد_topk_pipelines && ctx->device->pipeline_topk_f32[k_min_pipeline]) {
        أرجع خطأ;
    }
    أرجع سچ;
}

منطقي ggml_vk_can_fuse_rope_سیٹ_صفاں(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph,
                                           صحيح گرہ_idx) {
    ثابت ggml_tensor *rope = cgraph->nodes[گرہ_idx + 0];
    ثابت ggml_tensor *view = cgraph->nodes[گرہ_idx + 1];
    ثابت ggml_tensor *سیٹ_صفاں = cgraph->nodes[گرہ_idx + 2];

    // The set_rows epilogue uses one index per ne2 slice and does not encode ne3.
    إذا (rope->src[0]->ne[3] != 1) {
        أرجع خطأ;
    }

    إذا (سیٹ_صفاں->type != GGML_TYPE_F32 && سیٹ_صفاں->type != GGML_TYPE_F16) {
        أرجع خطأ;
    }

    // The shader reads each aligned I64 index as a uvec2 and uses its low 32 bits.
    إذا (سیٹ_صفاں->src[1]->type != GGML_TYPE_I64 || !ggml_is_contiguous(سیٹ_صفاں->src[1]) ||
        سیٹ_صفاں->nb[0] != ggml_type_ناپ(سیٹ_صفاں->type) || get_misalign_بائٹاں(ctx, سیٹ_صفاں->src[1]) != 0) {
        أرجع خطأ;
    }

    // SET_ROWS consumes one flattened [ne0*ne1] row for each ne2 slice.
    إذا (!ggml_is_contiguous(view) ||
        view->ne[0] != rope->ne[0] * rope->ne[1] || view->ne[1] != rope->ne[2] ||
        view->ne[2] != 1 || view->ne[3] != 1 ||
        ggml_nelements(سیٹ_صفاں->src[1]) != rope->ne[2]) {
        أرجع خطأ;
    }

    // Only norm/neox/mrope/imrope shaders have the fusion code
    ثابت صحيح mode = ((ثابت int32_t *) rope->op_params)[2];
    إذا (mode != GGML_ROPE_TYPE_عام && mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE && mode != GGML_ROPE_TYPE_IMROPE) {
        أرجع خطأ;
    }

    أرجع سچ;
}

منطقي ggml_vk_can_fuse_rms_norm_سیٹ_صفاں(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph,
                                               صحيح گرہ_idx) {
    ثابت ggml_tensor * rms = cgraph->nodes[گرہ_idx];
    ثابت ggml_tensor * view = cgraph->nodes[گرہ_idx + 1];
    ثابت ggml_tensor * سیٹ_صفاں = cgraph->nodes[گرہ_idx + 2];

    // The RMS kernel reads F32 and writes directly to the F32 or F16 SET_ROWS destination.
    إذا (rms->src[0]->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 ||
        (سیٹ_صفاں->type != GGML_TYPE_F32 && سیٹ_صفاں->type != GGML_TYPE_F16) ||
        سیٹ_صفاں->src[1]->type != GGML_TYPE_I64 || !ggml_is_contiguous(سیٹ_صفاں->src[1]) ||
        سیٹ_صفاں->nb[0] != ggml_type_ناپ(سیٹ_صفاں->type) || get_misalign_بائٹاں(ctx, سیٹ_صفاں->src[1]) != 0) {
        أرجع خطأ;
    }
    // As with the ROPE epilogue, each ne2 slice supplies one flattened row and ne3 is not encoded.
    إذا (rms->ne[3] != 1 || !ggml_is_contiguous(rms->src[0]) || !ggml_is_contiguous(view)) {
        أرجع خطأ;
    }
    إذا (view->ne[0] != rms->ne[0] * rms->ne[1] || view->ne[1] != rms->ne[2] ||
        view->ne[2] != 1 || view->ne[3] != 1 ||
        ggml_nelements(سیٹ_صفاں->src[1]) != rms->ne[2]) {
        أرجع خطأ;
    }

    أرجع سچ;
}

منطقي ggml_vk_can_fuse_snake(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح گرہ_idx) {
    GGML_بن_ورتیا(ctx);
    إذا (!ggml_can_fuse(cgraph, گرہ_idx, snake_نمونہ)) {
        أرجع خطأ;
    }

    ثابت ggml_tensor * mul0     = cgraph->nodes[گرہ_idx + 0];
    ثابت ggml_tensor * sin_گرہ = cgraph->nodes[گرہ_idx + 1];
    ثابت ggml_tensor * sqr      = cgraph->nodes[گرہ_idx + 2];
    ثابت ggml_tensor * mul1     = cgraph->nodes[گرہ_idx + 3];
    ثابت ggml_tensor * جوڑو_2      = cgraph->nodes[گرہ_idx + 4];

    ثابت ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ثابت ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    ثابت ggml_tensor * inv_b    = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];
    ثابت ggml_tensor * x_وچ_جوڑو = (جوڑو_2->src[0] == mul1) ? جوڑو_2->src[1] : جوڑو_2->src[0];

    إذا (x_وچ_جوڑو != x) {
        أرجع خطأ;
    }
    إذا (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_F16 && x->type != GGML_TYPE_BF16) {
        أرجع خطأ;
    }
    // Shader bindings: data_a is A_TYPE so it follows x's precision, while
    // data_b and data_c are hardcoded float, so the broadcast operands must
    // be F32 regardless of x's type.
    إذا (a->type     != GGML_TYPE_F32) أرجع خطأ;
    إذا (inv_b->type != GGML_TYPE_F32) أرجع خطأ;
    // Chain intermediates and output share x's precision (single A_TYPE / D_TYPE pipeline).
    إذا (mul0->type     != x->type) أرجع خطأ;
    إذا (sin_گرہ->type != x->type) أرجع خطأ;
    إذا (sqr->type      != x->type) أرجع خطأ;
    إذا (mul1->type     != x->type) أرجع خطأ;
    إذا (جوڑو_2->type      != x->type) أرجع خطأ;
    إذا (!ggml_are_same_shape(a, inv_b)) {
        أرجع خطأ;
    }
    إذا (a->ne[0] != 1 || a->ne[1] != x->ne[1]) {
        أرجع خطأ;
    }
    // Dispatch is 2D over (ne0, ne1), so x and add must be 2D and a / inv_b
    // must collapse to [1, C, 1, 1]. Higher dims are not handled by the shader.
    إذا (x->ne[2]     != 1 || x->ne[3]     != 1) أرجع خطأ;
    إذا (جوڑو_2->ne[2]   != 1 || جوڑو_2->ne[3]   != 1) أرجع خطأ;
    إذا (a->ne[2]     != 1 || a->ne[3]     != 1) أرجع خطأ;
    إذا (inv_b->ne[2] != 1 || inv_b->ne[3] != 1) أرجع خطأ;
    // Shader uses idx = i0 + i1 * ne0 and reads data_b[i1] / data_c[i1],
    // so every operand must be contiguous.
    إذا (!ggml_is_contiguous(x) || !ggml_is_contiguous(جوڑو_2) ||
        !ggml_is_contiguous(a) || !ggml_is_contiguous(inv_b)) {
        أرجع خطأ;
    }
    أرجع سچ;
}

منطقي ggml_vk_tensors_overlap(ثابت ggml_tensor * a, ثابت ggml_tensor * b, منطقي elementwise) {
    ggml_backend_vk_بفر_context * a_buf_ctx = (ggml_backend_vk_بفر_context *)a->بفر->context;
    vk_بفر a_buf = a_buf_ctx->dev_بفر;
    ggml_backend_vk_بفر_context * b_buf_ctx = (ggml_backend_vk_بفر_context *)b->بفر->context;
    vk_بفر b_buf = b_buf_ctx->dev_بفر;
    إذا (a_buf == b_buf) {
        خودکار a_ادھار = vk_tensor_وتھ(a) + a->view_offs;
        خودکار a_ناپ = ggml_nbytes(a);
        خودکار b_ادھار = vk_tensor_وتھ(b) + b->view_offs;
        خودکار b_ناپ = ggml_nbytes(b);

        إذا (elementwise && a_ادھار == b_ادھار && a_ناپ == b_ناپ) {
            أرجع خطأ;
        }

        إذا ((b_ادھار <= a_ادھار && a_ادھار < b_ادھار + b_ناپ) ||
            (a_ادھار <= b_ادھار && b_ادھار < a_ادھار + a_ناپ)) {
            أرجع سچ;
        }
    }
    أرجع خطأ;
}

منطقي ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph,
                                               صحيح گرہ_idx) {
    ثابت ggml_tensor *rms = cgraph->nodes[گرہ_idx + 0];
    ثابت ggml_tensor *mul = cgraph->nodes[گرہ_idx + 1];
    ثابت ggml_tensor *rope = cgraph->nodes[گرہ_idx + 2];

    ثابت صحيح mode = ((ثابت int32_t *) rope->op_params)[2];

    // noncontig tensors aren't tested, and don't seem common in practice
    إذا (!ggml_is_contiguous(rms) ||
        !ggml_is_contiguous(mul) ||
        !ggml_is_contiguous(rope)) {
        أرجع خطأ;
    }

    // only norm/neox are handled in the shader
    إذا (mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_عام) {
        أرجع خطأ;
    }

    // shared memory size for passing data from mul->rope
    إذا (mul->ne[0] > 1024) {
        أرجع خطأ;
    }

    // conditions for pipeline creation
    إذا (حجم(vk_op_rms_norm_mul_rope_ڈالو_constants) > ctx->device->properties.limits.max_ڈالو_Constants_ناپ) {
        أرجع خطأ;
    }

    أرجع سچ;
}

uint32_t ggml_vk_fuse_multi_جوڑو(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح گرہ_idx) {

    ثابت ggml_tensor *پہلا_گرہ = cgraph->nodes[گرہ_idx];
    إذا (پہلا_گرہ->op != GGML_OP_جوڑو) {
        أرجع 0;
    }

    إذا (!ctx->device->multi_جوڑو) {
        أرجع 0;
    }

    int32_t عدد_adds = 1;
    طالما (گرہ_idx + عدد_adds < cgraph->n_nodes &&
           cgraph->nodes[گرہ_idx + عدد_adds]->op == GGML_OP_جوڑو &&
           عدد_adds < MAX_FUSED_ADDS) {
        عدد_adds++;
    }

    // The shader currently requires same shapes (but different strides are allowed),
    // everything f32, and no misalignment
    لكل (int32_t i = 0; i < عدد_adds; ++i) {
        ثابت ggml_tensor *اگلا_گرہ = cgraph->nodes[گرہ_idx + i];
        إذا (!ggml_are_same_shape(پہلا_گرہ, اگلا_گرہ->src[0]) ||
            !ggml_are_same_shape(پہلا_گرہ, اگلا_گرہ->src[1]) ||
            اگلا_گرہ->type != GGML_TYPE_F32 ||
            اگلا_گرہ->src[0]->type != GGML_TYPE_F32 ||
            اگلا_گرہ->src[1]->type != GGML_TYPE_F32 ||
            get_misalign_بائٹاں(ctx, اگلا_گرہ) ||
            get_misalign_بائٹاں(ctx, اگلا_گرہ->src[0]) ||
            get_misalign_بائٹاں(ctx, اگلا_گرہ->src[1])) {
            عدد_adds = i;
        }
    }

    // Verify we can fuse these
    ggml_op adds[MAX_FUSED_ADDS];
    لكل (int32_t i = 0; i < عدد_adds; ++i) {
        adds[i] = GGML_OP_جوڑو;
    }

    // decrease num_adds if they can't all be fused
    طالما (عدد_adds > 1 && !ggml_can_fuse(cgraph, گرہ_idx, adds, عدد_adds)) {
        عدد_adds--;
    }

    // a single add is not "fused", so just return zero
    إذا (عدد_adds == 1) {
        أرجع 0;
    }
    أرجع عدد_adds;
}

ساكن int32_t لبھو_پہلا_سیٹ(uint32_t x) {
    int32_t ret = 0;
    إذا (!x) {
        أرجع -1;
    }
    طالما (!(x & 1)) {
        x >>= 1;
        ret++;
    }
    أرجع ret;
}

ساكن ggml_حالت ggml_backend_vk_گراف_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    VK_LOG_DEBUG("ggml_backend_vk_graph_compute(" << cgraph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ctx->device->diag_cgraph = خالی_سوچک;
    ctx->device->diag_prev_شروع = -1;
    ctx->device->diag_prev_سرا = -1;

    إذا (vk_instance.debug_utils_support) {
        ctx->device->debug_cmdbuf_idx = 0;
    }

    // queue scope, so it encloses every submit this evaluation makes.
    // closed when the function returns
    ggml_vk_debug_label قطار_dbg(ctx->device->compute_قطار->handle.get(), "ggml_backend_vk_graph_compute");

    ctx->prealloc_ناپ_جوڑو_rms_partials_وتھ = 0;
    ctx->do_جوڑو_rms_partials = خطأ;
    ctx->do_جوڑو_rms_partials_وتھ_calculation = خطأ;

    صحيح آخری_گرہ = cgraph->n_nodes - 1;

    // If the last op in the cgraph isn't backend GPU, the command buffer doesn't get closed properly
    طالما (آخری_گرہ > 0 && (ggml_vk_is_خالی_تھاواں(cgraph->nodes[آخری_گرہ]) || ((cgraph->nodes[آخری_گرہ]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0))) {
        آخری_گرہ -= 1;
    }

    // Reserve tensor context space for all nodes
    ctx->tensor_ctxs.resize(cgraph->n_nodes);

    منطقي پہلا_گرہ_وچ_batch = سچ; // true if next node will be first node in a batch
    صحيح submit_گرہ_idx = 0; // index to first node in a batch

    ggml_vk_submit_transfer_ctx(ctx);

    vk_context compute_ctx;
    إذا (vk_perf_logger_enabled) {
        // allocate/resize the query pool
        إذا (ctx->عدد_queries < cgraph->n_nodes + 1) {
            إذا (ctx->query_pool) {
                ctx->device->device.destroyQueryPool(ctx->query_pool);
            }
            vk::QueryPoolCreateInfo query_create_info;
            query_create_info.queryType = vk::QueryType::eTimestamp;
            query_create_info.query_گنتی = cgraph->n_nodes + 100;
            ctx->query_pool = ctx->device->device.createQueryPool(query_create_info);
            ctx->عدد_queries = query_create_info.query_گنتی;
            ctx->query_fusion_ناں.resize(ctx->عدد_queries);
            ctx->query_fusion_گرہ_گنتی.resize(ctx->عدد_queries);
            ctx->query_nodes.resize(ctx->عدد_queries);
            ctx->query_گرہ_idx.resize(ctx->عدد_queries);
        }

        ctx->device->device.resetQueryPool(ctx->query_pool, 0, cgraph->n_nodes+1);
        std::fill(ctx->query_fusion_ناں.begin(), ctx->query_fusion_ناں.end(), خالی_سوچک);
        std::fill(ctx->query_fusion_گرہ_گنتی.begin(), ctx->query_fusion_گرہ_گنتی.end(), 0);
        std::fill(ctx->query_nodes.begin(), ctx->query_nodes.end(), خالی_سوچک);
        std::fill(ctx->query_گرہ_idx.begin(), ctx->query_گرہ_idx.end(), 0);

        GGML_ASSERT(ctx->compute_ctx.expired());
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        ctx->query_idx = 0;
        compute_ctx->s->بفر->buf.writeTimestamp(vk::Pipeline_Stage_Flag_بٹ::eAllCommands, ctx->query_pool, ctx->query_idx++);
        ggml_vk_sync_buffers(ctx, compute_ctx);
    }

    ctx->prealloc_y_آخری_pipeline_ورتیا = خالی_سوچک;
    ctx->prealloc_y_آخری_tensor_ورتیا = خالی_سوچک;
    ctx->prealloc_y_آخری_k_padded = خطأ;

    إذا (ctx->prealloc_ناپ_جوڑو_rms_partials) {
        ggml_vk_preallocate_buffers(ctx, خالی_سوچک);
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        // initialize partial sums to zero.
        ggml_vk_بفر_memset_async(compute_ctx, ctx->prealloc_جوڑو_rms_partials, 0, 0, ctx->prealloc_ناپ_جوڑو_rms_partials);
        ggml_vk_sync_buffers(ctx, compute_ctx);
    }

    // Submit after enough work has accumulated, to overlap CPU cmdbuffer generation with GPU execution.
    // Estimate the amount of compute work using flops, and submit every 200 GFLOP
    // (and scaled down based on total graph flops, so smaller models submit earlier).
    // Also submit at least every 100 nodes, in case there are workloads without heavy compute.
    uint32_t submitted_nodes = 0;
    uint32_t submit_گنتی = 0;
    uint64_t batch_flops = 0;
    uint64_t کل_flops = 0;
    uint64_t flops_cap = 200'000'000'000ULL;

    // On weaker AMD GPUs larger submissions can hit a driver timeout, submit more often to avoid this
    إذا (ctx->device->vendor_id == VK_VENDOR_ID_AMD && ctx->device->shader_core_گنتی > 0) {
        إذا (ctx->device->architecture == AMD_GCN && ctx->device->shader_core_گنتی < 32) {
            flops_cap = 500'000'000ULL * ctx->device->shader_core_گنتی;
        } وإلا إذا (ctx->device->architecture != AMD_GCN && ctx->device->shader_core_گنتی < 24) {
            flops_cap = 2'000'000'000ULL * ctx->device->shader_core_گنتی;
        }
    }
    uint64_t flops_فی_submit = std::min(flops_cap, ctx->آخری_کل_flops / 40u);

    خودکار ثابت submit_after = [&](صحيح start, صحيح end) {
        إذا (ctx->device->serialize_submissions) {
            حاول {
                خودکار res = ctx->device->device.waitForFences({ ctx->fence }, سچ, UINT64_MAX);
                إذا (res != vk::نتیجہ_2::eSuccess) {
                    GGML_LOG_ERROR("ggml_vulkan: waitForFences error during serialized submission\n");
                    ارم vk::SystemError(vk::make_error_code(res), "ggml_vulkan: waitForFences during serialized submission");
                }
            } امسك (vk::Device_گواچا_Error &) {
                ggml_vk_print_device_fault_info(ctx->device);
                GGML_LOG_ERROR("ggml_vulkan: device lost on %s waiting for submission (nodes %d to %d):\n",
                        ctx->device->name.c_str(), start, end);
                ggml_vk_print_گرہ_list(cgraph, start, end);
                ارم;
            }
            ctx->device->device.resetFences({ ctx->fence });
            ctx->submit_pending = خطأ;
            ctx->device->diag_cgraph = cgraph;
            ctx->device->diag_prev_شروع = start;
            ctx->device->diag_prev_سرا = end;
        }
        پہلا_گرہ_وچ_batch = سچ;
        submitted_nodes = 0;
        batch_flops = 0;
        إذا (submit_گنتی < 3) {
            flops_فی_submit *= 2;
        }
        submit_گنتی++;
    };

    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        إذا (پہلا_گرہ_وچ_batch) {
            submit_گرہ_idx = i;
        }

        {
            خودکار گرہ_flops = ggml_vk_get_گرہ_flops(cgraph->nodes[i]);
            کل_flops += گرہ_flops;

            // Flush the current batch before recording a node that would push it over the flop threshold
            إذا (flops_فی_submit != 0 && submitted_nodes > 0 && batch_flops + گرہ_flops >= flops_فی_submit) {
                vk_context flush_ctx = ggml_vk_get_compute_ctx(ctx);
                ggml_vk_ctx_سرا(flush_ctx);
                flush_ctx->exit_tensor_idx = -1;
                ctx->compute_ctx.reset();
                ggml_vk_compute_forward(ctx, cgraph, cgraph->nodes[submit_گرہ_idx], submit_گرہ_idx, خطأ);
                submit_after(submit_گرہ_idx, i - 1);
                submit_گرہ_idx = i;
            }

            batch_flops += گرہ_flops;
        }

        // op_srcs_fused_elementwise indicates whether an op's srcs all contribute to
        // the fused result in an elementwise-way. This affects whether the memory for
        // the src is allowed to overlap the memory for the destination.
        // The array is sized to handle the largest fusion (asserted later).
        منطقي op_srcs_fused_elementwise[13];

        ctx->fused_topk_moe_منوال = TOPK_MOE_گنتی;
        ctx->fused_topk_moe_scale = خطأ;
        ctx->fused_topk_qsa = خطأ;
        ctx->fused_hc_post_gate = خطأ;
        ctx->fused_rms_norm_منوال = RMS_NORM_گنتی;
        ثابت حرف *fusion_string {};
        إذا (!ctx->device->disable_fusion) {
            uint32_t عدد_adds = ggml_vk_fuse_multi_جوڑو(ctx, cgraph, i);
            إذا (عدد_adds) {
                ctx->عدد_additional_fused_ops = عدد_adds - 1;
                fusion_string = "MULTI_ADD";
                std::fill_n(op_srcs_fused_elementwise, ctx->عدد_additional_fused_ops + 1, سچ);
            } وإلا إذا (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_جوڑو, GGML_OP_جوڑو })) {
                ctx->عدد_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ADD_ADD";
                op_srcs_fused_elementwise[0] = خطأ;
                op_srcs_fused_elementwise[1] = سچ;
                op_srcs_fused_elementwise[2] = سچ;
            } وإلا إذا (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_جوڑو })) {
                ctx->عدد_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ADD";
                op_srcs_fused_elementwise[0] = خطأ;
                op_srcs_fused_elementwise[1] = سچ;
            } وإلا إذا (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_جوڑو_ID, GGML_OP_MUL })) {
                ctx->عدد_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ID_ADD_ID_MUL";
                op_srcs_fused_elementwise[0] = خطأ;
                op_srcs_fused_elementwise[1] = سچ;
                op_srcs_fused_elementwise[2] = سچ;
            } وإلا إذا (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_جوڑو_ID })) {
                ctx->عدد_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_ADD_ID";
                op_srcs_fused_elementwise[0] = خطأ;
                op_srcs_fused_elementwise[1] = سچ;
            } وإلا إذا (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_MUL })) {
                ctx->عدد_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_MUL";
                op_srcs_fused_elementwise[0] = خطأ;
                op_srcs_fused_elementwise[1] = سچ;
            } وإلا إذا (ggml_can_fuse_subgraph(cgraph, i, rms_norm_mul_rope_view_سیٹ_صفاں_نمونہ, { i + 4 }) &&
                       ggml_check_کنارے(cgraph, i, rms_norm_mul_rope_view_سیٹ_صفاں_کنارے) &&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i) &&
                       ggml_vk_can_fuse_rope_سیٹ_صفاں(ctx, cgraph, i + 2)) {
                ctx->عدد_additional_fused_ops = 4;
                ctx->fused_rms_norm_منوال = RMS_NORM_MUL_ROPE_VIEW_سیٹ_صفاں;
                fusion_string = "RMS_NORM_MUL_ROPE_VIEW_SET_ROWS";
                op_srcs_fused_elementwise[0] = خطأ;
                op_srcs_fused_elementwise[1] = خطأ;
                op_srcs_fused_elementwise[2] = خطأ;
                op_srcs_fused_elementwise[3] = خطأ;
                op_srcs_fused_elementwise[4] = خطأ;
            } وإلا إذا (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }) &&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i)) {
                ctx->عدد_additional_fused_ops = 2;
                ctx->fused_rms_norm_منوال = RMS_NORM_MUL_ROPE;
                fusion_string = "RMS_NORM_MUL_ROPE";
                // rope is approximately elementwise - whole rows are done by a single workgroup and it's row-wise
                op_srcs_fused_elementwise[0] = خطأ;
                op_srcs_fused_elementwise[1] = سچ;
                op_srcs_fused_elementwise[2] = سچ;
            } وإلا إذا (ggml_can_fuse_subgraph(cgraph, i, hc_post_gate_نمونہ, { i + 3 }) &&
                       ggml_check_کنارے(cgraph, i, hc_post_gate_کنارے) &&
                       ggml_vk_can_fuse_hc_post_gate(cgraph, i)) {
                ctx->عدد_additional_fused_ops = hc_post_gate_نمونہ.size() - 1;
                ctx->fused_hc_post_gate = سچ;
                fusion_string = "HC_POST_GATE";
                std::fill_n(op_srcs_fused_elementwise, ctx->عدد_additional_fused_ops + 1, خطأ);
            } وإلا إذا (ggml_vk_can_fuse(ctx, cgraph, i, rms_norm_mul_جوڑو_mul_نمونہ)) {
                ctx->عدد_additional_fused_ops = 3;
                ctx->fused_rms_norm_منوال = RMS_NORM_MUL_جوڑو_MUL;
                fusion_string = "RMS_NORM_MUL_ADD_MUL";
                std::fill_n(op_srcs_fused_elementwise, 4, سچ);
            } وإلا إذا (ggml_vk_can_fuse(ctx, cgraph, i, rms_norm_mul_جوڑو_نمونہ)) {
                ctx->عدد_additional_fused_ops = 2;
                ctx->fused_rms_norm_منوال = RMS_NORM_MUL_جوڑو;
                fusion_string = "RMS_NORM_MUL_ADD";
                std::fill_n(op_srcs_fused_elementwise, 3, سچ);
            } وإلا إذا (ggml_can_fuse_subgraph(cgraph, i, rms_norm_view_سیٹ_صفاں_نمونہ, { i + 2 }) &&
                       ggml_check_کنارے(cgraph, i, rms_norm_view_سیٹ_صفاں_کنارے) &&
                       ggml_vk_can_fuse_rms_norm_سیٹ_صفاں(ctx, cgraph, i)) {
                ctx->عدد_additional_fused_ops = 2;
                ctx->fused_rms_norm_منوال = RMS_NORM_VIEW_سیٹ_صفاں;
                fusion_string = "RMS_NORM_VIEW_SET_ROWS";
                std::fill_n(op_srcs_fused_elementwise, 3, خطأ);
            } وإلا إذا (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                ctx->عدد_additional_fused_ops = 1;
                ctx->fused_rms_norm_منوال = RMS_NORM_MUL;
                fusion_string = "RMS_NORM_MUL";
                // rms_norm is not elementwise, but whole rows must be consumed and the scale factor computed before
                // they are overwritten, and one workgroup per row. So close enough.
                op_srcs_fused_elementwise[0] = سچ;
                op_srcs_fused_elementwise[1] = سچ;
            } وإلا إذا (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL })) {
                ctx->عدد_additional_fused_ops = 1;
                بدل (ggml_get_unary_op(cgraph->nodes[i])) {
                    حالة GGML_UNARY_OP_GELU:     fusion_string = "GELU_MUL";     اكسر;
                    حالة GGML_UNARY_OP_SIGMOID:  fusion_string = "SIGMOID_MUL";  اكسر;
                    حالة GGML_UNARY_OP_SILU:     fusion_string = "SILU_MUL";     اكسر;
                    افتراضي:                     fusion_string = "SOFTPLUS_MUL"; اكسر;
                }
                op_srcs_fused_elementwise[0] = سچ;
                op_srcs_fused_elementwise[1] = سچ;
            } وإلا إذا (ggml_vk_can_fuse_ssm_conv(ctx, cgraph, i, 2)) {
                ctx->عدد_additional_fused_ops = 2;
                fusion_string = "SSM_CONV_BIAS_SILU";
                // ssm_conv reads multiple input tokens per output, so it's not elementwise w.r.t. its srcs.
                // The downstream add and silu are elementwise on the conv output.
                op_srcs_fused_elementwise[0] = خطأ;
                op_srcs_fused_elementwise[1] = سچ;
                op_srcs_fused_elementwise[2] = سچ;
            } وإلا إذا (ggml_vk_can_fuse_ssm_conv(ctx, cgraph, i, 1)) {
                ctx->عدد_additional_fused_ops = 1;
                fusion_string = "SSM_CONV_SILU";
                op_srcs_fused_elementwise[0] = خطأ;
                op_srcs_fused_elementwise[1] = سچ;
            } وإلا إذا (ggml_can_fuse_subgraph(cgraph, i, rope_view_سیٹ_صفاں_نمونہ, { i + 2 }) &&
                       ggml_check_کنارے(cgraph, i, rope_view_سیٹ_صفاں_کنارے) &&
                       ggml_vk_can_fuse_rope_سیٹ_صفاں(ctx, cgraph, i)) {
                ctx->عدد_additional_fused_ops = 2;
                fusion_string = "ROPE_VIEW_SET_ROWS";
                op_srcs_fused_elementwise[0] = خطأ;
                op_srcs_fused_elementwise[1] = خطأ;
                op_srcs_fused_elementwise[2] = خطأ;
            } وإلا إذا (ggml_vk_can_fuse_snake(ctx, cgraph, i)) {
                ctx->عدد_additional_fused_ops = 4;
                fusion_string = "SNAKE";
                // elementwise=true: snake.comp is safe under exact aliasing because each
                // thread reads data_x[idx] into a register before writing data_d[idx]
                // with a data dependency on that register. The overlap check still
                // rejects partial overlaps (different base or size).
                std::fill_n(op_srcs_fused_elementwise, 5, سچ);
            } وإلا إذا (ggml_vk_can_fuse_topk_qsa(ctx, cgraph, i)) {
                ctx->عدد_additional_fused_ops = topk_qsa_نمونہ.size() - 1;
                ctx->fused_topk_qsa = سچ;
                fusion_string = "TOPK_QSA";
                std::fill_n(op_srcs_fused_elementwise, ctx->عدد_additional_fused_ops + 1, خطأ);
            } وإلا إذا (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax_norm, { i + 3, i + 9 }) &&
                       ggml_check_کنارے(cgraph, i, topk_moe_early_softmax_norm_کنارے) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX_NORM)) {
                ctx->عدد_additional_fused_ops = topk_moe_early_softmax_norm.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_نقاب |= 1 << 3;
                ctx->fused_topk_moe_منوال = TOPK_MOE_EARLY_SOFTMAX_NORM;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX_NORM";
                std::fill_n(op_srcs_fused_elementwise, ctx->عدد_additional_fused_ops + 1, خطأ);
            } وإلا إذا (ggml_can_fuse_subgraph(cgraph, i, topk_moe_sigmoid_norm_bias, { i + 4, i + 10 }) &&
                       ggml_check_کنارے(cgraph, i, topk_moe_sigmoid_norm_bias_کنارے) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_SIGMOID_NORM_BIAS)) {
                ctx->عدد_additional_fused_ops = topk_moe_sigmoid_norm_bias.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_نقاب |= 1 << 4;
                ctx->fused_topk_moe_منوال = TOPK_MOE_SIGMOID_NORM_BIAS;
                fusion_string = "TOPK_MOE_SIGMOID_NORM_BIAS";
                std::fill_n(op_srcs_fused_elementwise, ctx->عدد_additional_fused_ops + 1, خطأ);
            } وإلا إذا (ggml_can_fuse_subgraph(cgraph, i, topk_moe_sqrt_softplus_norm_bias, { i + 5, i + 11 }) &&
                       ggml_check_کنارے(cgraph, i, topk_moe_sqrt_softplus_norm_bias_کنارے) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS)) {
                ctx->عدد_additional_fused_ops = topk_moe_sqrt_softplus_norm_bias.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_نقاب |= 1 << 5;
                ctx->fused_topk_moe_منوال = TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS;
                fusion_string = "TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS";
                std::fill_n(op_srcs_fused_elementwise, ctx->عدد_additional_fused_ops + 1, خطأ);
            } وإلا إذا (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax, { i + 3, i + 4 }) &&
                       ggml_check_کنارے(cgraph, i, topk_moe_early_softmax_کنارے) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX)) {
                ctx->عدد_additional_fused_ops = topk_moe_early_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_نقاب |= 1 << 3;
                ctx->fused_topk_moe_منوال = TOPK_MOE_EARLY_SOFTMAX;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX";
                std::fill_n(op_srcs_fused_elementwise, ctx->عدد_additional_fused_ops + 1, خطأ);
            } وإلا إذا (ggml_can_fuse_subgraph(cgraph, i, topk_moe_late_softmax, { i + 1, i + 5 }) &&
                       ggml_check_کنارے(cgraph, i, topk_moe_late_softmax_کنارے) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_LATE_SOFTMAX)) {
                ctx->عدد_additional_fused_ops = topk_moe_late_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_نقاب |= 1 << 1;
                ctx->fused_topk_moe_منوال = TOPK_MOE_LATE_SOFTMAX;
                fusion_string = "TOPK_MOE_LATE_SOFTMAX";
                std::fill_n(op_srcs_fused_elementwise, ctx->عدد_additional_fused_ops + 1, خطأ);
            }
            إذا (ctx->fused_topk_moe_منوال != TOPK_MOE_گنتی) {
                // Look for an additional scale op to fuse - occurs in deepseek2 and nemotron3 nano.
                إذا (ggml_can_fuse_subgraph(cgraph, i + ctx->عدد_additional_fused_ops - 1, { GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE }, { i + ctx->عدد_additional_fused_ops + 1 }) ||
                    ggml_can_fuse_subgraph(cgraph, i + ctx->عدد_additional_fused_ops, { GGML_OP_GET_صفاں, GGML_OP_SCALE }, { i + ctx->عدد_additional_fused_ops + 1 })) {
                    ctx->fused_topk_moe_scale = سچ;
                    ctx->عدد_additional_fused_ops++;
                    op_srcs_fused_elementwise[ctx->عدد_additional_fused_ops] = خطأ;
                }
            }
        }
        GGML_ASSERT(ctx->عدد_additional_fused_ops < (صحيح)(حجم(op_srcs_fused_elementwise) / حجم(op_srcs_fused_elementwise[0])));
        ctx->fused_ops_write_نقاب |= 1 << ctx->عدد_additional_fused_ops;

        // Check whether fusion would overwrite src operands while they're still in use.
        // If so, disable fusion.
        إذا (ctx->عدد_additional_fused_ops) {
            // There are up to two output nodes - topk_moe has two.
            uint32_t بٹ = ctx->fused_ops_write_نقاب & ~(1 << ctx->عدد_additional_fused_ops);
            ggml_tensor *output_nodes[2] {};
            output_nodes[0] = cgraph->nodes[i + ctx->عدد_additional_fused_ops];
            إذا (بٹ) {
                صحيح output_idx = لبھو_پہلا_سیٹ(بٹ);
                GGML_ASSERT(بٹ == (1u << output_idx));
                output_nodes[1] = cgraph->nodes[i + output_idx];
            }

            منطقي need_disable = خطأ;

            لكل (صحيح j = 0; j < 2; ++j) {
                ggml_tensor *dst = output_nodes[j];
                إذا (!dst) {
                    استمر;
                }
                // Loop over all srcs of all nodes in the fusion. If the src overlaps
                // the destination and the src is not an intermediate node that's being
                // elided, then disable fusion.
                لكل (صحيح k = 0; k <= ctx->عدد_additional_fused_ops; ++k) {
                    لكل (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
                        ggml_tensor *src = cgraph->nodes[i + k]->src[s];
                        إذا (!src || src->op == GGML_OP_NONE) {
                            استمر;
                        }
                        إذا (ggml_vk_tensors_overlap(src, dst, op_srcs_fused_elementwise[k])) {
                            منطقي لبھیا = خطأ;
                            لكل (صحيح n = 0; n < k; ++n) {
                                إذا (cgraph->nodes[i + n] == src) {
                                    لبھیا = سچ;
                                    اكسر;
                                }
                            }
                            إذا (!لبھیا) {
                                need_disable = سچ;
                            }
                        }
                    }
                }
            }
            إذا (need_disable) {
                ctx->عدد_additional_fused_ops = 0;
                ctx->fused_ops_write_نقاب = 1;
                ctx->fused_topk_moe_منوال = TOPK_MOE_گنتی;
                ctx->fused_topk_moe_scale = خطأ;
                ctx->fused_topk_qsa = خطأ;
                ctx->fused_hc_post_gate = خطأ;
                ctx->fused_rms_norm_منوال = RMS_NORM_گنتی;
                fusion_string = خالی_سوچک;
            }
        }

        // Signal the almost_ready fence when the graph is mostly complete (< 20% remaining)
        منطقي almost_ready = (cgraph->n_nodes - i) < cgraph->n_nodes / 5;
        منطقي submit = (submitted_nodes >= ctx->device->max_nodes_فی_submit) ||
                      (flops_فی_submit != 0 && batch_flops >= flops_فی_submit) ||
                      (i + ctx->عدد_additional_fused_ops >= آخری_گرہ) ||
                      (almost_ready && !ctx->almost_ready_fence_pending);

        منطقي enqueued = ggml_vk_build_گراف(ctx, cgraph, i, cgraph->nodes[submit_گرہ_idx], submit_گرہ_idx, i + ctx->عدد_additional_fused_ops >= آخری_گرہ, almost_ready, submit);

        إذا (vk_perf_logger_enabled && enqueued) {
            compute_ctx = ggml_vk_get_compute_ctx(ctx);
            إذا (!vk_perf_logger_concurrent) {
                // track a single node/fusion for the current query
                ctx->query_nodes[ctx->query_idx] = cgraph->nodes[i];
                ctx->query_fusion_ناں[ctx->query_idx] = fusion_string;
                compute_ctx->s->بفر->buf.writeTimestamp(vk::Pipeline_Stage_Flag_بٹ::eAllCommands, ctx->query_pool, ctx->query_idx++);
                ggml_vk_sync_buffers(ctx, compute_ctx);
            } وإلا {
                // track a fusion string and number of fused ops for the current node_idx
                ctx->query_fusion_ناں[i] = fusion_string;
                ctx->query_fusion_گرہ_گنتی[i] = ctx->عدد_additional_fused_ops;
            }
        }

        إذا (enqueued) {
            ++submitted_nodes;

#جے_بنا_تعریف GGML_VULKAN_CHECK_RESULTS
            إذا (پہلا_گرہ_وچ_batch) {
                پہلا_گرہ_وچ_batch = خطأ;
            }
#جے_انت
        }

        إذا (submit && enqueued) {
            submit_after(submit_گرہ_idx, i + (صحيح)ctx->عدد_additional_fused_ops);
        }
        i += ctx->عدد_additional_fused_ops;
        ctx->عدد_additional_fused_ops = 0;
        ctx->fused_ops_write_نقاب = 0;
    }

    ctx->آخری_کل_flops = کل_flops;

    إذا (vk_perf_logger_enabled) {
        // End the command buffer and submit/wait
        GGML_ASSERT(!ctx->compute_ctx.expired());
        compute_ctx = ctx->compute_ctx.جندرا();
        ggml_vk_ctx_سرا(compute_ctx);

        ggml_vk_submit(compute_ctx, ctx->device->fence);
        VK_CHECK(ctx->device->device.waitForFences({ ctx->device->fence }, سچ, UINT64_MAX), "GGML_VULKAN_PERF waitForFences", ctx->device);
        ctx->device->device.resetFences({ ctx->device->fence });
        ctx->compute_ctx.reset();

        // Get the results and pass them to the logger
        std::vector<uint64_t> timestamps(cgraph->n_nodes + 1);
        VK_CHECK(ctx->device->device.getQueryPoolResults(ctx->query_pool, 0, ctx->query_idx, (cgraph->n_nodes + 1)*حجم(uint64_t), timestamps.انکڑے(), حجم(uint64_t), vk::Query_نتیجہ_Flag_بٹ::e64 | vk::Query_نتیجہ_Flag_بٹ::eWait), "get timestamp results", ctx->device);
        إذا (!vk_perf_logger_concurrent) {
            // Log each op separately
            لكل (صحيح i = 1; i < ctx->query_idx; i++) {
                خودکار گرہ = ctx->query_nodes[i];
                خودکار name = ctx->query_fusion_ناں[i];
                ctx->perf_logger->log_timing(گرہ, name, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod));
            }
        } وإلا {
            // Log each group of nodes
            صحيح prev_گرہ_idx = 0;
            لكل (صحيح i = 1; i < ctx->query_idx; i++) {
                خودکار cur_گرہ_idx = ctx->query_گرہ_idx[i];
                std::vector<ggml_tensor *> nodes;
                std::vector<ثابت حرف *> ناں;
                لكل (صحيح گرہ_idx = prev_گرہ_idx; گرہ_idx < cur_گرہ_idx; ++گرہ_idx) {
                    إذا (ggml_op_is_خالی_تھاواں(cgraph->nodes[گرہ_idx]->op)) {
                        استمر;
                    }
                    nodes.push_back(cgraph->nodes[گرہ_idx]);
                    ناں.push_back(ctx->query_fusion_ناں[گرہ_idx]);
                    گرہ_idx += ctx->query_fusion_گرہ_گنتی[گرہ_idx];
                }
                prev_گرہ_idx = cur_گرہ_idx;
                ctx->perf_logger->log_timing(nodes, ناں, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod));
            }
        }
        ctx->perf_logger->print_timings();
    }

    إذا (!ctx->device->support_async) {
        ggml_vk_synchronize(ctx);
    }

    أرجع GGML_حالت_SUCCESS;

    بن_ورتیا(backend);
}

فراغ ggml_vk_گراف_optimize(ggml_backend_t backend, بنية ggml_cgraph * graph, بنية ggml_backend_گراف_optimize_params * params)
{
    VK_LOG_DEBUG("ggml_vk_graph_optimize(" << graph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    إذا (ctx->device->disable_گراف_optimize) {
        أرجع;
    }

    خودکار ثابت &is_خالی_تھاواں = [](ثابت ggml_tensor * گرہ) -> منطقي {
        أرجع گرہ->op == GGML_OP_NONE || گرہ->op == GGML_OP_RESHAPE || گرہ->op == GGML_OP_TRANSPOSE || گرہ->op == GGML_OP_VIEW || گرہ->op == GGML_OP_ترتیب_بدلو;
    };

    خودکار ثابت &is_src_دا = [&is_خالی_تھاواں](ثابت ggml_tensor *dst, ثابت ggml_tensor *src) -> منطقي {
        خودکار ثابت &ادھار = [](ثابت ggml_tensor * tensor) {
            أرجع tensor->view_src ? tensor->view_src : tensor;
        };
        لكل (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            إذا (dst->src[s] == src) {
                أرجع سچ;
            }
            إذا (is_خالی_تھاواں(dst) || is_خالی_تھاواں(src)) {
                استمر;
            }
            // A source view of dst may read storage written through a different view by src.
            إذا (dst->src[s] && ادھار(dst->src[s]) == ادھار(src)) {
                أرجع سچ;
            }
            // Moving dst forward may overwrite storage still read through a view by src.
            إذا (src->src[s] && ادھار(dst) == ادھار(src->src[s])) {
                أرجع سچ;
            }
        }
        // implicit dependency if they view the same tensor
        إذا (ادھار(dst) == ادھار(src)) {
            أرجع سچ;
        }
        أرجع خطأ;
    };

    std::vector<ggml_tensor *> new_ترتیب;
    std::vector<منطقي> ورتیا(graph->n_nodes, خطأ);
    std::set<ggml_tensor *> ورتیا_گرہ_سیٹ;

    صحيح پہلا_بن_ورتیا = 0;

    // scheduled or zero-compute nodes in [lo, hi)
    خودکار ثابت &خالی_تھاواں_or_scheduled_between = [&](صحيح lo, صحيح hi) -> منطقي {
        لكل (صحيح v = lo; v < hi; ++v) {
            إذا (!ورتیا[v] && !is_خالی_تھاواں(graph->nodes[v])) {
                أرجع خطأ;
            }
        }
        أرجع سچ;
    };
    طالما (پہلا_بن_ورتیا < graph->n_nodes) {
        std::vector<صحيح> current_سیٹ;

        // Check for fusion patterns and avoid reordering them
        خودکار ثابت &میل_نمونہ = [&](ثابت std::initializer_list<ggml_op> &pattern, صحيح start) -> منطقي {
            إذا (start + (صحيح)pattern.size() <= graph->n_nodes) {
                منطقي is_نمونہ = سچ;
                لكل (size_t j = 0; j < pattern.size(); ++j) {
                    إذا (graph->nodes[start + j]->op != pattern.begin()[j] || ورتیا[start + j]) {
                        is_نمونہ = خطأ;
                    }
                }
                أرجع is_نمونہ;
            }
            أرجع خطأ;
        };

        خودکار ثابت &keep_نمونہ = [&](ثابت std::initializer_list<ggml_op> &pattern) -> منطقي {
            إذا (میل_نمونہ(pattern, پہلا_بن_ورتیا)) {
                لكل (size_t j = 0; j < pattern.size(); ++j) {
                    new_ترتیب.push_back(graph->nodes[پہلا_بن_ورتیا + j]);
                    ورتیا_گرہ_سیٹ.insert(graph->nodes[پہلا_بن_ورتیا + j]);
                    ورتیا[پہلا_بن_ورتیا + j] = سچ;
                }
                طالما (پہلا_بن_ورتیا < graph->n_nodes && ورتیا[پہلا_بن_ورتیا]) {
                    پہلا_بن_ورتیا++;
                }
                أرجع سچ;
            }
            أرجع خطأ;
        };

        خودکار ثابت &جوڑو_نمونہ_alloc_deps = [&](ثابت std::initializer_list<ggml_op> &pattern, صحيح آخری_گرہ) {
            // Keep external inputs alive through the fused output.
            std::set<ggml_tensor *> تکیا;
            لكل (size_t j = 0; j < pattern.size(); ++j) {
                ggml_tensor * گرہ = graph->nodes[پہلا_بن_ورتیا + j];
                لكل (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
                    ggml_tensor * src = گرہ->src[s];
                    إذا (src && تکیا.insert(src).second) {
                        params->جوڑو_alloc_dep(params->user_انکڑے, src, graph->nodes[آخری_گرہ]);
                    }
                }
                تکیا.insert(گرہ);
            }
        };

        خودکار ثابت &keep_topk_moe_نمونہ = [&](ثابت std::initializer_list<ggml_op> &pattern) -> منطقي {
            إذا (!میل_نمونہ(pattern, پہلا_بن_ورتیا)) {
                أرجع خطأ;
            }

            صحيح آخری_گرہ = پہلا_بن_ورتیا + (صحيح) pattern.size() - 1;
            // Some TOPK_MOE variants fuse a trailing scale.
            إذا (آخری_گرہ + 1 < graph->n_nodes && graph->nodes[آخری_گرہ + 1]->op == GGML_OP_SCALE) {
                آخری_گرہ++;
            }

            جوڑو_نمونہ_alloc_deps(pattern, آخری_گرہ);

            أرجع keep_نمونہ(pattern);
        };

        إذا (keep_topk_moe_نمونہ(topk_moe_early_softmax_norm)) {
            استمر;
        }
        إذا (keep_topk_moe_نمونہ(topk_moe_sigmoid_norm_bias)) {
            استمر;
        }
        إذا (keep_topk_moe_نمونہ(topk_moe_sqrt_softplus_norm_bias)) {
            استمر;
        }
        إذا (keep_topk_moe_نمونہ(topk_moe_early_softmax)) {
            استمر;
        }
        إذا (keep_topk_moe_نمونہ(topk_moe_late_softmax)) {
            استمر;
        }
        إذا (keep_نمونہ(snake_نمونہ)) {
            استمر;
        }
        إذا (keep_نمونہ(topk_qsa_نمونہ)) {
            استمر;
        }

        إذا (keep_نمونہ(rms_norm_mul_جوڑو_mul_نمونہ)) {
            استمر;
        }
        إذا (keep_نمونہ(rms_norm_mul_جوڑو_نمونہ)) {
            استمر;
        }
        إذا (keep_نمونہ(rms_norm_mul_rope_view_سیٹ_صفاں_نمونہ)) {
            استمر;
        }
        إذا (keep_نمونہ(rms_norm_view_سیٹ_صفاں_نمونہ)) {
            استمر;
        }
        إذا (keep_نمونہ(rope_view_سیٹ_صفاں_نمونہ)) {
            استمر;
        }
        إذا (میل_نمونہ(hc_post_gate_نمونہ, پہلا_بن_ورتیا)) {
            جوڑو_نمونہ_alloc_deps(hc_post_gate_نمونہ, پہلا_بن_ورتیا + (صحيح) hc_post_gate_نمونہ.size() - 1);
            keep_نمونہ(hc_post_gate_نمونہ);
            استمر;
        }

        // First, grab the next unused node.
        current_سیٹ.push_back(پہلا_بن_ورتیا);

        // Loop through the next N nodes. Grab any that don't depend on other nodes that
        // haven't already been run. Nodes that have already been run have used[i] set
        // to true. Allow nodes that depend on the previous node if it's a fusion pattern
        // that we support (e.g. RMS_NORM + MUL).
        // This first pass only grabs "real" (non-view nodes). Second pass grabs view nodes.
        // The goal is to not interleave real and view nodes in a way that breaks fusion.
        ثابت صحيح عدد_منزل_CHECK = 20;
        لكل (صحيح j = پہلا_بن_ورتیا+1; j < std::min(پہلا_بن_ورتیا + عدد_منزل_CHECK, graph->n_nodes); ++j) {
            إذا (ورتیا[j]) {
                استمر;
            }
            إذا (is_خالی_تھاواں(graph->nodes[j])) {
                استمر;
            }
            // Protect every interior QSA node (not just the start): the mask branch is
            // independent, so it gets pulled out and breaks keep_pattern otherwise.
            خودکار ثابت &وچ_qsa_نمونہ = [&](صحيح n) -> منطقي {
                لكل (صحيح o = 0; o < (صحيح) topk_qsa_نمونہ.size(); ++o) {
                    إذا (n - o >= 0 && میل_نمونہ(topk_qsa_نمونہ, n - o)) {
                        أرجع سچ;
                    }
                }
                أرجع خطأ;
            };
            إذا (میل_نمونہ(topk_moe_early_softmax_norm, j) ||
                میل_نمونہ(topk_moe_sigmoid_norm_bias, j) ||
                میل_نمونہ(topk_moe_sqrt_softplus_norm_bias, j) ||
                میل_نمونہ(topk_moe_early_softmax, j) ||
                میل_نمونہ(topk_moe_late_softmax, j) ||
                میل_نمونہ(snake_نمونہ, j) ||
                وچ_qsa_نمونہ(j) ||
                میل_نمونہ(rms_norm_mul_جوڑو_mul_نمونہ, j) ||
                میل_نمونہ(rms_norm_mul_جوڑو_نمونہ, j) ||
                میل_نمونہ(rms_norm_mul_rope_view_سیٹ_صفاں_نمونہ, j) ||
                میل_نمونہ(rms_norm_view_سیٹ_صفاں_نمونہ, j) ||
                میل_نمونہ(rope_view_سیٹ_صفاں_نمونہ, j) ||
                میل_نمونہ(hc_post_gate_نمونہ, j)) {
                استمر;
            }
            منطقي ok = سچ;
            لكل (صحيح c = پہلا_بن_ورتیا; c < j; ++c) {
                إذا (!ورتیا[c] &&
                    is_src_دا(graph->nodes[j], graph->nodes[c]) &&
                    !(c == current_سیٹ.back() && graph->nodes[c]->op == GGML_OP_RMS_NORM && graph->nodes[j]->op == GGML_OP_MUL && خالی_تھاواں_or_scheduled_between(c+1, j)) &&
                    !(c == current_سیٹ.back() && graph->nodes[c]->op == GGML_OP_UNARY && graph->nodes[j]->op == GGML_OP_MUL && خالی_تھاواں_or_scheduled_between(c+1, j)) &&
                    !(j == c+1 && c == current_سیٹ.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT && graph->nodes[j]->op == GGML_OP_جوڑو) &&
                    !(j == c+1 && c == current_سیٹ.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_جوڑو_ID) &&
                    !(j == c+1 && c == current_سیٹ.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_MUL) &&
                    !(j == c+1 && c == current_سیٹ.back() && graph->nodes[c]->op == GGML_OP_جوڑو && graph->nodes[j]->op == GGML_OP_جوڑو) &&
                    !(j == c+1 && c == current_سیٹ.back() && graph->nodes[c]->op == GGML_OP_SSM_CONV && graph->nodes[j]->op == GGML_OP_جوڑو) &&
                    !(j == c+1 && c == current_سیٹ.back() && graph->nodes[c]->op == GGML_OP_SSM_CONV && graph->nodes[j]->op == GGML_OP_UNARY)) {
                    ok = خطأ;
                    اكسر;
                }
            }
            إذا (ok) {
                current_سیٹ.push_back(j);

                صحيح rope_idx = j;

                // When we've found RMS_NORM + MUL, try to find a ROPE that uses it
                إذا (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_MUL &&
                    graph->nodes[j-1]->op == GGML_OP_RMS_NORM) {
                    لكل (صحيح k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        إذا (graph->nodes[k]->op == GGML_OP_ROPE &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // Check that other srcs are already valid
                            graph->nodes[k]->src[1]->op == GGML_OP_NONE &&
                            (graph->nodes[k]->src[2] == خالی_سوچک || graph->nodes[k]->src[2]->op == GGML_OP_NONE)) {
                            rope_idx = k;
                            current_سیٹ.push_back(rope_idx);
                            ورتیا[rope_idx] = سچ;
                            اكسر;
                        }
                    }
                }
                // Look for ROPE/RMS_NORM + VIEW + SET_ROWS and make them consecutive
                إذا (graph->nodes[rope_idx]->op == GGML_OP_ROPE || graph->nodes[rope_idx]->op == GGML_OP_RMS_NORM) {
                    صحيح view_idx = -1;
                    صحيح سیٹ_صفاں_idx = -1;
                    لكل (صحيح k = rope_idx + 1; k < std::min(rope_idx + 15, graph->n_nodes); ++k) {
                        إذا (ورتیا[k]) {
                            استمر;
                        }
                        إذا (view_idx == -1 && graph->nodes[k]->op == GGML_OP_VIEW && graph->nodes[k]->src[0] == graph->nodes[rope_idx]) {
                            view_idx = k;
                            استمر;
                        }
                        إذا (view_idx != -1 && graph->nodes[k]->op == GGML_OP_سیٹ_صفاں && graph->nodes[k]->src[0] == graph->nodes[view_idx]) {
                            سیٹ_صفاں_idx = k;
                            اكسر;
                        }
                    }
                    إذا (سیٹ_صفاں_idx != -1) {
                        ثابت صحيح گرہ_idxs[] = { rope_idx, view_idx, سیٹ_صفاں_idx };
                        ثابت ggml_op ops[] = { graph->nodes[rope_idx]->op, GGML_OP_VIEW, GGML_OP_سیٹ_صفاں };
                        منطقي can_pull = ggml_can_fuse_subgraph_ext(graph, گرہ_idxs, 3, ops, &سیٹ_صفاں_idx, 1);

                        لكل (صحيح c = rope_idx + 1; can_pull && c < سیٹ_صفاں_idx; ++c) {
                            إذا (!ورتیا[c] && c != view_idx && !is_خالی_تھاواں(graph->nodes[c]) &&
                                is_src_دا(graph->nodes[سیٹ_صفاں_idx], graph->nodes[c])) {
                                can_pull = خطأ;
                            }
                        }

                        إذا (can_pull) {
                            current_سیٹ.push_back(view_idx);
                            current_سیٹ.push_back(سیٹ_صفاں_idx);
                            ورتیا[view_idx] = سچ;
                            ورتیا[سیٹ_صفاں_idx] = سچ;
                        }
                    }
                }
                // Look for MUL_MAT_ID + ADD_ID + MUL
                إذا (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_جوڑو_ID &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT_ID) {
                    لكل (صحيح k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        إذا (graph->nodes[k]->op == GGML_OP_MUL &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || ورتیا_گرہ_سیٹ.find(graph->nodes[k]->src[1]) != ورتیا_گرہ_سیٹ.end())) {
                            current_سیٹ.push_back(k);
                            ورتیا[k] = سچ;
                            اكسر;
                        }
                    }
                }
                // Look for MUL_MAT + ADD + ADD
                إذا (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_جوڑو &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT) {
                    لكل (صحيح k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        إذا (graph->nodes[k]->op == GGML_OP_جوڑو &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || ورتیا_گرہ_سیٹ.find(graph->nodes[k]->src[1]) != ورتیا_گرہ_سیٹ.end())) {
                            current_سیٹ.push_back(k);
                            ورتیا[k] = سچ;
                            اكسر;
                        }
                    }
                }
                // SSM_CONV + ADD + UNARY: pull the consuming UNARY forward
                إذا (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_جوڑو &&
                    graph->nodes[j-1]->op == GGML_OP_SSM_CONV) {
                    لكل (صحيح k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        إذا (graph->nodes[k]->op == GGML_OP_UNARY &&
                            graph->nodes[k]->src[0] == graph->nodes[j]) {
                            current_سیٹ.push_back(k);
                            ورتیا[k] = سچ;
                            اكسر;
                        }
                    }
                }
                // UNARY + MUL: pull the consuming MUL forward
                إذا (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_UNARY) {
                    لكل (صحيح k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ggml_tensor * mul = graph->nodes[k];
                        إذا (mul->op != GGML_OP_MUL || (mul->src[0] != graph->nodes[j] && mul->src[1] != graph->nodes[j])) {
                            استمر;
                        }
                        ggml_tensor * other = (mul->src[0] == graph->nodes[j]) ? mul->src[1] : mul->src[0];
                        // the other src must either be weights or already processed
                        إذا (!(other->op == GGML_OP_NONE || ورتیا_گرہ_سیٹ.find(other) != ورتیا_گرہ_سیٹ.end())) {
                            استمر;
                        }
                        إذا (!ggml_vk_can_fuse_unary_mul(graph, j, k)) {
                            استمر;
                        }
                        current_سیٹ.push_back(k);
                        ورتیا[k] = سچ;
                        اكسر;
                    }
                }
            }
        }
        // Second pass grabs view nodes.
        // Skip this if it would break a fusion optimization (don't split up add->rms_norm or add->add).
        إذا (graph->nodes[current_سیٹ.back()]->op != GGML_OP_جوڑو) {
            لكل (صحيح j = پہلا_بن_ورتیا+1; j < std::min(پہلا_بن_ورتیا + عدد_منزل_CHECK, graph->n_nodes); ++j) {
                إذا (ورتیا[j]) {
                    استمر;
                }
                إذا (!is_خالی_تھاواں(graph->nodes[j])) {
                    استمر;
                }
                منطقي ok = سچ;
                لكل (صحيح c = پہلا_بن_ورتیا; c < j; ++c) {
                    منطقي c_وچ_current_سیٹ = std::find(current_سیٹ.begin(), current_سیٹ.end(), c) != current_سیٹ.end();
                    // skip views whose srcs haven't been processed.
                    إذا (!ورتیا[c] &&
                        is_src_دا(graph->nodes[j], graph->nodes[c]) &&
                        !c_وچ_current_سیٹ) {
                        ok = خطأ;
                        اكسر;
                    }
                }
                إذا (ok) {
                    current_سیٹ.push_back(j);
                }
            }
        }

        // Push the current set into new_order
        لكل (خودکار c : current_سیٹ) {
            new_ترتیب.push_back(graph->nodes[c]);
            ورتیا_گرہ_سیٹ.insert(graph->nodes[c]);
            ورتیا[c] = سچ;
        }
        طالما (پہلا_بن_ورتیا < graph->n_nodes && ورتیا[پہلا_بن_ورتیا]) {
            پہلا_بن_ورتیا++;
        }
    }
    // Replace the graph with the new order.
    لكل (صحيح i = 0; i < graph->n_nodes; ++i) {
        graph->nodes[i] = new_ترتیب[i];
    }
}

ساكن فراغ ggml_backend_vk_event_ریکارڈ(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_record(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    ggml_vk_submit_transfer_ctx(ctx);

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);
    خودکار* cmd_buf = compute_ctx->s->بفر; // retrieve pointer before it gets reset

    إذا (vkev->has_event) {
        // Move existing event into submitted
        vkev->events_submitted.push_back(vkev->event);
    }

    // Grab the next event and record it, create one if necessary
    إذا (vkev->events_free.خالی_تھاواں()) {
        vkev->event = ctx->device->device.createEvent({});
    } وإلا {
        vkev->event = vkev->events_free.back();
        vkev->events_free.pop_back();
    }

    vkev->has_event = سچ;

    ggml_vk_سیٹ_event(compute_ctx, vkev->event);

    vkev->tl_semaphore.value++;
    compute_ctx->s->signal_semaphores.push_back(vkev->tl_semaphore);
    ggml_vk_ctx_سرا(compute_ctx);

    ggml_vk_submit(compute_ctx, {});
    ctx->submit_pending = سچ;
    vkev->cmd_بفر = cmd_buf;
    vkev->cmd_بفر_use_counter = cmd_buf->use_counter;
    ctx->compute_ctx.reset();
}

ساكن فراغ ggml_backend_vk_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_wait(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    إذا (vkev->has_event) {
        // Wait for latest event
        ggml_vk_wait_events(compute_ctx, { vkev->event });

        إذا (ctx->device->async_use_transfer_قطار) {
            vk_context transfer_ctx = ggml_vk_get_transfer_ctx(ctx);
            transfer_ctx->s->wait_semaphores.push_back(vkev->tl_semaphore);
        }
    }
}

ساكن ggml_backend_i ggml_backend_vk_interface = {
    /* .get_name                = */ ggml_backend_vk_name,
    /* .free                    = */ ggml_backend_vk_free,
    /* .set_tensor_async        = */ ggml_backend_vk_سیٹ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_vk_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_vk_سیٹ_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_vk_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_vk_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_vk_synchronize,
    /* .graph_plan_create       = */ عدم,
    /* .graph_plan_free         = */ عدم,
    /* .graph_plan_update       = */ عدم,
    /* .graph_plan_compute      = */ عدم,
    /* .graph_compute           = */ ggml_backend_vk_گراف_compute,
    /* .event_record            = */ ggml_backend_vk_event_ریکارڈ,
    /* .event_wait              = */ ggml_backend_vk_event_wait,
    /* .graph_optimize          = */ ggml_vk_گراف_optimize,
};

ساكن ggml_guid_t ggml_backend_vk_guid() {
    ساكن ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x40, 0x3c, 0xe1, 0x02, 0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };
    أرجع &guid;
}

ggml_backend_t ggml_backend_vk_init(size_t dev_عدد) {
    VK_LOG_DEBUG("ggml_backend_vk_init(" << dev_عدد << ")");

    ggml_backend_vk_context * ctx = جديد ggml_backend_vk_context;
    ggml_vk_init(ctx, dev_عدد);

    ggml_backend_t vk_backend = جديد ggml_backend {
        /* .guid    = */ ggml_backend_vk_guid(),
        /* .iface   = */ ggml_backend_vk_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), dev_عدد),
        /* .context = */ ctx,
    };

    إذا (!ctx->device->support_async) {
        vk_backend->iface.get_tensor_async = خالی_سوچک;
    }

    أرجع vk_backend;
}

منطقي ggml_backend_is_vk(ggml_backend_t backend) {
    أرجع backend != عدم && ggml_guid_matches(backend->guid, ggml_backend_vk_guid());
}

صحيح ggml_backend_vk_get_device_گنتی() {
    أرجع ggml_vk_get_device_گنتی();
}

فراغ ggml_backend_vk_get_device_description(صحيح device, حرف * description, size_t description_ناپ) {
    GGML_ASSERT(device < (صحيح) vk_instance.device_indices.size());
    صحيح dev_idx = vk_instance.device_indices[device];
    ggml_vk_get_device_description(dev_idx, description, description_ناپ);
}

فراغ ggml_backend_vk_get_device_حافظہ(صحيح device, size_t * حرر, size_t * کل) {
    GGML_ASSERT(device < (صحيح) vk_instance.device_indices.size());
    GGML_ASSERT(device < (صحيح) vk_instance.device_supports_membudget.size());

    vk::PhysicalDevice vkdev = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device]];
    vk::Physical_Device_حافظہ_Budget_Properties_EXT budgetprops;
    vk::Physical_Device_حافظہ_Properties2 memprops = {};
    ثابت منطقي membudget_supported = vk_instance.device_supports_membudget[device];
    ثابت منطقي is_integrated_gpu = vkdev.getProperties().deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

    إذا (membudget_supported) {
        memprops.p_اگلا = &budgetprops;
    }
    vkdev.get_حافظہ_Properties2(&memprops);

    *کل = 0;
    *حرر = 0;

    لكل (uint32_t i = 0; i < memprops.حافظہ_Properties.حافظہ_ڈھیر_گنتی; ++i) {
        ثابت vk::حافظہ_ڈھیر & ڈھیر_2 = memprops.حافظہ_Properties.حافظہ_Heaps[i];

        إذا (is_integrated_gpu || (ڈھیر_2.flags & vk::حافظہ_ڈھیر_Flag_بٹ::eDeviceLocal)) {
            *کل += ڈھیر_2.size;

            إذا (membudget_supported && i < budgetprops.ڈھیر_Usage.size()) {
                // Driver may report heapBudget > heap.size so we add a safeguard here
                // See https://github.com/ggml-org/llama.cpp/issues/29277
                ثابت vk::Device_ناپ budget = std::min(budgetprops.ڈھیر_Budget[i], ڈھیر_2.size);
                *حرر += budget > budgetprops.ڈھیر_Usage[i] ? budget - budgetprops.ڈھیر_Usage[i] : 0;
            } وإلا {
                *حرر += ڈھیر_2.size;
            }
        }
    }
}

ساكن vk::PhysicalDeviceType ggml_backend_vk_get_device_type(صحيح device_idx) {
    GGML_ASSERT(device_idx >= 0 && device_idx < (صحيح) vk_instance.device_indices.size());

    vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

    vk::PhysicalDeviceProperties2 props = {};
    device.getProperties2(&props);

    أرجع props.properties.deviceType;
}

ساكن std::string ggml_backend_vk_get_device_pci_id(صحيح device_idx) {
    GGML_ASSERT(device_idx >= 0 && device_idx < (صحيح) vk_instance.device_indices.size());

    vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

    ثابت std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

    منطقي ext_support = خطأ;

    لكل (ثابت خودکار& properties : ext_props) {
        إذا (strcmp("VK_EXT_pci_bus_info", properties.extensionName) == 0) {
            ext_support = سچ;
            اكسر;
        }
    }

    إذا (!ext_support) {
        أرجع "";
    }

    vk::PhysicalDeviceProperties2 props = {};
    vk::PhysicalDevicePCIBusInfoPropertiesEXT pci_bus_info = {};

    props.p_اگلا = &pci_bus_info;

    device.getProperties2(&props);

    ثابت uint32_t pci_domain = pci_bus_info.pciDomain;
    ثابت uint32_t pci_bus = pci_bus_info.pciBus;
    ثابت uint32_t pci_device = pci_bus_info.pciDevice;
    ثابت uint8_t pci_function = (uint8_t) pci_bus_info.pciFunction; // pci function is between 0 and 7, prevent printf overflow warning

    حرف pci_bus_id[16] = {};
    snprintf(pci_bus_id, حجم(pci_bus_id), "%04x:%02x:%02x.%x", pci_domain, pci_bus, pci_device, pci_function);

    أرجع std::string(pci_bus_id);
}

ساكن ثابت حرف * ggml_backend_vk_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    أرجع ctx->name.c_str();
}

ساكن ثابت حرف * ggml_backend_vk_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    أرجع ctx->description.c_str();
}

ساكن فراغ ggml_backend_vk_device_get_حافظہ(ggml_backend_dev_t device, size_t * حرر, size_t * کل) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)device->context;
    ggml_backend_vk_get_device_حافظہ(ctx->device, حرر, کل);
}

ساكن ggml_backend_بفر_type_t ggml_backend_vk_device_get_بفر_type(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    أرجع ggml_backend_vk_بفر_type(ctx->device);
}

ساكن ggml_backend_بفر_type_t ggml_backend_vk_device_get_host_بفر_type(ggml_backend_dev_t dev) {
    بن_ورتیا(dev);
    أرجع ggml_backend_vk_host_بفر_type();
}

ساكن تعداد ggml_backend_dev_type ggml_backend_vk_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;

    أرجع ctx->is_integrated_gpu ? GGML_BACKEND_DEVICE_TYPE_IGPU : GGML_BACKEND_DEVICE_TYPE_GPU;
}

ساكن فراغ ggml_backend_vk_device_get_props(ggml_backend_dev_t dev, بنية ggml_backend_dev_props * props) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;

    props->name        = ggml_backend_vk_device_get_name(dev);
    props->description = ggml_backend_vk_device_get_description(dev);
    props->type        = ggml_backend_vk_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.خالی_تھاواں() ? خالی_سوچک : ctx->pci_bus_id.c_str();
    ggml_backend_vk_device_get_حافظہ(dev, &props->حافظہ_free, &props->حافظہ_کل);
    props->caps = {
        /* .async                 = */ سچ,
        /* .host_buffer           = */ سچ,
        /* .buffer_from_host_ptr  = */ خطأ,
        /* .events                = */ سچ,
        /* .mmap_support          = */ !ctx->is_integrated_gpu,
    };
}

ساكن ggml_backend_t ggml_backend_vk_device_init(ggml_backend_dev_t dev, ثابت حرف * params) {
    بن_ورتیا(params);
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    أرجع ggml_backend_vk_init(ctx->device);
}

ساكن منطقي ggml_backend_vk_device_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ثابت vk_device& device = ggml_vk_get_device(ctx->device);

    ثابت منطقي ورتدا_bda = (op->op == GGML_OP_IM2_خانہ || op->op == GGML_OP_IM2_خانہ_3_D) &&
                          device->shader_int64 && device->بفر_device_پتہ;

    خودکار ثابت & tensor_ناپ_supported = [&](size_t tensor_ناپ) {
        إذا (tensor_ناپ > device->max_بفر_ناپ) {
            أرجع خطأ;
        }
        // For im2col shaders using BDA, maxStorageBufferRange limit doesn't apply.
        // If shader64BitIndexing is enabled, maxStorageBufferRange limit doesn't apply.
        إذا (!ورتدا_bda && !device->shader_64b_indexing) {
            إذا (tensor_ناپ > device->properties.limits.max_Storage_بفر_Range) {
                أرجع خطأ;
            }
        }
        أرجع سچ;
    };
    // reject any tensors larger than the max buffer size
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        إذا (op->src[i] && !tensor_ناپ_supported(ggml_nbytes(op->src[i]))) {
            أرجع خطأ;
        }
    }
    إذا (!tensor_ناپ_supported(ggml_nbytes(op))) {
        أرجع خطأ;
    }

    بدل (op->op) {
        حالة GGML_OP_UNARY:
            بدل (ggml_get_unary_op(op)) {
                حالة GGML_UNARY_OP_EXP:
                حالة GGML_UNARY_OP_EXPM1:
                حالة GGML_UNARY_OP_ELU:
                حالة GGML_UNARY_OP_GELU:
                حالة GGML_UNARY_OP_GELU_ERF:
                حالة GGML_UNARY_OP_GELU_تیز:
                حالة GGML_UNARY_OP_SILU:
                حالة GGML_UNARY_OP_RELU:
                حالة GGML_UNARY_OP_XIELU:
                حالة GGML_UNARY_OP_NEG:
                حالة GGML_UNARY_OP_TANH:
                حالة GGML_UNARY_OP_SIGMOID:
                حالة GGML_UNARY_OP_HARDSIGMOID:
                حالة GGML_UNARY_OP_HARDSWISH:
                حالة GGML_UNARY_OP_ABS:
                حالة GGML_UNARY_OP_SOFTPLUS:
                حالة GGML_UNARY_OP_قدم:
                حالة GGML_UNARY_OP_گیڑا:
                حالة GGML_UNARY_OP_CEIL:
                حالة GGML_UNARY_OP_FLOOR:
                حالة GGML_UNARY_OP_TRUNC:
                حالة GGML_UNARY_OP_SGN:
                    أرجع (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type);
                افتراضي:
                    أرجع خطأ;
            }
        حالة GGML_OP_GLU:
            بدل (ggml_get_glu_op(op)) {
                حالة GGML_GLU_OP_GEGLU:
                حالة GGML_GLU_OP_REGLU:
                حالة GGML_GLU_OP_SWIGLU:
                حالة GGML_GLU_OP_SWIGLU_OAI:
                حالة GGML_GLU_OP_GEGLU_ERF:
                حالة GGML_GLU_OP_GEGLU_تیز:
                حالة GGML_GLU_OP_SWIGLU_CLAMP:
                    أرجع (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type) &&
                           (!op->src[1] || op->src[1]->type == op->src[0]->type);
                افتراضي:
                    أرجع خطأ;
            }
        حالة GGML_OP_MUL_MAT:
        حالة GGML_OP_MUL_MAT_ID:
            {
                ggml_type src0_type = op->src[0]->type;
                إذا (op->op == GGML_OP_MUL_MAT_ID) {
                    إذا (!ggml_vk_use_mul_mat_vec_id(op)) {
                        ثابت ggml_tensor * ids = op->src[2];
                        // Shaders store row IDs as uint16_t slot and token indices.
                        إذا (ids->ne[0] > int64_t(UINT16_MAX) + 1 || ids->ne[1] > int64_t(UINT16_MAX) + 1) {
                            أرجع خطأ;
                        }
                    }
                    إذا (!device->mul_mat_id_s[src0_type] && !device->mul_mat_id_m[src0_type] && !device->mul_mat_id_l[src0_type]) {
                        // If there's not enough shared memory for row_ids and the result tile, fallback to CPU
                        أرجع خطأ;
                    }
                    إذا (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                        أرجع خطأ;
                    }
                }
                بدل (src0_type) {
                    حالة GGML_TYPE_F32:
                    حالة GGML_TYPE_F16:
                    حالة GGML_TYPE_BF16:
                    حالة GGML_TYPE_Q1_0:
                    حالة GGML_TYPE_Q2_0:
                    حالة GGML_TYPE_Q4_0:
                    حالة GGML_TYPE_Q4_1:
                    حالة GGML_TYPE_Q5_0:
                    حالة GGML_TYPE_Q5_1:
                    حالة GGML_TYPE_Q8_0:
                    حالة GGML_TYPE_Q2_K:
                    حالة GGML_TYPE_Q3_K:
                    حالة GGML_TYPE_Q4_K:
                    حالة GGML_TYPE_Q5_K:
                    حالة GGML_TYPE_Q6_K:
                    حالة GGML_TYPE_IQ1_S:
                    حالة GGML_TYPE_IQ1_M:
                    حالة GGML_TYPE_IQ2_XXS:
                    حالة GGML_TYPE_IQ2_XS:
                    حالة GGML_TYPE_IQ2_S:
                    حالة GGML_TYPE_IQ3_XXS:
                    حالة GGML_TYPE_IQ3_S:
                    حالة GGML_TYPE_IQ4_XS:
                    حالة GGML_TYPE_IQ4_NL:
                    حالة GGML_TYPE_MXFP4:
                    حالة GGML_TYPE_NVFP4:
                    حالة GGML_TYPE_TQ1_0:
                    حالة GGML_TYPE_TQ2_0:
                        اكسر;
                    افتراضي:
                        أرجع خطأ;
                }
                بنية ggml_tensor * a;
                بنية ggml_tensor * b;
                إذا (op->op == GGML_OP_MUL_MAT) {
                    a = op->src[0];
                    b = op->src[1];
                } وإلا {
                    a = op->src[2];
                    b = op->src[1];
                }
                إذا (a->ne[3] != b->ne[3]) {
                    أرجع خطأ;
                }
                إذا (!(ggml_vk_dim01_contiguous(op->src[0]) || op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) ||
                    !(ggml_vk_dim01_contiguous(op->src[1]) || op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16)) {
                    أرجع خطأ;
                }
                إذا (op->src[0]->type == GGML_TYPE_BF16 && op->src[1]->type == GGML_TYPE_F16) {
                    // We currently don't have a bf16 x f16 shader, or an fp16->bf16 copy shader.
                    // So don't support this combination for now.
                    أرجع خطأ;
                }
                إذا (op->src[1]->type == GGML_TYPE_BF16 && op->src[0]->type != GGML_TYPE_BF16 && op->op != GGML_OP_MUL_MAT) {
                    // BF16 in src1 (widened to F32/F16) is only wired up for MUL_MAT
                    أرجع خطأ;
                }

                أرجع سچ;
            }
        حالة GGML_OP_FLASH_ATTN_EXT:
            {
                منطقي coopmat2 = device->coopmat2;
                uint32_t HSK = op->src[1]->ne[0];
                uint32_t HSV = op->src[2]->ne[0];
                إذا ((HSK % 8) != 0 || (HSV % 8) != 0) {
                    أرجع خطأ;
                }
                إذا (op->src[4] && op->src[4]->type != GGML_TYPE_F32) {
                    أرجع خطأ;
                }
                إذا (op->src[0]->type != GGML_TYPE_F32) {
                    أرجع خطأ;
                }
                إذا (op->type != GGML_TYPE_F32) {
                    أرجع خطأ;
                }
                إذا (op->src[3] && op->src[3]->type != GGML_TYPE_F16) {
                    أرجع خطأ;
                }
                خودکار fa_kv_ok = [](ggml_type t) {
                    بدل (t) {
                    حالة GGML_TYPE_F32:
                    حالة GGML_TYPE_F16:
                    حالة GGML_TYPE_BF16:
                    حالة GGML_TYPE_Q8_0:
                    حالة GGML_TYPE_Q5_1:
                    حالة GGML_TYPE_Q5_0:
                    حالة GGML_TYPE_Q4_1:
                    حالة GGML_TYPE_Q4_0:
                    حالة GGML_TYPE_IQ4_NL:
                        أرجع سچ;
                    افتراضي:
                        أرجع خطأ;
                    }
                };
                إذا (!fa_kv_ok(op->src[1]->type) || !fa_kv_ok(op->src[2]->type)) {
                    أرجع خطأ;
                }
                إذا ((op->src[1]->type == GGML_TYPE_BF16) != (op->src[2]->type == GGML_TYPE_BF16)) {
                    أرجع خطأ;
                }
                إذا (!coopmat2 && !(device->subgroup_shuffle && device->subgroup_vote)) {
                    // scalar/coopmat1 FA uses subgroupShuffle/subgroupAll
                    أرجع خطأ;
                }
                أرجع سچ;
            }
        حالة GGML_OP_GET_صفاں:
            {
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_F32:
                    حالة GGML_TYPE_F16:
                    حالة GGML_TYPE_BF16:
                    حالة GGML_TYPE_Q1_0:
                    حالة GGML_TYPE_Q2_0:
                    حالة GGML_TYPE_Q4_0:
                    حالة GGML_TYPE_Q4_1:
                    حالة GGML_TYPE_Q5_0:
                    حالة GGML_TYPE_Q5_1:
                    حالة GGML_TYPE_Q8_0:
                    حالة GGML_TYPE_Q2_K:
                    حالة GGML_TYPE_Q3_K:
                    حالة GGML_TYPE_Q4_K:
                    حالة GGML_TYPE_Q5_K:
                    حالة GGML_TYPE_Q6_K:
                    حالة GGML_TYPE_IQ1_S:
                    حالة GGML_TYPE_IQ1_M:
                    حالة GGML_TYPE_IQ2_XXS:
                    حالة GGML_TYPE_IQ2_XS:
                    حالة GGML_TYPE_IQ2_S:
                    حالة GGML_TYPE_IQ3_XXS:
                    حالة GGML_TYPE_IQ3_S:
                    حالة GGML_TYPE_IQ4_XS:
                    حالة GGML_TYPE_IQ4_NL:
                    حالة GGML_TYPE_MXFP4:
                    حالة GGML_TYPE_NVFP4:
                    حالة GGML_TYPE_TQ1_0:
                    حالة GGML_TYPE_TQ2_0:
                    حالة GGML_TYPE_I32:
                        أرجع سچ;
                    افتراضي:
                        أرجع خطأ;
                }
            }
        حالة GGML_OP_GET_صفاں_پٹھ:
            أرجع op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_سیٹ_صفاں:
            {
                إذا ((op->src[0]->type != GGML_TYPE_F32 && op->src[0]->type != GGML_TYPE_F16) ||
                    (op->src[1]->type != GGML_TYPE_I32 && op->src[1]->type != GGML_TYPE_I64)) {
                    أرجع خطأ;
                }
                بدل (op->type) {
                    حالة GGML_TYPE_F32:
                    حالة GGML_TYPE_F16:
                    حالة GGML_TYPE_BF16:
                    حالة GGML_TYPE_Q1_0:
                    حالة GGML_TYPE_Q2_0:
                    حالة GGML_TYPE_Q4_0:
                    حالة GGML_TYPE_Q4_1:
                    حالة GGML_TYPE_Q5_0:
                    حالة GGML_TYPE_Q5_1:
                    حالة GGML_TYPE_Q8_0:
                    حالة GGML_TYPE_IQ4_NL:
                        أرجع سچ;
                    افتراضي:
                        أرجع خطأ;
                }
            }
        حالة GGML_OP_CONT:
        حالة GGML_OP_CPY:
        حالة GGML_OP_DUP:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1] != خالی_سوچک ? op->src[1]->type : src0_type;

                إذا (src0_type == GGML_TYPE_F32) {
                    بدل (src1_type) {
                    حالة GGML_TYPE_F32:
                    حالة GGML_TYPE_F16:
                    حالة GGML_TYPE_BF16:
                    حالة GGML_TYPE_Q1_0:
                    حالة GGML_TYPE_Q2_0:
                    حالة GGML_TYPE_Q4_0:
                    حالة GGML_TYPE_Q4_1:
                    حالة GGML_TYPE_Q5_0:
                    حالة GGML_TYPE_Q5_1:
                    حالة GGML_TYPE_Q8_0:
                    حالة GGML_TYPE_IQ4_NL:
                        أرجع سچ;
                    افتراضي:
                        اكسر;
                    }
                }
                إذا (src1_type == GGML_TYPE_F32) {
                    بدل (src0_type) {
                    حالة GGML_TYPE_F16:
                    حالة GGML_TYPE_BF16:
                    حالة GGML_TYPE_Q1_0:
                    حالة GGML_TYPE_Q2_0:
                    حالة GGML_TYPE_Q4_0:
                    حالة GGML_TYPE_Q4_1:
                    حالة GGML_TYPE_Q5_0:
                    حالة GGML_TYPE_Q5_1:
                    حالة GGML_TYPE_Q8_0:
                    حالة GGML_TYPE_IQ4_NL:
                        أرجع سچ;
                    افتراضي:
                        اكسر;
                    }
                }

                إذا (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    أرجع سچ;
                }

                إذا (
                    (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) ||
                    (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32)
                ) {
                    أرجع سچ;
                }

                // We can handle copying from a type to the same type if it's
                // either not quantized or is quantized and contiguous.
                // We use f16 or f32 shaders to do the copy,
                // so the type/block size must be a multiple of 4.
                إذا (src0_type == src1_type &&
                    (!ggml_is_quantized(src0_type) || (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op))) &&
                    (ggml_type_ناپ(src0_type) % 2) == 0) {
                    أرجع سچ;
                }
                أرجع خطأ;
            }
        حالة GGML_OP_REPEAT:
            أرجع ggml_type_ناپ(op->type) == ggml_type_ناپ(op->src[0]->type) &&
                  (ggml_type_ناپ(op->type) == حجم(عائم) || ggml_type_ناپ(op->type) == 2);
        حالة GGML_OP_REPEAT_پٹھ:
            أرجع op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_ROPE:
            أرجع ggml_is_contiguous_صفاں(op) && ggml_is_contiguous_صفاں(op->src[0]);
        حالة GGML_OP_ROPE_پٹھ:
        حالة GGML_OP_NONE:
        حالة GGML_OP_RESHAPE:
        حالة GGML_OP_VIEW:
        حالة GGML_OP_ترتیب_بدلو:
        حالة GGML_OP_TRANSPOSE:
        حالة GGML_OP_RMS_NORM:
            أرجع سچ;
        حالة GGML_OP_گروہ_NORM:
            أرجع ggml_is_contiguous(op->src[0]);
        حالة GGML_OP_NORM:
        حالة GGML_OP_L2_NORM:
            أرجع op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        حالة GGML_OP_جوڑو:
        حالة GGML_OP_SUB:
        حالة GGML_OP_MUL:
        حالة GGML_OP_DIV:
            أرجع (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        حالة GGML_OP_جوڑو_ID:
            أرجع op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->src[2]->type == GGML_TYPE_I32 &&
                   op->type == GGML_TYPE_F32;
        حالة GGML_OP_SILU_پٹھ:
        حالة GGML_OP_RMS_NORM_پٹھ:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_SQR:
        حالة GGML_OP_SQRT:
        حالة GGML_OP_SIN:
        حالة GGML_OP_COS:
        حالة GGML_OP_CLAMP:
        حالة GGML_OP_LEAKY_RELU:
            أرجع (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->type == op->src[0]->type;
        حالة GGML_OP_OPT_قدم_ADAMW:
        حالة GGML_OP_OPT_قدم_SGD:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_باہرلا_PROD:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && op->type == GGML_TYPE_F32;
        حالة GGML_OP_LOG:
        حالة GGML_OP_TRI:
        حالة GGML_OP_DIAG:
            أرجع (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->type == op->src[0]->type;
        حالة GGML_OP_ARGSORT:
            {
                إذا (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
                    أرجع خطأ;
                }
                // pipeline_argsort_large_f32 requires vulkan memory model.
                إذا (device->vulkan_حافظہ_model) {
                    أرجع سچ;
                } وإلا {
                    أرجع op->ne[0] <= (1 << std::min(device->max_workgroup_ناپ_log2, عدد_argsort_pipelines - 1));
                }
            }
        حالة GGML_OP_چوٹی_K:
            {
                إذا (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
                    أرجع خطأ;
                }
                // large k falls back to radix-select
                ثابت uint32_t min_pipeline =
                    std::max((uint32_t) log2f(عائم(op->ne[0])) + 1, device->subgroup_ناپ_log2);
                إذا (min_pipeline < عدد_topk_pipelines && device->pipeline_topk_f32[min_pipeline]) {
                    أرجع سچ;
                }
                أرجع device->pipeline_topk_radix_f32 != خالی_سوچک;
            }
        حالة GGML_OP_UPSCALE:
            إذا (op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS) {
                إذا ((op->op_params[0] & 0xFF) != GGML_SCALE_منوال_BILINEAR) {
                    أرجع خطأ;
                }
            }
            أرجع op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_ACC:
            أرجع op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        حالة GGML_OP_سیٹ:
            أرجع op->src[0]->type == op->src[1]->type && op->src[0]->type == op->type &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_I32);
        حالة GGML_OP_CONCAT: {
            أرجع ggml_vk_concat_supported(op->src[0], op->src[1], op);
        }
        حالة GGML_OP_ADD1:
            أرجع (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16);
        حالة GGML_OP_ARANGE:
            أرجع op->type == GGML_TYPE_F32;
        حالة GGML_OP_FILL:
            أرجع op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
        حالة GGML_OP_SCALE:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_PAD:
        حالة GGML_OP_PAD_REFLECT_1D:
        حالة GGML_OP_ROLL:
            أرجع op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_DIAG_نقاب_INF:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_SOFT_MAX:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && (!op->src[1] || (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16));
        حالة GGML_OP_SOFT_MAX_پٹھ:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32;
        حالة GGML_OP_جوڑ:
            أرجع op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguously_allocated(op->src[0]);
        حالة GGML_OP_جوڑ_صفاں:
        حالة GGML_OP_اوسط:
            أرجع op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous_صفاں(op->src[0]);
        حالة GGML_OP_CUMSUM:
            {
                إذا (device->subgroup_arithmetic && device->subgroup_require_بھریا_support) {
                    أرجع op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous_صفاں(op->src[0]);
                }
                أرجع خطأ;
            }
        حالة GGML_OP_DSV4_HC_COMB:
        حالة GGML_OP_DSV4_HC_PRE:
        حالة GGML_OP_DSV4_HC_POST:
            {
                إذا (op->type != GGML_TYPE_F32) {
                    أرجع خطأ;
                }
                لكل (uint32_t i = 0; i < GGML_MAX_SRC; ++i) {
                    إذا (op->src[i] && op->src[i]->type != GGML_TYPE_F32) {
                        أرجع خطأ;
                    }
                }
                // hc is hardcoded to 4 in the shaders. ggml only constrains it
                // to 4 for COMB, so PRE/POST have to be checked here.
                إذا (op->op == GGML_OP_DSV4_HC_PRE && op->src[0]->ne[1] != 4) {
                    أرجع خطأ;
                }
                إذا (op->op == GGML_OP_DSV4_HC_POST && op->src[1]->ne[1] != 4) {
                    أرجع خطأ;
                }
                إذا (op->op == GGML_OP_DSV4_HC_COMB) {
                    أرجع device->pipeline_dsv4_hc_comb_f32 != خالی_سوچک;
                }
                أرجع سچ;
            }
        حالة GGML_OP_SOLVE_TRI:
            {
                إذا (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32) {
                    أرجع خطأ;
                }
                ثابت uint32_t N = op->src[0]->ne[0];
                ثابت uint32_t K = op->src[1]->ne[0];
                // K dimension limited to workgroup size
                إذا (K > 1u << device->max_workgroup_ناپ_log2) {
                    أرجع خطأ;
                }
                ثابت uint32_t batch_N = device->properties.limits.max_Compute_Shared_حافظہ_ناپ / ((N + K) * حجم(عائم));

                إذا (batch_N == 0) {
                    أرجع خطأ;
                }
                أرجع سچ;
            }
        حالة GGML_OP_ARGMAX:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_CROSS_ENTROPY_LOSS:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && ggml_are_same_shape(op->src[0], op->src[1])
                && ggml_is_contiguous(op) && ggml_is_scalar(op) && op->type == GGML_TYPE_F32;
        حالة GGML_OP_CROSS_ENTROPY_LOSS_پٹھ:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32 && ggml_is_scalar(op->src[0])
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[2]) && op->src[2]->type == GGML_TYPE_F32
                && ggml_are_same_shape(op->src[1], op->src[2])
                && ggml_are_same_shape(op->src[1], op)
                && ggml_is_contiguous(op) && op->type == GGML_TYPE_F32;
        حالة GGML_OP_گنتی_EQUAL:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_I32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_I32;
        حالة GGML_OP_IM2_خانہ:
            أرجع ggml_is_contiguous(op->src[1])
                && op->src[1]->type == GGML_TYPE_F32
                && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        حالة GGML_OP_IM2_خانہ_3_D:
            أرجع op->src[1]->type == GGML_TYPE_F32
                && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        حالة GGML_OP_TIMESTEP_EMBEDDING:
            أرجع op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_CONV_2D_DW:
            أرجع (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16)
                && op->src[1]->type == GGML_TYPE_F32;
        حالة GGML_OP_POOL_1D:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_POOL_2D:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_RWKV_WKV6:
        حالة GGML_OP_RWKV_WKV7:
            أرجع سچ; // all inputs are contiguous, see ggml.c
        حالة GGML_OP_GATED_خطی_ATTN:
            // the shader block size is hardcoded to head_size 64
            أرجع op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && op->src[0]->ne[0] == 64;
        حالة GGML_OP_LIGHTNING_INDEXER:
            {
                ثابت ggml_tensor * q = op->src[0];
                ثابت ggml_tensor * k = op->src[1];
                ثابت ggml_tensor * w = op->src[2];
                ثابت ggml_tensor * m = op->src[3];

                // the q/w/m types and the shape relationships between q, k, w, m and dst
                // are already asserted in ggml_lightning_indexer()
                إذا (!ggml_vk_lightning_indexer_k_type_supported(k->type) || !device->fp16) {
                    أرجع خطأ;
                }

                // the shader block size is hardcoded to head size 128
                إذا (q->ne[0] != 128) {
                    أرجع خطأ;
                }

                // the shader indexes the buffers by element stride, and is dispatched
                // without allow_misalign
                لكل (ثابت ggml_tensor * t : {q, k, w, m, op}) {
                    إذا (t->nb[0] != ggml_type_ناپ(t->type) ||
                        (vk_tensor_وتھ(t) + t->view_offs) % device->properties.limits.min_Storage_بفر_وتھ_Alignment != 0) {
                        أرجع خطأ;
                    }
                    // the strides get scaled down from bytes, so the division must be exact
                    لكل (صحيح i = 1; i < GGML_MAX_DIMS; ++i) {
                        إذا (t->nb[i] % ggml_type_ناپ(t->type) != 0) {
                            أرجع خطأ;
                        }
                    }
                }
                أرجع سچ;
            }
        حالة GGML_OP_GATED_DELTA_NET:
            {
                ثابت uint32_t S_v = op->src[2]->ne[0];
                إذا (S_v != 16 && S_v != 32 && S_v != 64 && S_v != 128) {
                    أرجع خطأ;
                }
                لكل (صحيح i = 0; i < 6; i++) {
                    إذا (op->src[i] == خالی_سوچک || op->src[i]->type != GGML_TYPE_F32) {
                        أرجع خطأ;
                    }
                }
                أرجع op->type == GGML_TYPE_F32;
            }
        حالة GGML_OP_SSM_جانچ:
            {
                لكل (صحيح i = 0; i < 6; i++) {
                    إذا (op->src[i] && ggml_is_quantized(op->src[i]->type)) {
                        أرجع خطأ;
                    }
                }
                إذا (op->src[6] && op->src[6]->type != GGML_TYPE_I32) {
                    أرجع خطأ;
                }
                إذا (op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    أرجع خطأ;
                }

                ثابت uint32_t d_state = op->src[0]->ne[0];
                ثابت uint32_t سر_dim = op->src[0]->ne[1];

                منطقي is_mamba2 = (op->src[3] && op->src[3]->nb[1] == حجم(عائم));
                إذا (!is_mamba2) {
                    أرجع خطأ;
                }

                إذا ((d_state != 128 && d_state != 256) || سر_dim % 16 != 0) {
                    أرجع خطأ;
                }

                size_t shmem_ناپ = d_state * حجم(عائم);

                إذا (shmem_ناپ > device->properties.limits.max_Compute_Shared_حافظہ_ناپ) {
                    أرجع خطأ;
                }

                إذا (!device->subgroup_basic) {
                    أرجع خطأ;
                }

                أرجع سچ;
            }
        حالة GGML_OP_SSM_CONV:
            أرجع op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_CONV_TRANSPOSE_1D:
            أرجع op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        حالة GGML_OP_COL2IM_1D:
            أرجع (op->src[0]->type == GGML_TYPE_F32 ||
                    op->src[0]->type == GGML_TYPE_F16 ||
                    op->src[0]->type == GGML_TYPE_BF16) &&
                   op->type == op->src[0]->type &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op);
        حالة GGML_OP_CONV_2D:
        حالة GGML_OP_CONV_TRANSPOSE_2D:
            {
                ثابت منطقي transpose = op->op == GGML_OP_CONV_TRANSPOSE_2D;
                ثابت int64_t باہر = !transpose ? op->src[0]->ne[3] : op->src[0]->ne[2];
                ثابت int64_t اندر  = !transpose ? op->src[0]->ne[2] : op->src[0]->ne[3];

                // Channel-contiguous format is not supported yet.
                أرجع ((op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                    (op->src[0]->nb[0] == حجم(عائم) || op->src[0]->nb[0] == حجم(ggml_fp16_t) ) &&
                    op->src[1]->type == GGML_TYPE_F32 &&
                    op->type == GGML_TYPE_F32 &&
                    باہر == op->ne[2] &&
                    اندر == op->src[1]->ne[2] &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op->src[1]) &&
                    ggml_is_contiguous(op));
            }
        حالة GGML_OP_CONV_3D:
            أرجع (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) &&
                ggml_is_contiguous(op);
        افتراضي:
            أرجع خطأ;
    }

    بن_ورتیا(dev);
}

ساكن منطقي ggml_backend_vk_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    إذا (buft->iface.get_name != ggml_backend_vk_بفر_type_name) {
        أرجع خطأ;
    }

    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ggml_backend_vk_بفر_type_context * buft_ctx = (ggml_backend_vk_بفر_type_context *)buft->context;

    أرجع buft_ctx->device->idx == ctx->device;
}

ساكن منطقي ggml_backend_vk_device_offload_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    ggml_backend_vk_device_context * dev_ctx = (ggml_backend_vk_device_context *)dev->context;

    أرجع ggml_vk_get_op_batch_ناپ(op) >= dev_ctx->op_offload_min_batch_ناپ;
}

ساكن ggml_backend_event_t ggml_backend_vk_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    خودکار device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = جديد vk_event;
    إذا (!vkev) {
        أرجع خالی_سوچک;
    }

    // No events initially, they get created on demand
    vkev->has_event = خطأ;

    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.سیٹ_PNext(&tci);
    vkev->tl_semaphore = { device->device.createSemaphore(ci), 0 };

    أرجع جديد ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ vkev,
    };
}

ساكن فراغ ggml_backend_vk_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    خودکار device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = (vk_event *)event->context;

    device->device.destroySemaphore(vkev->tl_semaphore.s);
    لكل (خودکار& event : vkev->events_free) {
        device->device.destroyEvent(event);
    }
    لكل (خودکار& event : vkev->events_submitted) {
        device->device.destroyEvent(event);
    }
    إذا (vkev->has_event) {
        device->device.destroyEvent(vkev->event);
    }
    مٹاؤ vkev;
    مٹاؤ event;
}

ساكن فراغ ggml_backend_vk_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_device_event_synchronize(backend=" << dev << ", event=" << event << ")");
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    خودکار device = ggml_vk_get_device(ctx->device);
    vk_event *vkev = (vk_event *)event->context;

    // Only do something if the event has actually been used
    إذا (vkev->has_event) {
        vk::Semaphore sem = vkev->tl_semaphore.s;
        uint64_t val = vkev->tl_semaphore.value;
        vk::SemaphoreWaitInfo swi{vk::SemaphoreWaitFlags{}, sem, val};
        VK_CHECK(device->device.waitSemaphores(swi, UINT64_MAX), "event_synchronize", device);

        // Reset and move submitted events
        لكل (خودکار& event : vkev->events_submitted) {
            device->device.resetEvent(event);
        }
        vkev->events_free.insert(vkev->events_free.end(), vkev->events_submitted.begin(), vkev->events_submitted.end());
        vkev->events_submitted.clear();

        // Finished using current command buffer so we flag for reuse
        إذا (vkev->cmd_بفر) {
            // Only flag for reuse if it hasn't been reused already
            إذا (vkev->cmd_بفر_use_counter == vkev->cmd_بفر->use_counter) {
                vkev->cmd_بفر->وچ_use = خطأ;
                vkev->cmd_بفر->buf.reset();
            }
            vkev->cmd_بفر = خالی_سوچک;
        }
    }
}

ساكن ggml_backend_بفر_t ggml_backend_vk_device_بفر_from_host_ptr(ggml_backend_dev_t dev, فراغ * ptr, size_t size, size_t max_tensor_ناپ) {
    VK_LOG_DEBUG("ggml_backend_vk_device_buffer_from_host_ptr(backend=" << dev << ", ptr=" << ptr << ", size=" << size << ")");
    GGML_بن_ورتیا(max_tensor_ناپ);

    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    خودکار device = ggml_vk_get_device(ctx->device);

    vk_بفر buf = ggml_vk_بفر_from_host_ptr(device, ptr, size);

    إذا (!buf) {
        أرجع {};
    }

    ggml_backend_vk_بفر_context * bufctx = جديد ggml_backend_vk_بفر_context(device, std::move(buf), device->name);

    ggml_backend_بفر_t ret = ggml_backend_بفر_init(ggml_backend_vk_device_get_بفر_type(dev), ggml_backend_vk_بفر_interface, bufctx, size);

    أرجع ret;
}

ساكن ثابت بنية ggml_backend_device_i ggml_backend_vk_device_i = {
    /* .get_name             = */ ggml_backend_vk_device_get_name,
    /* .get_description      = */ ggml_backend_vk_device_get_description,
    /* .get_memory           = */ ggml_backend_vk_device_get_حافظہ,
    /* .get_type             = */ ggml_backend_vk_device_get_type,
    /* .get_props            = */ ggml_backend_vk_device_get_props,
    /* .init_backend         = */ ggml_backend_vk_device_init,
    /* .get_buffer_type      = */ ggml_backend_vk_device_get_بفر_type,
    /* .get_host_buffer_type = */ ggml_backend_vk_device_get_host_بفر_type,
    /* .buffer_from_host_ptr = */ ggml_backend_vk_device_بفر_from_host_ptr,
    /* .supports_op          = */ ggml_backend_vk_device_supports_op,
    /* .supports_buft        = */ ggml_backend_vk_device_supports_buft,
    /* .offload_op           = */ ggml_backend_vk_device_offload_op,
    /* .event_new            = */ ggml_backend_vk_device_event_new,
    /* .event_free           = */ ggml_backend_vk_device_event_free,
    /* .event_synchronize    = */ ggml_backend_vk_device_event_synchronize,
};

ساكن ثابت حرف * ggml_backend_vk_reg_get_name(ggml_backend_reg_t reg) {
    بن_ورتیا(reg);
    أرجع GGML_VK_NAME;
}

ساكن size_t ggml_backend_vk_reg_get_device_گنتی(ggml_backend_reg_t reg) {
    بن_ورتیا(reg);
    أرجع ggml_backend_vk_get_device_گنتی();
}

ساكن ggml_backend_dev_t ggml_backend_vk_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    ساكن std::vector<ggml_backend_dev_t> devices;

    ساكن منطقي initialized = خطأ;

    {
        ساكن std::mutex mutex;
        std::lock_guard<std::mutex> جندرا(mutex);
        إذا (!initialized) {
            ثابت صحيح min_batch_ناپ = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;
            لكل (صحيح i = 0; i < ggml_backend_vk_get_device_گنتی(); i++) {
                ggml_backend_vk_device_context * ctx = جديد ggml_backend_vk_device_context;
                حرف desc[256];
                ggml_backend_vk_get_device_description(i, desc, حجم(desc));
                ctx->device = i;
                ctx->name = GGML_VK_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->is_integrated_gpu = ggml_backend_vk_get_device_type(i) == vk::PhysicalDeviceType::eIntegratedGpu;
                ctx->pci_bus_id = ggml_backend_vk_get_device_pci_id(i);
                ctx->op_offload_min_batch_ناپ = min_batch_ناپ;
                devices.push_back(جديد ggml_backend_device {
                    /* .iface   = */ ggml_backend_vk_device_i,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                });
            }
            initialized = سچ;
        }
    }

    GGML_ASSERT(device < devices.size());
    أرجع devices[device];
}

ساكن ثابت بنية ggml_backend_reg_i ggml_backend_vk_reg_i = {
    /* .get_name         = */ ggml_backend_vk_reg_get_name,
    /* .get_device_count = */ ggml_backend_vk_reg_get_device_گنتی,
    /* .get_device       = */ ggml_backend_vk_reg_get_device,
    /* .get_proc_address = */ عدم,
};

ggml_backend_reg_t ggml_backend_vk_reg() {
    ساكن ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_نسخہ,
        /* .iface       = */ ggml_backend_vk_reg_i,
        /* .context     = */ خالی_سوچک,
    };
    حاول {
        ggml_vk_instance_init();
        أرجع &reg;
    } امسك (ثابت vk::SystemError& e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: System error: " << e.what());
        أرجع خالی_سوچک;
    } امسك (ثابت std::exception &e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: " << e.what());
        أرجع خالی_سوچک;
    } امسك (...) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: unknown exception during Vulkan init");
        أرجع خالی_سوچک;
    }
}

منطقي ggml_vk_instance_layer_settings_دستیاب() {
#جے_تعریف_شدہ GGML_VULKAN_VALIDATE
    // Check if validation layer provides the extension
    ثابت std::string layer_name = "VK_LAYER_KHRONOS_validation";
    لكل (ثابت خودکار& layer : vk::enumerateInstanceLayerProperties()) {
        إذا (layer_name == layer.layerName.انکڑے()) {
            لكل (ثابت خودکار& ext : vk::enumerateInstanceExtensionProperties(layer_name)) {
                إذا (strcmp("VK_EXT_layer_settings", ext.extensionName.انکڑے()) == 0) {
                    أرجع سچ;
                }
            }
        }
    }

    std::cerr << "ggml_vulkan: WARNING: Validation layer or layer extension VK_EXT_layer_settings not found." << std::سطر_انت;
#جے_انت
    أرجع خطأ;
}

منطقي ggml_vk_instance_portability_enumeration_ext_دستیاب(ثابت std::vector<vk::ExtensionProperties>& instance_extensions) {
#جے_تعریف_شدہ __APPLE__
    // Check for portability enumeration extension for MoltenVK support
    لكل (ثابت خودکار& properties : instance_extensions) {
        إذا (strcmp("VK_KHR_portability_enumeration", properties.extensionName) == 0) {
            أرجع سچ;
        }
    }
    std::cerr << "ggml_vulkan: WARNING: Instance extension VK_KHR_portability_enumeration not found." << std::سطر_انت;
#جے_انت
    أرجع خطأ;

    بن_ورتیا(instance_extensions);
}

منطقي ggml_vk_instance_debug_utils_ext_دستیاب(
    ثابت std::vector<vk::ExtensionProperties> & instance_extensions) {
    // Check for portability enumeration extension for MoltenVK support
    لكل (ثابت خودکار & properties : instance_extensions) {
        إذا (strcmp("VK_EXT_debug_utils", properties.extensionName) == 0) {
            أرجع سچ;
        }
    }

    std::cerr << "ggml_vulkan: WARNING: Instance extension VK_EXT_debug_utils not found." << std::سطر_انت;
    أرجع خطأ;

    بن_ورتیا(instance_extensions);
}

منطقي ggml_vk_device_is_supported(ثابت vk::PhysicalDevice & vkdev) {
    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.p_اگلا = خالی_سوچک;
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.p_اگلا = &vk11_features;

    vkGetPhysicalDeviceFeatures2(vkdev, &device_features2);

    أرجع vk11_features.storage_Buffer16_ثنائی_ہندسہ_Access;
}

منطقي ggml_vk_khr_cooperative_میٹرکس_support(ثابت vk::PhysicalDeviceProperties& props, ثابت vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch) {
    بدل (props.vendorID) {
    حالة VK_VENDOR_ID_INTEL:
        // Only allowing Xe2/Xe3 GPU and integrated Xe GPUs at the moment since older hardware (ex. Arc A770) has performance regressions.
        أرجع (arch == vk_device_architecture::INTEL_XE2) ||
            (arch == vk_device_architecture::INTEL_XE1 && props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu && driver_props.driverID == vk::DriverId::eIntelProprietaryWindows);
    حالة VK_VENDOR_ID_AMD:
        إذا (driver_props.driverID == vk::DriverId::eAmdProprietary || driver_props.driverID == vk::DriverId::e_Amd_Open_سروت) {
            // Workaround for AMD proprietary driver reporting support on all GPUs
            أرجع arch == vk_device_architecture::AMD_RDNA3 || arch == vk_device_architecture::AMD_RDNA4;
        }
        أرجع سچ;
    حالة VK_VENDOR_ID_QUALCOMM:
        // Only allow Adreno GPUs with hardware matrix cores (Gen 6+).
        أرجع arch == vk_device_architecture::QUALCOMM_ADRENO;
    افتراضي:
        أرجع سچ;
    }
}

uint32_t ggml_vk_intel_shader_core_گنتی(ثابت vk::PhysicalDevice& vkdev) {
    VkPhysicalDeviceProperties2 props = vkdev.getProperties2();

    إذا (props.properties.vendorID != VK_VENDOR_ID_INTEL) {
        أرجع 0;
    }

    ثابت uint32_t device_id = props.properties.deviceID;

    بدل (device_id) {
    حالة 0x56A6:  // A310
        أرجع 6;
    حالة 0x5693:  // A370M
    حالة 0x56A5:  // A380
    حالة 0x56B1:  // Pro A40/A50
        أرجع 8;
    حالة 0x5697:  // A530M
        أرجع 12;
    حالة 0x5692:  // A550M
    حالة 0x56B3:  // Pro A60
        أرجع 16;
    حالة 0x56A2:  // A580
        أرجع 24;
    حالة 0x5691:  // A730M
    حالة 0x56A1:  // A750
        أرجع 28;
    حالة 0x56A0:  // A770
    حالة 0x5690:  // A770M
        أرجع 32;
    حالة 0xE212:  // Pro B50
        أرجع 16;
    حالة 0xE20C:  // B570
        أرجع 18;
    حالة 0xE20B:  // B580
    حالة 0xE211:  // Pro B60
    حالة 0xE222:  // Pro B65
        أرجع 20;
    حالة 0xE223:  // Pro B70
        أرجع 32;
    حالة 0xB080:  // PTL Xe3 LPG 2x6 (12 subslices)
        أرجع 12;
    افتراضي:
        أرجع 0;
    }
}

منطقي ggml_vk_intel_windows_driver_وچ_range(uint32_t driver_نسخہ, uint32_t ہیٹھلا_major, uint32_t ہیٹھلا_minor, uint32_t اتلا_major, uint32_t اتلا_minor) {
#إذا defined(_WIN32)
    // Intel Windows encodes xxx.yyyy as [31:14].[13:0].
    ثابت uint32_t major = driver_نسخہ >> 14;
    ثابت uint32_t minor = driver_نسخہ & 0x3fff;

    ثابت منطقي ge_ہیٹھلا = major > ہیٹھلا_major || (major == ہیٹھلا_major && minor >= ہیٹھلا_minor);
    ثابت منطقي lt_اتلا = major < اتلا_major || (major == اتلا_major && minor < اتلا_minor);

    أرجع ge_ہیٹھلا && lt_اتلا;
#وإلا
    GGML_بن_ورتیا(driver_نسخہ);
    GGML_بن_ورتیا(ہیٹھلا_major);
    GGML_بن_ورتیا(ہیٹھلا_minor);
    GGML_بن_ورتیا(اتلا_major);
    GGML_بن_ورتیا(اتلا_minor);
    أرجع سچ;
#جے_انت
}

GGML_BACKEND_DL_IMPL(ggml_backend_vk_reg)


// out-of-lined header method definitions

فراغ vk_قطار_handle_synchronized::submit(vk::ArrayProxy<ثابت vk::SubmitInfo> submits, vk::Fence fence) {
    // Workaround for NVIDIA driver bug
    std::unique_lock<std::mutex> device_guard;
    إذا (device_submit_mutex) {
        device_guard = std::unique_lock<std::mutex>(*device_submit_mutex);
    }
    std::lock_guard<std::mutex> guard(mutex);
    حاول {
        queue.submit(submits, fence);
    } امسك (vk::Device_گواچا_Error &) {
        إذا (خودکار dev = device.جندرا()) {
            ggml_vk_print_device_گواچا_info(dev);
        }
        ارم;
    }
}

فراغ vk_قطار_handle_unsynchronized::submit(vk::ArrayProxy<ثابت vk::SubmitInfo> submits, vk::Fence fence) {
    // Workaround for NVIDIA driver bug
    std::unique_lock<std::mutex> device_guard;
    إذا (device_submit_mutex) {
        device_guard = std::unique_lock<std::mutex>(*device_submit_mutex);
    }
    حاول {
        queue.submit(submits, fence);
    } امسك (vk::Device_گواچا_Error &) {
        إذا (خودکار dev = device.جندرا()) {
            ggml_vk_print_device_گواچا_info(dev);
        }
        ارم;
    }
}

vk_device_struct::~vk_device_struct() {
    // the build failed before the device was created, nothing to destroy
    إذا (device == VK_NULL_HANDLE) {
        أرجع;
    }

    VK_LOG_DEBUG("destroy device " << name);

    device.destroyFence(fence);

    ggml_vk_destroy_بفر(sync_staging);

    إذا (compute_قطار) compute_قطار->cmd_pool.destroy(device);
    إذا (transfer_قطار) transfer_قطار->cmd_pool.destroy(device);

    // Explicitly clear to ensure queues drop their shared_ptrs to handles
    // before the Vulkan logical device instance is destroyed
    compute_قطار.reset();
    transfer_قطار.reset();

    لكل (خودکار& pipeline : all_pipelines) {
        إذا (pipeline.expired()) {
            استمر;
        }

        vk_pipeline pl = pipeline.جندرا();
        ggml_vk_destroy_pipeline(device, pl);
    }
    all_pipelines.clear();

    device.destroy_Descriptor_سیٹ_Layout(dsl);

    device.destroy();
}

فراغ vk_perf_logger::print_timings(منطقي force) {
    إذا (timings.خالی_تھاواں()) {
        أرجع;
    }
    print_گنتی++;
    إذا ((print_گنتی % vk_perf_logger_frequency) != 0 && !force) {
        أرجع;
    }
    print_گنتی = 0;
    uint64_t کل_all_op_times = 0;
    std::cerr << "----------------\nVulkan Timings:" << std::سطر_انت;
    لكل (ثابت خودکار & t : timings) {
        uint64_t کل_op_times = 0;
        لكل (ثابت خودکار & ویلا : t.second) {
            کل_op_times += ویلا;
        }
        std::cerr << t.پہلا << ": " << t.second.size() << " x " << (کل_op_times / t.second.size() / 1000.0)
                  << " us = " << (کل_op_times / 1000.0) << " us";

        // If we have as many flops entries as timing entries for the op, then compute and log the flops/S.
        خودکار it = flops.find(t.پہلا);
        إذا (it != flops.end() && (it->second).size() == t.second.size()) {
            uint64_t کل_op_flops = 0;
            لكل (ثابت خودکار & elem : it->second) {
                کل_op_flops += elem;
            }
            std::cerr << " ("
                      << (مزدوج(کل_op_flops) / (1000.0 * 1000.0 * 1000.0)) /
                             (مزدوج(کل_op_times) / (1000.0 * 1000.0 * 1000.0))
                      << " GFLOPS/s)";
        }

        کل_all_op_times += کل_op_times;

        std::cerr << std::سطر_انت;
    }

    إذا (timings.size() > 0) {
        std::cerr << "Total time: " << کل_all_op_times / 1000.0 << " us." << std::سطر_انت;
    }

    timings.clear();
    flops.clear();
}

std::string vk_perf_logger::get_گرہ_fusion_name(ثابت ggml_tensor * گرہ, ثابت حرف *fusion_name, uint64_t *n_flops) {
    *n_flops = ggml_vk_get_گرہ_flops(گرہ);
    std::string fusion_str;
    إذا (fusion_name) {
        fusion_str = fusion_name + std::string(" ");
    }
    إذا (گرہ->op == GGML_OP_UNARY) {
        أرجع fusion_str + ggml_unary_op_name(ggml_get_unary_op(گرہ));
    }
    إذا (گرہ->op == GGML_OP_MUL_MAT || گرہ->op == GGML_OP_MUL_MAT_ID) {
        ثابت uint64_t m     = گرہ->ne[0];
        ثابت uint64_t n     = گرہ->ne[1];
        ثابت uint64_t k     = گرہ->src[1]->ne[0];
        ثابت uint64_t batch = گرہ->ne[2] * گرہ->ne[3];
        std::string    name  = ggml_op_name(گرہ->op);
        إذا ((گرہ->op == GGML_OP_MUL_MAT && n <= mul_mat_vec_max_خانے_گنتی) ||
            (گرہ->op == GGML_OP_MUL_MAT_ID && گرہ->src[2]->ne[1] == 1)) {
            name += "_VEC";
        }
        name += " ";
        name += ggml_type_name(گرہ->src[0]->type);
        name += " m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=" + std::to_string(k);
        إذا (گرہ->op == GGML_OP_MUL_MAT_ID) {
            name += " n_expert=" + std::to_string(گرہ->src[0]->ne[2]);
        }
        إذا (batch > 1) {
            name += " batch=" + std::to_string(batch);
        }
        أرجع fusion_str + name;
    }
    إذا (گرہ->op == GGML_OP_CONV_2D || گرہ->op == GGML_OP_CONV_TRANSPOSE_2D) {
        std::string   name    = ggml_op_name(گرہ->op);
        ثابت ggml_tensor * knl = گرہ->src[0];
        uint64_t      Cout    = گرہ->ne[2];
        uint64_t      ناپ_K  = گرہ->src[1]->ne[2] * knl->ne[0] * knl->ne[1];
        uint64_t      ناپ_N_2  = گرہ->ne[3] * گرہ->ne[0] * گرہ->ne[1];
        name += " M=Cout=" + std::to_string(Cout) + ", K=Cin*KW*KH=" + std::to_string(ناپ_K) +
                ", N=N*OW*OH=" + std::to_string(ناپ_N_2);
        أرجع fusion_str + name;
    }
    إذا (گرہ->op == GGML_OP_RMS_NORM) {
        std::string   name    = ggml_op_name(گرہ->op);
        name += "(" + std::to_string(گرہ->ne[0]) + "," + std::to_string(گرہ->ne[1]) + "," + std::to_string(گرہ->ne[2]) + "," + std::to_string(گرہ->ne[3]) + ")";
        أرجع fusion_str + name;
    }
    إذا (گرہ->op == GGML_OP_FLASH_ATTN_EXT) {
        ثابت ggml_tensor * dst = گرہ;
        ثابت ggml_tensor * q = گرہ->src[0];
        ثابت ggml_tensor * k = گرہ->src[1];
        ثابت ggml_tensor * v = گرہ->src[2];
        ثابت ggml_tensor * m = گرہ->src[3];
        std::stringstream name;
        name << fusion_str;
        name << ggml_op_name(گرہ->op) <<
            " dst(" << dst->ne[0] << "," << dst->ne[1] << "," << dst->ne[2] << "," << dst->ne[3] << "), " <<
            " q(" << q->ne[0] << "," << q->ne[1] << "," << q->ne[2] << "," << q->ne[3] << "), " <<
            " k(" << k->ne[0] << "," << k->ne[1] << "," << k->ne[2] << "," << k->ne[3] << "), " <<
            " v(" << v->ne[0] << "," << v->ne[1] << "," << v->ne[2] << "," << v->ne[3] << "), " <<
            " m(" << (m?m->ne[0]:0) << "," << (m?m->ne[1]:0) << "," << (m?m->ne[2]:0) << "," << (m?m->ne[3]:0) << ")";
        أرجع name.str();
    }
    إذا (گرہ->op == GGML_OP_چوٹی_K) {
        std::stringstream name;
        name << fusion_str;
        name << ggml_op_name(گرہ->op) <<
            " K=" << گرہ->ne[0] <<
            " (" << گرہ->src[0]->ne[0] << "," << گرہ->src[0]->ne[1] << "," << گرہ->src[0]->ne[2] << "," << گرہ->src[0]->ne[3] << ")";
        أرجع name.str();
    }
    أرجع fusion_str + ggml_op_name(گرہ->op);
}

ggml_backend_vk_بفر_context::~ggml_backend_vk_بفر_context() {
    ggml_vk_destroy_بفر(dev_بفر);
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_context & ctx, ثابت std::string & pipeline_name, uint32_t wg0, uint32_t wg1, uint32_t wg2) {
    إذا (!vk_instance.debug_utils_support || ctx->s == خالی_سوچک) {
        أرجع;
    }
    begin(ctx, pipeline_name + " (" + std::to_string(wg0) + "," + std::to_string(wg1) + "," + std::to_string(wg2) + ")");
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_context & ctx, ثابت ggml_cgraph * cgraph, صحيح گرہ_idx, صحيح n_fused) {
    إذا (!vk_instance.debug_utils_support || ctx->s == خالی_سوچک) {
        أرجع;
    }
    std::string name = ggml_op_name(cgraph->nodes[گرہ_idx]->op);
    لكل (صحيح i = 1; i <= n_fused; i++) {
        name += "+";
        name += ggml_op_name(cgraph->nodes[گرہ_idx + i]->op);
    }
    name += " ";
    name += cgraph->nodes[گرہ_idx]->name;
    begin(ctx, name);
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_قطار_handle * handle, ثابت حرف * name) {
    إذا (!vk_instance.debug_utils_support || handle == خالی_سوچک) {
        أرجع;
    }
    vk::DebugUtilsLabelEXT label = {};
    label.pLabelName = name;
    label.color = std::array<عائم, 4>{1.0f, 1.0f, 1.0f, 1.0f};

    qhandle = handle;
    std::lock_guard<vk_قطار_handle> guard(*qhandle);
    vk_instance.pfn_vk_قطار_Begin_Debug_Utils_Label_EXT(qhandle->queue, reinterpret_cast<VkDebugUtilsLabelEXT *>(&label));
}

فراغ ggml_vk_debug_label::close() {
    إذا (subctx != خالی_سوچک) {
        // close on the current command buffer, which may differ from the one begin used
        إذا (subctx->s != خالی_سوچک) {
            vk_instance.pfn_vk_Cmd_سرا_Debug_Utils_Label_EXT(subctx->s->بفر->buf);
        }
        subctx->debug_labels.pop_back();
        subctx = خالی_سوچک;
    }
    إذا (qhandle != خالی_سوچک) {
        std::lock_guard<vk_قطار_handle> guard(*qhandle);
        vk_instance.pfn_vk_قطار_سرا_Debug_Utils_Label_EXT(qhandle->queue);
        qhandle = خالی_سوچک;
    }
}

فراغ ggml_vk_debug_label::begin(vk_context & ctx, ثابت std::string & name) {
    إذا (!vk_instance.debug_utils_support || ctx->s == خالی_سوچک) {
        أرجع;
    }
    subctx = ctx.get();
    subctx->debug_labels.push_back(name);
    ggml_vk_cmd_label_begin(subctx->s->بفر->buf, subctx->debug_labels.back().c_str());
}


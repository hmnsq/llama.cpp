#अन्तर्भाव "ggml-vulkan-common.h"

स्थिर vk_device_architecture get_device_architecture(ध्रुव vk::PhysicalDevice& device) {
    vk::PhysicalDeviceProperties props = device.getProperties();

    यदि (props.vendorID == VK_VENDOR_ID_AMD) {
        ध्रुव std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        तार्किक amd_shader_core_properties = असत्य;
        तार्किक integer_dot_गुणनफलम् = असत्य;
        तार्किक subgroup_आकार_control = असत्य;
        तार्किक shader_float8 = असत्य;

        प्रति (ध्रुव स्वचालित& properties : ext_props) {
            यदि (वाक्यं_तुलय("VK_AMD_shader_core_properties", properties.extensionName) == 0) {
                amd_shader_core_properties = सत्य;
            } अन्यथा यदि (वाक्यं_तुलय("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0) {
                integer_dot_गुणनफलम् = सत्य;
            } अन्यथा यदि (वाक्यं_तुलय("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                subgroup_आकार_control = सत्य;
            } अन्यथा यदि (वाक्यं_तुलय("VK_EXT_shader_float8", properties.extensionName) == 0) {
                shader_float8 = सत्य;
            }
        }

        यदि (!amd_shader_core_properties || !integer_dot_गुणनफलम् || !subgroup_आकार_control) {
            प्रत्यागम vk_device_architecture::OTHER;
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::PhysicalDeviceShaderCorePropertiesAMD shader_core_props_amd;
        vk::Physical_Device_Shader_Integer_Dot_गुणनफलम्_Properties_KHR integer_dot_props;
        vk::Physical_Device_Subgroup_आकार_Control_Properties_EXT subgroup_आकार_control_props;

        props2.p_अग्र = &shader_core_props_amd;
        shader_core_props_amd.p_अग्र = &integer_dot_props;
        integer_dot_props.p_अग्र = &subgroup_आकार_control_props;

        device.getProperties2(&props2);

        यदि (subgroup_आकार_control_props.max_Subgroup_आकार == 64 && subgroup_आकार_control_props.min_Subgroup_आकार == 64) {
            प्रत्यागम vk_device_architecture::AMD_GCN;
        }
        यदि (subgroup_आकार_control_props.max_Subgroup_आकार == 64 && subgroup_आकार_control_props.min_Subgroup_आकार == 32) {
            // RDNA
            यदि (shader_core_props_amd.wavefronts_प्रत्येकम्_Simd == 20) {
                प्रत्यागम vk_device_architecture::AMD_RDNA1;
            }
            यदि (shader_float8) {
                प्रत्यागम vk_device_architecture::AMD_RDNA4;
            }
            यदि (integer_dot_props.integerDotProduct4x8BitPackedMixedSignednessAccelerated) {
                प्रत्यागम vk_device_architecture::AMD_RDNA3;
            }
            प्रत्यागम vk_device_architecture::AMD_RDNA2;
        }
    } अन्यथा यदि (props.vendorID == VK_VENDOR_ID_INTEL) {
        ध्रुव std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        तार्किक subgroup_आकार_control = असत्य;
        तार्किक integer_dot_गुणनफलम् = असत्य;

        प्रति (ध्रुव स्वचालित& properties : ext_props) {
            यदि (वाक्यं_तुलय("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                subgroup_आकार_control = सत्य;
            } अन्यथा यदि (वाक्यं_तुलय("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0) {
                integer_dot_गुणनफलम् = सत्य;
            }
        }

        यदि (!subgroup_आकार_control || !integer_dot_गुणनफलम्) {
            प्रत्यागम vk_device_architecture::OTHER;
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::Physical_Device_Subgroup_आकार_Control_Properties_EXT subgroup_आकार_control_props;
        vk::Physical_Device_Shader_Integer_Dot_गुणनफलम्_Properties_KHR integer_dot_props;

        props2.p_अग्र = &subgroup_आकार_control_props;
        subgroup_आकार_control_props.p_अग्र = &integer_dot_props;
        device.getProperties2(&props2);

        यदि (subgroup_आकार_control_props.min_Subgroup_आकार == 16) {
            // Xe2 architecture uses SIMD16 while previous Xe and Gen architecture uses SIMD8.
            // Minimum subgroup size matches the SIMD width so we distinguish architecture by checking this value.
            // https://www.intel.com/content/www/us/en/content-details/824434/2024-intel-tech-tour-xe2-and-lunar-lake-s-gpu.html
            // https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
            प्रत्यागम vk_device_architecture::INTEL_XE2;
        } अन्यथा यदि (subgroup_आकार_control_props.min_Subgroup_आकार == 8 &&
                 integer_dot_गुणनफलम् && integer_dot_props.integerDotProduct4x8BitPackedSignedAccelerated) {
            प्रत्यागम vk_device_architecture::INTEL_XE1;
        }
    } अन्यथा यदि (props.vendorID == VK_VENDOR_ID_NVIDIA) {
        ध्रुव std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        तार्किक cooperative_matrix = असत्य;
        तार्किक sm_builtins = असत्य;

        // Detect "pre-turing" based on lack of coopmat support.
        प्रति (ध्रुव स्वचालित& properties : ext_props) {
            यदि (वाक्यं_तुलय("VK_KHR_cooperative_matrix", properties.extensionName) == 0) {
                cooperative_matrix = सत्य;
            } अन्यथा यदि (वाक्यं_तुलय("VK_NV_shader_sm_builtins", properties.extensionName) == 0) {
                sm_builtins = सत्य;
            }
        }

        यदि (!cooperative_matrix) {
            प्रत्यागम vk_device_architecture::NVIDIA_PRE_TURING;
        }

        यदि (sm_builtins) {
            vk::PhysicalDeviceProperties2 props2;
            vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV sm_props;

            props2.p_अग्र = &sm_props;

            device.getProperties2(&props2);

            // Turing has 32, following architectures have 48
            यदि (sm_props.shader_Warps_प्रत्येकम्_SM == 32) {
                प्रत्यागम vk_device_architecture::NVIDIA_TURING;
            }
        }
    } अन्यथा यदि(props.vendorID == VK_VENDOR_ID_QUALCOMM){
        ध्रुव std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        तार्किक cooperative_matrix = असत्य;
        तार्किक cooperative_matrix_conversion = असत्य;

        प्रति (ध्रुव स्वचालित& properties : ext_props) {
            यदि (वाक्यं_तुलय("VK_KHR_cooperative_matrix", properties.extensionName) == 0) {
                cooperative_matrix = सत्य;
            } अन्यथा यदि (वाक्यं_तुलय("VK_QCOM_cooperative_matrix_conversion", properties.extensionName) == 0) {
                cooperative_matrix_conversion = सत्य;
            }
        }

        यदि (cooperative_matrix && cooperative_matrix_conversion) {
            प्रत्यागम vk_device_architecture::QUALCOMM_ADRENO;
        }
    }
    प्रत्यागम vk_device_architecture::OTHER;
}

तार्किक ggml_vk_lightning_indexer_k_type_supported(ggml_type type) {
    प्रत्यागम std::find(lightning_indexer_k_types.begin(), lightning_indexer_k_types.end(), type) != lightning_indexer_k_types.end();
}
शून्य ggml_vk_print_device_fault_info(ध्रुव vk_device& device) {
    यदि (!device->device_fault || !device->pfn_vkGetDeviceFaultInfoEXT) {
        प्रत्यागम;
    }

    VkDeviceFaultCountsEXT fault_counts {};
    fault_counts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
    Vk_परिणाम res = device->pfn_vkGetDeviceFaultInfoEXT(device->device, &fault_counts, शून्यसूचकः);
    यदि (res != VK_SUCCESS) {
        GGML_LOG_ERROR("ggml_vulkan: vkGetDeviceFaultInfoEXT (counts) failed: %d\n", res);
        प्रत्यागम;
    }

    std::vector<Vk_Device_Fault_स्थानाङ्कः_Info_EXT> स्थानाङ्कः_infos(fault_counts.स्थानाङ्कः_Info_गणनम्);
    std::vector<VkDeviceFaultVendorInfoEXT> vendor_infos(fault_counts.vendor_Info_गणनम्);

    VkDeviceFaultInfoEXT fault_info {};
    fault_info.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
    fault_info.p_स्थानाङ्कः_Infos = स्थानाङ्कः_infos.दत्त();
    fault_info.pVendorInfos = vendor_infos.दत्त();

    res = device->pfn_vkGetDeviceFaultInfoEXT(device->device, &fault_counts, &fault_info);
    यदि (res != VK_SUCCESS) {
        GGML_LOG_ERROR("ggml_vulkan: vkGetDeviceFaultInfoEXT (info) failed: %d\n", res);
        प्रत्यागम;
    }

    यदि (fault_counts.स्थानाङ्कः_Info_गणनम् == 0 && fault_counts.vendor_Info_गणनम् == 0 && fault_info.description[0] == '\0') {
        प्रत्यागम;
    }

    यदि (fault_info.description[0] != '\0') {
        GGML_LOG_ERROR("ggml_vulkan: device fault on %s: %s\n", device->name.c_str(), fault_info.description);
    }

    प्रति (uint32_t i = 0; i < fault_counts.स्थानाङ्कः_Info_गणनम्; i++) {
        ध्रुव स्वचालित& info = स्थानाङ्कः_infos[i];
        GGML_LOG_CONT("  address fault %u: type=%d address=0x%llx precision=0x%llx\n",
                i, (पूर्णाङ्क)info.स्थानाङ्कः_Type,
                (अचिह्नित दीर्घ दीर्घ)info.reported_स्थानाङ्कः,
                (अचिह्नित दीर्घ दीर्घ)info.स्थानाङ्कः_Precision);
    }
    प्रति (uint32_t i = 0; i < fault_counts.vendor_Info_गणनम्; i++) {
        ध्रुव स्वचालित& info = vendor_infos[i];
        GGML_LOG_CONT("  vendor fault %u: %s (code=0x%llx data=0x%llx)\n",
                i, info.description,
                (अचिह्नित दीर्घ दीर्घ)info.vendor_Fault_संहिता,
                (अचिह्नित दीर्घ दीर्घ)info.vendor_Fault_दत्त);
    }
}
uint64_t ggml_vk_get_ग्रन्थिः_flops(ध्रुव ggml_tensor * ग्रन्थिः) {
    यदि (ग्रन्थिः->op == GGML_OP_MUL_MAT || ग्रन्थिः->op == GGML_OP_MUL_MAT_ID) {
        ध्रुव uint64_t m     = ग्रन्थिः->ne[0];
        ध्रुव uint64_t n     = ग्रन्थिः->ne[1];
        ध्रुव uint64_t k     = ग्रन्थिः->src[1]->ne[0];
        ध्रुव uint64_t batch = ग्रन्थिः->ne[2] * ग्रन्थिः->ne[3];
        प्रत्यागम m * n * (k + (k - 1)) * batch;
    }
    यदि (ग्रन्थिः->op == GGML_OP_CONV_2D || ग्रन्थिः->op == GGML_OP_CONV_TRANSPOSE_2D) {
        ध्रुव ggml_tensor * knl = ग्रन्थिः->src[0];
        ध्रुव uint64_t Cout  = ग्रन्थिः->ne[2];
        ध्रुव uint64_t आकार_K = ग्रन्थिः->src[1]->ne[2] * knl->ne[0] * knl->ne[1];
        ध्रुव uint64_t आकार_N_2 = ग्रन्थिः->ne[3] * ग्रन्थिः->ne[0] * ग्रन्थिः->ne[1];
        प्रत्यागम Cout * आकार_N_2 * (आकार_K + (आकार_K - 1));
    }
    यदि (ग्रन्थिः->op == GGML_OP_CONV_3D) {
        ध्रुव ggml_tensor * knl = ग्रन्थिः->src[0];
        ध्रुव uint64_t OC     = ggml_get_op_params_i32(ग्रन्थिः, 11);
        ध्रुव uint64_t IC     = ggml_get_op_params_i32(ग्रन्थिः, 9);
        ध्रुव uint64_t आकार_K = IC * knl->ne[0] * knl->ne[1] * knl->ne[2];
        ध्रुव uint64_t आकार_N_2 = ग्रन्थिः->ne[3] / OC * ग्रन्थिः->ne[0] * ग्रन्थिः->ne[1] * ग्रन्थिः->ne[2];
        प्रत्यागम OC * आकार_N_2 * (आकार_K + (आकार_K - 1));
    }
    यदि (ग्रन्थिः->op == GGML_OP_FLASH_ATTN_EXT) {
        ध्रुव ggml_tensor * q = ग्रन्थिः->src[0];
        ध्रुव ggml_tensor * k = ग्रन्थिः->src[1];
        ध्रुव ggml_tensor * v = ग्रन्थिः->src[2];
        प्रत्यागम 2ull * q->ne[1] * q->ne[2] * (k->ne[0] + v->ne[0]) * k->ne[1] * q->ne[3];
    }
    प्रत्यागम 0;
}
शून्य ggml_vk_print_ग्रन्थिः_list(ध्रुव ggml_cgraph * cgraph, पूर्णाङ्क start, पूर्णाङ्क end) {
    uint64_t योग_flops = 0;
    पूर्णाङ्क n_ops = 0;
    प्रति (पूर्णाङ्क j = start; j <= end && j < cgraph->n_nodes; j++) {
        uint64_t flops = ggml_vk_get_ग्रन्थिः_flops(cgraph->nodes[j]);
        योग_flops += flops;
        n_ops++;
        यदि (flops > 0) {
            GGML_LOG_CONT("  node %d: %s (%s) [%.2f GFLOP]\n",
                    j, cgraph->nodes[j]->name, ggml_op_name(cgraph->nodes[j]->op),
                    flops / 1e9);
        } अन्यथा {
            GGML_LOG_CONT("  node %d: %s (%s)\n",
                    j, cgraph->nodes[j]->name, ggml_op_name(cgraph->nodes[j]->op));
        }
    }
    GGML_LOG_CONT("  total: %d ops, %.2f GFLOP\n", n_ops, योग_flops / 1e9);
}
शून्य ggml_vk_print_device_नष्टम्_info(ध्रुव vk_device& device) {
    ggml_vk_print_device_fault_info(device);
    यदि (device->serialize_submissions && device->diag_cgraph != शून्यसूचकः && device->diag_पूर्व_start >= 0) {
        GGML_LOG_ERROR("ggml_vulkan: device lost on %s, likely caused by previous submission (nodes %d to %d):\n",
                device->name.c_str(), device->diag_पूर्व_start, device->diag_पूर्व_अन्तः);
        ggml_vk_print_ग्रन्थिः_list(device->diag_cgraph, device->diag_पूर्व_start, device->diag_पूर्व_अन्तः);
    } अन्यथा {
        GGML_LOG_ERROR("ggml_vulkan: device lost on %s\n", device->name.c_str());
    }
}
शून्य * ध्रुव vk_ptr_आधारः = (शून्य *)(uintptr_t) 0x1000;  // NOLINT

uint64_t vk_tensor_विस्थापनम्(ध्रुव ggml_tensor * tensor) {
    यदि (tensor->view_src) {
        प्रत्यागम (uint8_t *) tensor->view_src->दत्त - (uint8_t *) vk_ptr_आधारः;
    }
    प्रत्यागम (uint8_t *) tensor->दत्त - (uint8_t *) vk_ptr_आधारः;
}

size_t ggml_vk_tensor_बफर_विस्थापनम्(ध्रुव ggml_backend_vk_context * ctx, ध्रुव ggml_tensor * t) {
    // vk_tensor_offset() is relative to vk_ptr_base, but mapped host tensors need an offset relative to their Vulkan buffer.
    यदि (ctx->device->uma) {
        vk_बफर buf = शून्यसूचकः;
        size_t off = 0;
        ggml_vk_host_get(ctx->device, t->दत्त, buf, off);
        यदि (buf) {
            प्रत्यागम off;
        }
    }
    प्रत्यागम (size_t)(vk_tensor_विस्थापनम्(t) + t->view_offs);
}
size_t ggml_vk_descriptor_विस्थापनम्(size_t tensor_विस्थापनम्, size_t alignment, size_t type_आकार) {
    // Move the descriptor back until its distance to the tensor is divisible by the tensor type size.
    size_t descriptor_विस्थापनम् = tensor_विस्थापनम् & ~(alignment - 1);
    यावत् ((tensor_विस्थापनम् - descriptor_विस्थापनम्) % type_आकार != 0) {
        GGML_ASSERT(descriptor_विस्थापनम् >= alignment);
        descriptor_विस्थापनम् -= alignment;
    }

    प्रत्यागम descriptor_विस्थापनम्;
}
uint32_t get_misalign_अष्टकानि(ध्रुव ggml_backend_vk_context * ctx, ध्रुव ggml_tensor * t) {
    ध्रुव size_t tensor_विस्थापनम् = ggml_vk_tensor_बफर_विस्थापनम्(ctx, t);
    ध्रुव size_t descriptor_विस्थापनम् = ggml_vk_descriptor_विस्थापनम्(
        tensor_विस्थापनम्, ctx->device->properties.limits.min_Storage_बफर_विस्थापनम्_Alignment, ggml_type_आकार(t->type));
    GGML_ASSERT(tensor_विस्थापनम् - descriptor_विस्थापनम् <= UINT32_MAX);
    प्रत्यागम tensor_विस्थापनम् - descriptor_विस्थापनम्;
}

uint32_t ggml_vk_concat_unit_आकार(ggml_type type) {
    ध्रुव uint32_t type_आकार = ggml_type_आकार(type);

    यदि (!ggml_is_quantized(type)) {
        प्रत्यागम type_आकार;
    }

    // Use the widest existing concat shader that evenly divides a quant block.
    यदि (type_आकार % 8 == 0) {
        प्रत्यागम 8;
    }
    यदि (type_आकार % 4 == 0) {
        प्रत्यागम 4;
    }
    यदि (type_आकार % 2 == 0) {
        प्रत्यागम 2;
    }
    प्रत्यागम 1;
}
तार्किक ggml_vk_concat_supported(ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ध्रुव ggml_tensor * dst) {
    यदि (src0->type != src1->type || src0->type != dst->type) {
        प्रत्यागम असत्य;
    }

    यदि (!ggml_is_quantized(src0->type)) {
        ध्रुव size_t type_आकार = ggml_type_आकार(src0->type);
        प्रत्यागम type_आकार == 1 || type_आकार == 2 || type_आकार == 4 || type_आकार == 8;
    }

    // Quantized tensor rows are block-aligned when created.
    प्रत्यागम ggml_is_contiguous_पङ्क्तिकाः(src0) && ggml_is_contiguous_पङ्क्तिकाः(src1) && ggml_is_contiguous_पङ्क्तिकाः(dst);
}
स्थिर तार्किक vk_instance_initialized = असत्य;

vk_instance_t vk_instance;

स्थिर Vk_Device_आकार ggml_vk_get_max_बफर_range(ध्रुव ggml_backend_vk_context * ctx, ध्रुव vk_बफर &buf, ध्रुव Vk_Device_आकार विस्थापनम्) {
    ध्रुव Vk_Device_आकार range = std::min(Vk_Device_आकार{buf->size - विस्थापनम्},
                                        Vk_Device_आकार{ctx->device->properties.limits.max_Storage_बफर_Range});
    प्रत्यागम range;
}

शून्य ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx) {
    // Use waitForFences while most of the graph executes. Hopefully the CPU can sleep
    // during this wait.
    यदि (ctx->almost_ready_fence_pending) {
        VK_CHECK(ctx->device->device.waitForFences({ ctx->almost_ready_fence }, सत्य, UINT64_MAX), "almost_ready_fence", ctx->device);
        ctx->device->device.resetFences({ ctx->almost_ready_fence });
        ctx->almost_ready_fence_pending = असत्य;
    }

    // Spin (w/pause) waiting for the graph to finish executing.
    vk::परिणाम_2 परिणाम;
    प्रति (;;) {
        यत्न {
            परिणाम = ctx->device->device.get_Fence_स्थितिः(ctx->fence);
        } गृहाण (vk::Device_नष्टम्_Error &) {
            ggml_vk_print_device_नष्टम्_info(ctx->device);
            GGML_LOG_ERROR("ggml_vulkan: getFenceStatus at %s:%d\n", __FILE__, __LINE__);
            क्षिप;
        }
        यदि (परिणाम == vk::परिणाम_2::eSuccess) {
            भङ्ग;
        }
        यदि (परिणाम != vk::परिणाम_2::eNotReady) {
            GGML_LOG_ERROR("ggml_vulkan: error %s at %s:%d\n", to_string(परिणाम).c_str(), __FILE__, __LINE__);
            क्षिप vk::SystemError(vk::make_error_code(परिणाम), "ggml_vulkan: getFenceStatus");
        }
        प्रति (uint32_t i = 0; i < 100; ++i) {
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

स्थिर तार्किक ggml_vk_strip_decode_सदिशम्(ध्रुव uint32_t * code, size_t word_गणनम्, std::vector<uint32_t> & बहिर्गमनम्) {
    स्थिर ध्रुव वर्ण k_Decode_सदिशम्_Ext[] = "SPV_NV_cooperative_matrix_decode_vector";

    यदि (word_गणनम् < 5) {
        प्रत्यागम असत्य;
    }

    तार्किक उपयुङ्क्ते_decode_सदिशम् = असत्य;
    प्रति (size_t pos = 5; pos < word_गणनम्; ) {
        uint32_t word = code[pos];
        uint32_t wc   = word >> spv::Word_गणनम्_सरणम्;
        uint32_t op   = word & spv::Op_संहिता_आवरणम्;
        GGML_ASSERT(wc > 0 && pos + wc <= word_गणनम्);
        यदि (op == spv::OpExtension && wc >= 2) {
            ध्रुव वर्ण * s = reinterpret_cast<ध्रुव वर्ण *>(&code[pos + 1]);
            यदि (वाक्यं_तुलय(s, k_Decode_सदिशम्_Ext) == 0) {
                उपयुङ्क्ते_decode_सदिशम् = सत्य;
                भङ्ग;
            }
        }
        pos += wc;
    }

    यदि (!उपयुङ्क्ते_decode_सदिशम्) {
        प्रत्यागम असत्य;
    }

    VK_LOG_DEBUG("ggml_vk_strip_decode_vector: stripping SPV_NV_cooperative_matrix_decode_vector");

    // Bulk-copy unchanged runs and only break the run when an instruction needs to
    // be dropped or patched. Use reserve + insert/push_back so the destination buffer
    // is touched exactly once (no zero-initialization pass from resize()).
    बहिर्गमनम्.clear();
    बहिर्गमनम्.reserve(word_गणनम्);

    size_t सततम्_start = 0;
    स्वचालित flush_सततम् = [&](size_t ऊर्ध्वम्_गन्तव्यम्) {
        यदि (ऊर्ध्वम्_गन्तव्यम् > सततम्_start) {
            बहिर्गमनम्.insert(बहिर्गमनम्.end(), code + सततम्_start, code + ऊर्ध्वम्_गन्तव्यम्);
        }
    };

    प्रति (size_t pos = 5; pos < word_गणनम्; ) {
        uint32_t word = code[pos];
        uint32_t wc   = word >> spv::Word_गणनम्_सरणम्;
        uint32_t op   = word & spv::Op_संहिता_आवरणम्;
        GGML_ASSERT(wc > 0 && pos + wc <= word_गणनम्);

        यदि (op == spv::OpExtension && wc >= 2) {
            ध्रुव वर्ण * s = reinterpret_cast<ध्रुव वर्ण *>(&code[pos + 1]);
            यदि (वाक्यं_तुलय(s, k_Decode_सदिशम्_Ext) == 0) {
                flush_सततम्(pos);
                pos += wc;
                सततम्_start = pos;
                अनुवर्त;
            }
        }

        यदि (op == spv::OpCapability && wc == 2 && code[pos + 1] == k_Spv_Capability_Cooperative_Matrix_Decode_सदिशम्_NV) {
            flush_सततम्(pos);
            pos += wc;
            सततम्_start = pos;
            अनुवर्त;
        }

        यदि (op == k_Spv_Op_Cooperative_Matrix_आरोपय_Tensor_NV) {
            // [opcode/wc][ResultType][Result][Pointer][Object][TensorLayout][MemOperand mask][mem extras...][TA mask][ta extras...]
            GGML_ASSERT(wc >= 8);

            uint32_t mem_आवरणम् = code[pos + 6];
            size_t   cur      = pos + 7;
            // Each of these MemoryAccess bits (when set) carries one trailing operand.
            cur += (mem_आवरणम् & 0x2)     ? 1 : 0; // Aligned
            cur += (mem_आवरणम् & 0x8)     ? 1 : 0; // MakePointerAvailable
            cur += (mem_आवरणम् & 0x10)    ? 1 : 0; // MakePointerVisible
            cur += (mem_आवरणम् & 0x10000) ? 1 : 0; // AliasScopeINTELMask
            cur += (mem_आवरणम् & 0x20000) ? 1 : 0; // NoAliasINTELMask
            GGML_ASSERT(cur < pos + wc);

            uint32_t ta_आवरणम् = code[cur];
            यदि ((ta_आवरणम् & k_Spv_Tensor_Addressing_Decode_सदिशम्_Func_Bit) == 0) {
                pos += wc;
                अनुवर्त; // leave instruction inside the current unchanged run
            }

            flush_सततम्(pos);

            // Append unchanged prefix of the instruction (header through the mem-extras).
            size_t inst_start = बहिर्गमनम्.size();
            size_t pre_n      = cur - pos;
            बहिर्गमनम्.insert(बहिर्गमनम्.end(), code + pos, code + pos + pre_n);

            // Emit TA mask with the DecodeVectorFunc bit cleared.
            बहिर्गमनम्.push_back(ta_आवरणम् & ~k_Spv_Tensor_Addressing_Decode_सदिशम्_Func_Bit);

            // TA extras: TensorView (0x1) and DecodeFunc (0x2) are kept verbatim;
            // DecodeVectorFunc (0x4) is dropped along with its trailing id operand.
            size_t keep_ta_extras = ((ta_आवरणम् & 0x1) ? 1 : 0) + ((ta_आवरणम् & 0x2) ? 1 : 0);
            यदि (keep_ta_extras) {
                बहिर्गमनम्.insert(बहिर्गमनम्.end(), code + cur + 1, code + cur + 1 + keep_ta_extras);
            }

            GGML_ASSERT(wc == pre_n + 1 + keep_ta_extras + 1);

            // Patch the instruction header with the new (one-shorter) word count.
            uint32_t new_wc = wc - 1;
            बहिर्गमनम्[inst_start] = (new_wc << spv::Word_गणनम्_सरणम्) | op;

            pos += wc;
            सततम्_start = pos;
            अनुवर्त;
        }

        pos += wc;
    }

    flush_सततम्(word_गणनम्);
    प्रत्यागम सत्य;
}

स्थिर तार्किक ggml_vk_roll_bk_loop(ध्रुव uint32_t * code, size_t word_गणनम्, std::vector<uint32_t> & बहिर्गमनम्) {
    यदि (word_गणनम् < 5) {
        प्रत्यागम असत्य;
    }

    संरचना vk_spv_loop {
        size_t   शीर्षकम्;
        size_t   end;
        uint32_t control;
    };

    std::vector<vk_spv_loop> loops;

    // Collect a list of all loops in the module.
    प्रति (size_t pos = 5; pos < word_गणनम्; ) {
        ध्रुव uint32_t wc = code[pos] >> spv::Word_गणनम्_सरणम्;
        ध्रुव uint32_t op = code[pos] & spv::Op_संहिता_आवरणम्;
        यदि (wc == 0 || pos + wc > word_गणनम्) {
            प्रत्यागम असत्य;
        }

        यदि (op == spv::OpLoopMerge && wc >= 4) { loops.push_back({ pos, 0, code[pos + 3] }); }

        यदि (op == spv::OpLabel && wc >= 2) {
            प्रति (स्वचालित & l : loops) {
                यदि (l.end == 0 && code[l.शीर्षकम् + 1] == code[pos + 1]) { l.end = pos; }
            }
        }

        pos += wc;
    }

    स्वचालित encloses = [](ध्रुव vk_spv_loop & a, ध्रुव vk_spv_loop & b) {
        प्रत्यागम a.शीर्षकम् < b.शीर्षकम् && b.शीर्षकम् < a.end;
    };

    // Find the BK loop.
    ध्रुव vk_spv_loop * bk = शून्यसूचकः;
    प्रति (ध्रुव स्वचालित & h : loops) {
        यदि (h.control != spv::Loop_Control_Unroll_आवरणम्) {
            अनुवर्त;
        }
        ध्रुव vk_spv_loop * parent = शून्यसूचकः;
        तार्किक has_child = असत्य;
        प्रति (ध्रुव स्वचालित & g : loops) {
            यदि (encloses(g, h) && (!parent || g.शीर्षकम् > parent->शीर्षकम्)) {
                parent = &g;
            }
            यदि (encloses(h, g)) {
                has_child = सत्य;
            }
        }
        // BK loop should be the last loop nested inside the loop with no hint
        // and have at least one child loop.
        यदि (parent &&
            parent->control == spv::Loop_Control_आवरणम्_None &&
            has_child &&
            (!bk || h.शीर्षकम् > bk->शीर्षकम्)) {
            bk = &h;
        }
    }
    यदि (!bk) {
        प्रत्यागम असत्य;
    }

    // set DontUnroll instead of Unroll
    बहिर्गमनम्.नियोजय(code, code + word_गणनम्);
    बहिर्गमनम्[bk->शीर्षकम् + 3] = spv::Loop_Control_Dont_Unroll_आवरणम्;
    प्रत्यागम सत्य;
}

स्थिर शून्य ggml_vk_create_pipeline_func(vk_device& device, vk_pipeline& pipeline, size_t spv_आकार, ध्रुव शून्य* spv_दत्त, ध्रुव std::string entrypoint,
                                         uint32_t parameter_गणनम्, std::array<uint32_t, 3> wg_denoms, std::vector<uint32_t> specialization_constants,
                                         तार्किक disable_robustness, तार्किक require_पूर्णम्_subgroups, uint32_t required_subgroup_आकार) {
    VK_LOG_DEBUG("ggml_vk_create_pipeline(" << device->name << ", " << pipeline->name << ", " << entrypoint << ", " << parameter_गणनम् <<
                 ", (" << wg_denoms[0] << "," << wg_denoms[1] << "," << wg_denoms[2] << "), specialization_constants, " <<
                 disable_robustness << ", " << require_पूर्णम्_subgroups << ", " << required_subgroup_आकार << ")");
    GGML_ASSERT(parameter_गणनम् > 0);
    GGML_ASSERT(parameter_गणनम् <= MAX_PARAMETER_गणनम्);
    GGML_ASSERT(wg_denoms[0] > 0 && wg_denoms[1] > 0 && wg_denoms[2] > 0); // NOLINT

    vk::ShaderModuleCreateInfo shader_module_create_info({}, spv_आकार, reinterpret_cast<ध्रुव uint32_t *>(spv_दत्त));

    // Patch SPIR-V to enable supported FP16 float controls, avoiding the need
    // for separate shader variants.
    std::vector<uint32_t> spirv;
    यदि (device->float_controls_rte_fp16 || device->float_controls_denorm_preserve_fp16) {
        ध्रुव uint32_t* spv_शब्दाः = reinterpret_cast<ध्रुव uint32_t *>(spv_दत्त);
        size_t word_गणनम् = spv_आकार / परिमाण(uint32_t);
        spirv.नियोजय(spv_शब्दाः, spv_शब्दाः + word_गणनम्);

        // Find insertion points respecting SPIR-V layout order:
        //   Header(5) -> OpCapability -> OpExtension -> ... -> OpEntryPoint -> OpExecutionMode -> ...
        size_t pos = 5; // skip header
        size_t cap_निवेशय_pos = pos;
        size_t ext_निवेशय_pos = pos;
        size_t exec_निवेशय_pos = pos;
        uint32_t प्रविष्टिः_point_id = 0;

        यावत् (pos < spirv.size()) {
            uint32_t opcode = spirv[pos] & spv::Op_संहिता_आवरणम्;
            uint32_t len    = spirv[pos] >> spv::Word_गणनम्_सरणम्;
            यदि (len == 0) भङ्ग;

            यदि (opcode == spv::OpCapability) {
                cap_निवेशय_pos = pos + len;
                ext_निवेशय_pos = pos + len;
            } अन्यथा यदि (opcode == spv::OpExtension) {
                ext_निवेशय_pos = pos + len;
            } अन्यथा यदि (opcode == spv::Op_प्रविष्टिः_Point) {
                प्रविष्टिः_point_id = spirv[pos + 2];
                exec_निवेशय_pos = pos + len;
            } अन्यथा यदि (opcode == spv::Op_Execution_बहुलकम् || opcode == spv::Op_Execution_बहुलकम्_Id) {
                exec_निवेशय_pos = pos + len;
            } अन्यथा यदि (प्रविष्टिः_point_id != 0) {
                भङ्ग;
            }

            pos += len;
        }

        // Insert from latest position first so earlier indices stay valid.

        यदि (device->float_controls_rte_fp16) {
            // OpExecutionMode %entrypoint RoundingModeRTE 16
            uint32_t exec_बहुलकम्[] = { (4u << spv::Word_गणनम्_सरणम्) | spv::Op_Execution_बहुलकम्, प्रविष्टिः_point_id, spv::Execution_बहुलकम्_Rounding_बहुलकम्_RTE, 16 };
            spirv.insert(spirv.begin() + exec_निवेशय_pos, std::begin(exec_बहुलकम्), std::end(exec_बहुलकम्));
        }

        यदि (device->float_controls_denorm_preserve_fp16) {
            // OpExecutionMode %entrypoint DenormPreserve 16
            uint32_t exec_बहुलकम्[] = { (4u << spv::Word_गणनम्_सरणम्) | spv::Op_Execution_बहुलकम्, प्रविष्टिः_point_id, spv::Execution_बहुलकम्_Denorm_Preserve, 16 };
            spirv.insert(spirv.begin() + exec_निवेशय_pos, std::begin(exec_बहुलकम्), std::end(exec_बहुलकम्));
        }

        // OpExtension "SPV_KHR_float_controls"
        ध्रुव वर्ण ext_str[] = "SPV_KHR_float_controls";
        size_t ext_str_शब्दाः = CEIL_DIV(परिमाण(ext_str), परिमाण(uint32_t));
        std::vector<uint32_t> extension(1 + ext_str_शब्दाः, 0);
        extension[0] = (uint32_t)((1 + ext_str_शब्दाः) << spv::Word_गणनम्_सरणम्) | spv::OpExtension;
        स्मृतिं_प्रतिलिख(&extension[1], ext_str, परिमाण(ext_str));
        spirv.insert(spirv.begin() + ext_निवेशय_pos, extension.begin(), extension.end());

        यदि (device->float_controls_rte_fp16) {
            // OpCapability RoundingModeRTE
            uint32_t capability[] = { (2u << spv::Word_गणनम्_सरणम्) | spv::OpCapability, spv::Capability_Rounding_बहुलकम्_RTE };
            spirv.insert(spirv.begin() + cap_निवेशय_pos, std::begin(capability), std::end(capability));
        }

        यदि (device->float_controls_denorm_preserve_fp16) {
            // OpCapability DenormPreserve
            uint32_t capability[] = { (2u << spv::Word_गणनम्_सरणम्) | spv::OpCapability, spv::CapabilityDenormPreserve };
            spirv.insert(spirv.begin() + cap_निवेशय_pos, std::begin(capability), std::end(capability));
        }

        shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * परिमाण(uint32_t), spirv.दत्त());
    }

#यदि defined(GGML_VULKAN_COOPMAT2_DECODE_सदिशम्_GLSLC_SUPPORT)
    यदि (device->coopmat2 && !device->coopmat2_decode_सदिशम्) {
        ध्रुव uint32_t * src   = spirv.रिक्तानि() ? reinterpret_cast<ध्रुव uint32_t *>(spv_दत्त) : spirv.दत्त();
        size_t           src_n = spirv.रिक्तानि() ? spv_आकार / परिमाण(uint32_t) : spirv.size();
        std::vector<uint32_t> stripped;
        यदि (ggml_vk_strip_decode_सदिशम्(src, src_n, stripped)) {
            spirv = std::move(stripped);
            shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * परिमाण(uint32_t), spirv.दत्त());
        }
    }
#यदि_अन्तः

#यदि VK_शीर्षकम्_संस्करणम् >= 287
    // Roll the mul_mm BK loop on Asahi Linux. Skip bf16 and the mul_mmq pipelines.
    यदि (device->driver_id == vk::DriverId::eMesaHoneykrisp &&
        pipeline->name.rfind("matmul", 0) == 0 &&
        pipeline->name.find("bf16") == std::string::npos &&
        pipeline->name.find("q8_1") == std::string::npos) {
        ध्रुव uint32_t * src   = spirv.रिक्तानि() ? reinterpret_cast<ध्रुव uint32_t *>(spv_दत्त) : spirv.दत्त();
        size_t           src_n = spirv.रिक्तानि() ? spv_आकार / परिमाण(uint32_t) : spirv.size();
        std::vector<uint32_t> rolled;
        यदि (ggml_vk_roll_bk_loop(src, src_n, rolled)) {
            spirv = std::move(rolled);
            shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * परिमाण(uint32_t), spirv.दत्त());
        }
    }
#यदि_अन्तः

    यत्न {
        pipeline->shader_module = device->device.createShaderModule(shader_module_create_info);
    } गृहाण (ध्रुव vk::SystemError& e) {
        std::cerr << "ggml_vulkan: shader module creation failed for " << pipeline->name << ": " << e.what() << std::पङ्क्त्यन्तः;
        क्षिप;
    }

    vk::प्रक्षिप_Constant_Range pcr(
        vk::Shader_Stage_Flag_द्वयङ्काः::eCompute,
        0,
        pipeline->प्रक्षिप_constant_आकार
    );

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info(vk::PipelineLayoutCreateFlags(), device->dsl, pcr);
    यत्न {
        pipeline->layout = device->device.createPipelineLayout(pipeline_layout_create_info);
    } गृहाण (ध्रुव vk::SystemError& e) {
        std::cerr << "ggml_vulkan: pipeline layout creation failed for " << pipeline->name << ": " << e.what() << std::पङ्क्त्यन्तः;
        क्षिप;
    }

    std::vector<vk::Specialization_Map_प्रविष्टिः> specialization_entries(specialization_constants.size());

    प्रति (size_t i = 0; i < specialization_constants.size(); i++) {
        specialization_entries[i].constantID = i;
        specialization_entries[i].विस्थापनम् = i * परिमाण(uint32_t);
        specialization_entries[i].size = परिमाण(uint32_t);
    }

    vk::SpecializationInfo specialization_info(
        specialization_entries.size(),
        specialization_entries.दत्त(),
        specialization_constants.size() * परिमाण(uint32_t),
        specialization_constants.दत्त()
    );

    vk::PipelineShaderStageCreateFlags pipeline_shader_stage_create_flags{};

    यदि (device->subgroup_require_पूर्णम्_support && require_पूर्णम्_subgroups) {
        pipeline_shader_stage_create_flags |= vk::Pipeline_Shader_Stage_Create_Flag_द्वयङ्काः::e_Require_पूर्णम्_Subgroups_EXT;
    }

    vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
            pipeline_shader_stage_create_flags,
            vk::Shader_Stage_Flag_द्वयङ्काः::eCompute,
            pipeline->shader_module,
            entrypoint.c_str(),
            &specialization_info);

    vk::Pipeline_Shader_Stage_Required_Subgroup_आकार_Create_Info_EXT pipeline_shader_stage_required_subgroup_आकार_create_info;
    pipeline_shader_stage_required_subgroup_आकार_create_info.required_Subgroup_आकार = required_subgroup_आकार;
    यदि (device->subgroup_आकार_control && required_subgroup_आकार > 0) {
        GGML_ASSERT(device->subgroup_min_आकार <= required_subgroup_आकार && required_subgroup_आकार <= device->subgroup_max_आकार);
        pipeline_shader_create_info.समुच्चयः_PNext(&pipeline_shader_stage_required_subgroup_आकार_create_info);
    }

    vk::ComputePipelineCreateInfo compute_pipeline_create_info(
        device->pipeline_executable_properties_support ?
            vk::Pipeline_Create_Flag_द्वयङ्काः::eCaptureStatisticsKHR :
            vk::PipelineCreateFlags{},
        pipeline_shader_create_info,
        pipeline->layout);

    vk::PipelineRobustnessCreateInfoEXT rci;

    यदि (device->pipeline_robustness && disable_robustness) {
        rci.storageBuffers = vk::Pipeline_Robustness_बफर_Behavior_EXT::eDisabled;
        rci.एकरूपम्_Buffers = vk::Pipeline_Robustness_बफर_Behavior_EXT::eDisabled;
        compute_pipeline_create_info.समुच्चयः_PNext(&rci);
    }

#यदि defined(VK_EXT_shader_64bit_indexing)
    vk::PipelineCreateFlags2CreateInfo pipelineFlags2CreateInfo;
    यदि (pipeline->is_64b_indexing)
    {
        pipelineFlags2CreateInfo.flags = vk::PipelineCreateFlagBits2::e64BitIndexingEXT;
        यदि (device->pipeline_executable_properties_support) {
            pipelineFlags2CreateInfo.flags |= vk::PipelineCreateFlagBits2::eCaptureStatisticsKHR;
        }
        pipelineFlags2CreateInfo.समुच्चयः_PNext(compute_pipeline_create_info.p_अग्र);
        compute_pipeline_create_info.समुच्चयः_PNext(&pipelineFlags2CreateInfo);
    }
#यदि_अन्तः

    यत्न {
        pipeline->pipeline = device->device.createComputePipeline(VK_NULL_HANDLE, compute_pipeline_create_info).value;
    } गृहाण (ध्रुव vk::SystemError& e) {
        std::cerr << "ggml_vulkan: compute pipeline creation failed for " << pipeline->name << ": " << e.what() << std::पङ्क्त्यन्तः;
        क्षिप;
    }

    यदि (vk_instance.debug_utils_support) {
        vk::DebugUtilsObjectNameInfoEXT duoni;
        duoni.objectType = vk::ObjectType::ePipeline;
        duoni.pObjectName = pipeline->name.c_str();
        duoni.objectHandle = /*reinterpret_cast*/(uint64_t)(static_cast<VkPipeline>(pipeline->pipeline));
        vk_instance.pfn_vk_समुच्चयः_Debug_Utils_Object_Name_EXT(device->device, &static_cast<VkDebugUtilsObjectNameInfoEXT &>(duoni));
    }

    यदि (device->pipeline_executable_properties_support) {
        vk::PipelineExecutableInfoKHR executableInfo;
        executableInfo.pipeline = pipeline->pipeline;

        स्वचालित statistics = device->device.getPipelineExecutableStatisticsKHR(executableInfo);

        तार्किक print_stats = !vk_pipeline_stats_filter.रिक्तानि() &&
                           pipeline->name.find(vk_pipeline_stats_filter) != std::string::npos;
        यदि (print_stats) {
            std::cerr << "ggml_vulkan: pipeline stats for " << pipeline->name << ":" << std::पङ्क्त्यन्तः;
        }

        प्रति (स्वचालित & s : statistics) {
            यदि (print_stats) {
                std::cerr << "ggml_vulkan:   " << s.name.दत्त() << ": ";
                विकल्प (s.format) {
                    पक्ष vk::PipelineExecutableStatisticFormatKHR::eBool32:
                        std::cerr << (s.value.b32 ? "true" : "false");
                        भङ्ग;
                    पक्ष vk::PipelineExecutableStatisticFormatKHR::eInt64:
                        std::cerr << s.value.i64;
                        भङ्ग;
                    पक्ष vk::PipelineExecutableStatisticFormatKHR::eUint64:
                        std::cerr << s.value.u64;
                        भङ्ग;
                    पक्ष vk::PipelineExecutableStatisticFormatKHR::eFloat64:
                        std::cerr << s.value.f64;
                        भङ्ग;
                }
                std::cerr << std::पङ्क्त्यन्तः;
            }
            // "Register Count" is reported by NVIDIA drivers.
            यदि (वाक्यं_तुलय(s.name, "Register Count") == 0) {
                VK_LOG_DEBUG(pipeline->name << " " << s.name << ": " << s.value.u64 << " registers");
                pipeline->पञ्जिकाकोशः_गणनम् = (uint32_t)s.value.u64;
            }
        }
    }

    {
        std::lock_guard<std::mutex> guard(device->compile_mutex);
        device->all_pipelines.push_back(pipeline);
        pipeline->compiled = सत्य;
        pipeline->compile_pending = असत्य;
    }
    device->compile_cv.notify_all();
}

शून्य ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_pipeline_destroy_pipeline(" << pipeline->name << ")");
    device.destroyPipelineLayout(pipeline->layout);

    device.destroyShaderModule(pipeline->shader_module);

    device.destroyPipeline(pipeline->pipeline);
}

शून्य ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n) {
    VK_LOG_DEBUG("ggml_pipeline_request_descriptor_sets(" << pipeline->name << ", " << n << ")");
    ctx->pipeline_descriptor_समुच्चयः_requirements += n;
    यदि (!pipeline->compiled) {
        ggml_vk_आरोपय_shaders(ctx->device, pipeline);
    }
    ggml_pipeline_allocate_descriptor_sets(ctx);
}

शून्य ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx) {

    यदि (ctx->descriptor_sets.size() >= ctx->pipeline_descriptor_समुच्चयः_requirements) {
        // Enough descriptors are available
        प्रत्यागम;
    }

    vk_device& device = ctx->device;

    // Grow by 50% to avoid frequent allocations
    uint32_t needed = std::max(3 * ctx->descriptor_sets.size() / 2, size_t{ctx->pipeline_descriptor_समुच्चयः_requirements});
    uint32_t गन्तव्यम्_alloc = needed - ctx->descriptor_sets.size();
    uint32_t pool_remaining = VK_DEVICE_DESCRIPTOR_POOL_आकार - ctx->descriptor_sets.size() % VK_DEVICE_DESCRIPTOR_POOL_आकार;
    uint32_t pool_idx = ctx->descriptor_sets.size() / VK_DEVICE_DESCRIPTOR_POOL_आकार;

    यावत् (गन्तव्यम्_alloc > 0) {
        ध्रुव uint32_t alloc_गणनम् = std::min(pool_remaining, गन्तव्यम्_alloc);
        गन्तव्यम्_alloc -= alloc_गणनम्;
        pool_remaining = VK_DEVICE_DESCRIPTOR_POOL_आकार;

        यदि (pool_idx >= ctx->descriptor_pools.size()) {
            vk::Descriptor_Pool_आकार descriptor_pool_आकार(vk::DescriptorType::e_Storage_बफर, MAX_PARAMETER_गणनम् * VK_DEVICE_DESCRIPTOR_POOL_आकार);
            vk::DescriptorPoolCreateInfo descriptor_pool_create_info({}, VK_DEVICE_DESCRIPTOR_POOL_आकार, descriptor_pool_आकार);
            ctx->descriptor_pools.push_back(device->device.createDescriptorPool(descriptor_pool_create_info));
        }

        std::vector<vk::Descriptor_समुच्चयः_Layout> layouts(alloc_गणनम्);
        प्रति (uint32_t i = 0; i < alloc_गणनम्; i++) {
            layouts[i] = device->dsl;
        }
        vk::Descriptor_समुच्चयः_Allocate_Info descriptor_समुच्चयः_alloc_info(ctx->descriptor_pools[pool_idx], alloc_गणनम्, layouts.दत्त());
        std::vector<vk::Descriptor_समुच्चयः> sets = device->device.allocateDescriptorSets(descriptor_समुच्चयः_alloc_info);
        ctx->descriptor_sets.insert(ctx->descriptor_sets.end(), sets.begin(), sets.end());

        pool_idx++;
    }
    ctx->descriptor_समुच्चयः_bindings.resize(ctx->descriptor_sets.size());
}

स्थिर vk_command_बफर* ggml_vk_create_cmd_बफर(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_create_cmd_buffer()");
    vk::Command_बफर_Allocate_Info command_बफर_alloc_info(
        p.pool,
        vk::Command_बफर_Level::ePrimary,
        1);
    ध्रुव std::vector<vk::Command_बफर> cmd_buffers = device->device.allocateCommandBuffers(command_बफर_alloc_info);
    p.cmd_buffers.push_back({ cmd_buffers.front(), 0, सत्य });
    प्रत्यागम &p.cmd_buffers[p.cmd_buffers.size()-1];
}

शून्य ggml_vk_submit(vk_context& ctx, vk::Fence fence) {
    यदि (ctx->seqs.रिक्तानि()) {
        यदि (fence) {
            ctx->p->q->handle->submit({}, fence);
        }
        प्रत्यागम;
    }
    VK_LOG_DEBUG("ggml_vk_submit(" << ctx << ", " << fence << ")");

    std::vector<std::vector<uint64_t>> tl_wait_vals;
    std::vector<std::vector<uint64_t>> tl_signal_vals;
    std::vector<std::vector<vk::Semaphore>> tl_wait_semaphores;
    std::vector<std::vector<vk::Semaphore>> tl_signal_semaphores;
    std::vector<vk::TimelineSemaphoreSubmitInfo> tl_submit_infos;
    std::vector<vk::SubmitInfo> submit_infos;
    पूर्णाङ्क idx = -1;
    std::vector<std::vector<vk::PipelineStageFlags>> stage_flags;

    size_t reserve = 0;

    प्रति (ध्रुव स्वचालित& sequence : ctx->seqs) {
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

    प्रति (ध्रुव स्वचालित& sequence : ctx->seqs) {
        प्रति (ध्रुव स्वचालित& submission : sequence) {
            stage_flags.push_back({});
            idx++;
            tl_wait_vals.push_back({});
            tl_wait_semaphores.push_back({});
            tl_signal_vals.push_back({});
            tl_signal_semaphores.push_back({});
            प्रति (size_t i = 0; i < submission.wait_semaphores.size(); i++) {
                stage_flags[idx].push_back(ctx->p->q->stage_flags);
                tl_wait_vals[idx].push_back(submission.wait_semaphores[i].value);
                tl_wait_semaphores[idx].push_back(submission.wait_semaphores[i].s);
            }
            प्रति (size_t i = 0; i < submission.signal_semaphores.size(); i++) {
                tl_signal_vals[idx].push_back(submission.signal_semaphores[i].value);
                tl_signal_semaphores[idx].push_back(submission.signal_semaphores[i].s);
            }
            tl_submit_infos.push_back({
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_vals[idx].दत्त(),
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_vals[idx].दत्त(),
            });
            tl_submit_infos[idx].sType = vk::StructureType::eTimelineSemaphoreSubmitInfo;
            tl_submit_infos[idx].p_अग्र = शून्यसूचकः;
            vk::SubmitInfo si{
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_semaphores[idx].दत्त(),
                stage_flags[idx].दत्त(),
                1,
                &submission.बफर->buf,
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_semaphores[idx].दत्त(),
            };
            si.समुच्चयः_PNext(&tl_submit_infos[idx]);
            submit_infos.push_back(si);
        }
    }

    ctx->p->q->handle->submit(submit_infos, fence);

    ctx->seqs.clear();
}

uint32_t ggml_vk_find_queue_family_सूचकांक(std::vector<vk::QueueFamilyProperties>& queue_family_props, ध्रुव vk::QueueFlags& required, ध्रुव vk::QueueFlags& avoid, int32_t compute_सूचकांक, uint32_t min_अङ्कः_queues) {
    VK_LOG_DEBUG("ggml_vk_find_queue_family_index()");
    ध्रुव uint32_t qfsize = queue_family_props.size();

    // Try with avoid preferences first
    प्रति (uint32_t i = 0; i < qfsize; i++) {
        यदि (queue_family_props[i].queue_गणनम् >= min_अङ्कः_queues && (compute_सूचकांक < 0 || i != (uint32_t) compute_सूचकांक) && queue_family_props[i].queueFlags & required && !(queue_family_props[i].queueFlags & avoid)) {
            प्रत्यागम i;
        }
    }

    // Fall back to only required
    प्रति (size_t i = 0; i < qfsize; i++) {
        यदि (queue_family_props[i].queue_गणनम् >= min_अङ्कः_queues && (compute_सूचकांक < 0 || i != (uint32_t) compute_सूचकांक) && queue_family_props[i].queueFlags & required) {
            प्रत्यागम i;
        }
    }

    // Fall back to reusing compute queue
    प्रति (size_t i = 0; i < qfsize; i++) {
        यदि (queue_family_props[i].queue_गणनम् >= min_अङ्कः_queues && queue_family_props[i].queueFlags & required) {
            प्रत्यागम i;
        }
    }

    // Fall back to ignoring min_num_queries
    प्रति (size_t i = 0; i < qfsize; i++) {
        यदि (queue_family_props[i].queueFlags & required) {
            प्रत्यागम i;
        }
    }

    // All commands that are allowed on a queue that supports transfer operations are also allowed on a queue that supports either graphics or compute operations.
    // Thus, if the capabilities of a queue family include VK_QUEUE_GRAPHICS_BIT or VK_QUEUE_COMPUTE_BIT, then reporting the VK_QUEUE_TRANSFER_BIT capability separately for that queue family is optional.
    यदि (compute_सूचकांक >= 0) {
        प्रत्यागम compute_सूचकांक;
    }

    std::cerr << "ggml_vulkan: No suitable queue family index found." << std::पङ्क्त्यन्तः;

    प्रति(स्वचालित &q_family : queue_family_props) {
        std::cerr << "Queue number: "  + std::to_string(q_family.queue_गणनम्) << " flags: " + to_string(q_family.queueFlags) << std::पङ्क्त्यन्तः;
    }
    abort();
}

std::unique_ptr<vk_queue> ggml_vk_create_queue(vk_device& device, uint32_t queue_family_सूचकांक, uint32_t queue_सूचकांक, vk::PipelineStageFlags&& stage_flags, तार्किक transfer_only) {
    VK_LOG_DEBUG("ggml_vk_create_queue()");
    std::lock_guard<std::recursive_mutex> guard(device->mutex);

    स्वचालित q = std::make_unique<vk_queue>();
    q->queue_family_सूचकांक = queue_family_सूचकांक;
    q->transfer_only = transfer_only;

    std::shared_ptr<vk_queue_handle> h;
    vk::DeviceQueueInfo2 queue_info2{};
    queue_info2.queue_Family_सूचकांक = queue_family_सूचकांक;
    queue_info2.queue_सूचकांक_2 = queue_सूचकांक;

    यदि (device->has_internally_synchronized_queues) {
        h = std::make_shared<vk_queue_handle_unsynchronized>();
        queue_info2.flags = eInternallySynchronizedKHR;
    } अन्यथा {
        h = std::make_shared<vk_queue_handle_synchronized>();
    }

    h->queue = device->device.getQueue2(queue_info2);
    h->device = device;
    // Avoid concurrent submissions on NVIDIA due to driver bug.
    यदि (device->vendor_id == VK_VENDOR_ID_NVIDIA) {
        h->device_submit_mutex = &device->queue_submit_mutex;
    }
    q->handle = h;

    q->cmd_pool.init(device, q.get());

    q->stage_flags = stage_flags;
    प्रत्यागम q;
}

std::unique_ptr<vk_queue> ggml_vk_create_aliased_queue(vk_device& device, ध्रुव std::unique_ptr<vk_queue>& source) {
    std::lock_guard<std::recursive_mutex> guard(device->mutex);
    स्वचालित q = std::make_unique<vk_queue>();
    q->handle = source->handle;
    q->queue_family_सूचकांक = source->queue_family_सूचकांक;
    q->stage_flags = source->stage_flags;
    q->transfer_only = source->transfer_only;
    q->cmd_pool.init(device, q.get());
    प्रत्यागम q;
}

vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p) {
    vk_context परिणाम = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_context(" << परिणाम << ")");
    ctx->gc.contexts.emplace_back(परिणाम);
    परिणाम->p = &p;
    प्रत्यागम परिणाम;
}

vk_context ggml_vk_create_temporary_context(vk_command_pool& p) {
    vk_context परिणाम = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_temporary_context(" << परिणाम << ")");
    परिणाम->p = &p;
    प्रत्यागम परिणाम;
}

स्थिर vk_semaphore * ggml_vk_create_द्विमान_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::e_द्विमान, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.समुच्चयः_PNext(&tci);
    vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
    ctx->gc.semaphores.push_back({ semaphore, 0 });
    प्रत्यागम &ctx->gc.semaphores[ctx->gc.semaphores.size() - 1];
}

स्थिर vk_semaphore * ggml_vk_create_timeline_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    यदि (ctx->semaphore_idx >= ctx->gc.tl_semaphores.size()) {
        vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
        vk::SemaphoreCreateInfo ci{};
        ci.समुच्चयः_PNext(&tci);
        vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
        ctx->gc.tl_semaphores.push_back({ semaphore, 0 });
    }
    प्रत्यागम &ctx->gc.tl_semaphores[ctx->semaphore_idx++];
}

स्थिर vk::Event ggml_vk_create_event(ggml_backend_vk_context * ctx) {
    यदि (ctx->event_idx >= ctx->gc.events.size()) {
        ctx->gc.events.push_back(ctx->device->device.createEvent({}));
    }
    प्रत्यागम ctx->gc.events[ctx->event_idx++];
}

शून्य ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_command_pool_cleanup()");

    // Requires command buffers to be done
    device->device.resetCommandPool(p.pool);
    // Don't clear the command buffers and mark them as not in use.
    // This allows us to reuse them
    प्रति (स्वचालित& cmd_बफर : p.cmd_buffers) {
        cmd_बफर.अन्तर्_use = असत्य;
    }
}

शून्य ggml_vk_queue_command_pools_cleanup(vk_device& device) {
    VK_LOG_DEBUG("ggml_vk_queue_command_pools_cleanup()");

    // Arbitrary frequency to cleanup/reuse command buffers
    स्थिर constexpr uint32_t cleanup_frequency = 10;

    यदि (device->compute_queue && device->compute_queue->cmd_pool.buffers_अन्तर्_use() >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->compute_queue->cmd_pool);
    }
    यदि (device->transfer_queue && device->transfer_queue->cmd_pool.buffers_अन्तर्_use() >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->transfer_queue->cmd_pool);
    }
}

vk_subbuffer ggml_vk_subbuffer(ध्रुव ggml_backend_vk_context* ctx, ध्रुव vk_बफर& buf, size_t विस्थापनम्) {
    प्रत्यागम { buf, विस्थापनम्, ggml_vk_get_max_बफर_range(ctx, buf, विस्थापनम्) };
}

शून्य ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx) {
    VK_LOG_DEBUG("ggml_vk_sync_buffers()");

    ध्रुव तार्किक transfer_queue = subctx->p->q->transfer_only;

    यदि (ctx) {
        ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_विभज_k_need_sync = असत्य;
    }

    subctx->s->बफर->buf.pipelineBarrier(
        subctx->p->q->stage_flags,
        subctx->p->q->stage_flags,
        {},
        { {
          { !transfer_queue ? (vk::Access_Flag_द्वयङ्काः::eShaderRead | vk::Access_Flag_द्वयङ्काः::eShaderWrite | vk::Access_Flag_द्वयङ्काः::eTransferRead | vk::Access_Flag_द्वयङ्काः::eTransferWrite) : (vk::Access_Flag_द्वयङ्काः::eTransferRead | vk::Access_Flag_द्वयङ्काः::eTransferWrite) },
          { !transfer_queue ? (vk::Access_Flag_द्वयङ्काः::eShaderRead | vk::Access_Flag_द्वयङ्काः::eShaderWrite | vk::Access_Flag_द्वयङ्काः::eTransferRead | vk::Access_Flag_द्वयङ्काः::eTransferWrite) : (vk::Access_Flag_द्वयङ्काः::eTransferRead | vk::Access_Flag_द्वयङ्काः::eTransferWrite) }
        } },
        {},
        {}
    );
}

स्थिर शून्य ggml_vk_reset_event(vk_context& ctx, vk::Event& event) {
    VK_LOG_DEBUG("ggml_vk_set_event()");

    ctx->s->बफर->buf.resetEvent(
        event,
        ctx->p->q->stage_flags
    );
}

शून्य ggml_vk_समुच्चयः_event(vk_context& ctx, vk::Event& event) {
    VK_LOG_DEBUG("ggml_vk_set_event()");

    ctx->s->बफर->buf.समुच्चयः_Event(
        event,
        ctx->p->q->stage_flags
    );
}

शून्य ggml_vk_wait_events(vk_context& ctx, std::vector<vk::Event>&& events) {
    VK_LOG_DEBUG("ggml_vk_wait_events()");
    यदि (events.रिक्तानि()) {
        प्रत्यागम;
    }

    ctx->s->बफर->buf.waitEvents(
        events,
        ctx->p->q->stage_flags,
        ctx->p->q->stage_flags,
        {},
        {},
        {}
    );
}

स्थिर vk_fa_tuning_params get_fa_tuning_params_scalar(ध्रुव vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_पङ्क्तिकाः, uint32_t n_kv, ggml_type k_type, ggml_type v_type, तार्किक f32acc) {

    vk_fa_tuning_params परिणाम{};
    परिणाम.path = FA_SCALAR;

    यदि (device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Disable subgroup use due to performance issues when enforcing subgroup sizes
        परिणाम.subgroup_आकार = 32;
        परिणाम.disable_subgroups = सत्य;
    } अन्यथा यदि (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN) {
        परिणाम.subgroup_आकार = n_पङ्क्तिकाः < 4 ? 32 : device->subgroup_आकार;
    } अन्यथा {
        परिणाम.subgroup_आकार = device->subgroup_आकार;
    }

    // Row split splits the workgroup so that synchronization only has to happen within subgroups, which avoids barriers
    uint32_t row_विभज_max_hsk = 64;
    यदि (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN && !device->uma) {
        row_विभज_max_hsk = n_पङ्क्तिकाः <= 8 ? 64 : 128;
    }
    परिणाम.row_विभज = (n_पङ्क्तिकाः < 4 || hsk <= row_विभज_max_hsk) ? 1 : 4;

    यदि (परिणाम.subgroup_आकार > 32 && (n_पङ्क्तिकाः < 4 || hsk < (परिणाम.row_विभज == 1 ? 128 : 64))) {
        परिणाम.workgroup_आकार = परिणाम.subgroup_आकार * 2;
    } अन्यथा {
        परिणाम.workgroup_आकार = परिणाम.subgroup_आकार * 4;
    }

    ध्रुव uint32_t D = hsk | hsv;

    ध्रुव तार्किक reduce_खण्डम्_पङ्क्तिकाः = D & 8 || n_kv < 1024 || device->vendor_id == VK_VENDOR_ID_INTEL;

    यदि (n_पङ्क्तिकाः == 1) {
        परिणाम.खण्डम्_पङ्क्तिकाः = 1;
        परिणाम.खण्डम्_स्तम्भसङ्ख्या = 64;
    } अन्यथा {
        // row_split 1 means higher register use per row, so block size has to be adjusted
        यदि (परिणाम.row_विभज == 1) {
            परिणाम.खण्डम्_पङ्क्तिकाः = n_पङ्क्तिकाः == 2 ? 2 : ((n_पङ्क्तिकाः <= 4 || reduce_खण्डम्_पङ्क्तिकाः) ? 4 : 8);
        } अन्यथा {
            परिणाम.खण्डम्_पङ्क्तिकाः = n_पङ्क्तिकाः <= 4 ? 4 : ((n_पङ्क्तिकाः <= 8 || reduce_खण्डम्_पङ्क्तिकाः) ? 8 : 16);
        }

        परिणाम.खण्डम्_स्तम्भसङ्ख्या = (D & 8) ? 64 : 32;
    }

    ध्रुव uint32_t D_lsb = D ^ (D & (D-1));  // extract lowest set bit

    परिणाम.d_विभज = std::min(std::min(परिणाम.subgroup_आकार, 8u), D_lsb / 4);

    परिणाम.shmem_staging = (device->vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) ? 1 : 0;

    यदि (!reduce_खण्डम्_पङ्क्तिकाः && !ggml_vk_flash_attn_scalar_shmem_support(device, परिणाम, hsk, hsv, f32acc, k_type, v_type)) {
        परिणाम.खण्डम्_पङ्क्तिकाः /= 2;
    }

    // On AMD RDNA, for small head sizes and big batch size the shader uses few registers, so too many subgroups get scheduled
    // at once and end up thrashing the cache. Fix this by setting a large (unused) shmem buffer that reduces occupancy.
    // This targets an occupancy of 4 subgroups per SIMD.
    यदि (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.max_Compute_Shared_स्मृतिः_आकार == 65536) {
        यदि (device->architecture != AMD_GCN && n_पङ्क्तिकाः >= 64 && hsk <= 128) {
            // 30kb target for hsk > 64, 26kb for <= 64 due to smaller workgroup size
            // Values are guessed, tested on RDNA2
            परिणाम.सीमा_occupancy_shmem = (hsk <= 64 ? 26 : 30) * 1024 / 4 / 4;
        } अन्यथा यदि (device->architecture == AMD_GCN && n_पङ्क्तिकाः <= 8 && hsk >= 256) {
            // Same thing for GCN, with an occupancy target of 2 subgroups per SIMD.
            // Here low-batch FA with large head size is affected.
            // n_rows < 4 switch because workgroup size switches from 128 to 256 there.
            परिणाम.सीमा_occupancy_shmem = (n_पङ्क्तिकाः < 4 ? 14 : 26) * 1024 / 4 / 4;
        }
    }

    प्रत्यागम परिणाम;
}

स्थिर vk_fa_tuning_params get_fa_tuning_params_coopmat1(ध्रुव vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_पङ्क्तिकाः, uint32_t n_kv, ggml_type k_type, ggml_type v_type, तार्किक f32acc) {
    GGML_अप्रयुक्तम्(n_पङ्क्तिकाः);
    GGML_अप्रयुक्तम्(n_kv);
    GGML_अप्रयुक्तम्(k_type);
    GGML_अप्रयुक्तम्(v_type);
    GGML_अप्रयुक्तम्(f32acc);

    vk_fa_tuning_params परिणाम{};
    परिणाम.path = FA_COOPMAT1;

    ध्रुव uint32_t D = hsk | hsv;

    ध्रुव uint32_t coopmat_खण्डम्_पङ्क्तिकाः = 16;
    ध्रुव uint32_t coopmat_खण्डम्_स्तम्भसङ्ख्या = 16;

    ध्रुव uint32_t अङ्कः_subgroups = 4;

    परिणाम.खण्डम्_पङ्क्तिकाः = coopmat_खण्डम्_पङ्क्तिकाः;
    परिणाम.खण्डम्_स्तम्भसङ्ख्या = coopmat_खण्डम्_स्तम्भसङ्ख्या * अङ्कः_subgroups;
    परिणाम.row_विभज = अङ्कः_subgroups;
    परिणाम.subgroup_आकार = device->subgroup_आकार;
    परिणाम.workgroup_आकार = अङ्कः_subgroups * परिणाम.subgroup_आकार;

    ध्रुव uint32_t D_lsb = D ^ (D & (D-1));  // extract lowest set bit
    परिणाम.d_विभज = std::min(std::min(परिणाम.subgroup_आकार, 8u), D_lsb / 4);

    परिणाम.shmem_staging = (device->vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) ? 1 : 0;

    प्रत्यागम परिणाम;
}

स्थिर vk_fa_tuning_params get_fa_tuning_params_coopmat2(ध्रुव vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_पङ्क्तिकाः, uint32_t n_kv, ggml_type k_type, ggml_type v_type, तार्किक f32acc) {
    GGML_अप्रयुक्तम्(n_kv);
    GGML_अप्रयुक्तम्(f32acc);

    vk_fa_tuning_params परिणाम{};
    परिणाम.path = FA_COOPMAT2;

    ध्रुव uint32_t D = hsk | hsv;

    ध्रुव तार्किक small_पङ्क्तिकाः = n_पङ्क्तिकाः < 32;

    यदि (small_पङ्क्तिकाः) {
        परिणाम.खण्डम्_पङ्क्तिकाः = 32;
        परिणाम.खण्डम्_स्तम्भसङ्ख्या = 32;
    } अन्यथा यदि (ggml_is_quantized(k_type) || ggml_is_quantized(v_type) || hsk >= 256 || hsv >= 256) {
        परिणाम.खण्डम्_पङ्क्तिकाः = (hsk >= 512 || hsv >= 512) ? 32 : 64;
        परिणाम.खण्डम्_स्तम्भसङ्ख्या = 32;
    } अन्यथा {
        परिणाम.खण्डम्_पङ्क्तिकाः = 64;
        परिणाम.खण्डम्_स्तम्भसङ्ख्या = 64;
    }

    परिणाम.subgroup_आकार = device->subgroup_आकार;
    परिणाम.workgroup_आकार = (small_पङ्क्तिकाः && (D % 32) == 0) ? 256 : 128;

    प्रत्यागम परिणाम;
}

vk_fa_tuning_params get_fa_tuning_params(ध्रुव vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_पङ्क्तिकाः, uint32_t n_kv, ggml_type k_type, ggml_type v_type, तार्किक f32acc) {
    Fa_संहिता_मार्गः path = device->coopmat2 ? FA_COOPMAT2 :
                      device->coopmat1_fa_support ? FA_COOPMAT1 : FA_SCALAR;

    यदि (path == FA_COOPMAT2 && k_type == GGML_TYPE_BF16 && !device->coopmat2_bf16_support) {
        path = FA_COOPMAT1;
    }
    यदि (path == FA_COOPMAT1 && k_type == GGML_TYPE_BF16 && !device->coopmat_bf16_support) {
        path = FA_SCALAR;
    }

    यदि (path == FA_COOPMAT1 && device->architecture == vk_device_architecture::NVIDIA_TURING) {
        // Nvidia compiler bug, see https://github.com/ggml-org/llama.cpp/pull/19075#issuecomment-3820716090
        path = FA_SCALAR;
    }

    यदि (path == FA_COOPMAT1) {
        तार्किक shape_ok = (f32acc && device->coopmat_support_16x16x16_f32acc) ||
                        (!f32acc && device->coopmat_support_16x16x16_f16acc);
        ध्रुव vk_fa_tuning_params params = get_fa_tuning_params_coopmat1(device, hsk, hsv, n_पङ्क्तिकाः, n_kv, k_type, v_type, f32acc);
        तार्किक shmem_ok = ggml_vk_flash_attn_coopmat_shmem_support(device, params, hsk, hsv, f32acc, k_type, v_type);

        यदि (!shape_ok || !shmem_ok) {
            path = FA_SCALAR;
        }
    }

    // scalar is faster than coopmat when N==1
    यदि (n_पङ्क्तिकाः == 1 && (path == FA_COOPMAT1 || path == FA_COOPMAT2)) {
        path = FA_SCALAR;
    }

    विकल्प (path) {
    पक्ष FA_SCALAR:
        प्रत्यागम get_fa_tuning_params_scalar(device, hsk, hsv, n_पङ्क्तिकाः, n_kv, k_type, v_type, f32acc);
    पक्ष FA_COOPMAT1:
        प्रत्यागम get_fa_tuning_params_coopmat1(device, hsk, hsv, n_पङ्क्तिकाः, n_kv, k_type, v_type, f32acc);
    पक्ष FA_COOPMAT2:
        प्रत्यागम get_fa_tuning_params_coopmat2(device, hsk, hsv, n_पङ्क्तिकाः, n_kv, k_type, v_type, f32acc);
    स्वभाव:
        क्षिप std::runtime_error("unsupported FaCodePath");
    }
}

vk_fa_pipeline_state get_fa_pipeline_state(ध्रुव vk_device& device, ध्रुव vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, तार्किक aligned, तार्किक f32acc,
                                                  तार्किक use_आवरणम्, तार्किक use_आवरणम्_opt, तार्किक use_logit_softcap, तार्किक use_sparse, ggml_type k_type, ggml_type v_type) {
    ध्रुव तार्किक old_amd_windows = device->vendor_id == VK_VENDOR_ID_AMD && device->driver_id == vk::DriverId::eAmdProprietary &&
                                 (device->architecture == AMD_GCN || device->architecture == AMD_RDNA1 || device->architecture == AMD_RDNA2);

    uint32_t flags = (use_आवरणम्_opt      ? 1 : 0) |
                     (use_आवरणम्          ? 2 : 0) |
                     (use_logit_softcap ? 4 : 0) |
                     (old_amd_windows   ? 8 : 0) |
                     (use_sparse        ? 16 : 0);

    ध्रुव uint32_t subgroup_आकार = params.disable_subgroups ? 0 : params.subgroup_आकार;

    प्रत्यागम vk_fa_pipeline_state{hsk, hsv, params.खण्डम्_पङ्क्तिकाः, params.खण्डम्_स्तम्भसङ्ख्या, params.d_विभज, params.row_विभज, params.shmem_staging, params.path, params.workgroup_आकार, subgroup_आकार, aligned, f32acc, flags, params.सीमा_occupancy_shmem, k_type, v_type};
}

स्थिर uint32_t fa_खण्डम्_अष्टकानि(ggml_type t) {
    यदि (t == GGML_TYPE_F32) {
        प्रत्यागम 16u;
    }
    प्रत्यागम (uint32_t) ggml_type_आकार(t);
}

स्थिर std::vector<uint32_t> get_fa_spec_constants(ध्रुव vk_fa_pipeline_state& state) {
    प्रत्यागम {
        /* 0 WorkGroupSize   */ state.workgroup_आकार,
        /* 1 Br              */ state.Br,
        /* 2 Bc              */ state.Bc,
        /* 3 HSK             */ state.HSK,
        /* 4 HSV             */ state.HSV,
        /* 5 Clamp           */ static_cast<uint32_t>(!state.aligned),
        /* 6 D_split         */ state.D_विभज,
        /* 7 row_split       */ state.row_विभज,
        /* 8 SubGroupSize    */ state.subgroup_आकार,
        /* 9 SHMEM_STAGING   */ state.shmem_staging ? 1u : 0u,
        /*10 Flags           */ state.flags,
        /*11 LIMIT_OCCUPANCY_SHMEM */ state.सीमा_occupancy_shmem,
        /*12 FaTypeK         */ static_cast<uint32_t>(state.k_type),
        /*13 FaTypeV         */ static_cast<uint32_t>(state.v_type),
        /*14 FaBlockBytesK   */ fa_खण्डम्_अष्टकानि(state.k_type),
        /*15 FaBlockBytesV   */ fa_खण्डम्_अष्टकानि(state.v_type),
    };
}

स्थिर तार्किक ggml_vk_matmul_shmem_support(ध्रुव vk_device& device, ध्रुव std::vector<uint32_t>& warptile, तार्किक mul_mat_id, ggml_type src0_type) {

    uint32_t lut_आकार = 0;
    विकल्प (src0_type) {
    पक्ष GGML_TYPE_IQ1_S:
    पक्ष GGML_TYPE_IQ1_M:
        // Regular matmul uses the compact uint16_t IQ1 grid; the expanded
        // uint32_t grid is only enabled for the q8_1/int-dot vector path.
        lut_आकार = 2*2048;
        भङ्ग;
    पक्ष GGML_TYPE_IQ2_XXS:
        lut_आकार = 8*256;
        भङ्ग;
    पक्ष GGML_TYPE_IQ2_XS:
        lut_आकार = 8*512;
        भङ्ग;
    पक्ष GGML_TYPE_IQ2_S:
        lut_आकार = 8*1024;
        भङ्ग;
    पक्ष GGML_TYPE_IQ3_XXS:
        lut_आकार = 4*256;
        भङ्ग;
    पक्ष GGML_TYPE_IQ3_S:
        lut_आकार = 4*512;
        भङ्ग;
    पक्ष GGML_TYPE_IQ4_NL:
    पक्ष GGML_TYPE_IQ4_XS:
    पक्ष GGML_TYPE_MXFP4:
        lut_आकार = 4*16;
        भङ्ग;
    पक्ष GGML_TYPE_NVFP4:
        // Same kvalues budget as MXFP4 plus ue4m3_fp32_lut[128] (types.glsl, DATA_A_NVFP4).
        lut_आकार = 4*16 + 128u * (uint32_t)परिमाण(भिन्न);
        भङ्ग;
    स्वभाव:
        भङ्ग;
    }

    // Needs to be kept up to date on shader changes
    // Needs to stay aligned with ggml_vk_mul_mm_spec.
    ध्रुव तार्किक intel_shmem_stride_pad_zero = device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
                                              device->driver_id == vk::DriverId::eIntelProprietaryWindows;
    ध्रुव uint32_t bank_conflict_विस्थापनम् = intel_shmem_stride_pad_zero ? 0 : (device->coopmat_support ? 8 : 1);
    ध्रुव uint32_t type_आकार = device->fp16 ? परिमाण(ggml_fp16_t) : परिमाण(भिन्न);
    ध्रुव uint32_t warps = warptile[0] / warptile[10];

    ध्रुव uint32_t आरोपय_bufs = (warptile[1] + warptile[2]) * (warptile[3] + bank_conflict_विस्थापनम्) * type_आकार;
    ध्रुव uint32_t mmid_row_ids = mul_mat_id ? (warptile[2] * 2 * परिमाण(uint16_t)) : 0;
    ध्रुव uint32_t coopmat_stage = device->coopmat_support ? warptile[7] * warptile[8] / warps * परिमाण(भिन्न) : 0;

    ध्रुव uint32_t योग_आकार = आरोपय_bufs + mmid_row_ids + coopmat_stage + lut_आकार;
    ध्रुव तार्किक supported = योग_आकार <= device->properties.limits.max_Compute_Shared_स्मृतिः_आकार;

    VK_LOG_DEBUG("ggml_vk_matmul_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", supported=" << supported);

    प्रत्यागम supported;
}

स्थिर तार्किक ggml_vk_matmul_int_shmem_support(ध्रुव vk_device& device, ध्रुव std::vector<uint32_t>& warptile, तार्किक mul_mat_id, ggml_type src0_type) {

    // FLOAT_TYPE in the shader is float16_t with fp16 support, otherwise float.
    ध्रुव uint32_t fp_आकार   = device->fp16 ? 2u : 4u;
    ध्रुव uint32_t fp_align  = fp_आकार;
    ध्रुव uint32_t fp2_आकार  = 2u * fp_आकार;
    ध्रुव uint32_t fp2_align = device->fp16 ? 4u : 8u;

    संरचना member { uint32_t size, align; };
    स्वचालित std430_आकार = [](std::initializer_list<member> members) {
        uint32_t off = 0, struct_align = 1;
        प्रति (ध्रुव स्वचालित &m : members) {
            off = (off + m.align - 1) & ~(m.align - 1);
            off += m.size;
            struct_align = std::max(struct_align, m.align);
        }
        प्रत्यागम (off + struct_align - 1) & ~(struct_align - 1);
    };

    uint32_t खण्डम्_a_आकार = 0;
    विकल्प (src0_type) {
        पक्ष GGML_TYPE_Q2_0:    खण्डम्_a_आकार = std430_आकार({{32, 4}, {fp_आकार,  fp_align}});                  भङ्ग; // qs[8] + dm
        पक्ष GGML_TYPE_Q4_0:    खण्डम्_a_आकार = std430_आकार({{16, 4}, {fp_आकार,  fp_align}});                  भङ्ग; // qs[16/4] + dm
        पक्ष GGML_TYPE_Q4_1:    खण्डम्_a_आकार = std430_आकार({{16, 4}, {fp2_आकार, fp2_align}});                 भङ्ग; // qs[16/4] + dm(vec2)
        पक्ष GGML_TYPE_Q5_0:    खण्डम्_a_आकार = std430_आकार({{16, 4}, {4, 4}, {fp_आकार,  fp_align}});          भङ्ग; // qs[16/4] + qh + dm
        पक्ष GGML_TYPE_Q5_1:    खण्डम्_a_आकार = std430_आकार({{16, 4}, {4, 4}, {fp2_आकार, fp2_align}});         भङ्ग; // qs[16/4] + qh + dm(vec2)
        पक्ष GGML_TYPE_Q8_0:    खण्डम्_a_आकार = std430_आकार({{32, 4}, {fp_आकार,  fp_align}});                  भङ्ग; // qs[8] + dm
        पक्ष GGML_TYPE_IQ4_XS:  खण्डम्_a_आकार = std430_आकार({{32, 4}, {fp_आकार,  fp_align}});                  भङ्ग; // qs[8] + d
        पक्ष GGML_TYPE_MXFP4:   खण्डम्_a_आकार = std430_आकार({{32, 4}, {fp_आकार,  fp_align}});                  भङ्ग; // qs[8] + d
        पक्ष GGML_TYPE_IQ4_NL:  खण्डम्_a_आकार = std430_आकार({{32, 4}, {fp_आकार,  fp_align}});                  भङ्ग; // qs[8] + d
        पक्ष GGML_TYPE_NVFP4:   खण्डम्_a_आकार = std430_आकार({{32, 4}, {fp2_आकार, fp2_align}});                 भङ्ग; // qs[8] + d_scales(vec2)
        पक्ष GGML_TYPE_Q2_K:    खण्डम्_a_आकार = std430_आकार({{ 8, 4}, {2, 2}, {fp2_आकार, fp2_align}});         भङ्ग; // qs[2] + scales(u8vec2) + dm(vec2)
        पक्ष GGML_TYPE_Q3_K:    खण्डम्_a_आकार = std430_आकार({{16, 4}, {fp2_आकार, fp2_align}});                 भङ्ग; // qs[4] + d_scales(vec2)
        पक्ष GGML_TYPE_Q4_K:    खण्डम्_a_आकार = std430_आकार({{16, 4}, {fp2_आकार, fp2_align}});                 भङ्ग; // qs[4] + dm(vec2)
        पक्ष GGML_TYPE_Q5_K:    खण्डम्_a_आकार = std430_आकार({{32, 4}, {fp2_आकार, fp2_align}});                 भङ्ग; // qs[8] + dm(vec2)
        पक्ष GGML_TYPE_Q6_K:    खण्डम्_a_आकार = std430_आकार({{32, 4}, {fp2_आकार, fp2_align}});                 भङ्ग; // qs[8] + d_scales(vec2)
        पक्ष GGML_TYPE_IQ3_S:   खण्डम्_a_आकार = std430_आकार({{32, 4}, {fp_आकार,  fp_align}});                  भङ्ग; // qs[8] + d
        स्वभाव:
            प्रत्यागम असत्य;
    }

    // IQ3_S also copies its 512-entry grid into shared memory (types.glsl, init_iq_shmem)
    ध्रुव uint32_t lut_आकार = (src0_type == GGML_TYPE_IQ3_S) ? 4*512 : 0;

    // block_b_cache: { int32_t qs[8]; FLOAT_TYPEV2 ds; }
    ध्रुव uint32_t खण्डम्_b_आकार = std430_आकार({{32, 4}, {fp2_आकार, fp2_align}});

    ध्रुव uint32_t BM = warptile[1];
    ध्रुव uint32_t BN = warptile[2];
    // mul_mmq.comp: BK_STEP=1 for MUL_MAT_ID, 4 otherwise.
    ध्रुव uint32_t BK_चरणः = mul_mat_id ? 1u : 4u;

    ध्रुव uint32_t buf_a_आकार = BM * BK_चरणः * खण्डम्_a_आकार;
    ध्रुव uint32_t buf_b_आकार = BN * BK_चरणः * खण्डम्_b_आकार;
    ध्रुव uint32_t mmid_row_ids = mul_mat_id ? (BN * 2u * (uint32_t)परिमाण(uint16_t)) : 0u;

    ध्रुव uint32_t योग_आकार = buf_a_आकार + buf_b_आकार + mmid_row_ids + lut_आकार;
    ध्रुव तार्किक supported = योग_आकार <= device->properties.limits.max_Compute_Shared_स्मृतिः_आकार;

    VK_LOG_DEBUG("ggml_vk_matmul_int_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", total=" << योग_आकार << ", supported=" << supported);

    प्रत्यागम supported;
}

स्थिर तार्किक ggml_vk_matmul_cm1_int_shmem_support(ध्रुव vk_device& device, ध्रुव std::vector<uint32_t>& warptile, तार्किक mul_mat_id, ggml_type src0_type) {

    तार्किक kscales2 = असत्य;    // two scale sets per block
    तार्किक has_dm   = असत्य;    // d+m as vec2 + b-side sum
    तार्किक has_kvalues = असत्य;
    विकल्प (src0_type) {
        पक्ष GGML_TYPE_Q4_0: पक्ष GGML_TYPE_Q5_0: पक्ष GGML_TYPE_Q8_0:
            भङ्ग;
        पक्ष GGML_TYPE_Q4_1: पक्ष GGML_TYPE_Q5_1:
        पक्ष GGML_TYPE_Q4_K: पक्ष GGML_TYPE_Q5_K:
            has_dm = सत्य;                          भङ्ग;
        पक्ष GGML_TYPE_IQ4_NL: पक्ष GGML_TYPE_IQ4_XS: पक्ष GGML_TYPE_MXFP4:
            has_kvalues = सत्य;                     भङ्ग;
        पक्ष GGML_TYPE_Q3_K: पक्ष GGML_TYPE_Q6_K:
            kscales2 = सत्य;                        भङ्ग;
        पक्ष GGML_TYPE_NVFP4:
            kscales2 = सत्य; has_kvalues = सत्य;    भङ्ग;
        स्वभाव:
            प्रत्यागम असत्य;
    }

    ध्रुव uint32_t BM         = warptile[1];
    ध्रुव uint32_t BN         = warptile[2];

    ध्रुव uint32_t BK      = 32;
    ध्रुव uint32_t BK_चरणः = mul_mat_id ? 2u : 4u;
    ध्रुव uint32_t QPITCH  = BK_चरणः * (BK / 4u) + 4u;
    ध्रुव uint32_t KSCALES = kscales2 ? 2u : 1u;

    uint32_t योग = 0;
    योग += BM * QPITCH * (uint32_t)परिमाण(uint32_t);   // buf_a_qs
    योग += BN * QPITCH * (uint32_t)परिमाण(uint32_t);   // buf_b_qs
    योग += has_dm ? (BM * BK_चरणः * 2u * (uint32_t)परिमाण(भिन्न))   // buf_a_dm (vec2)
                    : (BM * BK_चरणः * KSCALES * (uint32_t)परिमाण(भिन्न)); // buf_a_d
    योग += BN * BK_चरणः * (uint32_t)परिमाण(भिन्न);     // buf_b_d
    यदि (has_dm) {
        योग += BN * BK_चरणः * (uint32_t)परिमाण(भिन्न); // buf_b_s
    }
    यदि (has_kvalues) {
        योग += 16u * (uint32_t)परिमाण(int8_t);         // cm1_kvalues[16]
    }
    यदि (src0_type == GGML_TYPE_NVFP4 && !device->ocp_fp4) {
        योग += 128u * (uint32_t)परिमाण(भिन्न);         // ue4m3_fp32_lut[128]
    }
    यदि (mul_mat_id) {
        योग += BN * 2u * (uint32_t)परिमाण(uint16_t);   // row_ids[BN] (u16vec2)
    }

    ध्रुव तार्किक supported = योग <= device->properties.limits.max_Compute_Shared_स्मृतिः_आकार;

    VK_LOG_DEBUG("ggml_vk_matmul_cm1_int_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", total=" << योग << ", supported=" << supported);

    प्रत्यागम supported;
}

स्थिर ध्रुव std::unordered_map<std::string, uint32_t> rdna1_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
    {"argmax", 64}, {"mul_mat_vec", 64},
    {"mul_mat_vec_f16", 32}, {"mul_mat_vec_f32_f16", 32}
};

स्थिर ध्रुव std::unordered_map<std::string, uint32_t> rdna2_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
};

स्थिर std::vector<GpuPipelineConfig> gpu_pipeline_configs = {
    {
        vk_device_architecture::AMD_RDNA1,
        {
            rdna1_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_आकार
    },
    {
        vk_device_architecture::AMD_RDNA2,
        {
            rdna2_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_आकार
    },
};

uint32_t get_subgroup_आकार(ध्रुव std::string &pipeline_name, ध्रुव vk_device_architecture &arch) {
    प्रति (ध्रुव स्वचालित &config : gpu_pipeline_configs) {
        यदि (config.arch == arch) {
            स्वचालित pipIt = config.pipelines.find(pipeline_name);
            यदि (pipIt != config.pipelines.end()) {
                प्रत्यागम pipIt->second;
            }
            std::vector<std::pair<std::string, uint32_t>> sorted_pipelines(config.pipelines.begin(), config.pipelines.end());
            std::sort(sorted_pipelines.begin(), sorted_pipelines.end(),
                      [](ध्रुव स्वचालित &a, ध्रुव स्वचालित &b) { प्रत्यागम a.first.size() > b.first.size(); });
            प्रति (ध्रुव स्वचालित &प्रविष्टिः : sorted_pipelines) {
                यदि (pipeline_name.find(प्रविष्टिः.first) != std::string::npos) {
                    प्रत्यागम प्रविष्टिः.second;
                }
            }
            प्रत्यागम config.default_subgroup_आकार;
        }
    }
    प्रत्यागम 0; // If no matching configuration is found
}

स्थिर तार्किक ggml_vk_fa_type_needs_shmem(ggml_type type) {
    विकल्प (type) {
    पक्ष GGML_TYPE_IQ4_NL:
        प्रत्यागम सत्य;
    स्वभाव:
        प्रत्यागम असत्य;
    }
}

स्थिर तार्किक ggml_vk_fa_scalar_उपयुङ्क्ते_mmq(ध्रुव vk_device& device, ggml_type k_type, ggml_type v_type) {
#यदि defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    प्रत्यागम device->integer_dot_गुणनफलम् && device->subgroup_clustered &&
           !ggml_vk_fa_type_needs_shmem(v_type) &&
           (k_type == GGML_TYPE_Q4_0 || k_type == GGML_TYPE_Q4_1 ||
            k_type == GGML_TYPE_Q5_0 || k_type == GGML_TYPE_Q5_1 ||
            k_type == GGML_TYPE_Q8_0);
#अन्यथा
    GGML_अप्रयुक्तम्(device);
    GGML_अप्रयुक्तम्(k_type);
    GGML_अप्रयुक्तम्(v_type);
    प्रत्यागम असत्य;
#यदि_अन्तः
}

शून्य ggml_vk_आरोपय_shaders(vk_device& device, vk_pipeline requested) {
    VK_LOG_DEBUG("ggml_vk_load_shaders(" << device->name << ")");

    // some shaders have a minimum subgroup size
    ध्रुव uint32_t subgroup_आकार_8 = std::max(device->subgroup_आकार, 8u);
    ध्रुव uint32_t subgroup_आकार_16 = std::max(device->subgroup_आकार, 16u);
    ध्रुव uint32_t subgroup_आकार_32 = std::max(device->subgroup_आकार, 32u);

    // clamp WARP for l_/m_ warptiles so WM <= BM (breaks on subgroupSize > 64)
    ध्रुव uint32_t mm_warp_8  = std::min(subgroup_आकार_8,  64u);
    ध्रुव uint32_t mm_warp_16 = std::min(subgroup_आकार_16, 64u);

    ध्रुव uint32_t mul_mat_subgroup_आकार = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_आकार_control) ? device->subgroup_min_आकार : device->subgroup_आकार;
    ध्रुव uint32_t mul_mat_subgroup_आकार_8 = std::max(mul_mat_subgroup_आकार, 8u);
    ध्रुव uint32_t mul_mat_subgroup_आकार_16 = std::max(mul_mat_subgroup_आकार, 16u);
    ध्रुव uint32_t mul_mat_subgroup_आकार_32 = std::max(mul_mat_subgroup_आकार, 32u);
    ध्रुव uint32_t mul_mat_mm_warp_8  = std::min(mul_mat_subgroup_आकार_8,  64u);
    ध्रुव uint32_t mul_mat_mm_warp_16 = std::min(mul_mat_subgroup_आकार_16, 64u);

    ध्रुव तार्किक subgroup_min_आकार_16 = (!device->subgroup_आकार_control && device->subgroup_आकार >= 16) ||
                                      (device->subgroup_आकार_control && device->subgroup_max_आकार >= 16);

    // mulmat
    // Warptile layout (indices match mul_mm.comp constantIDs):
    //   [0..9]  : BLOCK_SIZE, BM, BN, BK, WM, WN, WMITER, TM, TN, TK
    //   [10]    : WARP / required_subgroup_size (read via WARP_SIZE_IDX)
    स्थिर constexpr size_t WARP_आकार_IDX = 10;
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
    ध्रुव तार्किक cm1_use_wave32 = device->vendor_id == VK_VENDOR_ID_AMD &&
                                device->subgroup_आकार_control &&
                                device->subgroup_min_आकार <= 32 && device->subgroup_max_आकार >= 32;
    ध्रुव uint32_t cm1_sg = cm1_use_wave32 ? 32 : device->subgroup_आकार;

    vk_pipeline wait_pipeline;
    CompileTask claimed_task {};
    तार्किक has_claimed_task = असत्य;

    // The rest of the walk reads and writes shared device state, so hold the
    // lock until we're done deciding what to compile.
    std::unique_lock<std::mutex> compile_तालकम्(device->compile_mutex);

    यदि (device->coopmat2) {
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
        ध्रुव uint32_t mmqid_bk = device->coopmat2_decode_सदिशम् ? 64u : 32u;
        l_warptile_mmqid = { 256, 128, 128, mmqid_bk, 1 };
        m_warptile_mmqid = { 256, 128, 64,  mmqid_bk, 0 };
        s_warptile_mmqid = { 256, 128, 64,  mmqid_bk, 0 };
        l_mmqid_wg_denoms = { 128, 128, 1 };
        m_mmqid_wg_denoms = { 128, 64, 1 };
        s_mmqid_wg_denoms = { 128, 64, 1 };

        l_align = 128;
        m_align =  64;
        s_align =  32;
    } अन्यथा {
        // Matrix cores require different warp group sizes
        ध्रुव uint32_t tm_l = device->coopmat_support ? device->coopmat_m : 4;
        ध्रुव uint32_t tm_m = device->coopmat_support ? device->coopmat_m : 4;
        ध्रुव uint32_t tm_s = device->coopmat_support ? device->coopmat_m : 2;
        ध्रुव uint32_t tn_l = device->coopmat_support ? device->coopmat_n : 4;
        ध्रुव uint32_t tn_m = device->coopmat_support ? device->coopmat_n : 2;
        ध्रुव uint32_t tn_s = device->coopmat_support ? device->coopmat_n : 2;
        ध्रुव uint32_t tk_l = device->coopmat_support ? device->coopmat_k : 1;
        ध्रुव uint32_t tk_m = device->coopmat_support ? device->coopmat_k : 1;
        ध्रुव uint32_t tk_s = device->coopmat_support ? device->coopmat_k : 1;

        ध्रुव uint32_t itm = device->coopmat_int_m;
        ध्रुव uint32_t itn = device->coopmat_int_n;
        ध्रुव uint32_t itk = device->coopmat_int_k;

        ध्रुव uint32_t s_warptile_wm = device->subgroup_आकार == 8 ? 8 : 32;

        l_warptile = { 128,             128, 128, 16, mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mm_warp_8 };
        m_warptile = { 128,              64,  64, 16, mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mm_warp_8 };
        s_warptile = { subgroup_आकार_32, 32,  32, 16, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, subgroup_आकार_8 };

        l_warptile_mmq = { 128,             128, 128, 32, mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mm_warp_8 };
        m_warptile_mmq = { 128,              64,  64, 32, mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mm_warp_8 };
        s_warptile_mmq = { subgroup_आकार_32, 32,  32, 32, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, subgroup_आकार_8 };

        // Integer MMQ has a smaller shared memory profile, but heavier register use
        l_warptile_mmq_int = { 128,             128, 128, 32, mm_warp_8 * 2, 64, 2, 4, 4, 1, mm_warp_8 };
        m_warptile_mmq_int = { 128,              64,  64, 32, mm_warp_8,     32, 2, 2, 2, 1, mm_warp_8 };
        s_warptile_mmq_int = { subgroup_आकार_32, 32,  32, 32, s_warptile_wm, 32, 2, 2, 1, 1, subgroup_आकार_8 };

        ध्रुव स्वचालित cm1_bs = [cm1_sg](uint32_t bm, uint32_t bn) {
            प्रत्यागम cm1_sg * (bm / std::min(cm1_sg, bm)) * (bn / 32);
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
        s_warptile_mmq_int_k = { subgroup_आकार_32,   32,  32, 32, s_warptile_wm, 32, 1, 2, 1, 1, subgroup_आकार_8 };

        l_warptile_id = { 128,                      128, 128, 16, mul_mat_mm_warp_16 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_mm_warp_16 };
        m_warptile_id = { 128,                       64,  64, 16, mul_mat_mm_warp_16,     32, 2, tm_m, tn_m, tk_m, mul_mat_mm_warp_16 };
        s_warptile_id = { mul_mat_subgroup_आकार_16,  32,  32, 16, s_warptile_wm,          32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_आकार_16 };

        l_warptile_mmqid = { 128,                       128, 128, 32, mul_mat_mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_mm_warp_8 };
        m_warptile_mmqid = { 128,                        64,  64, 32, mul_mat_mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mul_mat_mm_warp_8 };
        s_warptile_mmqid = { mul_mat_subgroup_आकार_32,   32,  32, 32, s_warptile_wm,         32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_आकार_8 };

        l_warptile_mmqid_int = { 128,                       128, 128, 32, mul_mat_mm_warp_8 * 2, 64, 2, 4, 4, 1, mul_mat_mm_warp_8 };
        m_warptile_mmqid_int = { 128,                        64,  64, 32, mul_mat_mm_warp_8,     32, 2, 2, 2, 1, mul_mat_mm_warp_8 };
        s_warptile_mmqid_int = { mul_mat_subgroup_आकार_32,   32,  32, 32, s_warptile_wm,         32, 2, 2, 1, 1, mul_mat_subgroup_आकार_8 };

        l_warptile_mmqid_int_k = { 128,                     128, 128, 32, mul_mat_mm_warp_16 * 2, 64, 1, 4, 4, 1, mul_mat_mm_warp_16 };
        m_warptile_mmqid_int_k = { 128,                      64,  64, 32, mul_mat_mm_warp_16,     32, 1, 2, 2, 1, mul_mat_mm_warp_16 };
        s_warptile_mmqid_int_k = { mul_mat_subgroup_आकार_32, 32,  32, 32, s_warptile_wm,          32, 1, 2, 1, 1, mul_mat_subgroup_आकार_16 };

        // chip specific tuning
        यदि ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
            m_warptile_mmq = m_warptile_mmq_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
            m_warptile_mmqid = m_warptile_mmqid_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
        } अन्यथा यदि (device->vendor_id == VK_VENDOR_ID_AMD && device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary) {
            // This is intentionally using tx_m values, slight performance increase
            l_warptile = { 256, 128, 128, 16, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
            l_warptile_mmq = l_warptile_mmq_int = { 256, 128, 128, 32, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
            l_warptile_mmq_int_k = { 256, 128, 128, 32, mm_warp_16, 64, 1, 4, 2, 1, mm_warp_16 };
        } अन्यथा यदि (device->vendor_id == VK_VENDOR_ID_QUALCOMM && device->coopmat_support) {
            m_warptile     = { 64, 64, 64, 16, 64, 64, 1, tm_l, tn_l, tk_l, 64 };
            m_warptile_mmq = { 64, 64, 64, 32, 64, 64, 1, tm_m, tn_m, tk_m, 64 };
        }

        l_mmq_wg_denoms = l_wg_denoms = {128, 128, 1 };
        m_mmq_wg_denoms = m_wg_denoms = { 64,  64, 1 };
        s_mmq_wg_denoms = s_wg_denoms = { 32,  32, 1 };
        l_align = 128;
        m_align =  64;
        s_align =  32;

        यदि (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support) {
            // Xe1/Xe2/Xe3 with coopmat enabled - warptile performance tuning
            l_warptile = { 512, 128, 128, 16, mm_warp_8, 32, 2, tm_l, tn_l, tk_l, mm_warp_8 };
            यदि (device->architecture == INTEL_XE1) {
                l_warptile_mmq  = { 512, 256, 128, 32, 32, 32, 2, tm_l, tn_l, tk_l, 16 };
                l_mmq_wg_denoms = { 256, 128, 1 };
                l_align         = 32;  //set as BK
            } अन्यथा {
                l_warptile_mmq  = { 512, 128, 256, 32, 32, 32, 2, tm_l, tn_l, tk_l, 16 };
                l_mmq_wg_denoms = { 128, 256, 1 };
                l_align         = 32;  //set as BK
            }
        }

        ध्रुव तार्किक use_cm1_int = device->coopmat_int_support &&
                                 (device->architecture == AMD_RDNA3 || device->architecture == AMD_RDNA4);

        प्रति (uint32_t i = 0; i < GGML_TYPE_गणनम्; ++i) {
            ggml_type t = (ggml_type)i;
            // Disable medium and large matrix multiplication if not enough shared memory is available
            // Check mmq warptiles as the largest configuration
            // Throw an error if not enough for any matrix multiplication is available
            यदि (!ggml_vk_matmul_shmem_support(device, s_warptile_mmq, असत्य, t)) {
                std::cerr << "ggml_vulkan: Error: Shared memory size too small for matrix multiplication." << std::पङ्क्त्यन्तः;
                क्षिप std::runtime_error("Shared memory size too small for matrix multiplication.");
            } अन्यथा यदि (!ggml_vk_matmul_shmem_support(device, m_warptile_mmq, असत्य, t)) {
                device->mul_mat_m[i] = असत्य;
                device->mul_mat_l[i] = असत्य;
            } अन्यथा यदि (!ggml_vk_matmul_shmem_support(device, l_warptile_mmq, असत्य, t)) {
                device->mul_mat_l[i] = असत्य;
            }

            // Disable mul_mat_id if not enough shared memory is available
            यदि (!ggml_vk_matmul_shmem_support(device, s_warptile_mmqid, सत्य, t)) {
                device->mul_mat_id_s[i] = असत्य;
                device->mul_mat_id_m[i] = असत्य;
                device->mul_mat_id_l[i] = असत्य;
            } अन्यथा यदि (!ggml_vk_matmul_shmem_support(device, m_warptile_mmqid, सत्य, t)) {
                device->mul_mat_id_m[i] = असत्य;
                device->mul_mat_id_l[i] = असत्य;
            } अन्यथा यदि (!ggml_vk_matmul_shmem_support(device, l_warptile_mmqid, सत्य, t)) {
                device->mul_mat_id_l[i] = असत्य;
            }

            // The q8_1 mmq path has its own (larger) shmem layout, check it separately.
            // K-quants and IQ3_S use the _int_k warptiles, others use _int.
            // cm1 splits k-tiles on the KSCALES==2 types and shares tiles between dense/id.
            ध्रुव तार्किक is_k_quant = (t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q3_K ||
                                     t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ||
                                     t == GGML_TYPE_Q6_K || t == GGML_TYPE_IQ3_S);
            ध्रुव तार्किक cm1_k_tile = (t == GGML_TYPE_Q3_K || t == GGML_TYPE_Q6_K ||
                                     t == GGML_TYPE_NVFP4);

            ध्रुव स्वचालित & s_int   = use_cm1_int ? (cm1_k_tile ? s_warptile_mmq_cm1_int_k : s_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? s_warptile_mmq_int_k     : s_warptile_mmq_int);
            ध्रुव स्वचालित & m_int   = use_cm1_int ? (cm1_k_tile ? m_warptile_mmq_cm1_int_k : m_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? m_warptile_mmq_int_k     : m_warptile_mmq_int);
            ध्रुव स्वचालित & l_int   = use_cm1_int ? (cm1_k_tile ? l_warptile_mmq_cm1_int_k : l_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? l_warptile_mmq_int_k     : l_warptile_mmq_int);
            ध्रुव स्वचालित & s_intid = use_cm1_int ? (cm1_k_tile ? s_warptile_mmq_cm1_int_k : s_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? s_warptile_mmqid_int_k   : s_warptile_mmqid_int);
            ध्रुव स्वचालित & m_intid = use_cm1_int ? (cm1_k_tile ? m_warptile_mmq_cm1_int_k : m_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? m_warptile_mmqid_int_k   : m_warptile_mmqid_int);
            ध्रुव स्वचालित & l_intid = use_cm1_int ? (cm1_k_tile ? l_warptile_mmq_cm1_int_k : l_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? l_warptile_mmqid_int_k   : l_warptile_mmqid_int);

            ध्रुव स्वचालित int_shmem_support = [&](ध्रुव std::vector<uint32_t>& wt, तार्किक id) {
                प्रत्यागम use_cm1_int ? ggml_vk_matmul_cm1_int_shmem_support(device, wt, id, t)
                                   : ggml_vk_matmul_int_shmem_support(device, wt, id, t);
            };

            यदि (!int_shmem_support(s_int, असत्य)) {
                device->mul_mat_s_int[i] = असत्य;
                device->mul_mat_m_int[i] = असत्य;
                device->mul_mat_l_int[i] = असत्य;
            } अन्यथा यदि (!int_shmem_support(m_int, असत्य)) {
                device->mul_mat_m_int[i] = असत्य;
                device->mul_mat_l_int[i] = असत्य;
            } अन्यथा यदि (!int_shmem_support(l_int, असत्य)) {
                device->mul_mat_l_int[i] = असत्य;
            }

            यदि (!int_shmem_support(s_intid, सत्य)) {
                device->mul_mat_id_s_int[i] = असत्य;
                device->mul_mat_id_m_int[i] = असत्य;
                device->mul_mat_id_l_int[i] = असत्य;
            } अन्यथा यदि (!int_shmem_support(m_intid, सत्य)) {
                device->mul_mat_id_m_int[i] = असत्य;
                device->mul_mat_id_l_int[i] = असत्य;
            } अन्यथा यदि (!int_shmem_support(l_intid, सत्य)) {
                device->mul_mat_id_l_int[i] = असत्य;
            }
        }
    }

    स्वचालित ध्रुव &ggml_vk_create_pipeline = [&](vk_device& device, vk_pipeline& आधारः_pipeline, ध्रुव वर्ण *name, size_t spv_आकार, ध्रुव शून्य* spv_दत्त, ध्रुव वर्ण *entrypoint,
                                              uint32_t parameter_गणनम्, uint32_t प्रक्षिप_constant_आकार, std::array<uint32_t, 3> wg_denoms, ध्रुव std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, तार्किक disable_robustness = असत्य, तार्किक require_पूर्णम्_subgroups = असत्य, uint32_t required_subgroup_आकार = 0) {

        यदि (!require_पूर्णम्_subgroups && required_subgroup_आकार == 0) {
            required_subgroup_आकार = get_subgroup_आकार(name, device->architecture);
        }

        vk_pipeline *ptr = &आधारः_pipeline;

        पूर्णाङ्क अङ्कः_pipelines = 1;
#यदि defined(VK_EXT_shader_64bit_indexing)
        यदि (device->shader_64b_indexing) {
            अङ्कः_pipelines = 2;
        }
#यदि_अन्तः
        प्रति (पूर्णाङ्क i = 0; i < अङ्कः_pipelines; ++i, ptr = &(*ptr)->अग्र) {
            vk_pipeline &pipeline = *ptr;
            यदि (!pipeline) {
                pipeline = std::make_shared<vk_pipeline_struct>();
            }
            यदि (!pipeline->initialized) {
                pipeline->name = name;
                pipeline->parameter_गणनम् = parameter_गणनम्;
                pipeline->प्रक्षिप_constant_आकार = प्रक्षिप_constant_आकार;
                pipeline->wg_denoms = wg_denoms;
                pipeline->align = align;
                pipeline->initialized = सत्य;
#यदि defined(VK_EXT_shader_64bit_indexing)
                pipeline->is_64b_indexing = (i == 1);
#यदि_अन्तः
            }

            // We only care about the pipeline this call asked for; the rest
            // (including the 64-bit indexing variant) are handled by their
            // own request_descriptor_sets / load_shaders calls.
            यदि (pipeline.get() != requested.get()) {
                अनुवर्त;
            }

            यदि (pipeline->compiled) {
                अनुवर्त;
            }

            wait_pipeline = pipeline;

            यदि (!pipeline->compile_pending) {
                pipeline->compile_pending = सत्य;
                claimed_task.pipeline = pipeline;
                claimed_task.spv_आकार = spv_आकार;
                claimed_task.spv_दत्त = spv_दत्त;
                claimed_task.entrypoint = entrypoint;
                claimed_task.parameter_गणनम् = parameter_गणनम्;
                claimed_task.wg_denoms = wg_denoms;
                claimed_task.specialization_constants = specialization_constants;
                claimed_task.disable_robustness = disable_robustness;
                claimed_task.require_पूर्णम्_subgroups = require_पूर्णम्_subgroups;
                claimed_task.required_subgroup_आकार = required_subgroup_आकार;
                has_claimed_task = सत्य;
            }
        }
    };

    स्वचालित ध्रुव &ggml_vk_create_pipeline2 = [&](vk_device& device, vk_pipeline& pipeline, ध्रुव std::string &name, size_t spv_आकार, ध्रुव शून्य* spv_दत्त, ध्रुव वर्ण *entrypoint,
                                              uint32_t parameter_गणनम्, uint32_t प्रक्षिप_constant_आकार, std::array<uint32_t, 3> wg_denoms, ध्रुव std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, तार्किक disable_robustness = असत्य, तार्किक require_पूर्णम्_subgroups = असत्य, uint32_t required_subgroup_आकार = 0) {
        प्रत्यागम ggml_vk_create_pipeline(device, pipeline, name.c_str(), spv_आकार, spv_दत्त, entrypoint,
                                       parameter_गणनम्, प्रक्षिप_constant_आकार, wg_denoms, specialization_constants,
                                       align, disable_robustness, require_पूर्णम्_subgroups, required_subgroup_आकार);
    };

    // FA scalar has two SPIR-V modules (MMQ vs non-MMQ); FA cm1 has one. K/V
    // quant type is selected at runtime via the FaTypeK / FaTypeV spec constants.

    प्रति (स्वचालित &fa : device->pipeline_flash_attn_f32_f16) {
        यदि (fa.first.path != FA_SCALAR) अनुवर्त;
        ध्रुव uint32_t Br = fa.first.Br;
        ध्रुव uint32_t Bc = fa.first.Bc;
        ध्रुव तार्किक aligned = fa.first.aligned;
        ध्रुव तार्किक f32acc = fa.first.f32acc;
        ध्रुव uint32_t fa_sgs = fa.first.subgroup_आकार;
        ध्रुव तार्किक fa_ds = fa.first.subgroup_आकार == 0;

        ध्रुव तार्किक bf16_kv = fa.first.k_type == GGML_TYPE_BF16;
        ध्रुव तार्किक use_mmq = ggml_vk_fa_scalar_उपयुङ्क्ते_mmq(device, fa.first.k_type, fa.first.v_type);
        ध्रुव शून्य * spv_दत्त = शून्यसूचकः;
        size_t spv_आकार = 0;
        ध्रुव वर्ण *name = शून्यसूचकः;
        यदि (bf16_kv) {
            spv_दत्त = flash_attn_f32_f16_fp32_दत्त;
            spv_आकार = flash_attn_f32_f16_fp32_len;
            name = aligned ? "flash_attn_f32_bf16_aligned" : "flash_attn_f32_bf16";
        } अन्यथा यदि (use_mmq) {
#यदि defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            यदि (device->fp16) {
                यदि (f32acc) { spv_दत्त = flash_attn_f32_f16_int8_दत्त;        spv_आकार = flash_attn_f32_f16_int8_len; }
                अन्यथा        { spv_दत्त = flash_attn_f32_f16_f16acc_int8_दत्त; spv_आकार = flash_attn_f32_f16_f16acc_int8_len; }
            } अन्यथा {
                spv_दत्त = flash_attn_f32_f16_fp32_int8_दत्त;
                spv_आकार = flash_attn_f32_f16_fp32_int8_len;
            }
#यदि_अन्तः
            name = aligned ? "flash_attn_f32_f16_aligned" : "flash_attn_f32_f16";
        } अन्यथा {
            यदि (device->fp16) {
                यदि (device->dot2_f16) {
                    यदि (f32acc) { spv_दत्त = flash_attn_f32_f16_dot2_दत्त;        spv_आकार = flash_attn_f32_f16_dot2_len; }
                    अन्यथा        { spv_दत्त = flash_attn_f32_f16_dot2_f16acc_दत्त; spv_आकार = flash_attn_f32_f16_dot2_f16acc_len; }
                } अन्यथा {
                    यदि (f32acc) { spv_दत्त = flash_attn_f32_f16_दत्त;        spv_आकार = flash_attn_f32_f16_len; }
                    अन्यथा        { spv_दत्त = flash_attn_f32_f16_f16acc_दत्त; spv_आकार = flash_attn_f32_f16_f16acc_len; }
                }
            } अन्यथा {
                spv_दत्त = flash_attn_f32_f16_fp32_दत्त;
                spv_आकार = flash_attn_f32_f16_fp32_len;
            }
            name = aligned ? "flash_attn_f32_f16_aligned" : "flash_attn_f32_f16";
        }
        ggml_vk_create_pipeline(device, fa.second, name, spv_आकार, spv_दत्त, "main", 8,
                                परिमाण(vk_flash_attn_प्रक्षिप_constants), {Br, 1, 1},
                                get_fa_spec_constants(fa.first), aligned ? Bc : 1, सत्य,
                                !fa_ds, !fa_ds ? fa_sgs : 0);
    }

#यदि defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    यदि (device->coopmat1_fa_support) {
        प्रति (स्वचालित &fa : device->pipeline_flash_attn_f32_f16) {
            यदि (fa.first.path != FA_COOPMAT1) अनुवर्त;
            ध्रुव uint32_t Br = fa.first.Br;
            ध्रुव uint32_t Bc = fa.first.Bc;
            ध्रुव तार्किक aligned = fa.first.aligned;
            ध्रुव तार्किक f32acc = fa.first.f32acc;
            ध्रुव uint32_t fa_sgs = fa.first.subgroup_आकार;
            ध्रुव तार्किक fa_ds = fa.first.subgroup_आकार == 0;

            ध्रुव तार्किक bf16_kv = fa.first.k_type == GGML_TYPE_BF16;

            ध्रुव शून्य * spv_दत्त;
            size_t spv_आकार;
            ध्रुव वर्ण *name;
            यदि (bf16_kv) {
#यदि defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                यदि (!device->coopmat_bf16_support) अनुवर्त;
                spv_दत्त = flash_attn_f32_f16_bf16_cm1_दत्त;
                spv_आकार = flash_attn_f32_f16_bf16_cm1_len;
                name = aligned ? "flash_attn_f32_bf16_aligned_cm1" : "flash_attn_f32_bf16_cm1";
#अन्यथा
                अनुवर्त;
#यदि_अन्तः
            } अन्यथा {
                यदि (f32acc) { spv_दत्त = flash_attn_f32_f16_cm1_दत्त;        spv_आकार = flash_attn_f32_f16_cm1_len; }
                अन्यथा        { spv_दत्त = flash_attn_f32_f16_f16acc_cm1_दत्त; spv_आकार = flash_attn_f32_f16_f16acc_cm1_len; }
                name = aligned ? "flash_attn_f32_f16_aligned_cm1" : "flash_attn_f32_f16_cm1";
            }
            ggml_vk_create_pipeline(device, fa.second, name, spv_आकार, spv_दत्त, "main", 8,
                                    परिमाण(vk_flash_attn_प्रक्षिप_constants), {Br, 1, 1},
                                    get_fa_spec_constants(fa.first), aligned ? Bc : 1, सत्य,
                                    !fa_ds, !fa_ds ? fa_sgs : 0);
        }
    }
#यदि_अन्तः

#यदि defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    यदि (device->coopmat2) {
        प्रति (स्वचालित &fa : device->pipeline_flash_attn_f32_f16) {
            यदि (fa.first.path != FA_COOPMAT2) अनुवर्त;
            ध्रुव uint32_t Br = fa.first.Br;
            ध्रुव uint32_t Bc = fa.first.Bc;
            ध्रुव तार्किक aligned = fa.first.aligned;
            ध्रुव तार्किक f32acc = fa.first.f32acc;

            ध्रुव तार्किक bf16_kv = fa.first.k_type == GGML_TYPE_BF16;
            ध्रुव शून्य * spv_दत्त;
            size_t spv_आकार;
            ध्रुव वर्ण * name;
            यदि (bf16_kv) {
#यदि defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                यदि (!device->coopmat2_bf16_support) अनुवर्त;
                spv_दत्त = flash_attn_f32_f16_bf16_cm2_दत्त;
                spv_आकार = flash_attn_f32_f16_bf16_cm2_len;
                name = aligned ? "flash_attn_f32_bf16_aligned_cm2" : "flash_attn_f32_bf16_cm2";
#अन्यथा
                अनुवर्त;
#यदि_अन्तः
            } अन्यथा यदि (aligned) {
                यदि (f32acc) { spv_दत्त = flash_attn_f32_f16_cm2_दत्त;        spv_आकार = flash_attn_f32_f16_cm2_len;        name = "flash_attn_f32_f16_aligned_f32acc_cm2"; }
                अन्यथा        { spv_दत्त = flash_attn_f32_f16_f16acc_cm2_दत्त; spv_आकार = flash_attn_f32_f16_f16acc_cm2_len; name = "flash_attn_f32_f16_aligned_f16acc_cm2"; }
            } अन्यथा {
                यदि (f32acc) { spv_दत्त = flash_attn_f32_f16_cm2_दत्त;        spv_आकार = flash_attn_f32_f16_cm2_len;        name = "flash_attn_f32_f16_f32acc_cm2"; }
                अन्यथा        { spv_दत्त = flash_attn_f32_f16_f16acc_cm2_दत्त; spv_आकार = flash_attn_f32_f16_f16acc_cm2_len; name = "flash_attn_f32_f16_f16acc_cm2"; }
            }
            ggml_vk_create_pipeline(device, fa.second, name, spv_आकार, spv_दत्त, "main", 8,
                                    परिमाण(vk_flash_attn_प्रक्षिप_constants), {Br, 1, 1},
                                    get_fa_spec_constants(fa.first), aligned ? Bc : 1, सत्य, असत्य, 0);
        }
    }
#यदि_अन्तः

    स्वचालित ध्रुव &ggml_vk_mul_mm_spec = [&device](std::vector<uint32_t> spec, तार्किक aligned) {
        spec.push_back(aligned ? 1u : 0u);  // constantID=11: ALIGNED
        यदि (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
            device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            spec.push_back(0u);  // constantID=12: SHMEM_STRIDE_PAD = 0
            spec.push_back(1u);  // constantID=13: APPLY_SLM_A_RESHAPE = true
        }
        प्रत्यागम spec;
    };

    स्वचालित ध्रुव &ggml_vk_mul_mm_spec_quant = [&device](std::vector<uint32_t> spec, तार्किक aligned, uint32_t type) {
        spec.push_back(aligned ? 1u : 0u);  // constantID=11: ALIGNED
        spec.push_back(type);               // constantID=12: MmTypeA
        यदि (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
            device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            spec.push_back(0u);  // constantID=13: SHMEM_STRIDE_PAD = 0
            spec.push_back(1u);  // constantID=14: APPLY_SLM_A_RESHAPE = true
        }
        प्रत्यागम spec;
    };

    स्थिर ध्रुव ggml_type non_lut_quant_types[] = {
        GGML_TYPE_Q1_0, GGML_TYPE_Q2_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
    };

#परिभाषय FOR_EACH_LUT_TYPE_NONFP4(X) \
    X(GGML_TYPE_IQ1_S,   iq1_s)   \
    X(GGML_TYPE_IQ1_M,   iq1_m)   \
    X(GGML_TYPE_IQ2_XXS, iq2_xxs) \
    X(GGML_TYPE_IQ2_XS,  iq2_xs)  \
    X(GGML_TYPE_IQ2_S,   iq2_s)   \
    X(GGML_TYPE_IQ3_XXS, iq3_xxs) \
    X(GGML_TYPE_IQ3_S,   iq3_s)   \
    X(GGML_TYPE_IQ4_XS,  iq4_xs)  \
    X(GGML_TYPE_IQ4_NL,  iq4_nl)
#परिभाषय FOR_EACH_LUT_FP4_TYPE(X) \
    X(GGML_TYPE_MXFP4,   mxfp4)   \
    X(GGML_TYPE_NVFP4,   nvfp4)
#परिभाषय FOR_EACH_LUT_TYPE(X) \
    FOR_EACH_LUT_TYPE_NONFP4(X)  \
    FOR_EACH_LUT_FP4_TYPE(X)

    ध्रुव पूर्णाङ्क mul_mat_id_param_गणनम् = 5;

    उपयुज्य spec_fn_t = std::function<std::vector<uint32_t>(ध्रुव std::vector<uint32_t>&, तार्किक)>;
    स्वचालित ध्रुव &create_mm_pipelines = [&](
        ध्रुव vk_matmul_pipeline_कुञ्चिका& कुञ्चिका,
        ध्रुव std::vector<vk_tile_config>& tile_configs,
        ध्रुव std::string& shader_name, size_t spv_len, ध्रुव शून्य* spv_दत्त,
        uint32_t प्रक्षिप_constant_आकार, uint32_t param_गणनम्,
        ध्रुव spec_fn_t& spec_fn,
        तार्किक disable_robustness = असत्य, तार्किक require_पूर्णम्_subgroups = असत्य, uint32_t required_subgroup_आकार = 0,
        तार्किक create_aligned = सत्य, तार्किक pin_subgroup_गन्तव्यम्_warp = असत्य
    ) {
        स्वचालित& vec = device->pipeline_matmul[कुञ्चिका];
        ध्रुव तार्किक first_call = vec.रिक्तानि();
        प्रति (size_t i = 0; i < tile_configs.size(); i++) {
            ध्रुव स्वचालित& tc = tile_configs[i];

            // Intel coopmat1 pins the required subgroup size to each warptile's WARP element.
            ध्रुव uint32_t rsgs = pin_subgroup_गन्तव्यम्_warp ? tc.warptile[WARP_आकार_IDX] : required_subgroup_आकार;
            ध्रुव तार्किक     rfs  = require_पूर्णम्_subgroups || pin_subgroup_गन्तव्यम्_warp;

            यदि (first_call) {
                vk_matmul_pipeline_pair pair{};
                pair.align = tc.align;
                std::string suffix = "_" + std::to_string(i);
                pair.unaligned = std::make_shared<vk_pipeline_struct>();
                यदि (create_aligned) {
                    pair.aligned = std::make_shared<vk_pipeline_struct>();
                }
                vec.push_back(pair);
            }

            ggml_vk_create_pipeline(device, vec[i].unaligned,
                vec[i].unaligned->name.रिक्तानि() ? (shader_name + "_" + std::to_string(i)).c_str() : vec[i].unaligned->name.c_str(),
                spv_len, spv_दत्त, "main", param_गणनम्, प्रक्षिप_constant_आकार,
                tc.wg_denoms, spec_fn(tc.warptile, असत्य), 1,
                disable_robustness, rfs, rsgs);

            यदि (vec[i].aligned) {
                ggml_vk_create_pipeline(device, vec[i].aligned,
                    vec[i].aligned->name.रिक्तानि() ? (shader_name + "_aligned_" + std::to_string(i)).c_str() : vec[i].aligned->name.c_str(),
                    spv_len, spv_दत्त, "main", param_गणनम्, प्रक्षिप_constant_आकार,
                    tc.wg_denoms, spec_fn(tc.warptile, सत्य), tc.align,
                    disable_robustness, rfs, rsgs);
            }
        }
    };

    स्वचालित filter_tc = [&](ध्रुव std::vector<vk_tile_config>& configs, ggml_type type, तार्किक is_id, तार्किक is_int = असत्य) -> std::vector<vk_tile_config> {
        std::vector<vk_tile_config> परिणाम;
        तार्किक enabled[3];
        यदि (is_int) {
            enabled[0] = is_id ? device->mul_mat_id_s_int[type] : device->mul_mat_s_int[type];
            enabled[1] = is_id ? device->mul_mat_id_m_int[type] : device->mul_mat_m_int[type];
            enabled[2] = is_id ? device->mul_mat_id_l_int[type] : device->mul_mat_l_int[type];
        } अन्यथा {
            enabled[0] = is_id ? device->mul_mat_id_s[type] : device->mul_mat_s[type];
            enabled[1] = is_id ? device->mul_mat_id_m[type] : device->mul_mat_m[type];
            enabled[2] = is_id ? device->mul_mat_id_l[type] : device->mul_mat_l[type];
        }
        प्रति (size_t i = 0; i < configs.size() && i < 3; i++) {
            यदि (enabled[i]) परिणाम.push_back(configs[i]);
        }
        प्रत्यागम परिणाम;
    };

    std::vector<vk_tile_config> tc_mm = {{s_warptile, s_wg_denoms, s_align}, {m_warptile, m_wg_denoms, m_align}, {l_warptile, l_wg_denoms, l_align}};
    std::vector<vk_tile_config> tc_mmq = {{s_warptile_mmq, s_mmq_wg_denoms, s_align}, {m_warptile_mmq, m_mmq_wg_denoms, m_align}, {l_warptile_mmq, l_mmq_wg_denoms, l_align}};

#यदि defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    यदि (device->coopmat2) {
        स्वचालित ध्रुव &ggml_vk_mul_mm_cm2_spec = [&](std::vector<uint32_t> spec, तार्किक aligned, uint32_t type = UINT32_MAX) {
            spec.push_back(aligned ? 1u : 0u);        // ALIGNED
            spec.push_back(device->subgroup_आकार);     // subgroup_size
            यदि (type != UINT32_MAX) {
                spec.push_back(type);                  // MmTypeA
                spec.push_back((uint32_t)ggml_type_आकार((ggml_type)type)); // MmABlockBytes
            }
            प्रत्यागम spec;
        };

        std::vector<vk_tile_config> tc_mmq_k = {{s_warptile_mmq_k, s_mmq_wg_denoms_k, s_align}, {m_warptile_mmq_k, m_mmq_wg_denoms_k, m_align}, {l_warptile_mmq_k, l_mmq_wg_denoms_k, l_align}};
        std::vector<vk_tile_config> tc_mmqid = {{s_warptile_mmqid, s_mmqid_wg_denoms, s_align}, {m_warptile_mmqid, m_mmqid_wg_denoms, m_align}, {l_warptile_mmqid, l_mmqid_wg_denoms, l_align}};

        spec_fn_t cm2_spec = [&](ध्रुव std::vector<uint32_t>& wt, तार्किक a) { प्रत्यागम ggml_vk_mul_mm_cm2_spec(wt, a); };

        // F16 x F16
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, असत्य, सत्य},  tc_mm, "matmul_f16_f16acc", matmul_f16_f16acc_cm2_len, matmul_f16_f16acc_cm2_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3, cm2_spec, सत्य);
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, असत्य, असत्य}, tc_mm, "matmul_f16",        matmul_f16_cm2_len,        matmul_f16_cm2_दत्त,        परिमाण(vk_mat_mat_प्रक्षिप_constants), 3, cm2_spec, सत्य);
#यदि defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        यदि (device->coopmat_bf16_support) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, असत्य, असत्य}, tc_mm, "matmul_bf16", matmul_bf16_cm2_len, matmul_bf16_cm2_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3, cm2_spec, सत्य);
        }
#यदि_अन्तः
        प्रति (ध्रुव स्वचालित type : non_lut_quant_types) {
            // regression in unified shader on Ampere
            यदि (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                अनुवर्त;
            }
            स्वचालित& tc = ((type >= GGML_TYPE_Q2_K && type <= GGML_TYPE_Q6_K) || type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0) ? tc_mmq_k : tc_mmq;
            spec_fn_t qs = [&, type](ध्रुव std::vector<uint32_t>& wt, तार्किक a) { प्रत्यागम ggml_vk_mul_mm_cm2_spec(wt, a, (uint32_t)type); };
            create_mm_pipelines({type, GGML_TYPE_F16, असत्य, सत्य},  tc, "matmul_quant_f16_f16acc", matmul_quant_f16_f16acc_cm2_len, matmul_quant_f16_f16acc_cm2_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3, qs, सत्य);
            create_mm_pipelines({type, GGML_TYPE_F16, असत्य, असत्य}, tc, "matmul_quant_f16",        matmul_quant_f16_cm2_len,        matmul_quant_f16_cm2_दत्त,        परिमाण(vk_mat_mat_प्रक्षिप_constants), 3, qs, सत्य);
        }
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, असत्य, सत्य},  tc_mmq_k, "matmul_q4_k_f16_f16acc", matmul_q4_k_f16_f16acc_cm2_len, matmul_q4_k_f16_f16acc_cm2_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3, cm2_spec, सत्य);
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, असत्य, असत्य}, tc_mmq_k, "matmul_q4_k_f16",        matmul_q4_k_f16_cm2_len,        matmul_q4_k_f16_cm2_दत्त,        परिमाण(vk_mat_mat_प्रक्षिप_constants), 3, cm2_spec, सत्य);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, असत्य, सत्य},  tc_mmq_k, "matmul_q5_k_f16_f16acc", matmul_q5_k_f16_f16acc_cm2_len, matmul_q5_k_f16_f16acc_cm2_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3, cm2_spec, सत्य);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, असत्य, असत्य}, tc_mmq_k, "matmul_q5_k_f16",        matmul_q5_k_f16_cm2_len,        matmul_q5_k_f16_cm2_दत्त,        परिमाण(vk_mat_mat_प्रक्षिप_constants), 3, cm2_spec, सत्य);
#परिभाषय X_CM2(TYPE, tstr) \
        { स्वचालित tc = filter_tc(tc_mmq, TYPE, असत्य); \
          यदि (!tc.रिक्तानि()) { \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, असत्य, सत्य},  tc, "matmul_" #tstr "_f16_f16acc", matmul_##tstr##_f16_f16acc_cm2_len, matmul_##tstr##_f16_f16acc_cm2_data, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3, cm2_spec, सत्य); \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, असत्य, असत्य}, tc, "matmul_" #tstr "_f16",        matmul_##tstr##_f16_cm2_len,        matmul_##tstr##_f16_cm2_data,        परिमाण(vk_mat_mat_प्रक्षिप_constants), 3, cm2_spec, सत्य); \
          } }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM2)
#यदि defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        यदि (device->ocp_fp4) {
#परिभाषय X_CM2_OCP(TYPE, tstr) \
            { स्वचालित tc = filter_tc(tc_mmq, TYPE, असत्य); \
              यदि (!tc.रिक्तानि()) { \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, असत्य, सत्य},  tc, "matmul_" #tstr "_f16_ocp_f16acc", matmul_##tstr##_f16_ocp_f16acc_cm2_len, matmul_##tstr##_f16_ocp_f16acc_cm2_data, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3, cm2_spec, सत्य); \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, असत्य, असत्य}, tc, "matmul_" #tstr "_f16_ocp",        matmul_##tstr##_f16_ocp_cm2_len,        matmul_##tstr##_f16_ocp_cm2_data,        परिमाण(vk_mat_mat_प्रक्षिप_constants), 3, cm2_spec, सत्य); \
              } }
            FOR_EACH_LUT_FP4_TYPE(X_CM2_OCP)
#अपरिभाषय X_CM2_OCP
        } अन्यथा
#यदि_अन्तः
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM2)
        }
#अपरिभाषय X_CM2

        GGML_ASSERT(device->subgroup_ballot);

        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, सत्य, सत्य},  tc_mm, "matmul_id_subgroup_f16_f16acc", matmul_id_subgroup_f16_f16acc_cm2_len, matmul_id_subgroup_f16_f16acc_cm2_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), 5, cm2_spec, सत्य);
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, सत्य, असत्य}, tc_mm, "matmul_id_subgroup_f16",        matmul_id_subgroup_f16_cm2_len,        matmul_id_subgroup_f16_cm2_दत्त,        परिमाण(vk_mat_mat_id_प्रक्षिप_constants), 5, cm2_spec, सत्य);
#यदि defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        यदि (device->coopmat_bf16_support) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, सत्य, असत्य}, tc_mm, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_cm2_len, matmul_id_subgroup_bf16_cm2_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), 5, cm2_spec, सत्य);
        }
#यदि_अन्तः
        प्रति (ध्रुव स्वचालित type : non_lut_quant_types) {
            यदि (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                अनुवर्त;
            }
            spec_fn_t qs_id = [&, type](ध्रुव std::vector<uint32_t>& wt, तार्किक a) { प्रत्यागम ggml_vk_mul_mm_cm2_spec(wt, a, (uint32_t)type); };
            create_mm_pipelines({type, GGML_TYPE_F16, सत्य, सत्य},  tc_mmqid, "matmul_id_subgroup_quant_f16_f16acc", matmul_id_subgroup_quant_f16_f16acc_cm2_len, matmul_id_subgroup_quant_f16_f16acc_cm2_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), 5, qs_id, सत्य);
            create_mm_pipelines({type, GGML_TYPE_F16, सत्य, असत्य}, tc_mmqid, "matmul_id_subgroup_quant_f16",        matmul_id_subgroup_quant_f16_cm2_len,        matmul_id_subgroup_quant_f16_cm2_दत्त,        परिमाण(vk_mat_mat_id_प्रक्षिप_constants), 5, qs_id, सत्य);
        }
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, सत्य, सत्य},  tc_mmqid, "matmul_id_subgroup_q4_k_f16_f16acc", matmul_id_subgroup_q4_k_f16_f16acc_cm2_len, matmul_id_subgroup_q4_k_f16_f16acc_cm2_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), 5, cm2_spec, सत्य);
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, सत्य, असत्य}, tc_mmqid, "matmul_id_subgroup_q4_k_f16",        matmul_id_subgroup_q4_k_f16_cm2_len,        matmul_id_subgroup_q4_k_f16_cm2_दत्त,        परिमाण(vk_mat_mat_id_प्रक्षिप_constants), 5, cm2_spec, सत्य);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, सत्य, सत्य},  tc_mmqid, "matmul_id_subgroup_q5_k_f16_f16acc", matmul_id_subgroup_q5_k_f16_f16acc_cm2_len, matmul_id_subgroup_q5_k_f16_f16acc_cm2_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), 5, cm2_spec, सत्य);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, सत्य, असत्य}, tc_mmqid, "matmul_id_subgroup_q5_k_f16",        matmul_id_subgroup_q5_k_f16_cm2_len,        matmul_id_subgroup_q5_k_f16_cm2_दत्त,        परिमाण(vk_mat_mat_id_प्रक्षिप_constants), 5, cm2_spec, सत्य);
#परिभाषय X_CM2_ID(TYPE, tstr) \
        { स्वचालित tc = filter_tc(tc_mmqid, TYPE, सत्य); \
          यदि (!tc.रिक्तानि()) { \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, सत्य, सत्य},  tc, "matmul_id_subgroup_" #tstr "_f16_f16acc", matmul_id_subgroup_##tstr##_f16_f16acc_cm2_len, matmul_id_subgroup_##tstr##_f16_f16acc_cm2_data, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), 5, cm2_spec, सत्य); \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, सत्य, असत्य}, tc, "matmul_id_subgroup_" #tstr "_f16",        matmul_id_subgroup_##tstr##_f16_cm2_len,        matmul_id_subgroup_##tstr##_f16_cm2_data,        परिमाण(vk_mat_mat_id_प्रक्षिप_constants), 5, cm2_spec, सत्य); \
          } }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM2_ID)
#यदि defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        यदि (device->ocp_fp4) {
#परिभाषय X_CM2_ID_OCP(TYPE, tstr) \
            { स्वचालित tc = filter_tc(tc_mmqid, TYPE, सत्य); \
              यदि (!tc.रिक्तानि()) { \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, सत्य, सत्य},  tc, "matmul_id_subgroup_" #tstr "_f16_ocp_f16acc", matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm2_len, matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm2_data, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), 5, cm2_spec, सत्य); \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, सत्य, असत्य}, tc, "matmul_id_subgroup_" #tstr "_f16_ocp",        matmul_id_subgroup_##tstr##_f16_ocp_cm2_len,        matmul_id_subgroup_##tstr##_f16_ocp_cm2_data,        परिमाण(vk_mat_mat_id_प्रक्षिप_constants), 5, cm2_spec, सत्य); \
              } }
            FOR_EACH_LUT_FP4_TYPE(X_CM2_ID_OCP)
#अपरिभाषय X_CM2_ID_OCP
        } अन्यथा
#यदि_अन्तः
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM2_ID)
        }
#अपरिभाषय X_CM2_ID
    } अन्यथा
#यदि_अन्तः  // defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
#यदि defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    यदि (device->coopmat_support) {
        spec_fn_t cm1_spec = [&](ध्रुव std::vector<uint32_t>& wt, तार्किक a) { प्रत्यागम ggml_vk_mul_mm_spec(wt, a); };

        // Intel coopmat1 pins each pipeline's required subgroup size to its warptile WARP element.
        ध्रुव तार्किक cm1_pin = device->vendor_id == VK_VENDOR_ID_INTEL;

        // Intel coopmat1 uses a dedicated large-tile config for quant matmul_id.
        std::vector<vk_tile_config> tc_mmq_id = tc_mmq;
        यदि (cm1_pin) {
            tc_mmq_id[2] = { { 512, 128, 128, 32, 32, 32, 2, device->coopmat_m, device->coopmat_n, device->coopmat_k, 32 }, { 128, 128, 1 }, 32 };
        }

        स्वचालित cm1_create = [&](vk_matmul_pipeline_कुञ्चिका कुञ्चिका, ध्रुव std::vector<vk_tile_config>& tc_आधारः,
                              ध्रुव std::string& name, size_t len, ध्रुव शून्य* दत्त, uint32_t pc_आकार, uint32_t pc) {
            स्वचालित tc = filter_tc(tc_आधारः, कुञ्चिका.type_a, कुञ्चिका.mul_mat_id);
            यदि (!tc.रिक्तानि()) create_mm_pipelines(कुञ्चिका, tc, name, len, दत्त, pc_आकार, pc, cm1_spec, असत्य, सत्य, 0, सत्य, cm1_pin);
        };
        स्वचालित cm1_create_quant = [&](vk_matmul_pipeline_कुञ्चिका कुञ्चिका, ध्रुव std::vector<vk_tile_config>& tc_आधारः,
                                    ध्रुव std::string& name, size_t len, ध्रुव शून्य* दत्त, uint32_t pc_आकार, uint32_t pc) {
            spec_fn_t qs = [&, type_a=कुञ्चिका.type_a](ध्रुव std::vector<uint32_t>& wt, तार्किक a) { प्रत्यागम ggml_vk_mul_mm_spec_quant(wt, a, (uint32_t)type_a); };
            स्वचालित tc = filter_tc(tc_आधारः, कुञ्चिका.type_a, कुञ्चिका.mul_mat_id);
            यदि (!tc.रिक्तानि()) create_mm_pipelines(कुञ्चिका, tc, name, len, दत्त, pc_आकार, pc, qs, असत्य, सत्य, 0, सत्य, cm1_pin);
        };
        // int8 MMQ helper: per-type cm1 shader, warptile passed as-is (carries DEVICE_ARCH in
        // spec constant WARP_SIZE_IDX+1), subgroup size pinned to the warptile WARP element, no aligned variant.
        स्वचालित cm1_create_mmq = [&](vk_matmul_pipeline_कुञ्चिका कुञ्चिका, ध्रुव std::vector<vk_tile_config>& tc_आधारः,
                                  ध्रुव std::string& name, size_t len, ध्रुव शून्य* दत्त, uint32_t pc_आकार, uint32_t pc) {
            spec_fn_t तटस्थः = [](ध्रुव std::vector<uint32_t>& wt, तार्किक) { प्रत्यागम wt; };
            स्वचालित tc = filter_tc(tc_आधारः, कुञ्चिका.type_a, कुञ्चिका.mul_mat_id, सत्य);
            यदि (!tc.रिक्तानि()) create_mm_pipelines(कुञ्चिका, tc, name, len, दत्त, pc_आकार, pc, तटस्थः, असत्य, असत्य, 0, असत्य, सत्य);
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
        ध्रुव तार्किक rdna3 = device->architecture == AMD_RDNA3;
        ध्रुव तार्किक rdna4 = device->architecture == AMD_RDNA4;

        cm1_create({GGML_TYPE_F32, GGML_TYPE_F32, असत्य, असत्य}, tc_mm, "matmul_f32_f32",     matmul_f32_f32_cm1_len,     matmul_f32_f32_cm1_दत्त,     परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
        cm1_create({GGML_TYPE_F32, GGML_TYPE_F16, असत्य, असत्य}, tc_mm, "matmul_f32_f16",     matmul_f32_f16_cm1_len,     matmul_f32_f16_cm1_दत्त,     परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
        यदि (device->coopmat_acc_f16_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, असत्य, सत्य},  tc_mm, "matmul_f16_f16acc", matmul_f16_f16acc_cm1_len, matmul_f16_f16acc_cm1_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, असत्य, सत्य},  tc_mm, "matmul_f16_f32_f16acc", matmul_f16_f32_f16acc_cm1_len, matmul_f16_f32_f16acc_cm1_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
        }
        यदि (device->coopmat_acc_f32_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, असत्य, असत्य}, tc_mm, "matmul_f16",      matmul_f16_cm1_len,      matmul_f16_cm1_दत्त,      परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, असत्य, असत्य}, tc_mm, "matmul_f16_f32",  matmul_f16_f32_cm1_len,  matmul_f16_f32_cm1_दत्त,  परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
        }
#यदि defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        यदि (device->coopmat_bf16_support) {
            cm1_create({GGML_TYPE_BF16, GGML_TYPE_BF16, असत्य, असत्य}, tc_mm, "matmul_bf16", matmul_bf16_cm1_len, matmul_bf16_cm1_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
        }
#यदि_अन्तः
        प्रति (ध्रुव स्वचालित type : non_lut_quant_types) {
            यदि (device->coopmat_acc_f16_support) {
                cm1_create_quant({type, GGML_TYPE_F32, असत्य, सत्य},  tc_mmq, "matmul_quant_f32_f16acc", matmul_quant_f32_f16acc_cm1_len, matmul_quant_f32_f16acc_cm1_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                cm1_create_quant({type, GGML_TYPE_F16, असत्य, सत्य},  tc_mmq, "matmul_quant_f16_f16acc", matmul_quant_f16_f16acc_cm1_len, matmul_quant_f16_f16acc_cm1_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            }
            यदि (device->coopmat_acc_f32_support) {
                cm1_create_quant({type, GGML_TYPE_F32, असत्य, असत्य}, tc_mmq, "matmul_quant_f32",        matmul_quant_f32_cm1_len,        matmul_quant_f32_cm1_दत्त,        परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                cm1_create_quant({type, GGML_TYPE_F16, असत्य, असत्य}, tc_mmq, "matmul_quant_f16",        matmul_quant_f16_cm1_len,        matmul_quant_f16_cm1_दत्त,        परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            }
        }
        // The _f16 variants provide the f16 B-type pipeline used when y_non_contig converts f32->f16.
#परिभाषय X_CM1(TYPE, tstr) \
        यदि (device->coopmat_acc_f16_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, असत्य, सत्य},  tc_mmq, "matmul_" #tstr "_f32_f16acc", matmul_##tstr##_f32_f16acc_cm1_len, matmul_##tstr##_f32_f16acc_cm1_data, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3); \
            cm1_create({TYPE, GGML_TYPE_F16, असत्य, सत्य},  tc_mmq, "matmul_" #tstr "_f16_f16acc", matmul_##tstr##_f16_f16acc_cm1_len, matmul_##tstr##_f16_f16acc_cm1_data, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3); \
        } \
        यदि (device->coopmat_acc_f32_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, असत्य, असत्य}, tc_mmq, "matmul_" #tstr "_f32",        matmul_##tstr##_f32_cm1_len,        matmul_##tstr##_f32_cm1_data,        परिमाण(vk_mat_mat_प्रक्षिप_constants), 3); \
            cm1_create({TYPE, GGML_TYPE_F16, असत्य, असत्य}, tc_mmq, "matmul_" #tstr "_f16",        matmul_##tstr##_f16_cm1_len,        matmul_##tstr##_f16_cm1_data,        परिमाण(vk_mat_mat_प्रक्षिप_constants), 3); \
        }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM1)
#यदि defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        यदि (device->ocp_fp4) {
#परिभाषय X_CM1_OCP(TYPE, tstr) \
            यदि (device->coopmat_acc_f16_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, असत्य, सत्य},  tc_mmq, "matmul_" #tstr "_f32_ocp_f16acc", matmul_##tstr##_f32_ocp_f16acc_cm1_len, matmul_##tstr##_f32_ocp_f16acc_cm1_data, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3); \
                cm1_create({TYPE, GGML_TYPE_F16, असत्य, सत्य},  tc_mmq, "matmul_" #tstr "_f16_ocp_f16acc", matmul_##tstr##_f16_ocp_f16acc_cm1_len, matmul_##tstr##_f16_ocp_f16acc_cm1_data, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3); \
            } \
            यदि (device->coopmat_acc_f32_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, असत्य, असत्य}, tc_mmq, "matmul_" #tstr "_f32_ocp",        matmul_##tstr##_f32_ocp_cm1_len,        matmul_##tstr##_f32_ocp_cm1_data,        परिमाण(vk_mat_mat_प्रक्षिप_constants), 3); \
                cm1_create({TYPE, GGML_TYPE_F16, असत्य, असत्य}, tc_mmq, "matmul_" #tstr "_f16_ocp",        matmul_##tstr##_f16_ocp_cm1_len,        matmul_##tstr##_f16_ocp_cm1_data,        परिमाण(vk_mat_mat_प्रक्षिप_constants), 3); \
            }
            FOR_EACH_LUT_FP4_TYPE(X_CM1_OCP)
#अपरिभाषय X_CM1_OCP
        } अन्यथा
#यदि_अन्तः
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM1)
        }
#अपरिभाषय X_CM1

        यदि (device->coopmat_int_support && (rdna3 || rdna4)) {
            cm1_create_mmq({GGML_TYPE_Q4_0,   GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_cm1_int,   "matmul_q4_0_q8_1",   matmul_q4_0_q8_1_cm1_len,   matmul_q4_0_q8_1_cm1_दत्त,   परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            यदि (!rdna4) { cm1_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_cm1_int,   "matmul_q4_1_q8_1",   matmul_q4_1_q8_1_cm1_len,   matmul_q4_1_q8_1_cm1_दत्त,   परिमाण(vk_mat_mat_प्रक्षिप_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q5_0,   GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_cm1_int,   "matmul_q5_0_q8_1",   matmul_q5_0_q8_1_cm1_len,   matmul_q5_0_q8_1_cm1_दत्त,   परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            यदि (!rdna4) { cm1_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_cm1_int,   "matmul_q5_1_q8_1",   matmul_q5_1_q8_1_cm1_len,   matmul_q5_1_q8_1_cm1_दत्त,   परिमाण(vk_mat_mat_प्रक्षिप_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q8_0,   GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_cm1_int,   "matmul_q8_0_q8_1",   matmul_q8_0_q8_1_cm1_len,   matmul_q8_0_q8_1_cm1_दत्त,   परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            cm1_create_mmq({GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_cm1_int,   "matmul_iq4_nl_q8_1", matmul_iq4_nl_q8_1_cm1_len, matmul_iq4_nl_q8_1_cm1_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            cm1_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_cm1_int,   "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_cm1_len, matmul_iq4_xs_q8_1_cm1_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            cm1_create_mmq({GGML_TYPE_MXFP4,  GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_cm1_int,   "matmul_mxfp4_q8_1",  matmul_mxfp4_q8_1_cm1_len,  matmul_mxfp4_q8_1_cm1_दत्त,  परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            cm1_create_mmq({GGML_TYPE_Q3_K,   GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_cm1_int_k, "matmul_q3_k_q8_1",   matmul_q3_k_q8_1_cm1_len,   matmul_q3_k_q8_1_cm1_दत्त,   परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            यदि (!rdna4) { cm1_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_cm1_int,   "matmul_q4_k_q8_1",   matmul_q4_k_q8_1_cm1_len,   matmul_q4_k_q8_1_cm1_दत्त,   परिमाण(vk_mat_mat_प्रक्षिप_constants), 3); }
            यदि (!rdna4) { cm1_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_cm1_int,   "matmul_q5_k_q8_1",   matmul_q5_k_q8_1_cm1_len,   matmul_q5_k_q8_1_cm1_दत्त,   परिमाण(vk_mat_mat_प्रक्षिप_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q6_K,   GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_cm1_int_k, "matmul_q6_k_q8_1",   matmul_q6_k_q8_1_cm1_len,   matmul_q6_k_q8_1_cm1_दत्त,   परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            यदि (!rdna4) { cm1_create_mmq({GGML_TYPE_NVFP4, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_cm1_int_k, "matmul_nvfp4_q8_1",  matmul_nvfp4_q8_1_cm1_len,  matmul_nvfp4_q8_1_cm1_दत्त,  परिमाण(vk_mat_mat_प्रक्षिप_constants), 3); }
        }

        GGML_ASSERT(device->subgroup_ballot);

        cm1_create({GGML_TYPE_F32, GGML_TYPE_F32, सत्य, असत्य}, tc_mm, "matmul_id_subgroup_f32_f32", matmul_id_subgroup_f32_f32_cm1_len, matmul_id_subgroup_f32_f32_cm1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
        यदि (device->coopmat_acc_f16_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, सत्य, सत्य},  tc_mm, "matmul_id_subgroup_f16_f16acc",     matmul_id_subgroup_f16_f16acc_cm1_len,     matmul_id_subgroup_f16_f16acc_cm1_दत्त,     परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, सत्य, सत्य},  tc_mm, "matmul_id_subgroup_f16_f32_f16acc", matmul_id_subgroup_f16_f32_f16acc_cm1_len, matmul_id_subgroup_f16_f32_f16acc_cm1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
        }
        यदि (device->coopmat_acc_f32_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, सत्य, असत्य}, tc_mm, "matmul_id_subgroup_f16",     matmul_id_subgroup_f16_cm1_len,     matmul_id_subgroup_f16_cm1_दत्त,     परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, सत्य, असत्य}, tc_mm, "matmul_id_subgroup_f16_f32", matmul_id_subgroup_f16_f32_cm1_len, matmul_id_subgroup_f16_f32_cm1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
        }
#यदि defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        यदि (device->coopmat_bf16_support) {
            cm1_create({GGML_TYPE_BF16, GGML_TYPE_BF16, सत्य, असत्य}, tc_mm, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_cm1_len, matmul_id_subgroup_bf16_cm1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
        }
#यदि_अन्तः
        प्रति (ध्रुव स्वचालित type : non_lut_quant_types) {
            यदि (device->coopmat_acc_f16_support) {
                cm1_create_quant({type, GGML_TYPE_F32, सत्य, सत्य},  tc_mmq_id, "matmul_id_subgroup_quant_f32_f16acc", matmul_id_subgroup_quant_f32_f16acc_cm1_len, matmul_id_subgroup_quant_f32_f16acc_cm1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                cm1_create_quant({type, GGML_TYPE_F16, सत्य, सत्य},  tc_mmq_id, "matmul_id_subgroup_quant_f16_f16acc", matmul_id_subgroup_quant_f16_f16acc_cm1_len, matmul_id_subgroup_quant_f16_f16acc_cm1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
            }
            यदि (device->coopmat_acc_f32_support) {
                cm1_create_quant({type, GGML_TYPE_F32, सत्य, असत्य}, tc_mmq_id, "matmul_id_subgroup_quant_f32",        matmul_id_subgroup_quant_f32_cm1_len,        matmul_id_subgroup_quant_f32_cm1_दत्त,        परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                cm1_create_quant({type, GGML_TYPE_F16, सत्य, असत्य}, tc_mmq_id, "matmul_id_subgroup_quant_f16",        matmul_id_subgroup_quant_f16_cm1_len,        matmul_id_subgroup_quant_f16_cm1_दत्त,        परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
            }
        }
        // The _f16 variants provide the f16 B-type pipeline used when y_non_contig converts f32->f16.
#परिभाषय X_CM1_ID(TYPE, tstr) \
        यदि (device->coopmat_acc_f16_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, सत्य, सत्य},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_f16acc", matmul_id_subgroup_##tstr##_f32_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f32_f16acc_cm1_data, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्); \
            cm1_create({TYPE, GGML_TYPE_F16, सत्य, सत्य},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_f16acc", matmul_id_subgroup_##tstr##_f16_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f16_f16acc_cm1_data, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्); \
        } \
        यदि (device->coopmat_acc_f32_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, सत्य, असत्य}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32",        matmul_id_subgroup_##tstr##_f32_cm1_len,        matmul_id_subgroup_##tstr##_f32_cm1_data,        परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्); \
            cm1_create({TYPE, GGML_TYPE_F16, सत्य, असत्य}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16",        matmul_id_subgroup_##tstr##_f16_cm1_len,        matmul_id_subgroup_##tstr##_f16_cm1_data,        परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्); \
        }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM1_ID)
#यदि defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        यदि (device->ocp_fp4) {
#परिभाषय X_CM1_ID_OCP(TYPE, tstr) \
            यदि (device->coopmat_acc_f16_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, सत्य, सत्य},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_ocp_f16acc", matmul_id_subgroup_##tstr##_f32_ocp_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f32_ocp_f16acc_cm1_data, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्); \
                cm1_create({TYPE, GGML_TYPE_F16, सत्य, सत्य},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_ocp_f16acc", matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm1_data, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्); \
            } \
            यदि (device->coopmat_acc_f32_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, सत्य, असत्य}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_ocp",        matmul_id_subgroup_##tstr##_f32_ocp_cm1_len,        matmul_id_subgroup_##tstr##_f32_ocp_cm1_data,        परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्); \
                cm1_create({TYPE, GGML_TYPE_F16, सत्य, असत्य}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_ocp",        matmul_id_subgroup_##tstr##_f16_ocp_cm1_len,        matmul_id_subgroup_##tstr##_f16_ocp_cm1_data,        परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्); \
            }
            FOR_EACH_LUT_FP4_TYPE(X_CM1_ID_OCP)
#अपरिभाषय X_CM1_ID_OCP
        } अन्यथा
#यदि_अन्तः
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM1_ID)
        }
#अपरिभाषय X_CM1_ID

        यदि (device->coopmat_int_support && (rdna3 || rdna4)) {
            cm1_create_mmq({GGML_TYPE_Q4_0,   GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_0_q8_1",   matmul_id_subgroup_q4_0_q8_1_cm1_len,   matmul_id_subgroup_q4_0_q8_1_cm1_दत्त,   परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
            cm1_create_mmq({GGML_TYPE_Q4_1,   GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_1_q8_1",   matmul_id_subgroup_q4_1_q8_1_cm1_len,   matmul_id_subgroup_q4_1_q8_1_cm1_दत्त,   परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
            cm1_create_mmq({GGML_TYPE_Q5_0,   GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_0_q8_1",   matmul_id_subgroup_q5_0_q8_1_cm1_len,   matmul_id_subgroup_q5_0_q8_1_cm1_दत्त,   परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
            cm1_create_mmq({GGML_TYPE_Q5_1,   GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_1_q8_1",   matmul_id_subgroup_q5_1_q8_1_cm1_len,   matmul_id_subgroup_q5_1_q8_1_cm1_दत्त,   परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
            cm1_create_mmq({GGML_TYPE_Q8_0,   GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmq_cm1_int,   "matmul_id_subgroup_q8_0_q8_1",   matmul_id_subgroup_q8_0_q8_1_cm1_len,   matmul_id_subgroup_q8_0_q8_1_cm1_दत्त,   परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
            cm1_create_mmq({GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmq_cm1_int,   "matmul_id_subgroup_iq4_nl_q8_1", matmul_id_subgroup_iq4_nl_q8_1_cm1_len, matmul_id_subgroup_iq4_nl_q8_1_cm1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
            cm1_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmq_cm1_int,   "matmul_id_subgroup_iq4_xs_q8_1", matmul_id_subgroup_iq4_xs_q8_1_cm1_len, matmul_id_subgroup_iq4_xs_q8_1_cm1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
            cm1_create_mmq({GGML_TYPE_MXFP4,  GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmq_cm1_int,   "matmul_id_subgroup_mxfp4_q8_1",  matmul_id_subgroup_mxfp4_q8_1_cm1_len,  matmul_id_subgroup_mxfp4_q8_1_cm1_दत्त,  परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
            cm1_create_mmq({GGML_TYPE_Q3_K,   GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmq_cm1_int_k, "matmul_id_subgroup_q3_k_q8_1",   matmul_id_subgroup_q3_k_q8_1_cm1_len,   matmul_id_subgroup_q3_k_q8_1_cm1_दत्त,   परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
            cm1_create_mmq({GGML_TYPE_Q4_K,   GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_k_q8_1",   matmul_id_subgroup_q4_k_q8_1_cm1_len,   matmul_id_subgroup_q4_k_q8_1_cm1_दत्त,   परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
            cm1_create_mmq({GGML_TYPE_Q5_K,   GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_k_q8_1",   matmul_id_subgroup_q5_k_q8_1_cm1_len,   matmul_id_subgroup_q5_k_q8_1_cm1_दत्त,   परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
            cm1_create_mmq({GGML_TYPE_Q6_K,   GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmq_cm1_int_k, "matmul_id_subgroup_q6_k_q8_1",   matmul_id_subgroup_q6_k_q8_1_cm1_len,   matmul_id_subgroup_q6_k_q8_1_cm1_दत्त,   परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
            यदि (!rdna4) { cm1_create_mmq({GGML_TYPE_NVFP4, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmq_cm1_int_k, "matmul_id_subgroup_nvfp4_q8_1",  matmul_id_subgroup_nvfp4_q8_1_cm1_len,  matmul_id_subgroup_nvfp4_q8_1_cm1_दत्त,  परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्); }
        }
    } अन्यथा
#यदि_अन्तः  // defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    {
        // Helper for subgroup path with dot2 selection and filtering
        स्वचालित sg_create = [&](vk_matmul_pipeline_कुञ्चिका कुञ्चिका, ध्रुव std::vector<vk_tile_config>& tc_आधारः,
                             ध्रुव std::string& name, size_t len, ध्रुव शून्य* दत्त, uint32_t pc_आकार, uint32_t pc,
                             uint32_t rsgs = 0) {
            स्वचालित tc = filter_tc(tc_आधारः, कुञ्चिका.type_a, कुञ्चिका.mul_mat_id);
            यदि (!tc.रिक्तानि()) create_mm_pipelines(कुञ्चिका, tc, name, len, दत्त, pc_आकार, pc,
                [&](ध्रुव std::vector<uint32_t>& wt, तार्किक a) { प्रत्यागम ggml_vk_mul_mm_spec(wt, a); },
                असत्य, rsgs > 0, rsgs);
        };
        स्वचालित sg_create_quant = [&](vk_matmul_pipeline_कुञ्चिका कुञ्चिका, ध्रुव std::vector<vk_tile_config>& tc_आधारः,
                                   ध्रुव std::string& name, size_t len, ध्रुव शून्य* दत्त, uint32_t pc_आकार, uint32_t pc,
                                   uint32_t rsgs = 0) {
            स्वचालित tc = filter_tc(tc_आधारः, कुञ्चिका.type_a, कुञ्चिका.mul_mat_id);
            यदि (!tc.रिक्तानि()) {
                spec_fn_t qs = [&, type_a=कुञ्चिका.type_a](ध्रुव std::vector<uint32_t>& wt, तार्किक a) { प्रत्यागम ggml_vk_mul_mm_spec_quant(wt, a, (uint32_t)type_a); };
                create_mm_pipelines(कुञ्चिका, tc, name, len, दत्त, pc_आकार, pc, qs, असत्य, rsgs > 0, rsgs);
            }
        };
#यदि defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        स्वचालित sg_create_mmq = [&](vk_matmul_pipeline_कुञ्चिका कुञ्चिका, ध्रुव std::vector<vk_tile_config>& tc_आधारः,
                                 ध्रुव std::string& name, size_t len, ध्रुव शून्य* दत्त, uint32_t pc_आकार, uint32_t pc,
                                 uint32_t rsgs = 0) {
            स्वचालित tc = filter_tc(tc_आधारः, कुञ्चिका.type_a, कुञ्चिका.mul_mat_id, सत्य);
            यदि (!tc.रिक्तानि()) {
                spec_fn_t तटस्थः = [](ध्रुव std::vector<uint32_t>& wt, तार्किक) { प्रत्यागम wt; };
                create_mm_pipelines(कुञ्चिका, tc, name, len, दत्त, pc_आकार, pc, तटस्थः, असत्य, rsgs > 0, rsgs, असत्य);
            }
        };
#यदि_अन्तः

        std::vector<vk_tile_config> tc_id = {{s_warptile_id, s_wg_denoms, s_align}, {m_warptile_id, m_wg_denoms, m_align}, {l_warptile_id, l_wg_denoms, l_align}};
        std::vector<vk_tile_config> tc_mmqid = {{s_warptile_mmqid, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid, l_mmq_wg_denoms, l_align}};

        यदि (device->fp16) {
            // FP16 subgroup path - with dot2 runtime selection
            #परिभाषय SPV_DOT2(NAME) (device->dot2_f16 ? NAME ## _dot2_len : NAME ## _len), (device->dot2_f16 ? NAME ## dot2_दत्त : NAME ## दत्त_2)
            #परिभाषय SPV_DOT2_F16ACC(NAME) (device->dot2_f16 ? NAME ## _dot2_f16acc_len : NAME ## _f16acc_len), (device->dot2_f16 ? NAME ## dot2_f16acc_दत्त : NAME ## f16acc_दत्त)

            sg_create({GGML_TYPE_F32, GGML_TYPE_F32, असत्य, असत्य}, tc_mm, "matmul_f32_f32", SPV_DOT2(matmul_f32_f32), परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            sg_create({GGML_TYPE_F32, GGML_TYPE_F16, असत्य, असत्य}, tc_mm, "matmul_f32_f16", SPV_DOT2(matmul_f32_f16), परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, असत्य, सत्य},  tc_mm, "matmul_f16_f16acc",     SPV_DOT2_F16ACC(matmul_f16),     परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, असत्य, असत्य}, tc_mm, "matmul_f16",            SPV_DOT2(matmul_f16),            परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, असत्य, सत्य},  tc_mm, "matmul_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_f16_f32), परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, असत्य, असत्य}, tc_mm, "matmul_f16_f32",        SPV_DOT2(matmul_f16_f32),        परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            // BF16 - no dot2
            sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, असत्य, असत्य}, tc_mm, "matmul_bf16", matmul_bf16_len, matmul_bf16_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);

            प्रति (ध्रुव स्वचालित type : non_lut_quant_types) {
                sg_create_quant({type, GGML_TYPE_F32, असत्य, सत्य},  tc_mmq, "matmul_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_quant_f32), परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_quant({type, GGML_TYPE_F32, असत्य, असत्य}, tc_mmq, "matmul_quant_f32",        SPV_DOT2(matmul_quant_f32),        परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            }
    #परिभाषय X_SG(TYPE, tstr) \
            sg_create({TYPE, GGML_TYPE_F32, असत्य, सत्य},  tc_mmq, "matmul_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_##tstr##_f32), परिमाण(vk_mat_mat_प्रक्षिप_constants), 3); \
            sg_create({TYPE, GGML_TYPE_F32, असत्य, असत्य}, tc_mmq, "matmul_" #tstr "_f32",        SPV_DOT2(matmul_##tstr##_f32),        परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            FOR_EACH_LUT_TYPE(X_SG)
#अपरिभाषय X_SG

#यदि defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            यदि (device->integer_dot_गुणनफलम्) {
                std::vector<vk_tile_config> tc_mmq_int = {{s_warptile_mmq_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int, l_mmq_wg_denoms, l_align}};
                std::vector<vk_tile_config> tc_mmq_int_k = {{s_warptile_mmq_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int_k, l_mmq_wg_denoms, l_align}};
                sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int, "matmul_q2_0_q8_1", matmul_q2_0_q8_1_len, matmul_q2_0_q8_1_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int, "matmul_q4_0_q8_1", matmul_q4_0_q8_1_len, matmul_q4_0_q8_1_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int, "matmul_q4_1_q8_1", matmul_q4_1_q8_1_len, matmul_q4_1_q8_1_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int, "matmul_q5_0_q8_1", matmul_q5_0_q8_1_len, matmul_q5_0_q8_1_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int, "matmul_q5_1_q8_1", matmul_q5_1_q8_1_len, matmul_q5_1_q8_1_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int, "matmul_q8_0_q8_1", matmul_q8_0_q8_1_len, matmul_q8_0_q8_1_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int, "matmul_mxfp4_q8_1", matmul_mxfp4_q8_1_len, matmul_mxfp4_q8_1_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int, "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_len, matmul_iq4_xs_q8_1_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int_k, "matmul_q2_k_q8_1", matmul_q2_k_q8_1_len, matmul_q2_k_q8_1_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int_k, "matmul_q3_k_q8_1", matmul_q3_k_q8_1_len, matmul_q3_k_q8_1_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int_k, "matmul_q4_k_q8_1", matmul_q4_k_q8_1_len, matmul_q4_k_q8_1_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int_k, "matmul_q5_k_q8_1", matmul_q5_k_q8_1_len, matmul_q5_k_q8_1_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int_k, "matmul_q6_k_q8_1", matmul_q6_k_q8_1_len, matmul_q6_k_q8_1_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int_k, "matmul_iq3_s_q8_1", matmul_iq3_s_q8_1_len, matmul_iq3_s_q8_1_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            }
#यदि_अन्तः

            यदि (device->subgroup_ballot && device->subgroup_require_पूर्णम्_support && subgroup_min_आकार_16) {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, सत्य, असत्य}, tc_id, "matmul_id_subgroup_f32_f32", SPV_DOT2(matmul_id_subgroup_f32_f32), परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, सत्य, सत्य},  tc_id, "matmul_id_subgroup_f16_f16acc",     SPV_DOT2_F16ACC(matmul_id_subgroup_f16),     परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, सत्य, असत्य}, tc_id, "matmul_id_subgroup_f16",            SPV_DOT2(matmul_id_subgroup_f16),            परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, सत्य, सत्य},  tc_id, "matmul_id_subgroup_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_f16_f32), परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, सत्य, असत्य}, tc_id, "matmul_id_subgroup_f16_f32",        SPV_DOT2(matmul_id_subgroup_f16_f32),        परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार_16);
                // BF16 id - no dot2
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, सत्य, असत्य}, tc_id, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_len, matmul_id_subgroup_bf16_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार_16);
                प्रति (ध्रुव स्वचालित type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, सत्य, सत्य},  tc_mmqid, "matmul_id_subgroup_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_quant_f32), परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार);
                    sg_create_quant({type, GGML_TYPE_F32, सत्य, असत्य}, tc_mmqid, "matmul_id_subgroup_quant_f32",        SPV_DOT2(matmul_id_subgroup_quant_f32),        परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार);
                }
        #परिभाषय X_SG_ID_SUB(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, सत्य, सत्य},  tc_mmqid, "matmul_id_subgroup_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_##tstr##_f32), परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार); \
                sg_create({TYPE, GGML_TYPE_F32, सत्य, असत्य}, tc_mmqid, "matmul_id_subgroup_" #tstr "_f32",        SPV_DOT2(matmul_id_subgroup_##tstr##_f32),        परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार);
                FOR_EACH_LUT_TYPE(X_SG_ID_SUB)
#अपरिभाषय X_SG_ID_SUB
#यदि defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
                यदि (device->integer_dot_गुणनफलम्) {
                    std::vector<vk_tile_config> tc_mmqid_int = {{s_warptile_mmqid_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int, l_mmq_wg_denoms, l_align}};
                    std::vector<vk_tile_config> tc_mmqid_int_k = {{s_warptile_mmqid_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int_k, l_mmq_wg_denoms, l_align}};
                    sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int, "matmul_id_subgroup_q2_0_q8_1", matmul_id_subgroup_q2_0_q8_1_len, matmul_id_subgroup_q2_0_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार);
                    sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int, "matmul_id_subgroup_q4_0_q8_1", matmul_id_subgroup_q4_0_q8_1_len, matmul_id_subgroup_q4_0_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार);
                    sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int, "matmul_id_subgroup_q4_1_q8_1", matmul_id_subgroup_q4_1_q8_1_len, matmul_id_subgroup_q4_1_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार);
                    sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int, "matmul_id_subgroup_q5_0_q8_1", matmul_id_subgroup_q5_0_q8_1_len, matmul_id_subgroup_q5_0_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार);
                    sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int, "matmul_id_subgroup_q5_1_q8_1", matmul_id_subgroup_q5_1_q8_1_len, matmul_id_subgroup_q5_1_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार);
                    sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int, "matmul_id_subgroup_q8_0_q8_1", matmul_id_subgroup_q8_0_q8_1_len, matmul_id_subgroup_q8_0_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार);
                    sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int, "matmul_id_subgroup_mxfp4_q8_1", matmul_id_subgroup_mxfp4_q8_1_len, matmul_id_subgroup_mxfp4_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार);
                    sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int, "matmul_id_subgroup_iq4_xs_q8_1", matmul_id_subgroup_iq4_xs_q8_1_len, matmul_id_subgroup_iq4_xs_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार);
                    sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int_k, "matmul_id_subgroup_q2_k_q8_1", matmul_id_subgroup_q2_k_q8_1_len, matmul_id_subgroup_q2_k_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार_16);
                    sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int_k, "matmul_id_subgroup_q3_k_q8_1", matmul_id_subgroup_q3_k_q8_1_len, matmul_id_subgroup_q3_k_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार_16);
                    sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int_k, "matmul_id_subgroup_q4_k_q8_1", matmul_id_subgroup_q4_k_q8_1_len, matmul_id_subgroup_q4_k_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार_16);
                    sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int_k, "matmul_id_subgroup_q5_k_q8_1", matmul_id_subgroup_q5_k_q8_1_len, matmul_id_subgroup_q5_k_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार_16);
                    sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int_k, "matmul_id_subgroup_q6_k_q8_1", matmul_id_subgroup_q6_k_q8_1_len, matmul_id_subgroup_q6_k_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार_16);
                    sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int_k, "matmul_id_subgroup_iq3_s_q8_1", matmul_id_subgroup_iq3_s_q8_1_len, matmul_id_subgroup_iq3_s_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार_16);
                }
#यदि_अन्तः
            } अन्यथा {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, सत्य, असत्य}, tc_mm, "matmul_id_f32_f32", SPV_DOT2(matmul_id_f32_f32), परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, सत्य, सत्य},  tc_mm, "matmul_id_f16_f16acc",     SPV_DOT2_F16ACC(matmul_id_f16),     परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, सत्य, असत्य}, tc_mm, "matmul_id_f16",            SPV_DOT2(matmul_id_f16),            परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, सत्य, सत्य},  tc_mm, "matmul_id_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_f16_f32), परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, सत्य, असत्य}, tc_mm, "matmul_id_f16_f32",        SPV_DOT2(matmul_id_f16_f32),        परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                // BF16 id - no dot2
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, सत्य, असत्य}, tc_mm, "matmul_id_bf16", matmul_id_bf16_len, matmul_id_bf16_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                प्रति (ध्रुव स्वचालित type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, सत्य, सत्य},  tc_mmqid, "matmul_id_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_quant_f32), परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                    sg_create_quant({type, GGML_TYPE_F32, सत्य, असत्य}, tc_mmqid, "matmul_id_quant_f32",        SPV_DOT2(matmul_id_quant_f32),        परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                }
        #परिभाषय X_SG_ID(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, सत्य, सत्य},  tc_mmqid, "matmul_id_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_##tstr##_f32), परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्); \
                sg_create({TYPE, GGML_TYPE_F32, सत्य, असत्य}, tc_mmqid, "matmul_id_" #tstr "_f32",        SPV_DOT2(matmul_id_##tstr##_f32),        परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                FOR_EACH_LUT_TYPE(X_SG_ID)
#अपरिभाषय X_SG_ID
#यदि defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
                यदि (device->integer_dot_गुणनफलम्) {
                    std::vector<vk_tile_config> tc_mmqid_int = {{s_warptile_mmqid_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int, l_mmq_wg_denoms, l_align}};
                    std::vector<vk_tile_config> tc_mmqid_int_k = {{s_warptile_mmqid_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int_k, l_mmq_wg_denoms, l_align}};
                    sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int, "matmul_id_q2_0_q8_1", matmul_id_q2_0_q8_1_len, matmul_id_q2_0_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                    sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int, "matmul_id_q4_0_q8_1", matmul_id_q4_0_q8_1_len, matmul_id_q4_0_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                    sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int, "matmul_id_q4_1_q8_1", matmul_id_q4_1_q8_1_len, matmul_id_q4_1_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                    sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int, "matmul_id_q5_0_q8_1", matmul_id_q5_0_q8_1_len, matmul_id_q5_0_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                    sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int, "matmul_id_q5_1_q8_1", matmul_id_q5_1_q8_1_len, matmul_id_q5_1_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                    sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int, "matmul_id_q8_0_q8_1", matmul_id_q8_0_q8_1_len, matmul_id_q8_0_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                    sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int, "matmul_id_mxfp4_q8_1", matmul_id_mxfp4_q8_1_len, matmul_id_mxfp4_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                    sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int, "matmul_id_iq4_xs_q8_1", matmul_id_iq4_xs_q8_1_len, matmul_id_iq4_xs_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                    sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int_k, "matmul_id_q2_k_q8_1", matmul_id_q2_k_q8_1_len, matmul_id_q2_k_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                    sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int_k, "matmul_id_q3_k_q8_1", matmul_id_q3_k_q8_1_len, matmul_id_q3_k_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                    sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int_k, "matmul_id_q4_k_q8_1", matmul_id_q4_k_q8_1_len, matmul_id_q4_k_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                    sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int_k, "matmul_id_q5_k_q8_1", matmul_id_q5_k_q8_1_len, matmul_id_q5_k_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                    sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int_k, "matmul_id_q6_k_q8_1", matmul_id_q6_k_q8_1_len, matmul_id_q6_k_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                    sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, सत्य, असत्य}, tc_mmqid_int_k, "matmul_id_iq3_s_q8_1", matmul_id_iq3_s_q8_1_len, matmul_id_iq3_s_q8_1_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                }
#यदि_अन्तः
            }
            #अपरिभाषय SPV_DOT2
            #अपरिभाषय SPV_DOT2_F16ACC
        } अन्यथा {
            // FP32-only fallback path
            sg_create({GGML_TYPE_F32, GGML_TYPE_F32, असत्य, असत्य}, tc_mm, "matmul_f32_f32", matmul_f32_f32_fp32_len, matmul_f32_f32_fp32_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            sg_create({GGML_TYPE_F32, GGML_TYPE_F16, असत्य, असत्य}, tc_mm, "matmul_f32_f16", matmul_f32_f16_fp32_len, matmul_f32_f16_fp32_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, असत्य, असत्य}, tc_mm, "matmul_f16",     matmul_f16_fp32_len,     matmul_f16_fp32_दत्त,     परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, असत्य, असत्य}, tc_mm, "matmul_f16_f32", matmul_f16_f32_fp32_len, matmul_f16_f32_fp32_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, असत्य, असत्य}, tc_mm, "matmul_bf16",  matmul_bf16_fp32_len,    matmul_bf16_fp32_दत्त,    परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);

            प्रति (ध्रुव स्वचालित type : non_lut_quant_types) {
                sg_create_quant({type, GGML_TYPE_F32, असत्य, असत्य}, tc_mmq, "matmul_quant_f32", matmul_quant_f32_fp32_len, matmul_quant_f32_fp32_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            }
    #परिभाषय X_SG_FP32(TYPE, tstr) \
            sg_create({TYPE, GGML_TYPE_F32, असत्य, असत्य}, tc_mmq, "matmul_" #tstr "_f32", matmul_##tstr##_f32_fp32_len, matmul_##tstr##_f32_fp32_data, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            FOR_EACH_LUT_TYPE(X_SG_FP32)
#अपरिभाषय X_SG_FP32

#यदि defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            यदि (device->integer_dot_गुणनफलम्) {
                std::vector<vk_tile_config> tc_mmq_int = {{s_warptile_mmq_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int, l_mmq_wg_denoms, l_align}};
                std::vector<vk_tile_config> tc_mmq_int_k = {{s_warptile_mmq_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int_k, l_mmq_wg_denoms, l_align}};
                sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int, "matmul_q2_0_q8_1", matmul_q2_0_q8_1_fp32_len, matmul_q2_0_q8_1_fp32_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int, "matmul_q4_0_q8_1", matmul_q4_0_q8_1_fp32_len, matmul_q4_0_q8_1_fp32_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int, "matmul_q4_1_q8_1", matmul_q4_1_q8_1_fp32_len, matmul_q4_1_q8_1_fp32_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int, "matmul_q5_0_q8_1", matmul_q5_0_q8_1_fp32_len, matmul_q5_0_q8_1_fp32_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int, "matmul_q5_1_q8_1", matmul_q5_1_q8_1_fp32_len, matmul_q5_1_q8_1_fp32_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int, "matmul_q8_0_q8_1", matmul_q8_0_q8_1_fp32_len, matmul_q8_0_q8_1_fp32_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int, "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_fp32_len, matmul_iq4_xs_q8_1_fp32_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int_k, "matmul_q2_k_q8_1", matmul_q2_k_q8_1_fp32_len, matmul_q2_k_q8_1_fp32_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int_k, "matmul_q3_k_q8_1", matmul_q3_k_q8_1_fp32_len, matmul_q3_k_q8_1_fp32_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int_k, "matmul_q4_k_q8_1", matmul_q4_k_q8_1_fp32_len, matmul_q4_k_q8_1_fp32_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int_k, "matmul_q5_k_q8_1", matmul_q5_k_q8_1_fp32_len, matmul_q5_k_q8_1_fp32_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int_k, "matmul_q6_k_q8_1", matmul_q6_k_q8_1_fp32_len, matmul_q6_k_q8_1_fp32_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, असत्य, असत्य}, tc_mmq_int_k, "matmul_iq3_s_q8_1", matmul_iq3_s_q8_1_fp32_len, matmul_iq3_s_q8_1_fp32_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3);
            }
#यदि_अन्तः

            यदि (device->subgroup_ballot && device->subgroup_require_पूर्णम्_support && subgroup_min_आकार_16) {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, सत्य, असत्य}, tc_id, "matmul_id_subgroup_f32_f32", matmul_id_subgroup_f32_f32_fp32_len, matmul_id_subgroup_f32_f32_fp32_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, सत्य, असत्य}, tc_id, "matmul_id_subgroup_f16",     matmul_id_subgroup_f16_fp32_len,     matmul_id_subgroup_f16_fp32_दत्त,     परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, सत्य, असत्य}, tc_id, "matmul_id_subgroup_f16_f32", matmul_id_subgroup_f16_f32_fp32_len, matmul_id_subgroup_f16_f32_fp32_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार_16);
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, सत्य, असत्य}, tc_id, "matmul_id_subgroup_bf16",  matmul_id_subgroup_bf16_fp32_len,    matmul_id_subgroup_bf16_fp32_दत्त,    परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार_16);
                प्रति (ध्रुव स्वचालित type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, सत्य, असत्य}, tc_mmqid, "matmul_id_subgroup_quant_f32", matmul_id_subgroup_quant_f32_fp32_len, matmul_id_subgroup_quant_f32_fp32_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार);
                }
        #परिभाषय X_SG_ID_SUB_FP32(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, सत्य, असत्य}, tc_mmqid, "matmul_id_subgroup_" #tstr "_f32", matmul_id_subgroup_##tstr##_f32_fp32_len, matmul_id_subgroup_##tstr##_f32_fp32_data, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, mul_mat_subgroup_आकार);
                FOR_EACH_LUT_TYPE(X_SG_ID_SUB_FP32)
#अपरिभाषय X_SG_ID_SUB_FP32
            } अन्यथा {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, सत्य, असत्य}, tc_mm, "matmul_id_f32_f32", matmul_id_f32_f32_fp32_len, matmul_id_f32_f32_fp32_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, सत्य, असत्य}, tc_mm, "matmul_id_f16",     matmul_id_f16_fp32_len,     matmul_id_f16_fp32_दत्त,     परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, सत्य, असत्य}, tc_mm, "matmul_id_f16_f32", matmul_id_f16_f32_fp32_len, matmul_id_f16_f32_fp32_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, सत्य, असत्य}, tc_mm, "matmul_id_bf16",  matmul_id_bf16_fp32_len,    matmul_id_bf16_fp32_दत्त,    परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                प्रति (ध्रुव स्वचालित type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, सत्य, असत्य}, tc_mmqid, "matmul_id_quant_f32", matmul_id_quant_f32_fp32_len, matmul_id_quant_f32_fp32_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                }
        #परिभाषय X_SG_ID_FP32(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, सत्य, असत्य}, tc_mmqid, "matmul_id_" #tstr "_f32", matmul_id_##tstr##_f32_fp32_len, matmul_id_##tstr##_f32_fp32_data, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्);
                FOR_EACH_LUT_TYPE(X_SG_ID_FP32)
#अपरिभाषय X_SG_ID_FP32
            }
        }
    }
#अपरिभाषय FOR_EACH_LUT_TYPE
#अपरिभाषय FOR_EACH_LUT_TYPE_NONFP4
#अपरिभाषय FOR_EACH_LUT_FP4_TYPE
    // BF16 fallback for coopmat devices without bf16 coopmat support
    यदि ((device->coopmat2 || device->coopmat_support)
#यदि defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        && !device->coopmat_bf16_support
#यदि_अन्तः
        ) {
        ध्रुव uint32_t s_warptile_wm_bf16 = device->subgroup_आकार == 8 ? 8 : 32;
        std::vector<vk_tile_config> tc_bf16_fb = {
            {{ subgroup_आकार_32, 32, 32, 16, s_warptile_wm_bf16, 32, 2, 2, 2, 1, subgroup_आकार_8 }, {32, 32, 1}, s_align},
            {{ 128, 64, 64, 16, mm_warp_8, 32, 2, 4, 2, 1, mm_warp_8 }, {64, 64, 1}, m_align},
            {{ 128, 128, 128, 16, mm_warp_8 * 2, 64, 2, 4, 4, 1, mm_warp_8 }, {128, 128, 1}, l_align},
        };
        स्वचालित tc_bf16_filtered = filter_tc(tc_bf16_fb, GGML_TYPE_BF16, असत्य);
        स्वचालित tc_bf16_id_filtered = filter_tc(tc_bf16_fb, GGML_TYPE_BF16, सत्य);
        spec_fn_t bf16_spec = [&](ध्रुव std::vector<uint32_t>& wt, तार्किक a) { प्रत्यागम ggml_vk_mul_mm_spec(wt, a); };
        यदि (!tc_bf16_filtered.रिक्तानि()) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, असत्य, असत्य}, tc_bf16_filtered, "matmul_bf16", matmul_bf16_fp32_len, matmul_bf16_fp32_दत्त, परिमाण(vk_mat_mat_प्रक्षिप_constants), 3, bf16_spec);
        }
        यदि (!tc_bf16_id_filtered.रिक्तानि()) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, सत्य, असत्य}, tc_bf16_id_filtered, "matmul_id_bf16", matmul_id_bf16_fp32_len, matmul_id_bf16_fp32_दत्त, परिमाण(vk_mat_mat_id_प्रक्षिप_constants), mul_mat_id_param_गणनम्, bf16_spec);
        }
    }

    // Set up tile selector functions
    यदि (device->coopmat2) {
        device->matmul_tile_selector = [](uint32_t m, uint32_t n, uint32_t /*k*/, uint32_t shader_core_गणनम्,
                                          ध्रुव std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            यदि (configs.size() <= 1) प्रत्यागम 0;
            uint32_t चरमः = (uint32_t)configs.size() - 1;
            यदि (configs.size() == 2) {
                uint32_t crossover = configs[0].unaligned->wg_denoms[1];
                प्रत्यागम (n > crossover) ? 1 : 0;
            }
            // 3+ configs: s=0, m=1, l=2
            ध्रुव uint32_t tiles_l = CEIL_DIV(m, configs[चरमः].unaligned->wg_denoms[0]) * CEIL_DIV(n, configs[चरमः].unaligned->wg_denoms[1]);
            ध्रुव uint32_t tiles_m = CEIL_DIV(m, configs[1].unaligned->wg_denoms[0]) * CEIL_DIV(n, configs[1].unaligned->wg_denoms[1]);
            uint32_t crossover_large = configs[1].unaligned->wg_denoms[1];
            तार्किक prefer_large = tiles_m > shader_core_गणनम् || tiles_l > shader_core_गणनम् ||
                                (tiles_l <= shader_core_गणनम् / 3 && tiles_m > shader_core_गणनम् / 2);
            यदि (n > crossover_large && prefer_large) प्रत्यागम चरमः;
            uint32_t crossover_medium_m = configs[0].unaligned->wg_denoms[0];
            uint32_t crossover_medium_n = configs[0].unaligned->wg_denoms[1];
            यदि (m > crossover_medium_m && n > crossover_medium_n) प्रत्यागम 1;
            प्रत्यागम 0;
        };
        device->matmul_id_tile_selector = [](uint32_t /*m*/, uint32_t n, uint32_t /*k*/, uint32_t /*shader_core_count*/,
                                             ध्रुव std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            यदि (configs.size() <= 1) प्रत्यागम 0;
            uint32_t चरमः = (uint32_t)configs.size() - 1;
            यदि (configs.size() == 2) {
                uint32_t crossover = configs[0].unaligned->wg_denoms[1];
                प्रत्यागम (n > crossover) ? 1 : 0;
            }
            uint32_t crossover_large = configs[1].unaligned->wg_denoms[1];
            यदि (n > crossover_large) प्रत्यागम चरमः;
            uint32_t crossover_medium = configs[0].unaligned->wg_denoms[1];
            यदि (n > crossover_medium) प्रत्यागम 1;
            प्रत्यागम 0;
        };
    } अन्यथा {
        device->matmul_tile_selector = [](uint32_t m, uint32_t n, uint32_t /*k*/, uint32_t /*shader_core_count*/,
                                          ध्रुव std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            यदि (configs.size() <= 1) प्रत्यागम 0;
            यदि (m <= 32 || n <= 32) प्रत्यागम 0;
            यदि (configs.size() == 2) प्रत्यागम 1;
            यदि (m <= 64 || n <= 64) प्रत्यागम 1;
            प्रत्यागम (uint32_t)configs.size() - 1;
        };
        device->matmul_id_tile_selector = device->matmul_tile_selector;
    }

    // mul mat vec

    // the number of rows computed per shader depends on GPU model and quant
    uint32_t rm_stdq = 1;
    uint32_t rm_kq = 2;
    uint32_t rm_stdq_int = 1;
    uint32_t rm_kq_int = 1;
    स्वचालित ध्रुव &rm_iq_int = [](uint32_t i) { प्रत्यागम i == 0 ? 8u : 4u; };
    यदि (device->vendor_id == VK_VENDOR_ID_AMD) {
        यदि (device->architecture == AMD_GCN) {
            rm_stdq = 2;
            rm_kq = 4;
            rm_stdq_int = 4;
        }
    } अन्यथा यदि (device->vendor_id == VK_VENDOR_ID_INTEL) {
        rm_stdq = 2;
        rm_stdq_int = 2;
    }
    // RDNA3/4: above four columns, static 4 rows for all types bench faster than the default
    ध्रुव तार्किक is_rdna3_or_4 = device->vendor_id == VK_VENDOR_ID_AMD && (device->architecture == AMD_RDNA3 || device->architecture == AMD_RDNA4);
    स्वचालित ध्रुव &rm_int_n = [&](uint32_t पङ्क्तिकाः, uint32_t i) { प्रत्यागम (is_rdna3_or_4 && i >= 4) ? 4u : पङ्क्तिकाः; };
    // RDNA3/4 and NVIDIA except pre-Turing: use 4 rows for MUL_MAT_ID MMVQ.
    स्वचालित ध्रुव &rm_id = [&](uint32_t पङ्क्तिकाः) {
        यदि (device->vendor_id == VK_VENDOR_ID_NVIDIA &&
            device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) {
            प्रत्यागम 4u;
        }
        प्रत्यागम is_rdna3_or_4 ? 4u : पङ्क्तिकाः;
    };
    uint32_t rm_iq = 2 * rm_kq;

    ध्रुव तार्किक use_subgroups = device->subgroup_arithmetic;
    // The Imagination proprietary compiler rejects the subgroup-only dequant mul_mat_vec
    // shaders that require a subgroup size >= 16; fall back to shared-memory reduction.
    ध्रुव तार्किक is_imagination_proprietary =
        device->driver_id == vk::DriverId::eImaginationProprietary;
    // Ensure a subgroup size >= 16 is available
    ध्रुव तार्किक use_subgroups16 = use_subgroups && subgroup_min_आकार_16 && !is_imagination_proprietary;

    ध्रुव uint32_t subgroup_आकार = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_आकार_control && device->subgroup_min_आकार <= 16 && device->subgroup_max_आकार >= 16) ? 16 : device->subgroup_आकार;
    ध्रुव uint32_t subgroup_size16 = std::max(subgroup_आकार, 16u);

    ध्रुव uint32_t force_subgroup_आकार = use_subgroups ? subgroup_आकार : 0;
    ध्रुव uint32_t force_subgroup_size16 = use_subgroups16 ? subgroup_size16 : 0;
    स्थिर constexpr uint32_t mul_mat_vec_अङ्कः_bindings = 5;
    स्थिर constexpr uint32_t mul_mat_vec_id_अङ्कः_bindings = 6;

#यदि defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
#परिभाषय OCP_DMMV_LEN(NAME, REDUC)  (device->ocp_fp4 ? NAME ## _ocp_len[REDUC]  : NAME ## _len[REDUC])
#परिभाषय OCP_DMMV_दत्त(NAME, REDUC) (device->ocp_fp4 ? NAME ## ocp_दत्त[REDUC] : NAME ## दत्त_2[REDUC])
#अन्यथा
#परिभाषय OCP_DMMV_LEN(NAME, REDUC)  NAME ## _len[REDUC]
#परिभाषय OCP_DMMV_दत्त(NAME, REDUC) NAME ## दत्त_2[REDUC]
#यदि_अन्तः

    प्रति (uint32_t w = 0; w < DMMV_WG_आकार_गणनम्; ++w) {
        ध्रुव uint32_t wg_आकार_subgroup   = (w == DMMV_WG_आकार_SUBGROUP) ? subgroup_आकार : (subgroup_आकार * 4);
        ध्रुव uint32_t wg_आकार_subgroup16 = (w == DMMV_WG_आकार_SUBGROUP) ? subgroup_size16 : (subgroup_size16 * 4);

        ध्रुव shader_reduction_बहुलकम् reduc = (use_subgroups && w == DMMV_WG_आकार_SUBGROUP) ? SHADER_REDUCTION_बहुलकम्_SUBGROUP :
                                            (use_subgroups && w == DMMV_WG_आकार_LARGE) ? SHADER_REDUCTION_बहुलकम्_HYBRID :
                                            SHADER_REDUCTION_बहुलकम्_SHMEM;

        ध्रुव shader_reduction_बहुलकम् reduc16 = (use_subgroups16 && w == DMMV_WG_आकार_SUBGROUP) ? SHADER_REDUCTION_बहुलकम्_SUBGROUP :
                                              (use_subgroups16 && w == DMMV_WG_आकार_LARGE) ? SHADER_REDUCTION_बहुलकम्_HYBRID :
                                              SHADER_REDUCTION_बहुलकम्_SHMEM;

        प्रति (uint32_t i = 0; i < mul_mat_vec_max_स्तम्भसङ्ख्या; ++i) {
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f32_f32",  arr_dmmv_f32_f32_f32_len[reduc],  arr_dmmv_f32_f32_f32_दत्त[reduc],  "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {1, 1, 1}, {wg_आकार_subgroup, 1, i+1}, 1, असत्य, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f32_f32",  arr_dmmv_f16_f32_f32_len[reduc],  arr_dmmv_f16_f32_f32_दत्त[reduc],  "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {2, 1, 1}, {wg_आकार_subgroup, 2, i+1}, 1, असत्य, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f32_f32", arr_dmmv_bf16_f32_f32_len[reduc], arr_dmmv_bf16_f32_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {2, 1, 1}, {wg_आकार_subgroup, 2, i+1}, 1, असत्य, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f32_f32", arr_dmmv_q1_0_f32_f32_len[reduc], arr_dmmv_q1_0_f32_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, सत्य, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_f32_f32", arr_dmmv_q2_0_f32_f32_len[reduc], arr_dmmv_q2_0_f32_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, सत्य, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f32_f32", arr_dmmv_q4_0_f32_f32_len[reduc], arr_dmmv_q4_0_f32_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, सत्य, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f32_f32", arr_dmmv_q4_1_f32_f32_len[reduc], arr_dmmv_q4_1_f32_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, सत्य, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f32_f32", arr_dmmv_q5_0_f32_f32_len[reduc], arr_dmmv_q5_0_f32_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, सत्य, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f32_f32", arr_dmmv_q5_1_f32_f32_len[reduc], arr_dmmv_q5_1_f32_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, सत्य, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f32_f32", arr_dmmv_q8_0_f32_f32_len[reduc], arr_dmmv_q8_0_f32_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {1*rm_stdq, 1, 1}, {wg_आकार_subgroup, 1*rm_stdq, i+1}, 1, सत्य, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f32_f32", arr_dmmv_q2_k_f32_f32_len[reduc16], arr_dmmv_q2_k_f32_f32_दत्त[reduc16], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f32_f32", arr_dmmv_q3_k_f32_f32_len[reduc16], arr_dmmv_q3_k_f32_f32_दत्त[reduc16], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f32_f32", arr_dmmv_q4_k_f32_f32_len[reduc16], arr_dmmv_q4_k_f32_f32_दत्त[reduc16], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f32_f32", arr_dmmv_q5_k_f32_f32_len[reduc16], arr_dmmv_q5_k_f32_f32_दत्त[reduc16], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f32_f32", arr_dmmv_q6_k_f32_f32_len[reduc16], arr_dmmv_q6_k_f32_f32_दत्त[reduc16], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_TQ1_0][i], "mul_mat_vec_tq1_0_f32_f32", arr_dmmv_tq1_0_f32_f32_len[reduc16], arr_dmmv_tq1_0_f32_f32_दत्त[reduc16], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_TQ2_0][i], "mul_mat_vec_tq2_0_f32_f32", arr_dmmv_tq2_0_f32_f32_len[reduc16], arr_dmmv_tq2_0_f32_f32_दत्त[reduc16], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f32_f32",   arr_dmmv_iq1_s_f32_f32_len[reduc16],   arr_dmmv_iq1_s_f32_f32_दत्त[reduc16],   "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f32_f32",   arr_dmmv_iq1_m_f32_f32_len[reduc16],   arr_dmmv_iq1_m_f32_f32_दत्त[reduc16],   "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f32_f32", arr_dmmv_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_iq2_xxs_f32_f32_दत्त[reduc16], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f32_f32",  arr_dmmv_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_iq2_xs_f32_f32_दत्त[reduc16],  "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f32_f32",   arr_dmmv_iq2_s_f32_f32_len[reduc16],   arr_dmmv_iq2_s_f32_f32_दत्त[reduc16],   "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f32_f32", arr_dmmv_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_iq3_xxs_f32_f32_दत्त[reduc16], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f32_f32",   arr_dmmv_iq3_s_f32_f32_len[reduc16],   arr_dmmv_iq3_s_f32_f32_दत्त[reduc16],   "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f32_f32",  arr_dmmv_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_iq4_xs_f32_f32_दत्त[reduc16],  "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f32_f32",  arr_dmmv_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_iq4_nl_f32_f32_दत्त[reduc16],  "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f32_f32",   OCP_DMMV_LEN(arr_dmmv_mxfp4_f32_f32, reduc16), OCP_DMMV_दत्त(arr_dmmv_mxfp4_f32_f32, reduc16), "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f32_f32",   OCP_DMMV_LEN(arr_dmmv_nvfp4_f32_f32, reduc16), OCP_DMMV_दत्त(arr_dmmv_nvfp4_f32_f32, reduc16), "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f16_f32",  arr_dmmv_f32_f16_f32_len[reduc],  arr_dmmv_f32_f16_f32_दत्त[reduc],  "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {1, 1, 1}, {wg_आकार_subgroup, 1, i+1}, 1, असत्य, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f16_f32",  arr_dmmv_f16_f16_f32_len[reduc],  arr_dmmv_f16_f16_f32_दत्त[reduc],  "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {2, 1, 1}, {wg_आकार_subgroup, 2, i+1}, 1, असत्य, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f16_f32", arr_dmmv_bf16_f16_f32_len[reduc], arr_dmmv_bf16_f16_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {2, 1, 1}, {wg_आकार_subgroup, 2, i+1}, 1, असत्य, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f16_f32", arr_dmmv_q1_0_f16_f32_len[reduc], arr_dmmv_q1_0_f16_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, सत्य, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_f16_f32", arr_dmmv_q2_0_f16_f32_len[reduc], arr_dmmv_q2_0_f16_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, सत्य, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f16_f32", arr_dmmv_q4_0_f16_f32_len[reduc], arr_dmmv_q4_0_f16_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, सत्य, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f16_f32", arr_dmmv_q4_1_f16_f32_len[reduc], arr_dmmv_q4_1_f16_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, सत्य, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f16_f32", arr_dmmv_q5_0_f16_f32_len[reduc], arr_dmmv_q5_0_f16_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, सत्य, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f16_f32", arr_dmmv_q5_1_f16_f32_len[reduc], arr_dmmv_q5_1_f16_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, सत्य, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f16_f32", arr_dmmv_q8_0_f16_f32_len[reduc], arr_dmmv_q8_0_f16_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {1*rm_stdq, 1, 1}, {wg_आकार_subgroup, 1*rm_stdq, i+1}, 1, सत्य, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f16_f32", arr_dmmv_q2_k_f16_f32_len[reduc16], arr_dmmv_q2_k_f16_f32_दत्त[reduc16], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f16_f32", arr_dmmv_q3_k_f16_f32_len[reduc16], arr_dmmv_q3_k_f16_f32_दत्त[reduc16], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f16_f32", arr_dmmv_q4_k_f16_f32_len[reduc16], arr_dmmv_q4_k_f16_f32_दत्त[reduc16], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f16_f32", arr_dmmv_q5_k_f16_f32_len[reduc16], arr_dmmv_q5_k_f16_f32_दत्त[reduc16], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f16_f32", arr_dmmv_q6_k_f16_f32_len[reduc16], arr_dmmv_q6_k_f16_f32_दत्त[reduc16], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_TQ1_0][i], "mul_mat_vec_tq1_0_f16_f32", arr_dmmv_tq1_0_f16_f32_len[reduc16], arr_dmmv_tq1_0_f16_f32_दत्त[reduc16], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_TQ2_0][i], "mul_mat_vec_tq2_0_f16_f32", arr_dmmv_tq2_0_f16_f32_len[reduc16], arr_dmmv_tq2_0_f16_f32_दत्त[reduc16], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f16_f32",   arr_dmmv_iq1_s_f16_f32_len[reduc16],   arr_dmmv_iq1_s_f16_f32_दत्त[reduc16],   "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f16_f32",   arr_dmmv_iq1_m_f16_f32_len[reduc16],   arr_dmmv_iq1_m_f16_f32_दत्त[reduc16],   "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f16_f32", arr_dmmv_iq2_xxs_f16_f32_len[reduc16], arr_dmmv_iq2_xxs_f16_f32_दत्त[reduc16], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f16_f32",  arr_dmmv_iq2_xs_f16_f32_len[reduc16],  arr_dmmv_iq2_xs_f16_f32_दत्त[reduc16],  "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f16_f32",   arr_dmmv_iq2_s_f16_f32_len[reduc16],   arr_dmmv_iq2_s_f16_f32_दत्त[reduc16],   "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f16_f32", arr_dmmv_iq3_xxs_f16_f32_len[reduc16], arr_dmmv_iq3_xxs_f16_f32_दत्त[reduc16], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f16_f32",   arr_dmmv_iq3_s_f16_f32_len[reduc16],   arr_dmmv_iq3_s_f16_f32_दत्त[reduc16],   "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f16_f32",  arr_dmmv_iq4_xs_f16_f32_len[reduc16],  arr_dmmv_iq4_xs_f16_f32_दत्त[reduc16],  "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f16_f32",  arr_dmmv_iq4_nl_f16_f32_len[reduc16],  arr_dmmv_iq4_nl_f16_f32_दत्त[reduc16],  "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f16_f32",   OCP_DMMV_LEN(arr_dmmv_mxfp4_f16_f32, reduc16), OCP_DMMV_दत्त(arr_dmmv_mxfp4_f16_f32, reduc16), "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f16_f32",   OCP_DMMV_LEN(arr_dmmv_nvfp4_f16_f32, reduc16), OCP_DMMV_दत्त(arr_dmmv_nvfp4_f16_f32, reduc16), "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, सत्य, use_subgroups16, force_subgroup_size16);

#यदि defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            यदि (device->integer_dot_गुणनफलम्) {
                ध्रुव uint32_t subgroup_आकार_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_आकार_control) ? device->subgroup_min_आकार : device->subgroup_आकार;
                ध्रुव uint32_t wg_आकार_subgroup_int = (w == DMMV_WG_आकार_SUBGROUP) ? subgroup_आकार_int : (subgroup_आकार_int * 4);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_q8_1_f32", arr_dmmv_q2_0_q8_1_f32_len[reduc], arr_dmmv_q2_0_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_int_n(2*rm_kq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(2*rm_kq_int, i), i+1}, 1, सत्य, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_q8_1_f32", arr_dmmv_q4_0_q8_1_f32_len[reduc], arr_dmmv_q4_0_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, सत्य, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_q8_1_f32", arr_dmmv_q4_1_q8_1_f32_len[reduc], arr_dmmv_q4_1_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, सत्य, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_q8_1_f32", arr_dmmv_q5_0_q8_1_f32_len[reduc], arr_dmmv_q5_0_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, सत्य, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_q8_1_f32", arr_dmmv_q5_1_q8_1_f32_len[reduc], arr_dmmv_q5_1_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, सत्य, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_q8_1_f32", arr_dmmv_q8_0_q8_1_f32_len[reduc], arr_dmmv_q8_0_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, सत्य, use_subgroups, subgroup_आकार_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_MXFP4][i], "mul_mat_vec_mxfp4_q8_1_f32", arr_dmmv_mxfp4_q8_1_f32_len[reduc], arr_dmmv_mxfp4_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_int_n(2*rm_stdq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(2*rm_stdq_int, i), i+1}, 1, सत्य, use_subgroups, subgroup_आकार_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_q8_1_f32", arr_dmmv_q2_k_q8_1_f32_len[reduc], arr_dmmv_q2_k_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_int_n(2*rm_kq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(2*rm_kq_int, i), i+1}, 1, सत्य, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_q8_1_f32", arr_dmmv_q3_k_q8_1_f32_len[reduc], arr_dmmv_q3_k_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, सत्य, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_q8_1_f32", arr_dmmv_q4_k_q8_1_f32_len[reduc], arr_dmmv_q4_k_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, सत्य, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_q8_1_f32", arr_dmmv_q5_k_q8_1_f32_len[reduc], arr_dmmv_q5_k_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, सत्य, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_q8_1_f32", arr_dmmv_q6_k_q8_1_f32_len[reduc], arr_dmmv_q6_k_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, सत्य, use_subgroups, subgroup_आकार_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_S][i], "mul_mat_vec_iq1_s_q8_1_f32", arr_dmmv_iq1_s_q8_1_f32_len[reduc], arr_dmmv_iq1_s_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {1*rm_iq_int(i), 1, 1}, {wg_आकार_subgroup_int, 1*rm_iq_int(i), i+1}, 1, सत्य, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_M][i], "mul_mat_vec_iq1_m_q8_1_f32", arr_dmmv_iq1_m_q8_1_f32_len[reduc], arr_dmmv_iq1_m_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {1*rm_iq_int(i), 1, 1}, {wg_आकार_subgroup_int, 1*rm_iq_int(i), i+1}, 1, सत्य, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ4_XS][i], "mul_mat_vec_iq4_xs_q8_1_f32", arr_dmmv_iq4_xs_q8_1_f32_len[reduc], arr_dmmv_iq4_xs_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_प्रक्षिप_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, सत्य, use_subgroups, subgroup_आकार_int);

            }
#यदि_अन्तः // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
        }

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F32 ], "mul_mat_vec_id_f32_f32",        arr_dmmv_id_f32_f32_f32_len[reduc],     arr_dmmv_id_f32_f32_f32_दत्त[reduc],     "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {1, 1, 1}, {wg_आकार_subgroup, 1}, 1, असत्य, use_subgroups, force_subgroup_आकार);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F16 ], "mul_mat_vec_id_f16_f32",        arr_dmmv_id_f16_f32_f32_len[reduc],     arr_dmmv_id_f16_f32_f32_दत्त[reduc],     "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {2, 1, 1}, {wg_आकार_subgroup, 2}, 1, असत्य, use_subgroups, force_subgroup_आकार);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_BF16], "mul_mat_vec_id_bf16_f32",       arr_dmmv_id_bf16_f32_f32_len[reduc],    arr_dmmv_id_bf16_f32_f32_दत्त[reduc],    "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {2, 1, 1}, {wg_आकार_subgroup, 2}, 1, असत्य, use_subgroups, force_subgroup_आकार);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q1_0], "mul_mat_vec_id_q1_0_f32",       arr_dmmv_id_q1_0_f32_f32_len[reduc],    arr_dmmv_id_q1_0_f32_f32_दत्त[reduc],    "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq}, 1, सत्य, use_subgroups, force_subgroup_आकार);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q2_0], "mul_mat_vec_id_q2_0_f32",       arr_dmmv_id_q2_0_f32_f32_len[reduc],    arr_dmmv_id_q2_0_f32_f32_दत्त[reduc],    "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq}, 1, सत्य, use_subgroups, force_subgroup_आकार);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_f32",       arr_dmmv_id_q4_0_f32_f32_len[reduc],    arr_dmmv_id_q4_0_f32_f32_दत्त[reduc],    "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq}, 1, सत्य, use_subgroups, force_subgroup_आकार);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_f32",       arr_dmmv_id_q4_1_f32_f32_len[reduc],    arr_dmmv_id_q4_1_f32_f32_दत्त[reduc],    "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq}, 1, सत्य, use_subgroups, force_subgroup_आकार);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_f32",       arr_dmmv_id_q5_0_f32_f32_len[reduc],    arr_dmmv_id_q5_0_f32_f32_दत्त[reduc],    "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq}, 1, सत्य, use_subgroups, force_subgroup_आकार);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_f32",       arr_dmmv_id_q5_1_f32_f32_len[reduc],    arr_dmmv_id_q5_1_f32_f32_दत्त[reduc],    "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq}, 1, सत्य, use_subgroups, force_subgroup_आकार);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_f32",       arr_dmmv_id_q8_0_f32_f32_len[reduc],    arr_dmmv_id_q8_0_f32_f32_दत्त[reduc],    "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {1*rm_stdq, 1, 1}, {wg_आकार_subgroup, 1*rm_stdq}, 1, सत्य, use_subgroups, force_subgroup_आकार);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_f32",       arr_dmmv_id_q2_k_f32_f32_len[reduc16],    arr_dmmv_id_q2_k_f32_f32_दत्त[reduc16],    "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq}, 1, सत्य, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_f32",       arr_dmmv_id_q3_k_f32_f32_len[reduc16],    arr_dmmv_id_q3_k_f32_f32_दत्त[reduc16],    "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq}, 1, सत्य, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_f32",       arr_dmmv_id_q4_k_f32_f32_len[reduc16],    arr_dmmv_id_q4_k_f32_f32_दत्त[reduc16],    "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq}, 1, सत्य, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_f32",       arr_dmmv_id_q5_k_f32_f32_len[reduc16],    arr_dmmv_id_q5_k_f32_f32_दत्त[reduc16],    "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq}, 1, सत्य, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_f32",       arr_dmmv_id_q6_k_f32_f32_len[reduc16],    arr_dmmv_id_q6_k_f32_f32_दत्त[reduc16],    "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq}, 1, सत्य, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_TQ1_0], "mul_mat_vec_id_tq1_0_f32",     arr_dmmv_id_tq1_0_f32_f32_len[reduc16],   arr_dmmv_id_tq1_0_f32_f32_दत्त[reduc16],   "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq}, 1, सत्य, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_TQ2_0], "mul_mat_vec_id_tq2_0_f32",     arr_dmmv_id_tq2_0_f32_f32_len[reduc16],   arr_dmmv_id_tq2_0_f32_f32_दत्त[reduc16],   "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq}, 1, सत्य, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_S],   "mul_mat_vec_id_iq1_s_f32",   arr_dmmv_id_iq1_s_f32_f32_len[reduc16],   arr_dmmv_id_iq1_s_f32_f32_दत्त[reduc16],   "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, सत्य, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_M],   "mul_mat_vec_id_iq1_m_f32",   arr_dmmv_id_iq1_m_f32_f32_len[reduc16],   arr_dmmv_id_iq1_m_f32_f32_दत्त[reduc16],   "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, सत्य, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XXS], "mul_mat_vec_id_iq2_xxs_f32", arr_dmmv_id_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq2_xxs_f32_f32_दत्त[reduc16], "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, सत्य, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XS],  "mul_mat_vec_id_iq2_xs_f32",  arr_dmmv_id_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq2_xs_f32_f32_दत्त[reduc16],  "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, सत्य, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_S],   "mul_mat_vec_id_iq2_s_f32",   arr_dmmv_id_iq2_s_f32_f32_len[reduc16],   arr_dmmv_id_iq2_s_f32_f32_दत्त[reduc16],   "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, सत्य, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_XXS], "mul_mat_vec_id_iq3_xxs_f32", arr_dmmv_id_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq3_xxs_f32_f32_दत्त[reduc16], "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, सत्य, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_S],   "mul_mat_vec_id_iq3_s_f32",   arr_dmmv_id_iq3_s_f32_f32_len[reduc16],   arr_dmmv_id_iq3_s_f32_f32_दत्त[reduc16],   "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, सत्य, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_XS],  "mul_mat_vec_id_iq4_xs_f32",  arr_dmmv_id_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq4_xs_f32_f32_दत्त[reduc16],  "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, सत्य, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_NL],  "mul_mat_vec_id_iq4_nl_f32",  arr_dmmv_id_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_id_iq4_nl_f32_f32_दत्त[reduc16],  "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, सत्य, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_MXFP4],   "mul_mat_vec_id_mxfp4_f32",   OCP_DMMV_LEN(arr_dmmv_id_mxfp4_f32_f32, reduc16), OCP_DMMV_दत्त(arr_dmmv_id_mxfp4_f32_f32, reduc16), "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, सत्य, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_NVFP4],   "mul_mat_vec_id_nvfp4_f32",   OCP_DMMV_LEN(arr_dmmv_id_nvfp4_f32_f32, reduc16), OCP_DMMV_दत्त(arr_dmmv_id_nvfp4_f32_f32, reduc16), "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, सत्य, use_subgroups16, force_subgroup_size16);

#यदि defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        यदि (device->integer_dot_गुणनफलम्) {
            ध्रुव uint32_t subgroup_आकार_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_आकार_control) ? device->subgroup_min_आकार : device->subgroup_आकार;
            ध्रुव uint32_t wg_आकार_subgroup_int = (w == DMMV_WG_आकार_SUBGROUP) ? subgroup_आकार_int : (subgroup_आकार_int * 4);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q2_0], "mul_mat_vec_id_q2_0_q8_1_f32", arr_dmmv_id_q2_0_q8_1_f32_len[reduc], arr_dmmv_id_q2_0_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_id(2*rm_kq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(2*rm_kq_int)}, 1, सत्य, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_q8_1_f32", arr_dmmv_id_q4_0_q8_1_f32_len[reduc], arr_dmmv_id_q4_0_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(1*rm_stdq_int)}, 1, सत्य, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_q8_1_f32", arr_dmmv_id_q4_1_q8_1_f32_len[reduc], arr_dmmv_id_q4_1_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(1*rm_stdq_int)}, 1, सत्य, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_q8_1_f32", arr_dmmv_id_q5_0_q8_1_f32_len[reduc], arr_dmmv_id_q5_0_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(1*rm_stdq_int)}, 1, सत्य, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_q8_1_f32", arr_dmmv_id_q5_1_q8_1_f32_len[reduc], arr_dmmv_id_q5_1_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(1*rm_stdq_int)}, 1, सत्य, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_q8_1_f32", arr_dmmv_id_q8_0_q8_1_f32_len[reduc], arr_dmmv_id_q8_0_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(1*rm_stdq_int)}, 1, सत्य, use_subgroups, subgroup_आकार_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_MXFP4], "mul_mat_vec_id_mxfp4_q8_1_f32", arr_dmmv_id_mxfp4_q8_1_f32_len[reduc], arr_dmmv_id_mxfp4_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_id(2*rm_stdq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(2*rm_stdq_int)}, 1, सत्य, use_subgroups, subgroup_आकार_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_q8_1_f32", arr_dmmv_id_q2_k_q8_1_f32_len[reduc], arr_dmmv_id_q2_k_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_id(2*rm_kq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(2*rm_kq_int)}, 1, सत्य, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_q8_1_f32", arr_dmmv_id_q3_k_q8_1_f32_len[reduc], arr_dmmv_id_q3_k_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(1*rm_kq_int)}, 1, सत्य, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_q8_1_f32", arr_dmmv_id_q4_k_q8_1_f32_len[reduc], arr_dmmv_id_q4_k_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(1*rm_kq_int)}, 1, सत्य, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_q8_1_f32", arr_dmmv_id_q5_k_q8_1_f32_len[reduc], arr_dmmv_id_q5_k_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(1*rm_kq_int)}, 1, सत्य, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_q8_1_f32", arr_dmmv_id_q6_k_q8_1_f32_len[reduc], arr_dmmv_id_q6_k_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(1*rm_kq_int)}, 1, सत्य, use_subgroups, subgroup_आकार_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_S], "mul_mat_vec_id_iq1_s_q8_1_f32", arr_dmmv_id_iq1_s_q8_1_f32_len[reduc], arr_dmmv_id_iq1_s_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {1*rm_iq_int(0), 1, 1}, {wg_आकार_subgroup_int, 1*rm_iq_int(0)}, 1, सत्य, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_M], "mul_mat_vec_id_iq1_m_q8_1_f32", arr_dmmv_id_iq1_m_q8_1_f32_len[reduc], arr_dmmv_id_iq1_m_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {1*rm_iq_int(0), 1, 1}, {wg_आकार_subgroup_int, 1*rm_iq_int(0)}, 1, सत्य, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ4_XS], "mul_mat_vec_id_iq4_xs_q8_1_f32", arr_dmmv_id_iq4_xs_q8_1_f32_len[reduc], arr_dmmv_id_iq4_xs_q8_1_f32_दत्त[reduc], "main", mul_mat_vec_id_अङ्कः_bindings, परिमाण(vk_mat_vec_id_प्रक्षिप_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(1*rm_stdq_int)}, 1, सत्य, use_subgroups, subgroup_आकार_int);
        }
#यदि_अन्तः // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
    }

#अपरिभाषय OCP_DMMV_दत्त
#अपरिभाषय OCP_DMMV_LEN

#यदि !defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    GGML_अप्रयुक्तम्(rm_stdq_int);
    GGML_अप्रयुक्तम्(rm_kq_int);
    GGML_अप्रयुक्तम्(is_rdna3_or_4);
    GGML_अप्रयुक्तम्(rm_int_n);
    GGML_अप्रयुक्तम्(rm_id);
    GGML_अप्रयुक्तम्(rm_iq_int);
#यदि_अन्तः

    // dequant shaders
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_F32 ], "f32_to_f16",   dequant_f32_len,  dequant_f32_दत्त,  "main", 2, 5 * परिमाण(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q1_0], "dequant_q1_0", dequant_q1_0_len, dequant_q1_0_दत्त, "main", 2, 5 * परिमाण(uint32_t), {256 * 8, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_0], "dequant_q2_0", dequant_q2_0_len, dequant_q2_0_दत्त, "main", 2, 5 * परिमाण(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_0], "dequant_q4_0", dequant_q4_0_len, dequant_q4_0_दत्त, "main", 2, 5 * परिमाण(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_1], "dequant_q4_1", dequant_q4_1_len, dequant_q4_1_दत्त, "main", 2, 5 * परिमाण(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_0], "dequant_q5_0", dequant_q5_0_len, dequant_q5_0_दत्त, "main", 2, 5 * परिमाण(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_1], "dequant_q5_1", dequant_q5_1_len, dequant_q5_1_दत्त, "main", 2, 5 * परिमाण(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q8_0], "dequant_q8_0", dequant_q8_0_len, dequant_q8_0_दत्त, "main", 2, 5 * परिमाण(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_transpose[GGML_TYPE_Q8_0], "dequant_q8_0_transpose", dequant_q8_0_transpose_len, dequant_q8_0_transpose_दत्त, "main", 2, 5 * परिमाण(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_K], "dequant_q2_k", dequant_q2_k_len, dequant_q2_k_दत्त, "main", 2, 5 * परिमाण(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q3_K], "dequant_q3_k", dequant_q3_k_len, dequant_q3_k_दत्त, "main", 2, 5 * परिमाण(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_K], "dequant_q4_k", dequant_q4_k_len, dequant_q4_k_दत्त, "main", 2, 5 * परिमाण(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_K], "dequant_q5_k", dequant_q5_k_len, dequant_q5_k_दत्त, "main", 2, 5 * परिमाण(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q6_K], "dequant_q6_k", dequant_q6_k_len, dequant_q6_k_दत्त, "main", 2, 5 * परिमाण(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_TQ1_0], "dequant_tq1_0", dequant_tq1_0_len, dequant_tq1_0_दत्त, "main", 2, 5 * परिमाण(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_TQ2_0], "dequant_tq2_0", dequant_tq2_0_len, dequant_tq2_0_दत्त, "main", 2, 5 * परिमाण(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_S],   "dequant_iq1_s",   dequant_iq1_s_len,   dequant_iq1_s_दत्त,   "main", 2, 5 * परिमाण(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_M],   "dequant_iq1_m",   dequant_iq1_m_len,   dequant_iq1_m_दत्त,   "main", 2, 5 * परिमाण(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XXS], "dequant_iq2_xxs", dequant_iq2_xxs_len, dequant_iq2_xxs_दत्त, "main", 2, 5 * परिमाण(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XS],  "dequant_iq2_xs",  dequant_iq2_xs_len,  dequant_iq2_xs_दत्त,  "main", 2, 5 * परिमाण(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_S],   "dequant_iq2_s",   dequant_iq2_s_len,   dequant_iq2_s_दत्त,   "main", 2, 5 * परिमाण(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_XXS], "dequant_iq3_xxs", dequant_iq3_xxs_len, dequant_iq3_xxs_दत्त, "main", 2, 5 * परिमाण(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_S],   "dequant_iq3_s",   dequant_iq3_s_len,   dequant_iq3_s_दत्त,   "main", 2, 5 * परिमाण(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_XS],  "dequant_iq4_xs",  dequant_iq4_xs_len,  dequant_iq4_xs_दत्त,  "main", 2, 5 * परिमाण(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_NL],  "dequant_iq4_nl",  dequant_iq4_nl_len,  dequant_iq4_nl_दत्त,  "main", 2, 5 * परिमाण(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_MXFP4],   "dequant_mxfp4",   dequant_mxfp4_len,   dequant_mxfp4_दत्त,   "main", 2, 5 * परिमाण(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_NVFP4],   "dequant_nvfp4",   dequant_nvfp4_len,   dequant_nvfp4_दत्त,   "main", 2, 5 * परिमाण(uint32_t), {256 * 16, 1, 1}, {}, 1);

    // get_rows
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_F32 ], "get_rows_f32",  get_पङ्क्तिकाः_f32_len,  get_पङ्क्तिकाः_f32_दत्त,  "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_F16 ], "get_rows_f16",  get_पङ्क्तिकाः_f16_len,  get_पङ्क्तिकाः_f16_दत्त,  "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_BF16], "get_rows_bf16", get_पङ्क्तिकाः_bf16_len, get_पङ्क्तिकाः_bf16_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_Q1_0], "get_rows_q1_0", get_पङ्क्तिकाः_q1_0_len, get_पङ्क्तिकाः_q1_0_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_Q2_0], "get_rows_q2_0", get_पङ्क्तिकाः_q2_0_len, get_पङ्क्तिकाः_q2_0_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_Q4_0], "get_rows_q4_0", get_पङ्क्तिकाः_q4_0_len, get_पङ्क्तिकाः_q4_0_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_Q4_1], "get_rows_q4_1", get_पङ्क्तिकाः_q4_1_len, get_पङ्क्तिकाः_q4_1_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_Q5_0], "get_rows_q5_0", get_पङ्क्तिकाः_q5_0_len, get_पङ्क्तिकाः_q5_0_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_Q5_1], "get_rows_q5_1", get_पङ्क्तिकाः_q5_1_len, get_पङ्क्तिकाः_q5_1_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_Q8_0], "get_rows_q8_0", get_पङ्क्तिकाः_q8_0_len, get_पङ्क्तिकाः_q8_0_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_Q2_K], "get_rows_q2_k", get_पङ्क्तिकाः_q2_k_len, get_पङ्क्तिकाः_q2_k_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_Q3_K], "get_rows_q3_k", get_पङ्क्तिकाः_q3_k_len, get_पङ्क्तिकाः_q3_k_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_Q4_K], "get_rows_q4_k", get_पङ्क्तिकाः_q4_k_len, get_पङ्क्तिकाः_q4_k_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_Q5_K], "get_rows_q5_k", get_पङ्क्तिकाः_q5_k_len, get_पङ्क्तिकाः_q5_k_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_Q6_K], "get_rows_q6_k", get_पङ्क्तिकाः_q6_k_len, get_पङ्क्तिकाः_q6_k_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_TQ1_0], "get_rows_tq1_0", get_पङ्क्तिकाः_tq1_0_len, get_पङ्क्तिकाः_tq1_0_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_TQ2_0], "get_rows_tq2_0", get_पङ्क्तिकाः_tq2_0_len, get_पङ्क्तिकाः_tq2_0_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_IQ1_S],   "get_rows_iq1_s",   get_पङ्क्तिकाः_iq1_s_len,   get_पङ्क्तिकाः_iq1_s_दत्त,   "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_IQ1_M],   "get_rows_iq1_m",   get_पङ्क्तिकाः_iq1_m_len,   get_पङ्क्तिकाः_iq1_m_दत्त,   "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs", get_पङ्क्तिकाः_iq2_xxs_len, get_पङ्क्तिकाः_iq2_xxs_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs",  get_पङ्क्तिकाः_iq2_xs_len,  get_पङ्क्तिकाः_iq2_xs_दत्त,  "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_IQ2_S],   "get_rows_iq2_s",   get_पङ्क्तिकाः_iq2_s_len,   get_पङ्क्तिकाः_iq2_s_दत्त,   "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs", get_पङ्क्तिकाः_iq3_xxs_len, get_पङ्क्तिकाः_iq3_xxs_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_IQ3_S],   "get_rows_iq3_s",   get_पङ्क्तिकाः_iq3_s_len,   get_पङ्क्तिकाः_iq3_s_दत्त,   "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs",  get_पङ्क्तिकाः_iq4_xs_len,  get_पङ्क्तिकाः_iq4_xs_दत्त,  "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl",  get_पङ्क्तिकाः_iq4_nl_len,  get_पङ्क्तिकाः_iq4_nl_दत्त,  "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_MXFP4],   "get_rows_mxfp4",   get_पङ्क्तिकाः_mxfp4_len,   get_पङ्क्तिकाः_mxfp4_दत्त,   "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_NVFP4],   "get_rows_nvfp4",   get_पङ्क्तिकाः_nvfp4_len,   get_पङ्क्तिकाः_nvfp4_दत्त,   "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः[GGML_TYPE_I32],     "get_rows_i32",     get_पङ्क्तिकाः_i32_len,     get_पङ्क्तिकाः_i32_दत्त,     "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_F32 ], "get_rows_f32_f32",  get_पङ्क्तिकाः_f32_f32_len,  get_पङ्क्तिकाः_f32_f32_दत्त,  "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_F16 ], "get_rows_f16_f32",  get_पङ्क्तिकाः_f16_f32_len,  get_पङ्क्तिकाः_f16_f32_दत्त,  "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_BF16], "get_rows_bf16_f32", get_पङ्क्तिकाः_bf16_f32_len, get_पङ्क्तिकाः_bf16_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_Q1_0], "get_rows_q1_0_f32", get_पङ्क्तिकाः_q1_0_f32_len, get_पङ्क्तिकाः_q1_0_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_Q2_0], "get_rows_q2_0_f32", get_पङ्क्तिकाः_q2_0_f32_len, get_पङ्क्तिकाः_q2_0_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_Q4_0], "get_rows_q4_0_f32", get_पङ्क्तिकाः_q4_0_f32_len, get_पङ्क्तिकाः_q4_0_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_Q4_1], "get_rows_q4_1_f32", get_पङ्क्तिकाः_q4_1_f32_len, get_पङ्क्तिकाः_q4_1_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_Q5_0], "get_rows_q5_0_f32", get_पङ्क्तिकाः_q5_0_f32_len, get_पङ्क्तिकाः_q5_0_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_Q5_1], "get_rows_q5_1_f32", get_पङ्क्तिकाः_q5_1_f32_len, get_पङ्क्तिकाः_q5_1_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_Q8_0], "get_rows_q8_0_f32", get_पङ्क्तिकाः_q8_0_f32_len, get_पङ्क्तिकाः_q8_0_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_Q2_K], "get_rows_q2_k_f32", get_पङ्क्तिकाः_q2_k_f32_len, get_पङ्क्तिकाः_q2_k_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_Q3_K], "get_rows_q3_k_f32", get_पङ्क्तिकाः_q3_k_f32_len, get_पङ्क्तिकाः_q3_k_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_Q4_K], "get_rows_q4_k_f32", get_पङ्क्तिकाः_q4_k_f32_len, get_पङ्क्तिकाः_q4_k_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_Q5_K], "get_rows_q5_k_f32", get_पङ्क्तिकाः_q5_k_f32_len, get_पङ्क्तिकाः_q5_k_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_Q6_K], "get_rows_q6_k_f32", get_पङ्क्तिकाः_q6_k_f32_len, get_पङ्क्तिकाः_q6_k_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_TQ1_0], "get_rows_tq1_0_f32", get_पङ्क्तिकाः_tq1_0_f32_len, get_पङ्क्तिकाः_tq1_0_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_TQ2_0], "get_rows_tq2_0_f32", get_पङ्क्तिकाः_tq2_0_f32_len, get_पङ्क्तिकाः_tq2_0_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_IQ1_S],   "get_rows_iq1_s_f32",   get_पङ्क्तिकाः_iq1_s_f32_len,   get_पङ्क्तिकाः_iq1_s_f32_दत्त,   "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_IQ1_M],   "get_rows_iq1_m_f32",   get_पङ्क्तिकाः_iq1_m_f32_len,   get_पङ्क्तिकाः_iq1_m_f32_दत्त,   "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs_f32", get_पङ्क्तिकाः_iq2_xxs_f32_len, get_पङ्क्तिकाः_iq2_xxs_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs_f32",  get_पङ्क्तिकाः_iq2_xs_f32_len,  get_पङ्क्तिकाः_iq2_xs_f32_दत्त,  "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_IQ2_S],   "get_rows_iq2_s_f32",   get_पङ्क्तिकाः_iq2_s_f32_len,   get_पङ्क्तिकाः_iq2_s_f32_दत्त,   "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs_f32", get_पङ्क्तिकाः_iq3_xxs_f32_len, get_पङ्क्तिकाः_iq3_xxs_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_IQ3_S],   "get_rows_iq3_s_f32",   get_पङ्क्तिकाः_iq3_s_f32_len,   get_पङ्क्तिकाः_iq3_s_f32_दत्त,   "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs_f32",  get_पङ्क्तिकाः_iq4_xs_f32_len,  get_पङ्क्तिकाः_iq4_xs_f32_दत्त,  "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl_f32",  get_पङ्क्तिकाः_iq4_nl_f32_len,  get_पङ्क्तिकाः_iq4_nl_f32_दत्त,  "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_MXFP4],   "get_rows_mxfp4_f32",   get_पङ्क्तिकाः_mxfp4_f32_len,   get_पङ्क्तिकाः_mxfp4_f32_दत्त,   "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_NVFP4],   "get_rows_nvfp4_f32",   get_पङ्क्तिकाः_nvfp4_f32_len,   get_पङ्क्तिकाः_nvfp4_f32_दत्त,   "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_पङ्क्तिकाः_पृष्ठभागः_f32, "get_rows_back_f32", get_पङ्क्तिकाः_पृष्ठभागः_f32_len, get_पङ्क्तिकाः_पृष्ठभागः_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {256, 1, 1}, {}, 1, सत्य);

    ggml_vk_create_pipeline(device, device->pipeline_matmul_विभज_k_reduce, "split_k_reduce", विभज_k_reduce_len, विभज_k_reduce_दत्त, "main", 2, 2 * परिमाण(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_flash_attn_विभज_k_reduce, "fa_split_k_reduce", fa_विभज_k_reduce_len, fa_विभज_k_reduce_दत्त, "main", 3, परिमाण(vk_op_flash_attn_विभज_k_reduce_प्रक्षिप_constants), {1, device->subgroup_आकार, 1}, {device->subgroup_आकार}, 1, सत्य);

#यदि defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    यदि (device->vendor_id == VK_VENDOR_ID_INTEL && (device->architecture == INTEL_XE2 || (device->architecture == INTEL_XE1 && device->coopmat_support && device->uma))) {
        स्वचालित ऊर्ध्वस्थम्_power_सम्बन्धि_2 = [&](uint32_t अन्तर्) {
            GGML_ASSERT(अन्तर् != 0);
            यदि (अन्तर् <= 1) प्रत्यागम 1u;
            uint32_t ret = अन्तर् - 1;
            ret |= ret >> 1;
            ret |= ret >> 2;
            ret |= ret >> 4;
            ret |= ret >> 8;
            ret |= ret >> 16;
            प्रत्यागम ret + 1;
            };

        uint32_t xe_native_sub_गणः_आकार = 16;
        यदि (device->architecture == INTEL_XE1) {
            xe_native_sub_गणः_आकार = 8;
        }

        प्रति (स्वचालित& it : device->pipeline_xe_fa_decode_dual_phases) {
            ध्रुव uint32_t विभज_p_chunk = 32;
            स्वचालित HdQk = it.first;
            स्वचालित& pipelines = it.second;
            uint32_t शीर्ष_dim_qk = std::get<0>(HdQk);
            uint32_t शीर्ष_dim_pv = std::get<1>(HdQk);
            uint32_t gqa_अनुपातः = std::get<2>(HdQk);
            uint32_t q_len = std::get<3>(HdQk);
            ध्रुव uint32_t बहिर्गमनम्_dim_प्रत्येकम्_wg = gqa_अनुपातः > 16 ? 8 : 16;
            uint32_t aligned_q_len = ऊर्ध्वस्थम्_power_सम्बन्धि_2(q_len);
            uint32_t गणः_sz_ph1 = std::min(std::max(aligned_q_len * xe_native_sub_गणः_आकार, 64u), 256u);
            uint32_t बहिर्गमनम्_प्रत्येकम्_wg_ph1 = std::min(q_len, 256u / xe_native_sub_गणः_आकार);
            uint32_t aligned_gqa_अनुपातः = ऊर्ध्वस्थम्_power_सम्बन्धि_2(gqa_अनुपातः);
            uint32_t विभज_p_प्रत्येकम्_iter_ph2 = 256;
            uint32_t विभज_p_प्रत्येकम्_warp = 16;
            uint32_t गणः_sz_ph2 = (विभज_p_प्रत्येकम्_iter_ph2 / विभज_p_प्रत्येकम्_warp) * xe_native_sub_गणः_आकार;
            uint32_t बहिर्गमनम्_प्रत्येकम्_wg_ph2 = std::min(std::max(16u / aligned_gqa_अनुपातः, 1u), q_len);
            ggml_vk_create_pipeline(device, pipelines.first, "xe_fa_decode_ph1", fa_decode_ph1_cm1_len, fa_decode_ph1_cm1_दत्त, "main", 5, परिमाण(vk_fa_xe_opt_प्रक्षिप_constants), { 1, 32, 1 }, { गणः_sz_ph1, gqa_अनुपातः, शीर्ष_dim_qk, xe_native_sub_गणः_आकार, विभज_p_chunk, बहिर्गमनम्_प्रत्येकम्_wg_ph1 }, 1, असत्य, सत्य, xe_native_sub_गणः_आकार);
            ggml_vk_create_pipeline(device, pipelines.second, "xe_fa_decode_ph2", fa_decode_ph2_cm1_len, fa_decode_ph2_cm1_दत्त, "main", 5, परिमाण(vk_fa_xe_opt_प्रक्षिप_constants), { 1, 1, 1 }, { गणः_sz_ph2, gqa_अनुपातः, शीर्ष_dim_pv, बहिर्गमनम्_प्रत्येकम्_wg_ph2, xe_native_sub_गणः_आकार, विभज_p_प्रत्येकम्_iter_ph2, विभज_p_chunk, बहिर्गमनम्_dim_प्रत्येकम्_wg }, 1, असत्य, सत्य, xe_native_sub_गणः_आकार);
        }
    }
#यदि_अन्तः

    प्रति (स्वचालित &it : device->pipeline_fa_आवरणम्_opt) {
        स्वचालित BrBc = it.first;
        ggml_vk_create_pipeline(device, it.second, "fa_mask_opt", fa_आवरणम्_opt_len, fa_आवरणम्_opt_दत्त, "main", 2, परिमाण(vk_op_flash_attn_आवरणम्_opt_प्रक्षिप_constants), {1, 1, 1}, {128, 128 / device->subgroup_आकार, BrBc.first, BrBc.second}, 1, सत्य, सत्य, device->subgroup_आकार);
    }

    {
        // Large workgroup so the per-row KV scan parallelizes; capped to device limits.
        ध्रुव uint32_t compact_max = std::min({1024u, device->properties.limits.max_Compute_Work_गणः_Invocations, device->properties.limits.max_Compute_Work_गणः_आकार[0]});

        // Fast ballot prefix-sum path when the device supports full subgroups; otherwise
        // a shared-memory prefix-sum fallback. Both emit a deterministic ascending list.
        device->fa_sparse_compact_use_subgroups = device->subgroup_ballot && device->subgroup_require_पूर्णम्_support;
        यदि (device->fa_sparse_compact_use_subgroups) {
            ध्रुव uint32_t compact_wg = std::max(device->subgroup_आकार, (compact_max / device->subgroup_आकार) * device->subgroup_आकार);
            ध्रुव uint32_t compact_अङ्कः_sg = compact_wg / device->subgroup_आकार;
            ggml_vk_create_pipeline(device, device->pipeline_fa_sparse_compact_subgroup, "fa_sparse_compact_subgroup", fa_sparse_compact_subgroup_len, fa_sparse_compact_subgroup_दत्त, "main", 2, परिमाण(vk_op_flash_attn_sparse_compact_प्रक्षिप_constants), {1, 1, 1}, {compact_wg, compact_अङ्कः_sg}, 1, सत्य, सत्य, device->subgroup_आकार);
        } अन्यथा {
            ggml_vk_create_pipeline(device, device->pipeline_fa_sparse_compact, "fa_sparse_compact", fa_sparse_compact_len, fa_sparse_compact_दत्त, "main", 2, परिमाण(vk_op_flash_attn_sparse_compact_प्रक्षिप_constants), {1, 1, 1}, {compact_max}, 1, सत्य);
        }
    }

    यदि (device->subgroup_clustered && device->subgroup_require_पूर्णम्_support) {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_subgroup_len, quantize_q8_1_x4_subgroup_दत्त, "main", 2, परिमाण(vk_quantize_q8_1_प्रक्षिप_constants), {32 * device->subgroup_आकार / 8, 1, 1}, { device->subgroup_आकार }, 1, सत्य, सत्य);
    } अन्यथा {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_len, quantize_q8_1_x4_दत्त, "main", 2, परिमाण(vk_quantize_q8_1_प्रक्षिप_constants), {32 * device->subgroup_आकार / 8, 1, 1}, { device->subgroup_आकार }, 1);
    }

    प्रति (uint32_t i = 0; i < p021_max_gqa_अनुपातः; ++i) {
        यदि (device->subgroup_arithmetic && device->subgroup_require_पूर्णम्_support) {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_subgroup_सङ्कलय_len, mul_mat_vec_p021_f16_f32_subgroup_सङ्कलय_दत्त, "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_p021_प्रक्षिप_constants), {1, 1, 1}, {device->subgroup_आकार, i + 1}, 1, सत्य, सत्य);
        } अन्यथा {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_len,              mul_mat_vec_p021_f16_f32_दत्त,              "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_p021_प्रक्षिप_constants), {1, 1, 1}, {device->subgroup_आकार, i + 1}, 1, सत्य);
        }
    }
    ggml_vk_create_pipeline(device, device->pipeline_mul_mat_vec_nc_f16_f32, "mul_mat_vec_nc_f16_f32", mul_mat_vec_nc_f16_f32_len, mul_mat_vec_nc_f16_f32_दत्त, "main", mul_mat_vec_अङ्कः_bindings, परिमाण(vk_mat_vec_nc_प्रक्षिप_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_norm_f32, "norm_f32", norm_f32_len, norm_f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_गणः_norm_f32, "group_norm_f32", गणः_norm_f32_len, गणः_norm_f32_दत्त, "main", 2, परिमाण(vk_op_प्रक्षिप_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_f32, "rms_norm_f32", rms_norm_f32_len, rms_norm_f32_दत्त, "main", 4, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, {0, 0}, 1, सत्य);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_f32, "rms_norm_mul_f32", rms_norm_f32_len, rms_norm_f32_दत्त, "main", 4, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, {0, 1}, 1, सत्य);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_सङ्कलय_f32, "rms_norm_mul_add_f32", rms_norm_mul_सङ्कलय_f32_len, rms_norm_mul_सङ्कलय_f32_दत्त, "main", 5, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, {0, 1, 0}, 1, सत्य);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_सङ्कलय_mul_f32, "rms_norm_mul_add_mul_f32", rms_norm_mul_सङ्कलय_f32_len, rms_norm_mul_सङ्कलय_f32_दत्त, "main", 5, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, {0, 1, 1}, 1, सत्य);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_सङ्कलय_partials_f32, "rms_norm_mul_add_partials_f32", rms_norm_mul_सङ्कलय_partials_f32_len, rms_norm_mul_सङ्कलय_partials_f32_दत्त, "main", 6, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, {0, 1, 0}, 1, सत्य);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_सङ्कलय_mul_partials_f32, "rms_norm_mul_add_mul_partials_f32", rms_norm_mul_सङ्कलय_partials_f32_len, rms_norm_mul_सङ्कलय_partials_f32_दत्त, "main", 6, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, {0, 1, 1}, 1, सत्य);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_समुच्चयः_पङ्क्तिकाः_f32_f32, "rms_norm_set_rows_f32_f32", rms_norm_समुच्चयः_पङ्क्तिकाः_f32_f32_len, rms_norm_समुच्चयः_पङ्क्तिकाः_f32_f32_दत्त, "main", 4, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, {0, 0}, 1, सत्य);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_समुच्चयः_पङ्क्तिकाः_f32_f16, "rms_norm_set_rows_f32_f16", rms_norm_समुच्चयः_पङ्क्तिकाः_f32_f16_len, rms_norm_समुच्चयः_पङ्क्तिकाः_f32_f16_दत्त, "main", 4, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, {0, 0}, 1, सत्य);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_partials_f32, "rms_norm_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_दत्त, "main", 4, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, {0, 0}, 1, सत्य);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_partials_f32, "rms_norm_mul_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_दत्त, "main", 4, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, {0, 1}, 1, सत्य);

    यदि (परिमाण(vk_op_rms_norm_mul_rope_प्रक्षिप_constants) <= device->properties.limits.max_प्रक्षिप_Constants_आकार) {
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f32, "rms_norm_mul_rope_f32_f32", rms_norm_mul_rope_f32_f32_len, rms_norm_mul_rope_f32_f32_दत्त, "main", 7, परिमाण(vk_op_rms_norm_mul_rope_प्रक्षिप_constants), {1, 1, 1}, {0, 1}, 1, सत्य);
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f16, "rms_norm_mul_rope_f32_f16", rms_norm_mul_rope_f32_f16_len, rms_norm_mul_rope_f32_f16_दत्त, "main", 7, परिमाण(vk_op_rms_norm_mul_rope_प्रक्षिप_constants), {1, 1, 1}, {0, 1}, 1, सत्य);
    }

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_पृष्ठभागः_f32, "rms_norm_back_f32", rms_norm_पृष्ठभागः_f32_len, rms_norm_पृष्ठभागः_f32_दत्त, "main", 3, परिमाण(vk_op_प्रक्षिप_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_l2_norm_f32, "l2_norm_f32", l2_norm_f32_len, l2_norm_f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_32, "cpy_32", cpy_32_len, cpy_32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_f16, "cpy_f32_f16", cpy_f32_f16_len, cpy_f32_f16_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_16, "cpy_16", cpy_16_len, cpy_16_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f16_f32, "cpy_f16_f32", cpy_f16_f32_len, cpy_f16_f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_bf16,"cpy_f32_bf16",cpy_f32_bf16_len,cpy_f32_bf16_दत्त,"main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_bf16_f32,"cpy_bf16_f32",cpy_bf16_f32_len,cpy_bf16_f32_दत्त,"main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_bf16_f16,"cpy_bf16_f16",cpy_bf16_f16_len,cpy_bf16_f16_दत्त,"main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_i32_f32, "cpy_i32_f32", cpy_i32_f32_len, cpy_i32_f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_i32, "cpy_f32_i32", cpy_f32_i32_len, cpy_f32_i32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_32, "contig_cpy_32", contig_cpy_32_len, contig_cpy_32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_f16, "contig_cpy_f32_f16", contig_cpy_f32_f16_len, contig_cpy_f32_f16_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_16, "contig_cpy_16", contig_cpy_16_len, contig_cpy_16_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f16_f32, "contig_cpy_f16_f32", contig_cpy_f16_f32_len, contig_cpy_f16_f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_bf16,"contig_cpy_f32_bf16",contig_cpy_f32_bf16_len,contig_cpy_f32_bf16_दत्त,"main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_bf16_f32,"contig_cpy_bf16_f32",contig_cpy_bf16_f32_len,contig_cpy_bf16_f32_दत्त,"main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_bf16_f16,"contig_cpy_bf16_f16",contig_cpy_bf16_f16_len,contig_cpy_bf16_f16_दत्त,"main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_i32_f32, "contig_cpy_i32_f32", contig_cpy_i32_f32_len, contig_cpy_i32_f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_i32, "contig_cpy_f32_i32", contig_cpy_f32_i32_len, contig_cpy_f32_i32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_32, "cpy_transpose_32", cpy_transpose_32_len, cpy_transpose_32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_16, "cpy_transpose_16", cpy_transpose_16_len, cpy_transpose_16_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_02_32, "cpy_transpose_02_32", cpy_transpose_02_32_len, cpy_transpose_02_32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_02_16, "cpy_transpose_02_16", cpy_transpose_02_16_len, cpy_transpose_02_16_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q1_0], "cpy_f32_q1_0", cpy_f32_q1_0_len, cpy_f32_q1_0_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q2_0], "cpy_f32_q2_0", cpy_f32_q2_0_len, cpy_f32_q2_0_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_0], "cpy_f32_q4_0", cpy_f32_q4_0_len, cpy_f32_q4_0_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_1], "cpy_f32_q4_1", cpy_f32_q4_1_len, cpy_f32_q4_1_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_0], "cpy_f32_q5_0", cpy_f32_q5_0_len, cpy_f32_q5_0_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_1], "cpy_f32_q5_1", cpy_f32_q5_1_len, cpy_f32_q5_1_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q8_0], "cpy_f32_q8_0", cpy_f32_q8_0_len, cpy_f32_q8_0_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_IQ4_NL], "cpy_f32_iq4_nl", cpy_f32_iq4_nl_len, cpy_f32_iq4_nl_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {32, 1, 1}, {}, 1);

#परिभाषय समुच्चयः_पङ्क्तिकाः_2(src_idx, src, itype) \
        ggml_vk_create_pipeline(device, device->pipeline_समुच्चयः_पङ्क्तिकाः ## itype [src_idx][GGML_TYPE_F32],  "set_rows_" #src "_f32" #itype,  समुच्चयः_पङ्क्तिकाः_3 ## src ## _f32 ## itype ## _len,  समुच्चयः_पङ्क्तिकाः_3 ## src ## _f32 ## itype ## दत्त_2,  "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, {1}, 1, सत्य); \
        ggml_vk_create_pipeline(device, device->pipeline_समुच्चयः_पङ्क्तिकाः ## itype [src_idx][GGML_TYPE_F16],  "set_rows_" #src "_f16" #itype,  समुच्चयः_पङ्क्तिकाः_3 ## src ## _f16 ## itype ## _len,  समुच्चयः_पङ्क्तिकाः_3 ## src ## _f16 ## itype ## दत्त_2,  "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, {1}, 1, सत्य); \
        ggml_vk_create_pipeline(device, device->pipeline_समुच्चयः_पङ्क्तिकाः ## itype [src_idx][GGML_TYPE_BF16], "set_rows_" #src "_bf16" #itype, समुच्चयः_पङ्क्तिकाः_3 ## src ## _bf16 ## itype ## _len, समुच्चयः_पङ्क्तिकाः_3 ## src ## _bf16 ## itype ## दत्त_2, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, {1}, 1, सत्य); \
        ggml_vk_create_pipeline(device, device->pipeline_समुच्चयः_पङ्क्तिकाः ## itype [src_idx][GGML_TYPE_Q1_0], "set_rows_" #src "_q1_0" #itype, समुच्चयः_पङ्क्तिकाः_3 ## src ## _q1_0 ## itype ## _len, समुच्चयः_पङ्क्तिकाः_3 ## src ## _q1_0 ## itype ## दत्त_2, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, {1}, 1, सत्य); \
        ggml_vk_create_pipeline(device, device->pipeline_समुच्चयः_पङ्क्तिकाः ## itype [src_idx][GGML_TYPE_Q2_0], "set_rows_" #src "_q2_0" #itype, समुच्चयः_पङ्क्तिकाः_3 ## src ## _q2_0 ## itype ## _len, समुच्चयः_पङ्क्तिकाः_3 ## src ## _q2_0 ## itype ## दत्त_2, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, {1}, 1, सत्य); \
        ggml_vk_create_pipeline(device, device->pipeline_समुच्चयः_पङ्क्तिकाः ## itype [src_idx][GGML_TYPE_Q4_0], "set_rows_" #src "_q4_0" #itype, समुच्चयः_पङ्क्तिकाः_3 ## src ## _q4_0 ## itype ## _len, समुच्चयः_पङ्क्तिकाः_3 ## src ## _q4_0 ## itype ## दत्त_2, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, {1}, 1, सत्य); \
        ggml_vk_create_pipeline(device, device->pipeline_समुच्चयः_पङ्क्तिकाः ## itype [src_idx][GGML_TYPE_Q4_1], "set_rows_" #src "_q4_1" #itype, समुच्चयः_पङ्क्तिकाः_3 ## src ## _q4_1 ## itype ## _len, समुच्चयः_पङ्क्तिकाः_3 ## src ## _q4_1 ## itype ## दत्त_2, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, {1}, 1, सत्य); \
        ggml_vk_create_pipeline(device, device->pipeline_समुच्चयः_पङ्क्तिकाः ## itype [src_idx][GGML_TYPE_Q5_0], "set_rows_" #src "_q5_0" #itype, समुच्चयः_पङ्क्तिकाः_3 ## src ## _q5_0 ## itype ## _len, समुच्चयः_पङ्क्तिकाः_3 ## src ## _q5_0 ## itype ## दत्त_2, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, {1}, 1, सत्य); \
        ggml_vk_create_pipeline(device, device->pipeline_समुच्चयः_पङ्क्तिकाः ## itype [src_idx][GGML_TYPE_Q5_1], "set_rows_" #src "_q5_1" #itype, समुच्चयः_पङ्क्तिकाः_3 ## src ## _q5_1 ## itype ## _len, समुच्चयः_पङ्क्तिकाः_3 ## src ## _q5_1 ## itype ## दत्त_2, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, {1}, 1, सत्य); \
        ggml_vk_create_pipeline(device, device->pipeline_समुच्चयः_पङ्क्तिकाः ## itype [src_idx][GGML_TYPE_Q8_0], "set_rows_" #src "_q8_0" #itype, समुच्चयः_पङ्क्तिकाः_3 ## src ## _q8_0 ## itype ## _len, समुच्चयः_पङ्क्तिकाः_3 ## src ## _q8_0 ## itype ## दत्त_2, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, {1}, 1, सत्य); \
        ggml_vk_create_pipeline(device, device->pipeline_समुच्चयः_पङ्क्तिकाः ## itype [src_idx][GGML_TYPE_IQ4_NL], "set_rows_" #src "_iq4_nl" #itype, समुच्चयः_पङ्क्तिकाः_3 ## src ## _iq4_nl ## itype ## _len, समुच्चयः_पङ्क्तिकाः_3 ## src ## _iq4_nl ## itype ## दत्त_2, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, {1}, 1, सत्य);

    समुच्चयः_पङ्क्तिकाः_2(0, f32, _i32)
    समुच्चयः_पङ्क्तिकाः_2(0, f32, _i64)
    समुच्चयः_पङ्क्तिकाः_2(1, f16, _i32)
    समुच्चयः_पङ्क्तिकाः_2(1, f16, _i64)
#अपरिभाषय समुच्चयः_पङ्क्तिकाः_2


    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q1_0], "cpy_q1_0_f32", cpy_q1_0_f32_len, cpy_q1_0_f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {(uint32_t)ggml_blck_आकार(GGML_TYPE_Q1_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q2_0], "cpy_q2_0_f32", cpy_q2_0_f32_len, cpy_q2_0_f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {(uint32_t)ggml_blck_आकार(GGML_TYPE_Q2_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_0], "cpy_q4_0_f32", cpy_q4_0_f32_len, cpy_q4_0_f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {(uint32_t)ggml_blck_आकार(GGML_TYPE_Q4_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_1], "cpy_q4_1_f32", cpy_q4_1_f32_len, cpy_q4_1_f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {(uint32_t)ggml_blck_आकार(GGML_TYPE_Q4_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_0], "cpy_q5_0_f32", cpy_q5_0_f32_len, cpy_q5_0_f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {(uint32_t)ggml_blck_आकार(GGML_TYPE_Q5_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_1], "cpy_q5_1_f32", cpy_q5_1_f32_len, cpy_q5_1_f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {(uint32_t)ggml_blck_आकार(GGML_TYPE_Q5_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q8_0], "cpy_q8_0_f32", cpy_q8_0_f32_len, cpy_q8_0_f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {(uint32_t)ggml_blck_आकार(GGML_TYPE_Q8_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_IQ4_NL], "cpy_iq4_nl_f32", cpy_iq4_nl_f32_len, cpy_iq4_nl_f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {(uint32_t)ggml_blck_आकार(GGML_TYPE_IQ4_NL), 1, 1}, {}, 1);

    स्वचालित get_suffix = [](तार्किक src0_f16, तार्किक src1_f16, तार्किक dst_f16) {
        std::string s;
        s += std::string(src0_f16 ? "_f16" : "_f32");
        s += std::string(src1_f16 ? "_f16" : "_f32");
        s += std::string(dst_f16 ? "_f16" : "_f32");
        प्रत्यागम s;
    };

#परिभाषय CREATE_द्विमान(name, namemod, spec, bindings) \
    प्रति (पूर्णाङ्क s0 : {0,1}) प्रति (पूर्णाङ्क s1 : {0,1}) प्रति (पूर्णाङ्क d : {0,1}) \
        ggml_vk_create_pipeline2(device, device->pipeline_ ## name ## namemod[s0][s1][d], \
                                #name + get_suffix(s0, s1, d) + #namemod, name ## _len[s0][s1][d], name ## दत्त_2[s0][s1][d], \
                                "main", (bindings), परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {512, 1, 1}, spec, 1);

    CREATE_द्विमान(सङ्कलय_2, , {0}, 4)
    CREATE_द्विमान(सङ्कलय_2, _norepeat, {1}, 4)
    CREATE_द्विमान(sub, , {0}, 3)
    CREATE_द्विमान(sub, _norepeat, {1}, 3)
    CREATE_द्विमान(mul, , {0}, 3)
    CREATE_द्विमान(mul, _norepeat, {1}, 3)
    CREATE_द्विमान(भाजय, , {0}, 3)
    CREATE_द्विमान(भाजय, _norepeat, {1}, 3)
    CREATE_द्विमान(सङ्कलय_rms, , {0}, 4)
    CREATE_द्विमान(सङ्कलय_rms, _norepeat, {1}, 4)
#अपरिभाषय CREATE_द्विमान

    यदि (device->multi_सङ्कलय) {
        प्रति (uint32_t i = 0; i < MAX_FUSED_ADDS; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_multi_सङ्कलय[i],     "multi_add_f32_"     + std::to_string(i+1), multi_सङ्कलय_f32_len,     multi_सङ्कलय_f32_दत्त,     "main", MAX_PARAMETER_गणनम्, परिमाण(vk_op_multi_सङ्कलय_प्रक्षिप_constants), {512, 1, 1}, {i+2}, 1);
            ggml_vk_create_pipeline2(device, device->pipeline_multi_सङ्कलय_rms[i], "multi_add_rms_f32_" + std::to_string(i+1), multi_सङ्कलय_rms_f32_len, multi_सङ्कलय_rms_f32_दत्त, "main", MAX_PARAMETER_गणनम्, परिमाण(vk_op_multi_सङ्कलय_प्रक्षिप_constants), {512, 1, 1}, {i+2}, 1);
        }
    }

    ggml_vk_create_pipeline(device, device->pipeline_सङ्कलय_id_f32, "add_id_f32", सङ्कलय_id_f32_len, सङ्कलय_id_f32_दत्त, "main", 4, परिमाण(vk_op_सङ्कलय_id_प्रक्षिप_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_acc_f32, "acc_f32", acc_f32_len, acc_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {512, 1, 1}, {0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_समुच्चयः_f32, "set_f32", acc_f32_len, acc_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {512, 1, 1}, {0, 0}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_concat_i8, "concat_i8", concat_i8_len, concat_i8_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i16, "concat_i16", concat_i16_len, concat_i16_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i32, "concat_i32", concat_i32_len, concat_i32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i64, "concat_i64", concat_i64_len, concat_i64_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_upscale_निकटतमः_f32, "upscale_f32", upscale_f32_len, upscale_f32_दत्त, "main", 2, परिमाण(vk_op_upscale_प्रक्षिप_constants), {512, 1, 1}, {GGML_SCALE_बहुलकम्_निकटतमः}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_f32, "upscale_f32", upscale_f32_len, upscale_f32_दत्त, "main", 2, परिमाण(vk_op_upscale_प्रक्षिप_constants), {512, 1, 1}, {GGML_SCALE_बहुलकम्_BILINEAR}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bicubic_f32, "upscale_f32", upscale_f32_len, upscale_f32_दत्त, "main", 2, परिमाण(vk_op_upscale_प्रक्षिप_constants), {512, 1, 1}, {GGML_SCALE_बहुलकम्_BICUBIC}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_antialias_f32, "upscale_f32", upscale_f32_len, upscale_f32_दत्त, "main", 2, परिमाण(vk_op_upscale_प्रक्षिप_constants), {512, 1, 1}, {GGML_SCALE_बहुलकम्_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_scale_f32, "scale_f32", scale_f32_len, scale_f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_log[0], "log_f32", log_f32_len, log_f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_log[1], "log_f16", log_f16_len, log_f16_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_tri[0], "tri_f32", tri_f32_len, tri_f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_tri[1], "tri_f16", tri_f16_len, tri_f16_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag[0], "diag_f32", diag_f32_len, diag_f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_diag[1], "diag_f16", diag_f16_len, diag_f16_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pad_f32, "pad_f32", pad_f32_len, pad_f32_दत्त, "main", 2, परिमाण(vk_op_pad_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_pad_reflect_1d_f32, "pad_reflect_1d_f32", pad_reflect_1d_f32_len, pad_reflect_1d_f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_roll_f32, "roll_f32", roll_f32_len, roll_f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_repeat_i32, "repeat_i32", repeat_i32_len, repeat_i32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_repeat_पृष्ठभागः_f32, "repeat_back_f32", repeat_पृष्ठभागः_f32_len, repeat_पृष्ठभागः_f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_repeat_i16, "repeat_i16", repeat_i16_len, repeat_i16_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);

#परिभाषय CREATE_UNARY(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## f32_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## f16_दत्त, "main", 2, परिमाण(vk_op_unary_प्रक्षिप_constants), {512, 1, 1}, {}, 1);

    CREATE_UNARY(elu)
    CREATE_UNARY(gelu)
    CREATE_UNARY(gelu_erf)
    CREATE_UNARY(gelu_शीघ्र)
    CREATE_UNARY(silu)
    CREATE_UNARY(relu)
    CREATE_UNARY(sqr)
    CREATE_UNARY(वर्गमूलम्)
    CREATE_UNARY(ज्या)
    CREATE_UNARY(कोटिज्या)
    CREATE_UNARY(clamp)
    CREATE_UNARY(leaky_relu)
    CREATE_UNARY(xielu)
    CREATE_UNARY(neg)
    CREATE_UNARY(tanh)
    CREATE_UNARY(sigmoid)
    CREATE_UNARY(hardsigmoid)
    CREATE_UNARY(hardswish)
    CREATE_UNARY(निरपेक्षम्)
    CREATE_UNARY(softplus)
    CREATE_UNARY(चरणः)
    CREATE_UNARY(round)
    CREATE_UNARY(ceil)
    CREATE_UNARY(floor)
    CREATE_UNARY(trunc)
    CREATE_UNARY(sgn)
    CREATE_UNARY(घातांकः)
    CREATE_UNARY(expm1)
#अपरिभाषय CREATE_UNARY

// spec constants: {norepeat, op_on_b}
#परिभाषय CREATE_UNARY_MUL(name, idx) \
    प्रति (पूर्णाङ्क dt = 0; dt < 2; ++dt) { \
        ध्रुव size_t len_ = dt ? name ## _mul_f16_len : name ## _mul_f32_len; \
        ध्रुव अचिह्नित वर्ण * दत्त_3 = dt ? name ## mul_f16_दत्त : name ## mul_f32_दत्त; \
        ध्रुव std::string dts_ = dt ? "f16" : "f32"; \
        प्रति (पूर्णाङ्क ob = 0; ob < 2; ++ob) \
            प्रति (पूर्णाङ्क nr = 0; nr < 2; ++nr) \
                ggml_vk_create_pipeline(device, device->pipeline_unary_mul[(idx)][dt][nr][ob], \
                    (#name "_mul" + std::string(ob ? "_b" : "") + "_" + dts_ + (nr ? "_norepeat" : "")).c_str(), \
                    len_, दत्त_3, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {512, 1, 1}, \
                    { (uint32_t) nr, (uint32_t) ob }, 1); \
    }

    CREATE_UNARY_MUL(gelu, 0)
    CREATE_UNARY_MUL(sigmoid, 1)
    CREATE_UNARY_MUL(silu, 2)
    CREATE_UNARY_MUL(softplus, 3)
#अपरिभाषय CREATE_UNARY_MUL

    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f16, "add1_f16_f16", add1_f16_f16_len, add1_f16_f16_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f32, "add1_f16_f32", add1_f16_f32_len, add1_f16_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f32_f32, "add1_f32_f32", add1_f32_f32_len, add1_f32_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_arange_f32, "arange_f32", arange_f32_len, arange_f32_दत्त, "main", 1, परिमाण(vk_op_प्रक्षिप_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_fill_f32, "fill_f32", fill_f32_len, fill_f32_दत्त, "main", 1, परिमाण(vk_op_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_fill_f16, "fill_f16", fill_f16_len, fill_f16_दत्त, "main", 1, परिमाण(vk_op_प्रक्षिप_constants), {512, 1, 1}, {}, 1);

#परिभाषय CREATE_GLU(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## f32_दत्त, "main", 3, परिमाण(vk_op_glu_प्रक्षिप_constants), {512, 1, 1}, {}, 1, सत्य);   \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## f16_दत्त, "main", 3, परिमाण(vk_op_glu_प्रक्षिप_constants), {512, 1, 1}, {}, 1, सत्य);

    CREATE_GLU(geglu)
    CREATE_GLU(reglu)
    CREATE_GLU(swiglu)
    CREATE_GLU(swiglu_oai)
    CREATE_GLU(swiglu_clamp)
    CREATE_GLU(geglu_erf)
    CREATE_GLU(geglu_शीघ्र)
#अपरिभाषय CREATE_GLU

    ggml_vk_create_pipeline(device, device->pipeline_silu_पृष्ठभागः_f32, "silu_back_f32", silu_पृष्ठभागः_f32_len, silu_पृष्ठभागः_f32_दत्त, "main", 3, परिमाण(vk_op_प्रक्षिप_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag_आवरणम्_inf_f32, "diag_mask_inf_f32", diag_आवरणम्_inf_f32_len, diag_आवरणम्_inf_f32_दत्त, "main", 2, परिमाण(vk_op_diag_आवरणम्_प्रक्षिप_constants), {1, 512, 1}, {}, 1, सत्य);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32, "soft_max_f32", soft_max_f32_len, soft_max_f32_दत्त, "main", 4, परिमाण(vk_op_soft_max_प्रक्षिप_constants), {1, 1, 1}, { device->subgroup_आकार }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_wg512, "soft_max_f32_wg512", soft_max_f32_len, soft_max_f32_दत्त, "main", 4, परिमाण(vk_op_soft_max_प्रक्षिप_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16, "soft_max_f32_f16", soft_max_f32_f16_len, soft_max_f32_f16_दत्त, "main", 4, परिमाण(vk_op_soft_max_प्रक्षिप_constants), {1, 1, 1}, { device->subgroup_आकार }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16_wg512, "soft_max_f32_f16_wg512", soft_max_f32_f16_len, soft_max_f32_f16_दत्त, "main", 4, परिमाण(vk_op_soft_max_प्रक्षिप_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_पृष्ठभागः_f32, "soft_max_back_f32", soft_max_पृष्ठभागः_f32_len, soft_max_पृष्ठभागः_f32_दत्त, "main", 3, परिमाण(vk_op_प्रक्षिप_constants), {1, 1, 1}, { device->subgroup_आकार }, 1, सत्य);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32,     "soft_max_large1_f32",     soft_max_large1_f32_len,     soft_max_large1_f32_दत्त,     "main", 6, परिमाण(vk_op_soft_max_प्रक्षिप_constants), {1, 1, 1}, { 128, 4 }, 1, सत्य);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32,     "soft_max_large2_f32",     soft_max_large2_f32_len,     soft_max_large2_f32_दत्त,     "main", 6, परिमाण(vk_op_soft_max_प्रक्षिप_constants), {1, 1, 1}, { 128, 4 }, 1, सत्य);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32,     "soft_max_large3_f32",     soft_max_large3_f32_len,     soft_max_large3_f32_दत्त,     "main", 6, परिमाण(vk_op_soft_max_प्रक्षिप_constants), {1, 1, 1}, { 128, 4 }, 1, सत्य);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32_f16, "soft_max_large1_f32_f16", soft_max_large1_f32_f16_len, soft_max_large1_f32_f16_दत्त, "main", 6, परिमाण(vk_op_soft_max_प्रक्षिप_constants), {1, 1, 1}, { 128, 4 }, 1, सत्य);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32_f16, "soft_max_large2_f32_f16", soft_max_large2_f32_f16_len, soft_max_large2_f32_f16_दत्त, "main", 6, परिमाण(vk_op_soft_max_प्रक्षिप_constants), {1, 1, 1}, { 128, 4 }, 1, सत्य);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32_f16, "soft_max_large3_f32_f16", soft_max_large3_f32_f16_len, soft_max_large3_f32_f16_दत्त, "main", 6, परिमाण(vk_op_soft_max_प्रक्षिप_constants), {1, 1, 1}, { 128, 4 }, 1, सत्य);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32, "rope_norm_f32", rope_norm_f32_len, rope_norm_f32_दत्त, "main", 5, परिमाण(vk_op_rope_प्रक्षिप_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32, "rope_neox_f32", rope_neox_f32_len, rope_neox_f32_दत्त, "main", 5, परिमाण(vk_op_rope_प्रक्षिप_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32, "rope_multi_f32", rope_multi_f32_len, rope_multi_f32_दत्त, "main", 5, परिमाण(vk_op_rope_प्रक्षिप_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f32, "rope_vision_f32", rope_vision_f32_len, rope_vision_f32_दत्त, "main", 5, परिमाण(vk_op_rope_प्रक्षिप_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f16, "rope_norm_f16", rope_norm_f16_len, rope_norm_f16_दत्त, "main", 5, परिमाण(vk_op_rope_प्रक्षिप_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f16, "rope_neox_f16", rope_neox_f16_len, rope_neox_f16_दत्त, "main", 5, परिमाण(vk_op_rope_प्रक्षिप_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f16, "rope_multi_f16", rope_multi_f16_len, rope_multi_f16_दत्त, "main", 5, परिमाण(vk_op_rope_प्रक्षिप_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f16, "rope_vision_f16", rope_vision_f16_len, rope_vision_f16_दत्त, "main", 5, परिमाण(vk_op_rope_प्रक्षिप_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32_f16, "rope_norm_f32_f16", rope_norm_f32_f16_len, rope_norm_f32_f16_दत्त, "main", 5, परिमाण(vk_op_rope_प्रक्षिप_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32_f16, "rope_neox_f32_f16", rope_neox_f32_f16_len, rope_neox_f32_f16_दत्त, "main", 5, परिमाण(vk_op_rope_प्रक्षिप_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32_f16, "rope_multi_f32_f16", rope_multi_f32_f16_len, rope_multi_f32_f16_दत्त, "main", 5, परिमाण(vk_op_rope_प्रक्षिप_constants), {1, 512, 1}, {}, 1);

    प्रति (uint32_t i = 0; i < अङ्कः_argsort_pipelines; ++i) {
        uint32_t खण्डम्_आकार_2 = 1u << std::min(i, device->max_workgroup_आकार_log2);
        यदि (i <= device->max_workgroup_आकार_log2 &&
            2 * परिमाण(पूर्णाङ्क) * खण्डम्_आकार_2 <= device->properties.limits.max_Compute_Shared_स्मृतिः_आकार) {
            ध्रुव uint32_t NCOLS_PADDED_LOG2 = i;
            ggml_vk_create_pipeline2(device, device->pipeline_argsort_f32[i], "argsort_f32_"+std::to_string(i), argsort_f32_len, argsort_f32_दत्त, "main", 3, परिमाण(vk_op_argsort_प्रक्षिप_constants), {खण्डम्_आकार_2, 1, 1}, {खण्डम्_आकार_2, NCOLS_PADDED_LOG2}, 1, सत्य);
        }
        ध्रुव uint32_t WG_UNROLL_गुणकः = खण्डम्_आकार_2 > 1 ? 2 : 1;
        खण्डम्_आकार_2 /= WG_UNROLL_गुणकः;
        ggml_vk_create_pipeline2(device, device->pipeline_argsort_large_f32[i], "argsort_large_f32_"+std::to_string(i), argsort_large_f32_len, argsort_large_f32_दत्त, "main", 3, परिमाण(vk_op_argsort_प्रक्षिप_constants), {खण्डम्_आकार_2 * WG_UNROLL_गुणकः, 1, 1}, {खण्डम्_आकार_2, WG_UNROLL_गुणकः}, 1, सत्य);
    }

    प्रति (uint32_t i = 0; i < अङ्कः_topk_pipelines; ++i) {
        ध्रुव uint32_t खण्डम्_आकार_2 = 1u << i;
        ध्रुव uint32_t NCOLS_PADDED_LOG2 = i;
        यदि (i <= device->max_workgroup_आकार_log2) {
            uint32_t nary_shmem = 2 * परिमाण(पूर्णाङ्क) * खण्डम्_आकार_2 +
                                  परिमाण(पूर्णाङ्क) * device->subgroup_आकार +
                                  2 * परिमाण(पूर्णाङ्क) +
                                  2 * (खण्डम्_आकार_2 / device->subgroup_आकार) * परिमाण(पूर्णाङ्क);
            यदि (device->subgroup_arithmetic && device->subgroup_require_पूर्णम्_support && device->subgroup_shuffle && device->subgroup_ballot &&
                nary_shmem <= device->properties.limits.max_Compute_Shared_स्मृतिः_आकार) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_nary_अन्वेषणम्_f32_len, topk_nary_अन्वेषणम्_f32_दत्त, "main", 2, परिमाण(vk_op_topk_प्रक्षिप_constants), {खण्डम्_आकार_2, 1, 1}, {खण्डम्_आकार_2, device->subgroup_आकार, device->subgroup_आकार_log2}, 1, सत्य, सत्य, device->subgroup_आकार);
            } अन्यथा यदि (2 * परिमाण(पूर्णाङ्क) * खण्डम्_आकार_2 <= device->properties.limits.max_Compute_Shared_स्मृतिः_आकार) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_argsort_f32_len, topk_argsort_f32_दत्त, "main", 2, परिमाण(vk_op_topk_प्रक्षिप_constants), {खण्डम्_आकार_2, 1, 1}, {खण्डम्_आकार_2, NCOLS_PADDED_LOG2}, 1, सत्य);
            }
        }
    }

    // large-k fallback: one workgroup per row, radix-select instead of a full sort. The QSA
    // variant (spec constant 1) additionally gathers the qwen4 indexer input on the fly.
    {
        ध्रुव uint32_t खण्डम्_आकार_2 = 1u << std::min(10u, device->max_workgroup_आकार_log2);
        ggml_vk_create_pipeline2(device, device->pipeline_topk_radix_f32, "topk_radix_f32", topk_radix_select_f32_len, topk_radix_select_f32_दत्त, "main", 5, परिमाण(vk_op_topk_radix_प्रक्षिप_constants), {खण्डम्_आकार_2, 1, 1}, {खण्डम्_आकार_2, 0}, 1, सत्य);
        ggml_vk_create_pipeline2(device, device->pipeline_topk_radix_qsa, "topk_radix_qsa", topk_radix_select_f32_len, topk_radix_select_f32_दत्त, "main", 5, परिमाण(vk_op_topk_radix_प्रक्षिप_constants), {खण्डम्_आकार_2, 1, 1}, {खण्डम्_आकार_2, 1}, 1, सत्य);
    }

    ggml_vk_create_pipeline(device, device->pipeline_argmax_f32, "argmax_f32", argmax_f32_len, argmax_f32_दत्त, "main", 2, परिमाण(vk_op_प्रक्षिप_constants), {1, 1, 1}, { device->subgroup_आकार }, 1);

    ggml_vk_create_pipeline(device, device->pipeline_योग_पङ्क्तिकाः_f32, "sum_rows_f32", योग_पङ्क्तिकाः_f32_len, योग_पङ्क्तिकाः_f32_दत्त, "main", 2, परिमाण(vk_op_योग_पङ्क्तिकाः_प्रक्षिप_constants), {1, 1, 1}, { device->subgroup_आकार }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_f32, "cross_entropy_loss_f32", cross_entropy_loss_f32_len, cross_entropy_loss_f32_दत्त, "main", 3, परिमाण(vk_op_प्रक्षिप_constants), {1, 1, 1}, { device->subgroup_आकार }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_f32_wg512, "cross_entropy_loss_f32_wg512", cross_entropy_loss_f32_len, cross_entropy_loss_f32_दत्त, "main", 3, परिमाण(vk_op_प्रक्षिप_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_पृष्ठभागः_f32, "cross_entropy_loss_back_f32", cross_entropy_loss_पृष्ठभागः_f32_len, cross_entropy_loss_पृष्ठभागः_f32_दत्त, "main", 4, परिमाण(vk_op_प्रक्षिप_constants), {1, 1, 1}, { device->subgroup_आकार }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_पृष्ठभागः_f32_wg512, "cross_entropy_loss_back_f32_wg512", cross_entropy_loss_पृष्ठभागः_f32_len, cross_entropy_loss_पृष्ठभागः_f32_दत्त, "main", 4, परिमाण(vk_op_प्रक्षिप_constants), {1, 1, 1}, { 512 }, 1);
    // Intel Windows driver in range [32.0.101.8509, 32.0.101.8860) will crash when using fwht kernels so we gate that here
    ध्रुव तार्किक can_use_fwht = device->driver_id != vk::DriverId::eIntelProprietaryWindows ||
        !ggml_vk_intel_windows_driver_अन्तर्_range(device->properties.driver_संस्करणम्_2, 101, 8509, 101, 8860);
    यदि (can_use_fwht && device->subgroup_basic && device->subgroup_shuffle) {
        पूर्णाङ्क idx = 0;
        प्रति (uint32_t n : {64, 128, 256, 512}) {
            यदि (device->subgroup_आकार <= n) {
                ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_f32", fwht_f32_len, fwht_f32_दत्त, "main", 2, परिमाण(vk_op_fwht_प्रक्षिप_constants), {1, 1, 1}, { device->subgroup_आकार, n }, 1, सत्य, सत्य, device->subgroup_आकार);
            }
            ++idx;
        }
    } अन्यथा यदि (can_use_fwht) {
        पूर्णाङ्क idx = 0;
        प्रति (uint32_t n : {64, 128, 256, 512}) {
            ध्रुव uint32_t खण्डम्_आकार_3 = std::min(device->subgroup_आकार, n);
            ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_shmem_f32", fwht_shmem_f32_len, fwht_shmem_f32_दत्त, "main", 2, परिमाण(vk_op_fwht_प्रक्षिप_constants), {1, 1, 1}, { खण्डम्_आकार_3, n }, 1);
            ++idx;
        }
    }

    ध्रुव uint32_t cumsum_elem_प्रत्येकम्_thread = (device->vendor_id == VK_VENDOR_ID_AMD || device->vendor_id == VK_VENDOR_ID_INTEL) ? 2 : 4;
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_f32,       "cumsum_f32", cumsum_f32_len, cumsum_f32_दत्त, "main", 2, परिमाण(vk_op_योग_पङ्क्तिकाः_प्रक्षिप_constants), {1, 1, 1}, { 256, device->subgroup_आकार, cumsum_elem_प्रत्येकम्_thread }, 1, सत्य, सत्य, device->subgroup_आकार);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_small_f32, "cumsum_f32", cumsum_f32_len, cumsum_f32_दत्त, "main", 2, परिमाण(vk_op_योग_पङ्क्तिकाः_प्रक्षिप_constants), {1, 1, 1}, { 128, device->subgroup_आकार, 1 }, 1, सत्य, सत्य, device->subgroup_आकार);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass1_f32, "cumsum_multipass1_f32", cumsum_multipass1_f32_len, cumsum_multipass1_f32_दत्त, "main", 3, परिमाण(vk_op_योग_पङ्क्तिकाः_प्रक्षिप_constants), {256, 1, 1}, { 256, device->subgroup_आकार }, 1, सत्य, सत्य, device->subgroup_आकार);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass2_f32, "cumsum_multipass2_f32", cumsum_multipass2_f32_len, cumsum_multipass2_f32_दत्त, "main", 3, परिमाण(vk_op_योग_पङ्क्तिकाः_प्रक्षिप_constants), {256, 1, 1}, { 256, device->subgroup_आकार }, 1, सत्य, सत्य, device->subgroup_आकार);

    ggml_vk_create_pipeline(device, device->pipeline_गणनम्_equal_i32, "count_equal_i32", गणनम्_equal_i32_len, गणनम्_equal_i32_दत्त, "main", 3, परिमाण(vk_op_प्रक्षिप_constants), {512, 1, 1}, { device->subgroup_आकार }, 1);

    यदि (device->subgroup_arithmetic && device->subgroup_vote && device->subgroup_require_पूर्णम्_support) {
        ggml_vk_create_pipeline(device, device->pipeline_गणनम्_experts, "count_experts", गणनम्_experts_subgroup_len, गणनम्_experts_subgroup_दत्त, "main", 2, परिमाण(vk_op_गणनम्_experts_प्रक्षिप_constants), {1, 1, 1}, {}, 1, सत्य, सत्य);
    } अन्यथा {
        ggml_vk_create_pipeline(device, device->pipeline_गणनम्_experts, "count_experts", गणनम्_experts_len, गणनम्_experts_दत्त, "main", 2, परिमाण(vk_op_गणनम्_experts_प्रक्षिप_constants), {1, 1, 1}, {}, 1, सत्य);
    }

    // comb holds a token's 4x4 matrix in one 16-lane slice of a subgroup, so it
    // needs at least 16 lanes, pinned to a known size.
    यदि (device->subgroup_basic && device->subgroup_shuffle && device->subgroup_require_पूर्णम्_support && device->subgroup_आकार >= 16) {
        ध्रुव uint32_t tokens_प्रत्येकम्_workgroup = 4 * (device->subgroup_आकार / 16);
        ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_comb_f32, "dsv4_hc_comb_f32", dsv4_hc_comb_f32_len, dsv4_hc_comb_f32_दत्त, "main", 4, परिमाण(vk_op_dsv4_hc_comb_प्रक्षिप_constants), {tokens_प्रत्येकम्_workgroup, 1, 1}, { device->subgroup_आकार }, 1, सत्य, सत्य, device->subgroup_आकार);
    }

    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_pre_f32,        "dsv4_hc_pre_f32",        dsv4_hc_pre_f32_len,  dsv4_hc_pre_f32_दत्त,  "main", 3, परिमाण(vk_op_dsv4_hc_pre_प्रक्षिप_constants),  {256, 1, 1}, { 256, 0 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_pre_gated_f32,  "dsv4_hc_pre_gated_f32",  dsv4_hc_pre_f32_len,  dsv4_hc_pre_f32_दत्त,  "main", 3, परिमाण(vk_op_dsv4_hc_pre_प्रक्षिप_constants),  {256, 1, 1}, { 256, 1 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_post_f32,       "dsv4_hc_post_f32",       dsv4_hc_post_f32_len, dsv4_hc_post_f32_दत्त, "main", 5, परिमाण(vk_op_dsv4_hc_post_प्रक्षिप_constants), {256, 1, 1}, { 256, 1 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_post_nocomb_f32,"dsv4_hc_post_nocomb_f32",dsv4_hc_post_f32_len, dsv4_hc_post_f32_दत्त, "main", 5, परिमाण(vk_op_dsv4_hc_post_प्रक्षिप_constants), {256, 1, 1}, { 256, 0 }, 1);

    प्रति (स्वचालित &s : device->pipeline_solve_tri_f32) {
        ध्रुव vk_solve_tri_pipeline_state &state = s.first;

        // Max number of rows to load at a time, limited by shared memory
        ध्रुव uint32_t batch_N = device->properties.limits.max_Compute_Shared_स्मृतिः_आकार / ((state.N + state.K) * परिमाण(भिन्न));
        // Need at least K invocations, and prefer a minimum of 128 to spread out loading shared memory
        ध्रुव uint32_t खण्डम्_आकार_3 = std::max(128u, 1u << (uint32_t)ceilf(log2f(भिन्न(state.K))));

        ggml_vk_create_pipeline(
            device, s.second, "solve_tri_f32",
            solve_tri_f32_len, solve_tri_f32_दत्त, "main", 3,
            परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {1, 1, 1}, { 0, state.N, state.K, batch_N, खण्डम्_आकार_3 }, 1, सत्य);
    }

#परिभाषय IM2COL(bda) \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32, "im2col_f32", im2col_f32 ## bda ## _len, im2col_f32 ## bda ## दत्त_2, "main", 2, परिमाण(vk_op_im2col_प्रक्षिप_constants), {512, 1, 1}, { device->subgroup_आकार }, 1, सत्य);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32, "im2col_3d_f32", im2col_3d_f32 ## bda ## _len, im2col_3d_f32 ## bda ## दत्त_2, "main", 2, परिमाण(vk_op_im2col_3d_प्रक्षिप_constants), {512, 1, 1}, { 512 }, 1, सत्य);      \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32_f16, "im2col_f32_f16", im2col_f32_f16 ## bda ## _len, im2col_f32_f16 ## bda ## दत्त_2, "main", 2, परिमाण(vk_op_im2col_प्रक्षिप_constants), {512, 1, 1}, { device->subgroup_आकार }, 1, सत्य);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32_f16, "im2col_3d_f32_f16", im2col_3d_f32_f16 ## bda ## _len, im2col_3d_f32_f16 ## bda ## दत्त_2, "main", 2, परिमाण(vk_op_im2col_3d_प्रक्षिप_constants), {512, 1, 1}, { 512 }, 1, सत्य);
    यदि (device->shader_int64 && device->बफर_device_स्थानाङ्कः) {
        IM2COL(_bda)
    } अन्यथा {
        IM2COL()
    }

    ggml_vk_create_pipeline(device, device->pipeline_timestep_embedding_f32, "timestep_embedding_f32", timestep_embedding_f32_len, timestep_embedding_f32_दत्त, "main", 2, परिमाण(vk_op_timestep_embedding_प्रक्षिप_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_conv_transpose_1d_f32, "conv_transpose_1d_f32", conv_transpose_1d_f32_len, conv_transpose_1d_f32_दत्त, "main", 3, परिमाण(vk_op_conv_transpose_1d_प्रक्षिप_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_f32,  "col2im_1d_f32",  col2im_1d_f32_len,  col2im_1d_f32_दत्त,  "main", 2, परिमाण(vk_op_col2im_1d_प्रक्षिप_constants), {256, 1, 1}, {}, 1, सत्य);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_f16,  "col2im_1d_f16",  col2im_1d_f16_len,  col2im_1d_f16_दत्त,  "main", 2, परिमाण(vk_op_col2im_1d_प्रक्षिप_constants), {256, 1, 1}, {}, 1, सत्य);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_bf16, "col2im_1d_bf16", col2im_1d_bf16_len, col2im_1d_bf16_दत्त, "main", 2, परिमाण(vk_op_col2im_1d_प्रक्षिप_constants), {256, 1, 1}, {}, 1, सत्य);

    ggml_vk_create_pipeline(device, device->pipeline_बहिर्गमनम्_prod_f32, "out_prod_f32", बहिर्गमनम्_prod_f32_len, बहिर्गमनम्_prod_f32_दत्त, "main", 3, परिमाण(vk_op_द्विमान_प्रक्षिप_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_snake_f32,  "snake_f32",  snake_f32_len,  snake_f32_दत्त,  "main", 4, परिमाण(vk_op_snake_प्रक्षिप_constants), {256, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_snake_f16,  "snake_f16",  snake_f16_len,  snake_f16_दत्त,  "main", 4, परिमाण(vk_op_snake_प्रक्षिप_constants), {256, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_snake_bf16, "snake_bf16", snake_bf16_len, snake_bf16_दत्त, "main", 4, परिमाण(vk_op_snake_प्रक्षिप_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pool1d_f32, "pool1d_f32", pool1d_f32_len, pool1d_f32_दत्त, "main", 2, परिमाण(vk_op_pool1d_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_pool2d_f32, "pool2d_f32", pool2d_f32_len, pool2d_f32_दत्त, "main", 2, परिमाण(vk_op_pool2d_प्रक्षिप_constants), {1, 1, 1}, {512}, 1);
    यदि (device->subgroup_arithmetic && device->subgroup_require_पूर्णम्_support) {
        ggml_vk_create_pipeline(device, device->pipeline_pool2d_subgroup_f32, "pool2d_subgroup_f32", pool2d_subgroup_f32_len, pool2d_subgroup_f32_दत्त, "main", 2, परिमाण(vk_op_pool2d_प्रक्षिप_constants), {1, 1, 1}, {128}, 1, असत्य, सत्य, device->subgroup_आकार);
    }

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv6_f32, "rwkv_wkv6_f32", rwkv_wkv6_f32_len, rwkv_wkv6_f32_दत्त, "main", 7, परिमाण(vk_op_rwkv_wkv6_प्रक्षिप_constants), {1, 1, 1}, {device->subgroup_आकार}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv7_f32, "rwkv_wkv7_f32", rwkv_wkv7_f32_len, rwkv_wkv7_f32_दत्त, "main", 8, परिमाण(vk_op_rwkv_wkv7_प्रक्षिप_constants), {1, 1, 1}, {device->subgroup_आकार}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_gated_रैखिकम्_attn_f32, "gated_linear_attn_f32", gated_रैखिकम्_attn_f32_len, gated_रैखिकम्_attn_f32_दत्त, "main", 6, परिमाण(vk_op_gated_रैखिकम्_attn_प्रक्षिप_constants), {1, 1, 1}, {}, 1);

    प्रति (ggml_type k_type : lightning_indexer_k_types) {
        ध्रुव std::string name = "lightning_indexer_" + std::string(ggml_type_name(k_type)) + "_k_f32";
        ggml_vk_create_pipeline(device, device->pipeline_lightning_indexer_f32[k_type], name.c_str(), lightning_indexer_f32_len, lightning_indexer_f32_दत्त, "main", 5, परिमाण(vk_op_lightning_indexer_प्रक्षिप_constants), {1, 1, 1}, {(uint32_t)k_type, fa_खण्डम्_अष्टकानि(k_type)}, 1, सत्य);
    }

    {
        ध्रुव uint32_t gdn_sizes[] = {16, 32, 64, 128};
        ध्रुव वर्ण * gdn_नामानि[][2] = {
            {"gated_delta_net_f32_d16",     "gated_delta_net_f32_d16_kda"},
            {"gated_delta_net_f32_d32",     "gated_delta_net_f32_d32_kda"},
            {"gated_delta_net_f32_d64",     "gated_delta_net_f32_d64_kda"},
            {"gated_delta_net_f32_d128",    "gated_delta_net_f32_d128_kda"},
        };
        प्रति (uint32_t si = 0; si < 4; si++) {
            ध्रुव uint32_t S_V = gdn_sizes[si];
            GGML_ASSERT(is_pow2(S_V));

            // Intel Xe regresses at SIMD32 for this scan; prefer a narrower subgroup.
            uint32_t gdn_subgroup_आकार = device->subgroup_आकार;
            यदि (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_आकार_control &&
                device->subgroup_min_आकार <= 16u && device->subgroup_max_आकार >= 16u) {
                gdn_subgroup_आकार = 16u;
            }
            uint32_t lanes_प्रत्येकम्_column;
            यदि (device->vendor_id == VK_VENDOR_ID_INTEL) {
                // Intel Xe: full-width reduction (min rows/lane) ~10x over the COLS_PER_WG=8 rule.
                lanes_प्रत्येकम्_column = std::min(gdn_subgroup_आकार, S_V);
            } अन्यथा यदि (S_V >= 128u && device->subgroup_clustered) {
                // COLS_PER_WG=8: measured optimum (Ampere sg32->4, Vega20 sg64->8).
                lanes_प्रत्येकम्_column = std::max(1u, gdn_subgroup_आकार / 8u);
            } अन्यथा {
                // Use largest power-of-two that divides both S_V and subgroup_size so that
                // (1) S_V % lanes_per_column == 0 and (2) S_V % (subgroup_size / lanes_per_column) == 0.
                // This means we don't need extra bounds checking logic in the shader.
                lanes_प्रत्येकम्_column = std::min(S_V, gdn_subgroup_आकार);
            }

            // gated_delta_net.comp relies on S_V % COLS_PER_WG == 0 and
            // S_V % LANES_PER_COLUMN == 0 to avoid bounds checks.
            यावत् (lanes_प्रत्येकम्_column > 1u) {
                ध्रुव तार्किक valid_lanes = (gdn_subgroup_आकार % lanes_प्रत्येकम्_column) == 0 &&
                                         (S_V % lanes_प्रत्येकम्_column) == 0;
                ध्रुव uint32_t स्तम्भसङ्ख्या_प्रत्येकम्_wg = valid_lanes ? gdn_subgroup_आकार / lanes_प्रत्येकम्_column : 0;
                यदि (valid_lanes && स्तम्भसङ्ख्या_प्रत्येकम्_wg > 0 && (S_V % स्तम्भसङ्ख्या_प्रत्येकम्_wg) == 0) {
                    भङ्ग;
                }
                lanes_प्रत्येकम्_column >>= 1u;
            }

            GGML_ASSERT((gdn_subgroup_आकार % lanes_प्रत्येकम्_column) == 0);
            GGML_ASSERT((S_V % lanes_प्रत्येकम्_column) == 0);
            GGML_ASSERT((S_V % (gdn_subgroup_आकार / lanes_प्रत्येकम्_column)) == 0);

            ध्रुव तार्किक need_partial_subgroup_reduce = lanes_प्रत्येकम्_column != 1u && lanes_प्रत्येकम्_column < gdn_subgroup_आकार;
            ध्रुव तार्किक use_clustered_reduce = device->subgroup_arithmetic && device->subgroup_clustered && need_partial_subgroup_reduce;
            ध्रुव तार्किक use_subgroup_reduce = device->subgroup_arithmetic && !need_partial_subgroup_reduce;
            ध्रुव तार्किक use_subgroup_ops = use_clustered_reduce || use_subgroup_reduce;
            size_t gdn_len;
            ध्रुव शून्य * gdn_दत्त;
            यदि (use_clustered_reduce) {
                gdn_len = gated_delta_net_f32_len;
                gdn_दत्त = (ध्रुव शून्य *)gated_delta_net_f32_दत्त;
            } अन्यथा यदि (use_subgroup_reduce) {
                gdn_len = gated_delta_net_f32_nocluster_len;
                gdn_दत्त = (ध्रुव शून्य *)gated_delta_net_f32_nocluster_दत्त;
            } अन्यथा {
                gdn_len = gated_delta_net_f32_shmem_len;
                gdn_दत्त = (ध्रुव शून्य *)gated_delta_net_f32_shmem_दत्त;
            }

            ध्रुव uint32_t स्तम्भसङ्ख्या_प्रत्येकम्_wg = gdn_subgroup_आकार / lanes_प्रत्येकम्_column;
            ध्रुव std::array<uint32_t, 3> wg_denoms = {1u, 1u, स्तम्भसङ्ख्या_प्रत्येकम्_wg};

            प्रति (uint32_t kda = 0; kda < 2; kda++) {
                ggml_vk_create_pipeline(device, device->pipeline_gated_delta_net[si][kda],
                    gdn_नामानि[si][kda], gdn_len, gdn_दत्त, "main", 7, परिमाण(vk_op_gated_delta_net_प्रक्षिप_constants),
                    wg_denoms, {S_V, kda, gdn_subgroup_आकार, lanes_प्रत्येकम्_column}, 1, सत्य, use_subgroup_ops, gdn_subgroup_आकार);
            }
        }
    }

    यदि (device->subgroup_arithmetic && device->subgroup_require_पूर्णम्_support) {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_अवलोकनम्_f32_d128, "ssm_scan_128_f32", ssm_अवलोकनम्_subgroup_f32_len, ssm_अवलोकनम्_subgroup_f32_दत्त, "main", 8, परिमाण(vk_op_ssm_अवलोकनम्_प्रक्षिप_constants), {1, 1, 1}, {128, device->subgroup_आकार}, 1, सत्य, सत्य);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_अवलोकनम्_f32_d256, "ssm_scan_256_f32", ssm_अवलोकनम्_subgroup_f32_len, ssm_अवलोकनम्_subgroup_f32_दत्त, "main", 8, परिमाण(vk_op_ssm_अवलोकनम्_प्रक्षिप_constants), {1, 1, 1}, {256, device->subgroup_आकार}, 1, सत्य, सत्य);
    } अन्यथा {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_अवलोकनम्_f32_d128, "ssm_scan_128_f32", ssm_अवलोकनम्_f32_len, ssm_अवलोकनम्_f32_दत्त, "main", 8, परिमाण(vk_op_ssm_अवलोकनम्_प्रक्षिप_constants), {1, 1, 1}, {128, device->subgroup_आकार, 16}, 1, सत्य, सत्य);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_अवलोकनम्_f32_d256, "ssm_scan_256_f32", ssm_अवलोकनम्_f32_len, ssm_अवलोकनम्_f32_दत्त, "main", 8, परिमाण(vk_op_ssm_अवलोकनम्_प्रक्षिप_constants), {1, 1, 1}, {256, device->subgroup_आकार, 16}, 1, सत्य, सत्य);
    }

    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_f32,           "ssm_conv_f32",           ssm_conv_f32_len, ssm_conv_f32_दत्त, "main", 4, परिमाण(vk_op_ssm_conv_प्रक्षिप_constants), {32, 16, 1}, {32, 16, 0, 0}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_silu_f32,      "ssm_conv_silu_f32",      ssm_conv_f32_len, ssm_conv_f32_दत्त, "main", 4, परिमाण(vk_op_ssm_conv_प्रक्षिप_constants), {32, 16, 1}, {32, 16, 0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_bias_silu_f32, "ssm_conv_bias_silu_f32", ssm_conv_f32_len, ssm_conv_f32_दत्त, "main", 4, परिमाण(vk_op_ssm_conv_प्रक्षिप_constants), {32, 16, 1}, {32, 16, 1, 1}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_चरणः_adamw_f32, "opt_step_adamw_f32", opt_चरणः_adamw_f32_len, opt_चरणः_adamw_f32_दत्त, "main", 5, परिमाण(vk_op_प्रक्षिप_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_चरणः_sgd_f32, "opt_step_sgd_f32", opt_चरणः_sgd_f32_len, opt_चरणः_sgd_f32_दत्त, "main", 3, परिमाण(vk_op_प्रक्षिप_constants), {512, 1, 1}, {}, 1);

    // conv2d, conv_transpose_2d, conv3d
    प्रति (uint32_t s = 0; s < CONV_SHAPE_गणनम्; ++s) {
        // smaller WG for the small-tile fallback gives more concurrent WGs per SM
        uint32_t conv2d_WG_आकार  = (s == CONV_SHAPE_64x32) ? 128 : 256;
        uint32_t use_collectives = 0;  // Enables subgroup ops for preventing the re-calculation of indices.
        uint32_t conv2d_TS_K     = (s == CONV_SHAPE_64x32) ? 4 : 8;
        uint32_t conv2d_SHMEM_PAD = 4;
        vk_conv_खण्डम्_आकार conv2d_BS = vk_conv_खण्डम्_sizes[s];
        तार्किक conv2d_UNROLL = सत्य;

#यदि defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        यदि (device->coopmat2) {
            conv2d_SHMEM_PAD = 8; // 8 float16_t
        }
#यदि_अन्तः

        यदि (device->vendor_id == VK_VENDOR_ID_INTEL) {
            conv2d_SHMEM_PAD = 0;
            conv2d_UNROLL = असत्य;
        } अन्यथा यदि (device->vendor_id == VK_VENDOR_ID_AMD) {
            conv2d_SHMEM_PAD = device->architecture == vk_device_architecture::AMD_GCN ? 1 : 4;
            यदि (s == CONV_SHAPE_128x128 && device->architecture != vk_device_architecture::AMD_GCN) {
                conv2d_UNROLL = असत्य;
            }
        }

        // Use collectives on pre-Turing NVIDIA GPUs and GCN AMD cards, which had slower integer math.
        तार्किक allow_collectives_nv = device->vendor_id != VK_VENDOR_ID_NVIDIA ||
                                    device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        तार्किक allow_collectives_amd = device->vendor_id != VK_VENDOR_ID_AMD ||
                                     device->architecture == vk_device_architecture::AMD_GCN;

        यदि (device->subgroup_shuffle &&
            device->vendor_id != VK_VENDOR_ID_INTEL &&   // Do not enable collectives on Intel, see PR 14316.
            allow_collectives_nv &&
            allow_collectives_amd) {
            use_collectives = 1;
            conv2d_BS.CRS   = std::min(
                device->subgroup_आकार,
                conv2d_BS.CRS);  // CRS block size should be capped at subgroup size for correctness when shuffle is used.
        }

        // cm1 is used only when cm2 is unavailable; capped at 64x128 (due to shared memory size).
        // Requires 16x16x16 f16-acc since that's the fragment shape hard-coded in the shader.
        // Subgroup size must be 32 or 64 (to keep WG_SIZE sane) and we need
        // subgroup_size_control to force the driver to actually use it.
        तार्किक conv2d_use_cm1 = असत्य;
#यदि defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        conv2d_use_cm1 = !device->coopmat2 &&
                         device->coopmat_support && device->coopmat_support_16x16x16_f16acc &&
                         device->subgroup_आकार_control &&
                         (device->subgroup_आकार == 32 || device->subgroup_आकार == 64) &&
                         s != CONV_SHAPE_128x128;
#यदि_अन्तः

        ध्रुव uint32_t conv2d_cm1_shmem_pad = 8;

        स्वचालित shmem_req = [&](uint32_t pad, तार्किक csh_store, तार्किक fp16_shmem) {
            ध्रुव uint32_t elem_आकार = fp16_shmem ? (uint32_t)परिमाण(uint16_t) : (uint32_t)परिमाण(भिन्न);
            ध्रुव uint32_t csh_elems = csh_store ? conv2d_BS.K * conv2d_BS.NPQ : 0u;
            प्रत्यागम (conv2d_BS.K * (conv2d_BS.CRS + pad) + conv2d_BS.CRS * (conv2d_BS.NPQ + pad) + csh_elems) * elem_आकार;
        };

        // 2D, transpose-2D, and 3D conv use the same KxCRS @ CRSxNPQ shmem
        // layout. cm1 needs Csh for output, so check before applying cm1 params.
        यदि (conv2d_use_cm1 && device->properties.limits.max_Compute_Shared_स्मृतिः_आकार < shmem_req(conv2d_cm1_shmem_pad, सत्य, सत्य)) {
            conv2d_use_cm1 = असत्य;
        }

        uint32_t conv2d_WM = 16, conv2d_WN = 16;  // cm1 subgroup tile, ignored otherwise
        यदि (conv2d_use_cm1) {
            conv2d_SHMEM_PAD = conv2d_cm1_shmem_pad;
            // 16x16x16 fragments; pick WM/WN to keep WG_SIZE at 256
            // (i.e. 8 subgroups for sg=32, 4 subgroups for sg=64).
            ध्रुव तार्किक sg64 = (device->subgroup_आकार == 64);
            विकल्प (s) {
                पक्ष CONV_SHAPE_64x32:   conv2d_WM = sg64 ? 32 : 16; conv2d_WN = 16; भङ्ग;
                पक्ष CONV_SHAPE_64x128:  conv2d_WM = 32; conv2d_WN = sg64 ? 64 : 32; भङ्ग;
                पक्ष CONV_SHAPE_32x256:  conv2d_WM = sg64 ? 16 : 32; conv2d_WN = sg64 ? 128 : 32; भङ्ग;
                स्वभाव: भङ्ग;
            }
            ध्रुव uint32_t warps_M = conv2d_BS.K / conv2d_WM;
            ध्रुव uint32_t warps_N = conv2d_BS.NPQ / conv2d_WN;
            conv2d_WG_आकार         = warps_M * warps_N * device->subgroup_आकार;
        }

        // stage cm2 accumulator through shmem for coalesced global stores;
        // skipped on 128x128 where the extra Csh footprint hurts occupancy.
        // cm1 always uses the staged path.
        uint32_t conv2d_csh_store = (device->coopmat2 && s != CONV_SHAPE_128x128) ? 1u : 0u;
        यदि (conv2d_use_cm1) {
            conv2d_csh_store = 1;
        }

        // shmem is fp16 on cm2/cm1 (matches Csh), fp32 on scalar
        ध्रुव तार्किक conv2d_use_fp16_shmem = device->coopmat2 || conv2d_use_cm1;

        // shrink CRS if the non-cm1 config still doesn't fit
        यदि (device->properties.limits.max_Compute_Shared_स्मृतिः_आकार < shmem_req(conv2d_SHMEM_PAD, conv2d_csh_store, conv2d_use_fp16_shmem)) {
            GGML_ASSERT(!conv2d_use_cm1);
            conv2d_BS.CRS = 8;
            यदि (use_collectives) {
                conv2d_BS.CRS = std::min(device->subgroup_आकार, conv2d_BS.CRS);
            }
            conv2d_csh_store = 0;
        }

        std::array<uint32_t, 3> wg_denoms = { conv2d_BS.K, 1, 1 };
        std::vector<uint32_t> spec_constants = { conv2d_WG_आकार, conv2d_BS.K, conv2d_BS.CRS, conv2d_BS.NPQ, conv2d_TS_K, use_collectives, conv2d_SHMEM_PAD };

        // cm1 needs a fixed subgroup width to match the WG_SIZE we computed
        ध्रुव uint32_t conv2d_required_subgroup_आकार = conv2d_use_cm1 ? device->subgroup_आकार : 0;

#परिभाषय CREATE_CONV(name, type_suffix, spv_suffix) \
        प्रति (स्वचालित &c : device->pipeline_##name##type_suffix[s]) { \
            ध्रुव vk_conv2d_pipeline_state &state = c.first;  \
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
                परिमाण(vk_op_conv2d_प्रक्षिप_constants), wg_denoms, spec_constants_cpy, 1, सत्य, use_collectives || conv2d_required_subgroup_आकार, conv2d_required_subgroup_आकार);    \
        }
#परिभाषय CREATE_CONVS(spv_suffix) \
        CREATE_CONV(conv2d, _f32, spv_suffix) \
        CREATE_CONV(conv2d, _f16_f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f16_f32, spv_suffix)
#यदि defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        यदि (device->coopmat2) {
            CREATE_CONVS(_cm2)
        } अन्यथा
#यदि_अन्तः
#यदि defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        यदि (conv2d_use_cm1) {
            CREATE_CONVS(_cm1)
        } अन्यथा
#यदि_अन्तः
        यदि (conv2d_UNROLL) {
            CREATE_CONVS(_unroll)
        } अन्यथा {
            CREATE_CONVS( )
        }
#अपरिभाषय CREATE_CONV
#अपरिभाषय CREATE_CONVS

        std::vector<uint32_t> conv3d_spec_constants = { conv2d_WG_आकार, conv2d_BS.K, conv2d_BS.CRS, conv2d_BS.NPQ, conv2d_TS_K, conv2d_SHMEM_PAD };
#परिभाषय CREATE_CONV3D(type_suffix, spv_suffix) \
        प्रति (स्वचालित &c : device->pipeline_conv3d##type_suffix[s]) { \
            ध्रुव vk_conv3d_pipeline_state &state = c.first; \
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
                परिमाण(vk_op_conv3d_प्रक्षिप_constants), wg_denoms, spec_constants_cpy, 1, सत्य, conv2d_required_subgroup_आकार != 0, conv2d_required_subgroup_आकार); \
        }
#यदि defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        यदि (device->coopmat2) {
            CREATE_CONV3D(_f32, _cm2)
            CREATE_CONV3D(_f16_f32, _cm2)
        } अन्यथा
#यदि_अन्तः
#यदि defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        यदि (conv2d_use_cm1) {
            CREATE_CONV3D(_f32, _cm1)
            CREATE_CONV3D(_f16_f32, _cm1)
        } अन्यथा
#यदि_अन्तः
        यदि (conv2d_UNROLL) {
            CREATE_CONV3D(_f32, _unroll)
            CREATE_CONV3D(_f16_f32, _unroll)
        } अन्यथा {
            CREATE_CONV3D(_f32, )
            CREATE_CONV3D(_f16_f32, )
        }
#अपरिभाषय CREATE_CONV3D
    }

    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f32, "conv2d_dw_whcn_f32", conv2d_dw_whcn_f32_len, conv2d_dw_whcn_f32_दत्त, "main", 3, परिमाण(vk_op_conv2d_dw_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f32, "conv2d_dw_cwhn_f32", conv2d_dw_cwhn_f32_len, conv2d_dw_cwhn_f32_दत्त, "main", 3, परिमाण(vk_op_conv2d_dw_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f16_f32, "conv2d_dw_whcn_f16_f32", conv2d_dw_whcn_f16_f32_len, conv2d_dw_whcn_f16_f32_दत्त, "main", 3, परिमाण(vk_op_conv2d_dw_प्रक्षिप_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f16_f32, "conv2d_dw_cwhn_f16_f32", conv2d_dw_cwhn_f16_f32_len, conv2d_dw_cwhn_f16_f32_दत्त, "main", 3, परिमाण(vk_op_conv2d_dw_प्रक्षिप_constants), {512, 1, 1}, {}, 1);

    प्रति (uint32_t use_प्रक्षिप = 0; use_प्रक्षिप < 2; ++use_प्रक्षिप) {
        प्रति (uint32_t i = 0; i < अङ्कः_topk_moe_pipelines; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_topk_moe[i][use_प्रक्षिप], "topk_moe_f32_"+std::to_string(i), topk_moe_f32_len, topk_moe_f32_दत्त, "main", 4, परिमाण(vk_op_topk_moe_प्रक्षिप_constants), {1, 1, 1}, {device->subgroup_आकार, 1u<<i, use_प्रक्षिप}, 1, सत्य, सत्य, device->subgroup_आकार);
        }
    }

    // Drop compile_mutex so other threads can walk while we compile.
    compile_तालकम्.unlock();

    // Compile what we claimed; create_pipeline_func reacquires compile_mutex
    // at the end to flip compile_pending/compiled and notify waiters.
    यदि (has_claimed_task) {
        स्वचालित & task = claimed_task;
        ggml_vk_create_pipeline_func(device, task.pipeline, task.spv_आकार, task.spv_दत्त,
                                     task.entrypoint, task.parameter_गणनम्, task.wg_denoms,
                                     task.specialization_constants, task.disable_robustness,
                                     task.require_पूर्णम्_subgroups, task.required_subgroup_आकार);
    }

    // Another thread may be compiling the pipeline we need; block on it here.
    यदि (wait_pipeline) {
        std::unique_lock<std::mutex> wait_तालकम्(device->compile_mutex);
        device->compile_cv.प्रतीक्षस्व(wait_तालकम्, [&] {
            प्रत्यागम wait_pipeline->compiled.load();
        });
    }
}

स्थिर तार्किक ggml_vk_external_स्मृतिः_host_usable(vk::DriverId driver_id) {
    // Disable external_memory_host until https://github.com/KhronosGroup/MoltenVK/pull/2622
    // is available in the Vulkan SDK.
    प्रत्यागम driver_id != vk::DriverId::eMoltenvk;
}

स्थिर तार्किक ggml_vk_host_बफर_import_usable(uint32_t vendor_id) {
    // amdgpu re-faults all pages of every imported host allocation on each submission
    प्रत्यागम vendor_id != VK_VENDOR_ID_AMD;
}

vk_device ggml_vk_get_device(size_t idx) {
    VK_LOG_DEBUG("ggml_vk_get_device(" << idx << ")");

    स्थिर std::mutex mutex;
    std::lock_guard<std::mutex> तालकम्(mutex);

    यदि (vk_instance.devices[idx] == शून्यसूचकः) {
        VK_LOG_DEBUG("Initializing new vk_device");
        vk_device device = std::make_shared<vk_device_struct>();

        device->स्मृतिः_logger = std::unique_ptr<vk_स्मृतिः_logger>(नूतन vk_स्मृतिः_logger());

        size_t dev_अङ्कः = vk_instance.device_indices[idx];

        std::vector<vk::PhysicalDevice> physical_devices = vk_instance.instance.enumeratePhysicalDevices();

        यदि (dev_अङ्कः >= physical_devices.size()) {
            std::cerr << "ggml_vulkan: Device with index " << dev_अङ्कः << " does not exist." << std::पङ्क्त्यन्तः;
            क्षिप std::runtime_error("Device not found");
        }

        device->physical_device = physical_devices[dev_अङ्कः];
        ध्रुव std::vector<vk::ExtensionProperties> ext_props = device->physical_device.enumerateDeviceExtensionProperties();

        device->architecture = get_device_architecture(device->physical_device);

        ध्रुव वर्ण* GGML_VK_PREFER_HOST_स्मृतिः = getenv("GGML_VK_PREFER_HOST_MEMORY");
        device->prefer_host_स्मृतिः = GGML_VK_PREFER_HOST_स्मृतिः != शून्यसूचकः;

        ध्रुव वर्ण* GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM = getenv("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM");
        device->disable_host_visible_vidmem = GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM != शून्यसूचकः;

        ध्रुव वर्ण* GGML_VK_ALLOW_SYSMEM_FALLBACK = getenv("GGML_VK_ALLOW_SYSMEM_FALLBACK");
        device->allow_sysmem_fallback = GGML_VK_ALLOW_SYSMEM_FALLBACK != शून्यसूचकः;

        ध्रुव वर्ण* GGML_VK_DISABLE_GRAPH_OPTIMIZE = getenv("GGML_VK_DISABLE_GRAPH_OPTIMIZE");
        device->disable_graph_optimize = GGML_VK_DISABLE_GRAPH_OPTIMIZE != शून्यसूचकः;

        तार्किक fp16_storage = असत्य;
        तार्किक fp16_compute = असत्य;
        तार्किक maintenance4_support = असत्य;
        तार्किक sm_builtins = असत्य;
        तार्किक amd_shader_core_properties2 = असत्य;
        तार्किक pipeline_robustness = असत्य;
        तार्किक coopmat2_support = असत्य;
        तार्किक coopmat2_decode_सदिशम्_support = असत्य;
        तार्किक pipeline_executable_properties_support = असत्य;
        तार्किक internally_sync_support = असत्य;
        device->coopmat_support = असत्य;
        device->integer_dot_गुणनफलम् = असत्य;
        device->shader_64b_indexing = असत्य;
        तार्किक bfloat16_support = असत्य;
        तार्किक dot2_f16_support = असत्य;
        तार्किक ocp_microscaling_extension = असत्य;
        तार्किक shader_float8_extension = असत्य;

        प्रति (ध्रुव स्वचालित& properties : ext_props) {
            यदि (वाक्यं_तुलय("VK_KHR_maintenance4", properties.extensionName) == 0) {
                maintenance4_support = सत्य;
            } अन्यथा यदि (वाक्यं_तुलय("VK_KHR_16bit_storage", properties.extensionName) == 0) {
                fp16_storage = सत्य;
            } अन्यथा यदि (वाक्यं_तुलय("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
                fp16_compute = सत्य;
            } अन्यथा यदि (वाक्यं_तुलय("VK_NV_shader_sm_builtins", properties.extensionName) == 0) {
                sm_builtins = सत्य;
            } अन्यथा यदि (वाक्यं_तुलय("VK_AMD_shader_core_properties2", properties.extensionName) == 0) {
                amd_shader_core_properties2 = सत्य;
            } अन्यथा यदि (वाक्यं_तुलय("VK_EXT_pipeline_robustness", properties.extensionName) == 0) {
                pipeline_robustness = सत्य;
            } अन्यथा यदि (वाक्यं_तुलय("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                device->subgroup_आकार_control = सत्य;
#यदि defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            } अन्यथा यदि (वाक्यं_तुलय("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT")) {
                device->coopmat_support = सत्य;
                device->coopmat_m = 0;
                device->coopmat_n = 0;
                device->coopmat_k = 0;
#यदि_अन्तः
#यदि defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            } अन्यथा यदि (वाक्यं_तुलय("VK_NV_cooperative_matrix2", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT2")) {
                coopmat2_support = सत्य;
#यदि_अन्तः
            } अन्यथा यदि (वाक्यं_तुलय(VK_NV_COOPERATIVE_MATRIX_DECODE_सदिशम्_EXTENSION_NAME, properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR")) {
                coopmat2_decode_सदिशम्_support = सत्य;
#यदि defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            } अन्यथा यदि (वाक्यं_तुलय("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
                device->integer_dot_गुणनफलम् = सत्य;
#यदि_अन्तः
#यदि defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
            } अन्यथा यदि (वाक्यं_तुलय("VK_KHR_shader_bfloat16", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_BFLOAT16")) {
                bfloat16_support = सत्य;
#यदि_अन्तः
#यदि defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT)
            } अन्यथा यदि (वाक्यं_तुलय(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME, properties.extensionName) == 0) {
                ocp_microscaling_extension = सत्य;
#यदि_अन्तः
#यदि defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
            } अन्यथा यदि (वाक्यं_तुलय(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, properties.extensionName) == 0) {
                shader_float8_extension = सत्य;
#यदि_अन्तः
            } अन्यथा यदि (वाक्यं_तुलय("VK_VALVE_shader_mixed_float_dot_product", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_DOT2")) {
                dot2_f16_support = सत्य;
            } अन्यथा यदि (वाक्यं_तुलय("VK_KHR_pipeline_executable_properties", properties.extensionName) == 0) {
                pipeline_executable_properties_support = सत्य;
            } अन्यथा यदि (वाक्यं_तुलय("VK_EXT_memory_priority", properties.extensionName) == 0 &&
                       getenv("GGML_VK_ENABLE_MEMORY_PRIORITY")) {
                device->स्मृतिः_प्राथमिकता = सत्य;
            } अन्यथा यदि (वाक्यं_तुलय("VK_EXT_external_memory_host", properties.extensionName) == 0) {
                device->external_स्मृतिः_host = सत्य;
#यदि defined(VK_EXT_shader_64bit_indexing)
            } अन्यथा यदि (वाक्यं_तुलय("VK_EXT_shader_64bit_indexing", properties.extensionName) == 0) {
                device->shader_64b_indexing = सत्य;
#यदि_अन्तः
            } अन्यथा यदि (वाक्यं_तुलय(VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME, properties.extensionName) == 0) {
                internally_sync_support = सत्य;
            } अन्यथा यदि (वाक्यं_तुलय("VK_EXT_device_fault", properties.extensionName) == 0) {
                device->device_fault = सत्य;
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
        vk::Physical_Device_Subgroup_आकार_Control_Properties_EXT subgroup_आकार_control_props;
        vk::Physical_Device_Shader_Integer_Dot_गुणनफलम्_Properties_KHR shader_integer_dot_गुणनफलम्_props;
        vk::Physical_Device_External_स्मृतिः_Host_Properties_EXT external_स्मृतिः_host_props;

        props2.p_अग्र = &props3;
        props3.p_अग्र = &subgroup_props;
        subgroup_props.p_अग्र = &driver_props;
        driver_props.p_अग्र = &vk11_props;
        vk11_props.p_अग्र = &vk12_props;

        Vk_आधारः_बहिर्गमनम्_Structure * चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&vk12_props;

        यदि (maintenance4_support) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&props4;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&props4;
        }
        यदि (sm_builtins) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&sm_props;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&sm_props;
        }
        यदि (amd_shader_core_properties2) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&amd_shader_core_properties2_props;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&amd_shader_core_properties2_props;
        }
        यदि (device->subgroup_आकार_control) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&subgroup_आकार_control_props;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&subgroup_आकार_control_props;
        }

#यदि defined(VK_NV_cooperative_matrix2)
        vk::PhysicalDeviceCooperativeMatrix2PropertiesNV coopmat2_props;
        यदि (coopmat2_support) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&coopmat2_props;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&coopmat2_props;
        }
#यदि_अन्तः

        यदि (device->integer_dot_गुणनफलम्) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&shader_integer_dot_गुणनफलम्_props;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&shader_integer_dot_गुणनफलम्_props;
        }

        यदि (device->external_स्मृतिः_host) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&external_स्मृतिः_host_props;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&external_स्मृतिः_host_props;
        }

        device->physical_device.getProperties2(&props2);
        device->properties = props2.properties;
        device->vendor_id = device->properties.vendorID;
        device->driver_id = driver_props.driverID;

        device->external_स्मृतिः_host = device->external_स्मृतिः_host && ggml_vk_external_स्मृतिः_host_usable(device->driver_id);

        // Implementing the async backend interfaces seems broken on older Intel HW,
        // see https://github.com/ggml-org/llama.cpp/issues/17302.
        device->support_async = (device->vendor_id != VK_VENDOR_ID_INTEL ||
                                 std::string(device->properties.deviceName.दत्त()).find("(DG1)") == std::string::npos) &&
                                getenv("GGML_VK_DISABLE_ASYNC") == शून्यसूचकः;

        यदि (!device->support_async) {
            GGML_LOG_DEBUG("ggml_vulkan: WARNING: Async execution disabled on certain Intel devices.\n");
        }

        ध्रुव वर्ण* GGML_VK_FORCE_MAX_आवण्टनम्_आकार = getenv("GGML_VK_FORCE_MAX_ALLOCATION_SIZE");

        यदि (GGML_VK_FORCE_MAX_आवण्टनम्_आकार != शून्यसूचकः) {
            device->max_स्मृतिः_आवण्टनम्_आकार = std::stoull(GGML_VK_FORCE_MAX_आवण्टनम्_आकार);
        } अन्यथा यदि (maintenance4_support) {
            device->max_स्मृतिः_आवण्टनम्_आकार = std::min(props3.max_स्मृतिः_आवण्टनम्_आकार_2, props4.max_बफर_आकार_2);
        } अन्यथा {
            device->max_स्मृतिः_आवण्टनम्_आकार = props3.max_स्मृतिः_आवण्टनम्_आकार_2;
        }

        ध्रुव वर्ण* GGML_VK_FORCE_MAX_बफर_आकार = getenv("GGML_VK_FORCE_MAX_BUFFER_SIZE");

        यदि (GGML_VK_FORCE_MAX_बफर_आकार != शून्यसूचकः) {
            device->max_बफर_आकार = std::stoull(GGML_VK_FORCE_MAX_बफर_आकार);
        } अन्यथा यदि (maintenance4_support) {
            device->max_बफर_आकार = props4.max_बफर_आकार_2;
        } अन्यथा {
            device->max_बफर_आकार = device->max_स्मृतिः_आवण्टनम्_आकार;
        }

        ध्रुव वर्ण* GGML_VK_SUBALLOCATION_खण्डम्_आकार = getenv("GGML_VK_SUBALLOCATION_BLOCK_SIZE");

        यदि (GGML_VK_SUBALLOCATION_खण्डम्_आकार != शून्यसूचकः) {
            device->suballocation_खण्डम्_आकार = std::stoull(GGML_VK_SUBALLOCATION_खण्डम्_आकार);
        } अन्यथा {
            // Limit batching of allocations to 1GB by default to avoid fragmentation issues
            device->suballocation_खण्डम्_आकार = 1024*1024*1024;
        }
        device->suballocation_खण्डम्_आकार = std::min(device->suballocation_खण्डम्_आकार, device->max_स्मृतिः_आवण्टनम्_आकार);

        device->subgroup_आकार = subgroup_props.subgroup_आकार_2;
        device->subgroup_आकार_log2 = uint32_t(log2f(भिन्न(device->subgroup_आकार)));
        device->uma = device->properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
        यदि (sm_builtins) {
            device->shader_core_गणनम् = sm_props.shaderSMCount;
        } अन्यथा यदि (amd_shader_core_properties2) {
            device->shader_core_गणनम् = amd_shader_core_properties2_props.सक्रिय_Compute_Unit_गणनम्;
        } अन्यथा यदि (device->vendor_id == VK_VENDOR_ID_INTEL) {
            device->shader_core_गणनम् = ggml_vk_intel_shader_core_गणनम्(device->physical_device);
        } अन्यथा {
            device->shader_core_गणनम् = 0;
        }
        device->float_controls_rte_fp16 = vk12_props.shader_Rounding_बहुलकम्_RTEFloat16;
        device->float_controls_denorm_preserve_fp16 = vk12_props.shaderDenormPreserveFloat16;

        device->subgroup_basic = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_द्वयङ्काः::eCompute) &&
                                 (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_द्वयङ्काः::eBasic);
        device->subgroup_arithmetic = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_द्वयङ्काः::eCompute) &&
                                      (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_द्वयङ्काः::eArithmetic);
#यदि_परिभाषितम् __APPLE__
        // Workaround for subgroup arithmetic failing on MoltenVK with AMD GPUs (issue 15846)
        यदि (device->vendor_id == VK_VENDOR_ID_AMD) {
            device->subgroup_arithmetic = असत्य;
        }
#यदि_अन्तः
        device->subgroup_shuffle = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_द्वयङ्काः::eCompute) &&
                                   (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_द्वयङ्काः::eShuffle);
#यदि_परिभाषितम् __APPLE__
        यदि (device->vendor_id == VK_VENDOR_ID_AMD) {
            device->subgroup_shuffle = असत्य;
        }
#यदि_अन्तः
        device->subgroup_clustered = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_द्वयङ्काः::eCompute) &&
                                     (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_द्वयङ्काः::eClustered);

        device->subgroup_ballot = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_द्वयङ्काः::eCompute) &&
                                  (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_द्वयङ्काः::eBallot);

        device->subgroup_vote = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_द्वयङ्काः::eCompute) &&
                                (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_द्वयङ्काः::eVote);

        // Submit at least every 100 nodes, in case there are workloads without as much matmul.
        device->max_nodes_प्रत्येकम्_submit = 100;
        ध्रुव वर्ण* GGML_VK_MAX_NODES_प्रत्येकम्_SUBMIT = getenv("GGML_VK_MAX_NODES_PER_SUBMIT");
        यदि (GGML_VK_MAX_NODES_प्रत्येकम्_SUBMIT != शून्यसूचकः) {
            uint32_t max_nodes_प्रत्येकम्_submit = std::stoul(GGML_VK_MAX_NODES_प्रत्येकम्_SUBMIT);
            device->max_nodes_प्रत्येकम्_submit = std::max(max_nodes_प्रत्येकम्_submit, 1u);
        }

        ध्रुव तार्किक force_disable_f16 = getenv("GGML_VK_DISABLE_F16") != शून्यसूचकः;

        device->fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

        यदि (!ggml_vk_khr_cooperative_matrix_support(device->properties, driver_props, device->architecture)) {
            device->coopmat_support = असत्य;
        }

        device->integer_dot_गुणनफलम् = device->integer_dot_गुणनफलम् && shader_integer_dot_गुणनफलम्_props.integerDotProduct4x8BitPackedSignedAccelerated;

        device->min_imported_host_pointer_alignment = external_स्मृतिः_host_props.minImportedHostPointerAlignment;

        device->max_workgroup_आकार_log2 = uint32_t(log2f(भिन्न(device->properties.limits.max_Compute_Work_गणः_Invocations)));

        std::vector<vk::QueueFamilyProperties> queue_family_props = device->physical_device.getQueueFamilyProperties();

        // Try to find a non-graphics compute queue and transfer-focused queues
        // Allow overriding avoiding the graphics queue because it can increase performance on RADV
        ध्रुव तार्किक allow_graphics_queue = (getenv("GGML_VK_ALLOW_GRAPHICS_QUEUE") != शून्यसूचकः);
        ध्रुव vk::Queue_Flag_द्वयङ्काः graphics_flag = allow_graphics_queue ? (vk::Queue_Flag_द्वयङ्काः)0 : vk::Queue_Flag_द्वयङ्काः::eGraphics;
        ध्रुव uint32_t compute_queue_family_सूचकांक = ggml_vk_find_queue_family_सूचकांक(queue_family_props, vk::Queue_Flag_द्वयङ्काः::eCompute, graphics_flag, -1, 1);
        ध्रुव uint32_t transfer_queue_family_सूचकांक = ggml_vk_find_queue_family_सूचकांक(queue_family_props, vk::Queue_Flag_द्वयङ्काः::eTransfer, vk::Queue_Flag_द्वयङ्काः::eCompute | graphics_flag, compute_queue_family_सूचकांक, 1);

        ध्रुव भिन्न priorities[] = { 1.0f, 1.0f };
        device->single_queue = compute_queue_family_सूचकांक == transfer_queue_family_सूचकांक && queue_family_props[compute_queue_family_सूचकांक].queue_गणनम् == 1;

        std::vector<vk::DeviceQueueCreateInfo> device_queue_create_infos;
        vk::DeviceCreateInfo device_create_info{};
        std::vector<ध्रुव वर्ण *> device_extensions;
        vk::PhysicalDeviceFeatures device_features = device->physical_device.getFeatures();

        VkPhysicalDeviceFeatures2 device_features2;
        device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        device_features2.p_अग्र = शून्यसूचकः;
        device_features2.features = (VkPhysicalDeviceFeatures)device_features;

        VkPhysicalDeviceVulkan11Features vk11_features;
        vk11_features.p_अग्र = शून्यसूचकः;
        vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        device_features2.p_अग्र = &vk11_features;

        VkPhysicalDeviceVulkan12Features vk12_features;
        vk12_features.p_अग्र = शून्यसूचकः;
        vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vk11_features.p_अग्र = &vk12_features;

        चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&vk12_features;

        VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR internally_synchronized_queues_features{};
        internally_synchronized_queues_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR;
        internally_synchronized_queues_features.p_अग्र = शून्यसूचकः;
        internally_synchronized_queues_features.internallySynchronizedQueues = VK_FALSE;

        यदि (internally_sync_support) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&internally_synchronized_queues_features;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&internally_synchronized_queues_features;
            device_extensions.push_back(VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME);
        }

        VkPhysicalDevicePipelineRobustnessFeaturesEXT pl_robustness_features;
        pl_robustness_features.p_अग्र = शून्यसूचकः;
        pl_robustness_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES_EXT;
        pl_robustness_features.pipelineRobustness = VK_FALSE;

        यदि (pipeline_robustness) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&pl_robustness_features;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&pl_robustness_features;
            device_extensions.push_back("VK_EXT_pipeline_robustness");
        }

        Vk_Physical_Device_स्मृतिः_प्राथमिकता_Features_EXT स्मृतिः_प्राथमिकता_features;
        स्मृतिः_प्राथमिकता_features.p_अग्र = शून्यसूचकः;
        स्मृतिः_प्राथमिकता_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_स्मृतिः_प्राथमिकता_FEATURES_EXT;
        स्मृतिः_प्राथमिकता_features.स्मृतिः_प्राथमिकता_2 = VK_FALSE;
        यदि (device->स्मृतिः_प्राथमिकता) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&स्मृतिः_प्राथमिकता_features;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&स्मृतिः_प्राथमिकता_features;
            device_extensions.push_back("VK_EXT_memory_priority");
        }

        Vk_Physical_Device_Subgroup_आकार_Control_Features_EXT subgroup_आकार_control_features;
        subgroup_आकार_control_features.p_अग्र = शून्यसूचकः;
        subgroup_आकार_control_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_आकार_CONTROL_FEATURES_EXT;
        subgroup_आकार_control_features.compute_पूर्णम्_Subgroups = असत्य;
        subgroup_आकार_control_features.subgroup_आकार_Control = असत्य;

        यदि (device->subgroup_आकार_control) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&subgroup_आकार_control_features;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&subgroup_आकार_control_features;
        }

#यदि defined(VK_KHR_cooperative_matrix)
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmat_features;
        coopmat_features.p_अग्र = शून्यसूचकः;
        coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        coopmat_features.cooperativeMatrix = VK_FALSE;

        यदि (device->coopmat_support) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&coopmat_features;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&coopmat_features;
        }
#यदि_अन्तः

#यदि defined(VK_NV_cooperative_matrix2)
        VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features {};
        coopmat2_features.p_अग्र = शून्यसूचकः;
        coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV;
        यदि (coopmat2_support) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&coopmat2_features;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&coopmat2_features;
            device_extensions.push_back("VK_NV_cooperative_matrix2");
        }
#यदि_अन्तः

        Vk_Physical_Device_Cooperative_Matrix_Decode_सदिशम्_Features_NV coopmat2_decode_सदिशम्_features {};
        coopmat2_decode_सदिशम्_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_DECODE_सदिशम्_FEATURES_NV;
        यदि (coopmat2_decode_सदिशम्_support) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&coopmat2_decode_सदिशम्_features;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&coopmat2_decode_सदिशम्_features;
            device_extensions.push_back(VK_NV_COOPERATIVE_MATRIX_DECODE_सदिशम्_EXTENSION_NAME);
        }

#यदि defined(VK_KHR_shader_bfloat16)
        VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features {};
        bfloat16_features.p_अग्र = शून्यसूचकः;
        bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
        यदि (bfloat16_support) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&bfloat16_features;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&bfloat16_features;
            device_extensions.push_back("VK_KHR_shader_bfloat16");
        }
#यदि_अन्तः

        VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT ocp_microscaling_features {};
        ocp_microscaling_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT;
        यदि (ocp_microscaling_extension) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&ocp_microscaling_features;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&ocp_microscaling_features;
            device_extensions.push_back(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME);
        }

        VkPhysicalDeviceShaderFloat8FeaturesEXT shader_float8_features {};
        shader_float8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
        यदि (shader_float8_extension) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&shader_float8_features;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&shader_float8_features;
            device_extensions.push_back(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
        }

        VkPhysicalDeviceMaintenance4Features maint4_features {};
        maint4_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
        यदि (maintenance4_support) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&maint4_features;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&maint4_features;
            device_extensions.push_back("VK_KHR_maintenance4");
        }

        Vk_Physical_Device_Shader_Integer_Dot_गुणनफलम्_Features_KHR shader_integer_dot_गुणनफलम्_features {};
        shader_integer_dot_गुणनफलम्_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_गुणनफलम्_FEATURES_KHR;
        यदि (device->integer_dot_गुणनफलम्) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&shader_integer_dot_गुणनफलम्_features;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&shader_integer_dot_गुणनफलम्_features;
            device_extensions.push_back("VK_KHR_shader_integer_dot_product");
        }

        Vk_Physical_Device_Shader_Mixed_Float_Dot_गुणनफलम्_Features_VALVE dot2_features {};
        dot2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_गुणनफलम्_FEATURES_VALVE;
        यदि (dot2_f16_support) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&dot2_features;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&dot2_features;
            device_extensions.push_back("VK_VALVE_shader_mixed_float_dot_product");
        }

        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pep_features {};
        pep_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR;
        यदि (pipeline_executable_properties_support) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&pep_features;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&pep_features;
            device_extensions.push_back("VK_KHR_pipeline_executable_properties");
        }

        यदि (device->external_स्मृतिः_host) {
            device_extensions.push_back("VK_EXT_external_memory_host");
        }

#यदि defined(VK_EXT_shader_64bit_indexing)
        VkPhysicalDeviceShader64BitIndexingFeaturesEXT shader_64bit_indexing_features {};
        shader_64bit_indexing_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_64_BIT_INDEXING_FEATURES_EXT;
        यदि (device->shader_64b_indexing) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&shader_64bit_indexing_features;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&shader_64bit_indexing_features;
            device_extensions.push_back("VK_EXT_shader_64bit_indexing");
        }
#यदि_अन्तः

        VkPhysicalDeviceFaultFeaturesEXT fault_features {};
        fault_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT;
        यदि (device->device_fault) {
            चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&fault_features;
            चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&fault_features;
            device_extensions.push_back("VK_EXT_device_fault");
        }

        vkGetPhysicalDeviceFeatures2(device->physical_device, &device_features2);

        device->device_fault = device->device_fault && fault_features.deviceFault;

        device->has_internally_synchronized_queues = internally_synchronized_queues_features.internallySynchronizedQueues;

        // Build queue create infos only after querying whether internally synchronized queues are enabled.
        // getQueue2() later uses the same flag, so creation/retrieval must stay consistent.
        vk::DeviceQueueCreateFlags queue_flags = device->has_internally_synchronized_queues ?
                                                eInternallySynchronizedKHR :
                                                vk::DeviceQueueCreateFlags();

        यदि (compute_queue_family_सूचकांक != transfer_queue_family_सूचकांक) {
            device_queue_create_infos.push_back({queue_flags, compute_queue_family_सूचकांक, 1, priorities});
            device_queue_create_infos.push_back({queue_flags, transfer_queue_family_सूचकांक, 1, priorities + 1});
        } अन्यथा यदि(!device->single_queue) {
            device_queue_create_infos.push_back({queue_flags, compute_queue_family_सूचकांक, 2, priorities});
        } अन्यथा {
            device_queue_create_infos.push_back({queue_flags, compute_queue_family_सूचकांक, 1, priorities});
        }

        device->pipeline_executable_properties_support = pipeline_executable_properties_support;

        device->fp16 = device->fp16 && vk12_features.shaderFloat16;

#यदि defined(VK_KHR_shader_bfloat16)
        device->bf16 = bfloat16_support && bfloat16_features.shaderBFloat16Type;
#अन्यथा
        device->bf16 = असत्य;
#यदि_अन्तः

        device->dot2_f16 = dot2_f16_support && dot2_features.shader_Mixed_Float_Dot_गुणनफलम्_Float16_Acc_Float32;
        device->ocp_fp4 = ocp_microscaling_extension && ocp_microscaling_features.shaderFloat4 &&
                          shader_float8_extension && shader_float8_features.shaderFloat8 &&
                          !getenv("GGML_VK_DISABLE_OCP_FP4");

        device->pipeline_robustness = pl_robustness_features.pipelineRobustness;

        device->multi_सङ्कलय = vk12_props.shader_Rounding_बहुलकम्_RTEFloat16 &&
                            device->properties.limits.max_प्रक्षिप_Constants_आकार >= परिमाण(vk_op_multi_सङ्कलय_प्रक्षिप_constants) &&
                            getenv("GGML_VK_DISABLE_MULTI_ADD") == शून्यसूचकः;

        device->shader_int64 = device_features2.features.shaderInt64;
        device->बफर_device_स्थानाङ्कः = vk12_features.बफर_Device_स्थानाङ्कः;
        device->vulkan_स्मृतिः_model = vk12_features.vulkan_स्मृतिः_Model;

        यदि (device->subgroup_आकार_control) {
            device->subgroup_min_आकार = subgroup_आकार_control_props.min_Subgroup_आकार;
            device->subgroup_max_आकार = subgroup_आकार_control_props.max_Subgroup_आकार;
            device_extensions.push_back("VK_EXT_subgroup_size_control");
        }

        device->subgroup_आकार_control = device->subgroup_आकार_control &&
                (subgroup_आकार_control_props.required_Subgroup_आकार_Stages & vk::Shader_Stage_Flag_द्वयङ्काः::eCompute) &&
                subgroup_आकार_control_features.subgroup_आकार_Control;

        device->subgroup_require_पूर्णम्_support = subgroup_आकार_control_features.compute_पूर्णम्_Subgroups;

#यदि defined(VK_KHR_cooperative_matrix)
        device->coopmat_support = device->coopmat_support && coopmat_features.cooperativeMatrix;
        device->coopmat1_fa_support = device->coopmat_support && device->subgroup_require_पूर्णम्_support;
#यदि_अन्तः

        यदि (coopmat2_support) {
#यदि defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            यदि (coopmat2_features.cooperativeMatrixWorkgroupScope &&
                coopmat2_features.cooperativeMatrixFlexibleDimensions &&
                coopmat2_features.cooperativeMatrixReductions &&
                coopmat2_features.cooperativeMatrixConversions &&
                coopmat2_features.cooperative_Matrix_प्रत्येकम्_Element_Operations &&
                coopmat2_features.cooperativeMatrixTensorAddressing &&
                coopmat2_features.cooperative_Matrix_खण्डम्_Loads &&
                vk12_features.बफर_Device_स्थानाङ्कः) {

                std::vector<VkCooperativeMatrixFlexibleDimensionsPropertiesNV> flexible_dimensions;
                uint32_t count = 0;

                PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV
                    _vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV =
                        (PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV)
                        vk_instance.instance.getProcAddr("vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV");

                _vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(device->physical_device, &count, शून्यसूचकः);

                VkCooperativeMatrixFlexibleDimensionsPropertiesNV रिक्तानि_prop {};
                रिक्तानि_prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_FLEXIBLE_DIMENSIONS_PROPERTIES_NV;
                flexible_dimensions.resize(count, रिक्तानि_prop);

                _vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(device->physical_device, &count, flexible_dimensions.दत्त());

                तार्किक प्राप्तम्_fp16_128 = असत्य,
                     प्राप्तम्_fp16_256 = असत्य,
                     प्राप्तम्_fp32_128 = असत्य,
                     प्राप्तम्_fp32_256 = असत्य;
                तार्किक प्राप्तम्_bf16_128 = असत्य,
                     प्राप्तम्_bf16_256 = असत्य;
                // need to support fp16*fp16 with fp16/fp32 accumulator, for workgroupsize 128
                // with 32x16x16 and 256 with 32x32x16.
                प्रति (स्वचालित &prop : flexible_dimensions) {
                    यदि (prop.saturatingAccumulation == VK_FALSE &&
                        prop.scope == VK_SCOPE_WORKGROUP_KHR) {

                        यदि (prop.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                            prop.BType == VK_COMPONENT_TYPE_FLOAT16_KHR) {

                            यदि (prop.workgroupInvocations == 128 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 16 &&
                                prop.KGranularity <= 16) {
                                यदि (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    prop.परिणाम_Type == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                                    प्राप्तम्_fp16_128 = सत्य;
                                }
                                यदि (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                    prop.परिणाम_Type == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                                    प्राप्तम्_fp32_128 = सत्य;
                                }
                            }
                            यदि (prop.workgroupInvocations == 256 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 32 &&
                                prop.KGranularity <= 16) {
                                यदि (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    prop.परिणाम_Type == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                                    प्राप्तम्_fp16_256 = सत्य;
                                }
                                यदि (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                    prop.परिणाम_Type == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                                    प्राप्तम्_fp32_256 = सत्य;
                                }
                            }
                        }

#यदि defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                        यदि (bfloat16_support &&
                            prop.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                            prop.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                            prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                            prop.परिणाम_Type == VK_COMPONENT_TYPE_FLOAT32_KHR) {

                            यदि (prop.workgroupInvocations == 128 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 16 &&
                                prop.KGranularity <= 16) {
                                प्राप्तम्_bf16_128 = सत्य;
                            }
                            यदि (prop.workgroupInvocations == 256 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 32 &&
                                prop.KGranularity <= 16) {
                                प्राप्तम्_bf16_256 = सत्य;
                            }
                        }
#यदि_अन्तः
                    }
                }
                यदि (प्राप्तम्_fp16_128 && प्राप्तम्_fp16_256 &&
                    प्राप्तम्_fp32_128 && प्राप्तम्_fp32_256 &&
                    coopmat2_props.cooperativeMatrixFlexibleDimensionsMaxDimension >= 512) {
                    device->coopmat2 = सत्य;
                    device->coopmat2_bf16_support = प्राप्तम्_bf16_128 && प्राप्तम्_bf16_256;
                    device->coopmat2_decode_सदिशम् = coopmat2_decode_सदिशम्_support && coopmat2_decode_सदिशम्_features.cooperative_Matrix_Decode_सदिशम्;
                }
            }
#यदि_अन्तः
        }

        यदि (!vk11_features.storageBuffer16BitAccess) {
            std::cerr << "ggml_vulkan: device " << GGML_VK_NAME << idx << " does not support 16-bit storage." << std::पङ्क्त्यन्तः;
            क्षिप std::runtime_error("Unsupported device");
        }

        device_extensions.push_back("VK_KHR_16bit_storage");

#यदि_परिभाषितम् GGML_VULKAN_VALIDATE
        device_extensions.push_back("VK_KHR_shader_non_semantic_info");
#यदि_अन्तः

        यदि (device->fp16) {
            device_extensions.push_back("VK_KHR_shader_float16_int8");
        }

#यदि defined(VK_KHR_cooperative_matrix)
        यदि (device->coopmat_support) {
            // Query supported shapes
            std::vector<VkCooperativeMatrixPropertiesKHR> cm_props;

            PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR =
                (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(vk_instance.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");

            uint32_t cm_props_अङ्कः;

            pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device->physical_device, &cm_props_अङ्कः, शून्यसूचकः);

            cm_props.resize(cm_props_अङ्कः);

            प्रति (स्वचालित& prop : cm_props) {
                prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            }

            pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device->physical_device, &cm_props_अङ्कः, cm_props.दत्त());

            VK_LOG_DEBUG("ggml_vulkan: Cooperative Matrix Shapes: " << cm_props.size());

            प्रति (स्वचालित& prop : cm_props) {
                VK_LOG_DEBUG("ggml_vulkan: M: " << prop.MSize << " N: " << prop.NSize << " K: " << prop.KSize << " A: " << vk::to_string((vk::ComponentTypeKHR)prop.AType) << " B: " << vk::to_string((vk::ComponentTypeKHR)prop.BType) << " C: " << vk::to_string((vk::ComponentTypeKHR)prop.CType) << " Result: " << vk::to_string((vk::ComponentTypeKHR)prop.परिणाम_Type) << " saturatingAccumulation: " << prop.saturatingAccumulation << " scope: " << vk::to_string((vk::ScopeKHR)prop.scope));

                यदि ((vk::ComponentTypeKHR)prop.AType == vk::ComponentTypeKHR::eFloat16 &&
                    (vk::ComponentTypeKHR)prop.BType == vk::ComponentTypeKHR::eFloat16 &&
                    (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
                ) {
                    यदि ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat32 &&
                        (vk::ComponentTypeKHR)prop.परिणाम_Type == vk::ComponentTypeKHR::eFloat32) {
                        // coopmat sizes not set yet
                        यदि (device->coopmat_m == 0) {
                            device->coopmat_acc_f32_support = सत्य;
                            device->coopmat_m = prop.MSize;
                            device->coopmat_n = prop.NSize;
                            device->coopmat_k = prop.KSize;
                        } अन्यथा यदि (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                            // Only enable if shape is identical
                            device->coopmat_acc_f32_support = सत्य;
                        }
                        यदि (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
                            device->coopmat_support_16x16x16_f32acc = सत्य;
                        }
                    } अन्यथा यदि ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat16 &&
                               (vk::ComponentTypeKHR)prop.परिणाम_Type == vk::ComponentTypeKHR::eFloat16) {
                        // coopmat sizes not set yet
                        यदि (device->coopmat_m == 0) {
                            device->coopmat_acc_f16_support = सत्य;
                            device->coopmat_m = prop.MSize;
                            device->coopmat_n = prop.NSize;
                            device->coopmat_k = prop.KSize;
                        } अन्यथा यदि (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                            // Only enable if shape is identical
                            device->coopmat_acc_f16_support = सत्य;
                        }
                        यदि (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
                            device->coopmat_support_16x16x16_f16acc = सत्य;
                        }
                    }
                } अन्यथा यदि ((vk::ComponentTypeKHR)prop.AType      == vk::ComponentTypeKHR::eSint8 &&
                           (vk::ComponentTypeKHR)prop.BType      == vk::ComponentTypeKHR::eSint8 &&
                           (vk::ComponentTypeKHR)prop.CType      == vk::ComponentTypeKHR::eSint32 &&
                           (vk::ComponentTypeKHR)prop.परिणाम_Type == vk::ComponentTypeKHR::eSint32 &&
                           (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup &&
                           device->coopmat_int_m == 0
                ) {
                    device->coopmat_int_support = सत्य;
                    device->coopmat_int_m = prop.MSize;
                    device->coopmat_int_n = prop.NSize;
                    device->coopmat_int_k = prop.KSize;
                }
#यदि defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                यदि (bfloat16_support &&
                    prop.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    prop.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    prop.परिणाम_Type == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
                ) {
                    // coopmat sizes not set yet
                    यदि (device->coopmat_m == 0) {
                        device->coopmat_bf16_support = सत्य;
                        device->coopmat_m = prop.MSize;
                        device->coopmat_n = prop.NSize;
                        device->coopmat_k = prop.KSize;
                    } अन्यथा यदि (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                        // Only enable if shape is identical
                        device->coopmat_bf16_support = सत्य;
                    }
                }
#यदि_अन्तः
            }

            यदि (device->coopmat_m == 0 || !device->coopmat_acc_f32_support) {
                // No suitable matmul mode found
                GGML_LOG_DEBUG("ggml_vulkan: WARNING: No suitable matrix core mode found. Disabling matrix cores.\n");
                device->coopmat_support = असत्य;
            }
        }

        यदि (device->coopmat_support) {
            device_extensions.push_back("VK_KHR_cooperative_matrix");
        }
#यदि_अन्तः
        device->name = GGML_VK_NAME + std::to_string(idx);

        device_create_info
            .समुच्चयः_Flags(vk::DeviceCreateFlags())
            .समुच्चयः_Queue_Create_Infos(device_queue_create_infos)
            .समुच्चयः_PEnabled_Extension_नामानि(device_extensions);
        device_create_info.समुच्चयः_PNext(&device_features2);
        device->device = device->physical_device.createDevice(device_create_info);

        यदि (device->device_fault) {
            device->pfn_vkGetDeviceFaultInfoEXT = (PFN_vkGetDeviceFaultInfoEXT)
                vkGetDeviceProcAddr(device->device, "vkGetDeviceFaultInfoEXT");
        }

        // Queues
        device->compute_queue = ggml_vk_create_queue(device, compute_queue_family_सूचकांक, 0, { vk::Pipeline_Stage_Flag_द्वयङ्काः::eComputeShader | vk::Pipeline_Stage_Flag_द्वयङ्काः::eTransfer }, असत्य);

        // Shaders
        // Disable matmul tile sizes early if performance low or not supported
        प्रति (uint32_t i = 0; i < GGML_TYPE_गणनम्; ++i) {
            विकल्प (device->vendor_id) {
#यदि_अपरिभाषितम् GGML_VULKAN_सततम्_TESTS
            पक्ष VK_VENDOR_ID_AMD:
                device->mul_mat_l[i]    = device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary;
                device->mul_mat_m[i]    = सत्य;
                device->mul_mat_s[i]    = सत्य;
                device->mul_mat_id_l[i] = असत्य;
                device->mul_mat_id_m[i] = सत्य;
                device->mul_mat_id_s[i] = सत्य;
                भङ्ग;
            पक्ष VK_VENDOR_ID_INTEL: {
                // Current Windows driver does not expose BF16 support.
                // We only want to use l_warptile if coopmat is available
                ध्रुव तार्किक use_l_warptile = (i == GGML_TYPE_BF16) ? (device->coopmat_bf16_support && device->coopmat_support) : device->coopmat_support;
                device->mul_mat_l[i] = use_l_warptile;
                device->mul_mat_id_l[i] = use_l_warptile;
                device->mul_mat_m[i] = सत्य;
                device->mul_mat_s[i] = सत्य;
                device->mul_mat_id_m[i] = सत्य;
                device->mul_mat_id_s[i] = सत्य;
                भङ्ग;
            }
            पक्ष VK_VENDOR_ID_APPLE:
                device->mul_mat_l[i] = असत्य;
                device->mul_mat_m[i] = सत्य;
                device->mul_mat_s[i] = असत्य;
                device->mul_mat_id_l[i] = असत्य;
                device->mul_mat_id_m[i] = सत्य;
                device->mul_mat_id_s[i] = असत्य;
                भङ्ग;
            पक्ष VK_VENDOR_ID_QUALCOMM:
                device->mul_mat_l[i] = असत्य;
                device->mul_mat_m[i] = सत्य;
                device->mul_mat_s[i] = !device->coopmat_support;
                device->mul_mat_id_l[i] = असत्य;
                device->mul_mat_id_m[i] = सत्य;
                device->mul_mat_id_s[i] = !device->coopmat_support;
                भङ्ग;
            पक्ष VK_VENDOR_ID_SAMSUNG:
                device->mul_mat_l[i] = असत्य;
                device->mul_mat_m[i] = सत्य;
                device->mul_mat_s[i] = सत्य;
                device->mul_mat_id_l[i] = असत्य;
                device->mul_mat_id_m[i] = सत्य;
                device->mul_mat_id_s[i] = सत्य;
                भङ्ग;
#यदि_अन्तः
            स्वभाव:
                device->mul_mat_l[i] = सत्य;
                device->mul_mat_m[i] = सत्य;
                device->mul_mat_s[i] = सत्य;
                device->mul_mat_id_l[i] = सत्य;
                device->mul_mat_id_m[i] = सत्य;
                device->mul_mat_id_s[i] = सत्य;
                भङ्ग;
            }

#यदि VK_शीर्षकम्_संस्करणम् >= 287
            // Honeykrisp driver for Asahi Linux doesn't report VK_VENDOR_ID_APPLE.
            // Check for Honeykrisp driver and force same configuration as the VK_VENDOR_ID_APPLE case.
            यदि (device->driver_id == vk::DriverId::eMesaHoneykrisp) {
                device->mul_mat_l[i] = असत्य;
                device->mul_mat_m[i] = सत्य;
                device->mul_mat_s[i] = असत्य;
                device->mul_mat_id_l[i] = असत्य;
                device->mul_mat_id_m[i] = सत्य;
                device->mul_mat_id_s[i] = असत्य;
            }
#यदि_अन्तः

            device->mul_mat_l_int[i]    = device->mul_mat_l[i];
            device->mul_mat_m_int[i]    = device->mul_mat_m[i];
            device->mul_mat_s_int[i]    = device->mul_mat_s[i];
            device->mul_mat_id_l_int[i] = device->mul_mat_id_l[i];
            device->mul_mat_id_m_int[i] = device->mul_mat_id_m[i];
            device->mul_mat_id_s_int[i] = device->mul_mat_id_s[i];
        }


        std::vector<vk::Descriptor_समुच्चयः_Layout_Binding> dsl_binding;
        std::vector<vk::DescriptorBindingFlags> dsl_binding_flags;
        प्रति (uint32_t i = 0; i < MAX_PARAMETER_गणनम्; i++) {
            dsl_binding.push_back({i, vk::DescriptorType::e_Storage_बफर, 1, vk::Shader_Stage_Flag_द्वयङ्काः::eCompute});
            dsl_binding_flags.push_back({});
        }

        vk::Descriptor_समुच्चयः_Layout_Binding_Flags_Create_Info dslbfci = { dsl_binding_flags };

        vk::Descriptor_समुच्चयः_Layout_Create_Info descriptor_समुच्चयः_layout_create_info(
            {},
            dsl_binding);
        descriptor_समुच्चयः_layout_create_info.समुच्चयः_PNext(&dslbfci);
        device->dsl = device->device.create_Descriptor_समुच्चयः_Layout(descriptor_समुच्चयः_layout_create_info);

        ggml_vk_आरोपय_shaders(device);

        // Prefer a dedicated transfer queue on AMD dGPUs (non-GCN) when graphics queue use is disabled.
        ध्रुव तार्किक prefers_transfer_queue =
            device->vendor_id == VK_VENDOR_ID_AMD &&
            device->architecture != AMD_GCN &&
            !device->uma &&
            !allow_graphics_queue;

        यदि (!device->single_queue) {
            ध्रुव uint32_t transfer_queue_सूचकांक = compute_queue_family_सूचकांक == transfer_queue_family_सूचकांक ? 1 : 0;
            device->transfer_queue = ggml_vk_create_queue(device, transfer_queue_family_सूचकांक, transfer_queue_सूचकांक, { vk::Pipeline_Stage_Flag_द्वयङ्काः::eTransfer }, सत्य);

            device->async_use_transfer_queue = prefers_transfer_queue || (getenv("GGML_VK_ASYNC_USE_TRANSFER_QUEUE") != शून्यसूचकः);
        } अन्यथा {
            device->transfer_queue = ggml_vk_create_aliased_queue(device, device->compute_queue);

            device->async_use_transfer_queue = असत्य;
        }

        device->fence = device->device.createFence({});

        device->बफर_type = {
            /* .iface    = */ ggml_backend_vk_बफर_type_interface,
            /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), idx),
            /* .context  = */ नूतन ggml_backend_vk_बफर_type_context{ device->name, device },
        };

        device->idx = idx;

        device->serialize_submissions = getenv("GGML_VK_SERIALIZE_SUBMISSIONS") != शून्यसूचकः;

        device->disable_fusion = getenv("GGML_VK_DISABLE_FUSION") != शून्यसूचकः;

        device->disable_descriptor_reuse = getenv("GGML_VK_DISABLE_DESCRIPTOR_REUSE") != शून्यसूचकः;

        device->सङ्कलय_rms_fusion = !device->disable_fusion &&
                                 device->subgroup_arithmetic &&
                                 device->vendor_id != VK_VENDOR_ID_INTEL;
        device->partials_binding_alignment =
            std::max(4u, (uint32_t)device->properties.limits.min_Storage_बफर_विस्थापनम्_Alignment);

        device->mmvq_बहुलकम् = 0;
        यदि (getenv("GGML_VK_DISABLE_MMVQ")) {
            device->mmvq_बहुलकम् = -1;
        } अन्यथा यदि (getenv("GGML_VK_FORCE_MMVQ")) {
            device->mmvq_बहुलकम् = 1;
        }

        // publish the device only when it is fully built
        vk_instance.devices[idx] = device;

        प्रत्यागम device;
    }

    प्रत्यागम vk_instance.devices[idx];
}

स्थिर शून्य ggml_vk_print_gpu_info(size_t idx) {
    GGML_ASSERT(idx < vk_instance.device_indices.size());
    size_t dev_अङ्कः = vk_instance.device_indices[idx];
    VK_LOG_DEBUG("ggml_vk_print_gpu_info(" << dev_अङ्कः << ")");
    GGML_ASSERT(vk_instance_initialized);

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    यदि (dev_अङ्कः >= devices.size()) {
        std::cerr << "ggml_vulkan: Device with index " << dev_अङ्कः << " does not exist." << std::पङ्क्त्यन्तः;
        क्षिप std::runtime_error("Device not found");
    }

    vk::PhysicalDevice physical_device = devices[dev_अङ्कः];
    std::vector<vk::ExtensionProperties> ext_props = physical_device.enumerateDeviceExtensionProperties();

    तार्किक fp16_storage = असत्य;
    तार्किक fp16_compute = असत्य;
    तार्किक coopmat_support = असत्य;
    तार्किक coopmat2_support = असत्य;
    तार्किक coopmat2_decode_सदिशम्_support = असत्य;
    तार्किक integer_dot_गुणनफलम् = असत्य;
    तार्किक bfloat16_support = असत्य;
    तार्किक dot2_f16_support = असत्य;
    तार्किक ocp_microscaling_extension = असत्य;
    तार्किक shader_float8_extension = असत्य;

    प्रति (स्वचालित properties : ext_props) {
        यदि (वाक्यं_तुलय("VK_KHR_16bit_storage", properties.extensionName) == 0) {
            fp16_storage = सत्य;
        } अन्यथा यदि (वाक्यं_तुलय("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
            fp16_compute = सत्य;
#यदि defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
       } अन्यथा यदि (वाक्यं_तुलय("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT")) {
            coopmat_support = सत्य;
#यदि_अन्तः
#यदि defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        } अन्यथा यदि (वाक्यं_तुलय("VK_NV_cooperative_matrix2", properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT2")) {
            coopmat2_support = सत्य;
#यदि_अन्तः
        } अन्यथा यदि (वाक्यं_तुलय(VK_NV_COOPERATIVE_MATRIX_DECODE_सदिशम्_EXTENSION_NAME, properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR")) {
            coopmat2_decode_सदिशम्_support = सत्य;
#यदि defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        } अन्यथा यदि (वाक्यं_तुलय("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
            integer_dot_गुणनफलम् = सत्य;
#यदि_अन्तः
#यदि defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        } अन्यथा यदि (वाक्यं_तुलय("VK_KHR_shader_bfloat16", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_BFLOAT16")) {
            bfloat16_support = सत्य;
#यदि_अन्तः
#यदि defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT)
        } अन्यथा यदि (वाक्यं_तुलय(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME, properties.extensionName) == 0) {
            ocp_microscaling_extension = सत्य;
#यदि_अन्तः
#यदि defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        } अन्यथा यदि (वाक्यं_तुलय(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, properties.extensionName) == 0) {
            shader_float8_extension = सत्य;
#यदि_अन्तः
        } अन्यथा यदि (वाक्यं_तुलय("VK_VALVE_shader_mixed_float_dot_product", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_DOT2")) {
            dot2_f16_support = सत्य;
        }
    }

    ध्रुव vk_device_architecture device_architecture = get_device_architecture(physical_device);

    ध्रुव वर्ण* GGML_VK_DISABLE_F16 = getenv("GGML_VK_DISABLE_F16");
    तार्किक force_disable_f16 = GGML_VK_DISABLE_F16 != शून्यसूचकः;

    तार्किक fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

    vk::PhysicalDeviceProperties2 props2;
    vk::PhysicalDeviceMaintenance3Properties props3;
    vk::PhysicalDeviceSubgroupProperties subgroup_props;
    vk::PhysicalDeviceDriverProperties driver_props;
    vk::Physical_Device_Shader_Integer_Dot_गुणनफलम्_Properties_KHR shader_integer_dot_गुणनफलम्_props;
    props2.p_अग्र = &props3;
    props3.p_अग्र = &subgroup_props;
    subgroup_props.p_अग्र = &driver_props;

    // Pointer to the last chain element
    Vk_आधारः_बहिर्गमनम्_Structure * चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&driver_props;

    यदि (integer_dot_गुणनफलम्) {
        चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&shader_integer_dot_गुणनफलम्_props;
        चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&shader_integer_dot_गुणनफलम्_props;
    }

    physical_device.getProperties2(&props2);

    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    device_features2.p_अग्र = शून्यसूचकः;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.p_अग्र = शून्यसूचकः;
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.p_अग्र = &vk11_features;

    VkPhysicalDeviceVulkan12Features vk12_features;
    vk12_features.p_अग्र = शून्यसूचकः;
    vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk11_features.p_अग्र = &vk12_features;

    // Pointer to the last chain element
    चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&vk12_features;

#यदि defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmat_features;
    coopmat_features.p_अग्र = शून्यसूचकः;
    coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    coopmat_features.cooperativeMatrix = VK_FALSE;

    यदि (coopmat_support) {
        चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&coopmat_features;
        चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&coopmat_features;
    }
#यदि_अन्तः

    Vk_Physical_Device_Shader_Integer_Dot_गुणनफलम्_Features_KHR shader_integer_dot_गुणनफलम्_features {};
    shader_integer_dot_गुणनफलम्_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_गुणनफलम्_FEATURES_KHR;
    यदि (integer_dot_गुणनफलम्) {
        चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&shader_integer_dot_गुणनफलम्_features;
        चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&shader_integer_dot_गुणनफलम्_features;
    }

#यदि defined(VK_KHR_shader_bfloat16)
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features {};
    bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
    यदि (bfloat16_support) {
        चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&bfloat16_features;
        चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&bfloat16_features;
    }
#यदि_अन्तः

#यदि defined(VK_NV_cooperative_matrix2)
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features {};
    coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV;
    यदि (coopmat2_support) {
        चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&coopmat2_features;
        चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&coopmat2_features;
    }
#यदि_अन्तः

    Vk_Physical_Device_Cooperative_Matrix_Decode_सदिशम्_Features_NV coopmat2_decode_सदिशम्_features {};
    coopmat2_decode_सदिशम्_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_DECODE_सदिशम्_FEATURES_NV;
    यदि (coopmat2_decode_सदिशम्_support) {
        चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&coopmat2_decode_सदिशम्_features;
        चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&coopmat2_decode_सदिशम्_features;
    }

    Vk_Physical_Device_Shader_Mixed_Float_Dot_गुणनफलम्_Features_VALVE dot2_features {};
    dot2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_गुणनफलम्_FEATURES_VALVE;
    यदि (dot2_f16_support) {
        चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&dot2_features;
        चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&dot2_features;
    }

#यदि defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT ocp_microscaling_features {};
    ocp_microscaling_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT;
    VkPhysicalDeviceShaderFloat8FeaturesEXT shader_float8_features {};
    shader_float8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
    यदि (ocp_microscaling_extension) {
        चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&ocp_microscaling_features;
        चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&ocp_microscaling_features;
    }
    यदि (shader_float8_extension) {
        चरमः_struct->p_अग्र = (Vk_आधारः_बहिर्गमनम्_Structure *)&shader_float8_features;
        चरमः_struct = (Vk_आधारः_बहिर्गमनम्_Structure *)&shader_float8_features;
    }
#यदि_अन्तः

    vkGetPhysicalDeviceFeatures2(physical_device, &device_features2);

    fp16 = fp16 && vk12_features.shaderFloat16;

#यदि defined(VK_KHR_shader_bfloat16)
    तार्किक bf16 = bfloat16_support && bfloat16_features.shaderBFloat16Type;
#अन्यथा
    तार्किक bf16 = असत्य;
#यदि_अन्तः

    uint32_t default_subgroup_आकार = get_subgroup_आकार("", device_architecture);
    ध्रुव size_t subgroup_आकार = (default_subgroup_आकार != 0) ? default_subgroup_आकार : subgroup_props.subgroup_आकार_2;
    ध्रुव तार्किक uma = props2.properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

    integer_dot_गुणनफलम् = integer_dot_गुणनफलम्
                       && shader_integer_dot_गुणनफलम्_props.integerDotProduct4x8BitPackedSignedAccelerated
                       && shader_integer_dot_गुणनफलम्_features.shader_Integer_Dot_गुणनफलम्;

    coopmat_support = coopmat_support
#यदि defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                   && coopmat_features.cooperativeMatrix
#यदि_अन्तः
                   && ggml_vk_khr_cooperative_matrix_support(props2.properties, driver_props, device_architecture);

#यदि defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    coopmat2_support = coopmat2_support &&
                       coopmat2_features.cooperativeMatrixWorkgroupScope &&
                       coopmat2_features.cooperativeMatrixFlexibleDimensions &&
                       coopmat2_features.cooperativeMatrixReductions &&
                       coopmat2_features.cooperativeMatrixConversions &&
                       coopmat2_features.cooperative_Matrix_प्रत्येकम्_Element_Operations &&
                       coopmat2_features.cooperativeMatrixTensorAddressing &&
                       coopmat2_features.cooperative_Matrix_खण्डम्_Loads;
#अन्यथा
    coopmat2_support = असत्य;
#यदि_अन्तः

    coopmat2_decode_सदिशम्_support = coopmat2_decode_सदिशम्_support && coopmat2_decode_सदिशम्_features.cooperative_Matrix_Decode_सदिशम्;
#यदि !defined(GGML_VULKAN_COOPMAT2_DECODE_सदिशम्_GLSLC_SUPPORT)
    coopmat2_decode_सदिशम्_support = असत्य;
#यदि_अन्तः

    std::string matrix_cores = coopmat2_support ? (coopmat2_decode_सदिशम्_support ? "NV_coopmat2v" : "NV_coopmat2")
                             : coopmat_support  ? "KHR_coopmat"
                             : "none";

    तार्किक dot2_f16 = dot2_f16_support && dot2_features.shader_Mixed_Float_Dot_गुणनफलम्_Float16_Acc_Float32;
    ध्रुव वर्ण *fp16_str = fp16 ? (dot2_f16 ? "dot2" : "1") : "0";
#यदि defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    ध्रुव तार्किक fp4 = ocp_microscaling_extension && ocp_microscaling_features.shaderFloat4 &&
                     shader_float8_extension && shader_float8_features.shaderFloat8 &&
                     !getenv("GGML_VK_DISABLE_OCP_FP4");
#अन्यथा
    GGML_अप्रयुक्तम्(ocp_microscaling_extension);
    GGML_अप्रयुक्तम्(shader_float8_extension);
    ध्रुव तार्किक fp4 = असत्य;
#यदि_अन्तः

    std::string device_name = props2.properties.deviceName.दत्त();
    GGML_LOG_DEBUG("ggml_vulkan: %zu = %s (%s) | uma: %d | fp16: %s | bf16: %d | fp4: %d | warp size: %zu | shared memory: %d | int dot: %d | matrix cores: %s\n",
              idx, device_name.c_str(), driver_props.driverName.दत्त(), uma, fp16_str, bf16, fp4, subgroup_आकार,
              props2.properties.limits.max_Compute_Shared_स्मृतिः_आकार, integer_dot_गुणनफलम्, matrix_cores.c_str());

    यदि (props2.properties.deviceType == vk::PhysicalDeviceType::eCpu) {
        GGML_LOG_DEBUG("ggml_vulkan: Warning: Device type is CPU. This is probably not the device you want.\n");
    }
}

स्थिर DispatchLoaderDynamic ggml_vk_default_dispatcher_instance;

DispatchLoaderDynamic & ggml_vk_default_dispatcher() {
    प्रत्यागम ggml_vk_default_dispatcher_instance;
}

शून्य ggml_vk_instance_init() {
    यदि (vk_instance_initialized) {
        प्रत्यागम;
    }
    VK_LOG_DEBUG("ggml_vk_instance_init()");

    // See https://github.com/KhronosGroup/Vulkan-Hpp?tab=readme-ov-file#extensions--per-device-function-pointers-
    ggml_vk_default_dispatcher_instance.init(vkGetInstanceProcAddr);

    // vkEnumerateInstanceVersion is Vulkan 1.1. A null value indicated Vulkan 1.0.
    यदि (ggml_vk_default_dispatcher_instance.vk_Enumerate_Instance_संस्करणम् == शून्यसूचकः) {
        GGML_LOG_ERROR("ggml_vulkan: Error: Vulkan 1.2 required.");
        क्षिप vk::SystemError(vk::परिणाम_2::eErrorFeatureNotPresent, "Vulkan 1.2 required");
    }

    uint32_t api_संस्करणम् = vk::enumerate_Instance_संस्करणम्();

    यदि (api_संस्करणम् < VK_API_संस्करणम्_1_2) {
        GGML_LOG_ERROR("ggml_vulkan: Error: Vulkan 1.2 required.");
        क्षिप vk::SystemError(vk::परिणाम_2::eErrorFeatureNotPresent, "Vulkan 1.2 required");
    }

    vk::ApplicationInfo app_info{ "ggml-vulkan", 1, शून्यसूचकः, 0, api_संस्करणम् };

    ध्रुव std::vector<vk::ExtensionProperties> instance_extensions = vk::enumerateInstanceExtensionProperties();
    ध्रुव तार्किक layer_settings = ggml_vk_instance_layer_settings_उपलब्धम्();
#यदि_परिभाषितम् __APPLE__
    ध्रुव तार्किक portability_enumeration_ext = ggml_vk_instance_portability_enumeration_ext_उपलब्धम्(instance_extensions);
#यदि_अन्तः
    ध्रुव तार्किक debug_utils_ext = ggml_vk_instance_debug_utils_ext_उपलब्धम्(instance_extensions) && getenv("GGML_VK_DEBUG_MARKERS") != शून्यसूचकः;
    std::vector<ध्रुव वर्ण*> layers;

    यदि (layer_settings) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }
    std::vector<ध्रुव वर्ण*> extensions;
    यदि (layer_settings) {
        extensions.push_back("VK_EXT_layer_settings");
    }
#यदि_परिभाषितम् __APPLE__
    यदि (portability_enumeration_ext) {
        extensions.push_back("VK_KHR_portability_enumeration");
    }
#यदि_अन्तः
    यदि (debug_utils_ext) {
        extensions.push_back("VK_EXT_debug_utils");
    }
    VkBool32 enable_best_practice = layer_settings;
    std::vector<vk::LayerSettingEXT> settings = {
        {
            "VK_LAYER_KHRONOS_validation",
            "validate_best_practices",
            vk::LayerSettingTypeEXT::eBool32,
            1,
            &enable_best_practice
        },
    };
    vk::LayerSettingsCreateInfoEXT layer_setting_info(settings);
    vk::InstanceCreateInfo instance_create_info(vk::InstanceCreateFlags{}, &app_info, layers, extensions, &layer_setting_info);
#यदि_परिभाषितम् __APPLE__
    यदि (portability_enumeration_ext) {
        instance_create_info.flags |= vk::Instance_Create_Flag_द्वयङ्काः::eEnumeratePortabilityKHR;
    }
#यदि_अन्तः

    vk_instance.instance = vk::createInstance(instance_create_info);
    vk_instance_initialized = सत्य;

    यदि (debug_utils_ext) {
        vk_instance.debug_utils_support              = सत्य;
        vk_instance.pfn_vk_समुच्चयः_Debug_Utils_Object_Name_EXT = (PFN_vk_समुच्चयः_Debug_Utils_Object_Name_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkSetDebugUtilsObjectNameEXT");
        vk_instance.pfn_vkQueueBeginDebugUtilsLabelEXT = (PFN_vkQueueBeginDebugUtilsLabelEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkQueueBeginDebugUtilsLabelEXT");
        vk_instance.pfn_vk_Queue_अन्तः_Debug_Utils_Label_EXT = (PFN_vk_Queue_अन्तः_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkQueueEndDebugUtilsLabelEXT");
        vk_instance.pfn_vkCmdBeginDebugUtilsLabelEXT = (PFN_vkCmdBeginDebugUtilsLabelEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdBeginDebugUtilsLabelEXT");
        vk_instance.pfn_vk_Cmd_अन्तः_Debug_Utils_Label_EXT =   (PFN_vk_Cmd_अन्तः_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdEndDebugUtilsLabelEXT");
        vk_instance.pfn_vk_Cmd_निवेशय_Debug_Utils_Label_EXT = (PFN_vk_Cmd_निवेशय_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdInsertDebugUtilsLabelEXT");
    }

    vk_perf_logger_enabled = getenv("GGML_VK_PERF_LOGGER") != शून्यसूचकः;
    vk_perf_logger_concurrent = getenv("GGML_VK_PERF_LOGGER_CONCURRENT") != शून्यसूचकः;
    vk_enable_sync_logger = getenv("GGML_VK_SYNC_LOGGER") != शून्यसूचकः;
    vk_स्मृतिः_logger_enabled = getenv("GGML_VK_MEMORY_LOGGER") != शून्यसूचकः;
    ध्रुव वर्ण* GGML_VK_PIPELINE_STATS = getenv("GGML_VK_PIPELINE_STATS");
    यदि (GGML_VK_PIPELINE_STATS != शून्यसूचकः) {
        vk_pipeline_stats_filter = GGML_VK_PIPELINE_STATS;
    }
    ध्रुव वर्ण* GGML_VK_PERF_LOGGER_FREQUENCY = getenv("GGML_VK_PERF_LOGGER_FREQUENCY");

    यदि (GGML_VK_PERF_LOGGER_FREQUENCY != शून्यसूचकः) {
        vk_perf_logger_frequency = std::stoul(GGML_VK_PERF_LOGGER_FREQUENCY);
    }

    // See https://github.com/KhronosGroup/Vulkan-Hpp?tab=readme-ov-file#extensions--per-device-function-pointers-
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_instance.instance);

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    // Emulate behavior of CUDA_VISIBLE_DEVICES for Vulkan
    वर्ण * devices_env = getenv("GGML_VK_VISIBLE_DEVICES");
    यदि (devices_env != शून्यसूचकः) {
        size_t अङ्कः_उपलब्धम्_devices = devices.size();

        std::string devices(devices_env);
        std::replace(devices.begin(), devices.end(), ',', ' ');

        std::stringstream ss(devices);
        size_t tmp;
        यावत् (ss >> tmp) {
            यदि(tmp >= अङ्कः_उपलब्धम्_devices) {
                std::cerr << "ggml_vulkan: Invalid device index " << tmp << " in GGML_VK_VISIBLE_DEVICES." << std::पङ्क्त्यन्तः;
                क्षिप std::runtime_error("Invalid Vulkan device index");
            }
            vk_instance.device_indices.push_back(tmp);
        }
    } अन्यथा {
        // If no vulkan devices are found, return early
        यदि (devices.रिक्तानि()) {
            GGML_LOG_INFO("ggml_vulkan: No devices found.\n");
            प्रत्यागम;
        }

        // Default to using all dedicated GPUs
        प्रति (size_t i = 0; i < devices.size(); i++) {
            vk::PhysicalDeviceProperties2 new_props;
            vk::PhysicalDeviceDriverProperties new_driver;
            vk::PhysicalDeviceIDProperties new_id;
            new_props.p_अग्र = &new_driver;
            new_driver.p_अग्र = &new_id;
            devices[i].getProperties2(&new_props);

            यदि ((new_props.properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu || new_props.properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) && ggml_vk_device_is_supported(devices[i])) {
                // Check if there are two physical devices corresponding to the same GPU
                // This handles the case where the same GPU appears with different drivers (e.g., RADV + AMDVLK on Linux),
                // see https://github.com/ggml-org/llama.cpp/pull/7582 for original deduplication.
                // MoltenVK on macOS may report the same UUID for distinct GPUs on multi-GPU cards,
                // see https://github.com/KhronosGroup/MoltenVK/issues/2683. Skip when both old/new
                // driver is MoltenVK
                स्वचालित old_device = std::find_if(
                    vk_instance.device_indices.begin(),
                    vk_instance.device_indices.end(),
                    [&devices, &new_id, &new_driver](ध्रुव size_t k){
                        vk::PhysicalDeviceProperties2 old_props;
                        vk::PhysicalDeviceDriverProperties old_driver;
                        vk::PhysicalDeviceIDProperties old_id;
                        old_props.p_अग्र = &old_driver;
                        old_driver.p_अग्र = &old_id;
                        devices[k].getProperties2(&old_props);

                        तार्किक same_uuid = std::equal(std::begin(old_id.deviceUUID), std::end(old_id.deviceUUID), std::begin(new_id.deviceUUID));
                        same_uuid = same_uuid || (
                            old_id.deviceLUIDValid && new_id.deviceLUIDValid &&
                            std::equal(std::begin(old_id.deviceLUID), std::end(old_id.deviceLUID), std::begin(new_id.deviceLUID))
                        );
                        तार्किक both_molten_vk = (new_driver.driverID == vk::DriverId::eMoltenvk && old_driver.driverID == vk::DriverId::eMoltenvk);

                        प्रत्यागम same_uuid && !both_molten_vk;
                    }
                );
                यदि (old_device == vk_instance.device_indices.end()) {
                    vk_instance.device_indices.push_back(i);
                } अन्यथा {
                    // There can be two physical devices corresponding to the same GPU if there are 2 different drivers
                    // This can cause error when splitting layers aross the devices, need to keep only 1
                    VK_LOG_DEBUG("Device " << i << " and device " << *old_device << " have the same deviceUUID");

                    vk::PhysicalDeviceProperties2 old_props;
                    vk::PhysicalDeviceDriverProperties old_driver;
                    old_props.p_अग्र = &old_driver;
                    devices[*old_device].getProperties2(&old_props);

                    std::map<vk::DriverId, पूर्णाङ्क> driver_priorities {};
                    पूर्णाङ्क old_प्राथमिकता = std::numeric_limits<पूर्णाङ्क>::max();
                    पूर्णाङ्क new_प्राथमिकता = std::numeric_limits<पूर्णाङ्क>::max();

                    // Check https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDriverId.html for the list of driver id
                    // Smaller number -> higher priority
                    विकल्प (old_props.properties.vendorID) {
                        पक्ष VK_VENDOR_ID_AMD:
                            driver_priorities[vk::DriverId::eMesaRadv] = 1;
                            driver_priorities[vk::DriverId::eAmdOpenSource] = 2;
                            driver_priorities[vk::DriverId::eAmdProprietary] = 3;
                            भङ्ग;
                        पक्ष VK_VENDOR_ID_INTEL:
                            driver_priorities[vk::DriverId::eIntelOpenSourceMESA] = 1;
                            driver_priorities[vk::DriverId::eIntelProprietaryWindows] = 2;
                            भङ्ग;
                        पक्ष VK_VENDOR_ID_NVIDIA:
                            driver_priorities[vk::DriverId::eNvidiaProprietary] = 1;
#यदि defined(VK_API_संस्करणम्_1_3) && VK_शीर्षकम्_संस्करणम् >= 235
                            driver_priorities[vk::DriverId::eMesaNvk] = 2;
#यदि_अन्तः
                            भङ्ग;
                        पक्ष VK_VENDOR_ID_QUALCOMM:
                            driver_priorities[vk::DriverId::eQualcommProprietary] = 1;
                            driver_priorities[vk::DriverId::eMesaTurnip] = 2;
                            भङ्ग;
                    }
                    driver_priorities[vk::DriverId::eMesaDozen] = 100;

                    यदि (driver_priorities.count(old_driver.driverID)) {
                        old_प्राथमिकता = driver_priorities[old_driver.driverID];
                    }
                    यदि (driver_priorities.count(new_driver.driverID)) {
                        new_प्राथमिकता = driver_priorities[new_driver.driverID];
                    }

                    यदि (new_प्राथमिकता < old_प्राथमिकता) {
                        स्वचालित r = std::remove(vk_instance.device_indices.begin(), vk_instance.device_indices.end(), *old_device);
                        vk_instance.device_indices.erase(r, vk_instance.device_indices.end());
                        vk_instance.device_indices.push_back(i);

                        VK_LOG_DEBUG("Prioritize device " << i << " driver " << new_driver.driverName << " over device " << *old_device << " driver " << old_driver.driverName);
                    }
                    अन्यथा {
                        VK_LOG_DEBUG("Prioritize device " << *old_device << " driver " << old_driver.driverName << " over device " << i << " driver " << new_driver.driverName << std::पङ्क्त्यन्तः);
                    }
                }
            }
        }

        // If no GPUs found, fall back to the first non-CPU device.
        // If only CPU devices are available, return without devices.
        यदि (vk_instance.device_indices.रिक्तानि()) {
            प्रति (size_t i = 0; i < devices.size(); i++) {
                यदि (devices[i].getProperties().deviceType != vk::PhysicalDeviceType::eCpu) {
                    vk_instance.device_indices.push_back(i);
                    भङ्ग;
                }
            }
        }

        यदि (vk_instance.device_indices.रिक्तानि()) {
            GGML_LOG_INFO("ggml_vulkan: No devices found.\n");
            प्रत्यागम;
        }
    }
    GGML_LOG_DEBUG("ggml_vulkan: Found %zu Vulkan devices:\n", vk_instance.device_indices.size());

    size_t host_बफर_import_गणनम् = 0;
    प्रति (size_t i = 0; i < vk_instance.device_indices.size(); i++) {
        vk::PhysicalDevice vkdev = devices[vk_instance.device_indices[i]];
        std::vector<vk::ExtensionProperties> extensionprops = vkdev.enumerateDeviceExtensionProperties();

        तार्किक membudget_supported = असत्य;
        तार्किक external_स्मृतिः_host_supported = असत्य;
        प्रति (ध्रुव स्वचालित & ext : extensionprops) {
            यदि (वाक्यं_तुलय(VK_EXT_स्मृतिः_BUDGET_EXTENSION_NAME, ext.extensionName) == 0) {
                membudget_supported = सत्य;
            } अन्यथा यदि (वाक्यं_तुलय("VK_EXT_external_memory_host", ext.extensionName) == 0) {
                external_स्मृतिः_host_supported = सत्य;
            }
        }

        vk_instance.device_supports_membudget.push_back(membudget_supported);

        तार्किक host_बफर_import = असत्य;
        यदि (external_स्मृतिः_host_supported) {
            vk::PhysicalDeviceProperties2 props2;
            vk::PhysicalDeviceDriverProperties driver_props;
            vk::Physical_Device_External_स्मृतिः_Host_Properties_EXT ext_props;
            props2.p_अग्र = &driver_props;
            driver_props.p_अग्र = &ext_props;
            vkdev.getProperties2(&props2);
            host_बफर_import = ggml_vk_external_स्मृतिः_host_usable(driver_props.driverID) &&
                                 ggml_vk_host_बफर_import_usable(props2.properties.vendorID);
            यदि (host_बफर_import) {
                vk_instance.host_import_alignment = std::max<size_t>(vk_instance.host_import_alignment, ext_props.minImportedHostPointerAlignment);
                host_बफर_import_गणनम्++;
            }
        }
        vk_instance.device_host_बफर_import.push_back(host_बफर_import);

        ggml_vk_print_gpu_info(i);
    }

    vk_instance.multi_device = host_बफर_import_गणनम् > 1 && vk_instance.device_host_बफर_import[0];
}

शून्य ggml_vk_init(ggml_backend_vk_context * ctx, size_t idx) {
    VK_LOG_DEBUG("ggml_vk_init(" << ctx->name << ", " << idx << ")");
    ggml_vk_instance_init();
    GGML_ASSERT(idx < vk_instance.device_indices.size());

    ctx->name = GGML_VK_NAME + std::to_string(idx);

    ctx->device = ggml_vk_get_device(idx);

    ctx->semaphore_idx = 0;
    ctx->event_idx = 0;

    ctx->prealloc_आकार_x = 0;
    ctx->prealloc_आकार_y = 0;
    ctx->prealloc_आकार_विभज_k = 0;
    // Fixed size of 1KB, for deterministic behavior
    ctx->prealloc_आकार_सङ्कलय_rms_partials = 1024;

    ctx->fence = ctx->device->device.createFence({});
    ctx->almost_ready_fence = ctx->device->device.createFence({});

    ctx->compute_cmd_pool.init(ctx->device, ctx->device->compute_queue.get());
    यदि (ctx->device->async_use_transfer_queue) {
        vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
        vk::SemaphoreCreateInfo ci{};
        ci.समुच्चयः_PNext(&tci);
        ctx->transfer_semaphore.s = ctx->device->device.createSemaphore(ci);
        ctx->transfer_semaphore.value = 0;

        ctx->transfer_cmd_pool.init(ctx->device, ctx->device->transfer_queue.get());
    }

    यदि (vk_perf_logger_enabled) {
        ctx->perf_logger = std::unique_ptr<vk_perf_logger>(नूतन vk_perf_logger());
    }

#यदि_परिभाषितम् GGML_VULKAN_CHECK_RESULTS
    ध्रुव वर्ण* skip_checks = getenv("GGML_VULKAN_SKIP_CHECKS");
    vk_skip_checks = (skip_checks == रिक्त ? 0 : atoi(skip_checks));
    ध्रुव वर्ण* output_tensor = getenv("GGML_VULKAN_OUTPUT_TENSOR");
    vk_output_tensor = (output_tensor == रिक्त ? 0 : atoi(output_tensor));
#यदि_अन्तः
}

vk_pipeline ggml_vk_get_गन्तव्यम्_fp16(ggml_backend_vk_context * ctx, ggml_type type) {
    VK_LOG_DEBUG("ggml_vk_get_to_fp16()");
    विकल्प (type) {
        पक्ष GGML_TYPE_F32:
        पक्ष GGML_TYPE_Q1_0:
        पक्ष GGML_TYPE_Q2_0:
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q4_1:
        पक्ष GGML_TYPE_Q5_0:
        पक्ष GGML_TYPE_Q5_1:
        पक्ष GGML_TYPE_Q8_0:
        पक्ष GGML_TYPE_Q2_K:
        पक्ष GGML_TYPE_Q3_K:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q5_K:
        पक्ष GGML_TYPE_Q6_K:
        पक्ष GGML_TYPE_IQ1_S:
        पक्ष GGML_TYPE_IQ1_M:
        पक्ष GGML_TYPE_IQ2_XXS:
        पक्ष GGML_TYPE_IQ2_XS:
        पक्ष GGML_TYPE_IQ2_S:
        पक्ष GGML_TYPE_IQ3_XXS:
        पक्ष GGML_TYPE_IQ3_S:
        पक्ष GGML_TYPE_IQ4_XS:
        पक्ष GGML_TYPE_IQ4_NL:
        पक्ष GGML_TYPE_MXFP4:
        पक्ष GGML_TYPE_NVFP4:
        पक्ष GGML_TYPE_TQ1_0:
        पक्ष GGML_TYPE_TQ2_0:
            भङ्ग;
        स्वभाव:
            प्रत्यागम शून्यसूचकः;
    }

    प्रत्यागम ctx->device->pipeline_dequant[type];
}

स्थिर vk_pipeline ggml_vk_get_dequantize_mul_mat_vec(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type, uint32_t अङ्कः_स्तम्भसङ्ख्या, uint32_t m, uint32_t k) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec()");
    GGML_ASSERT(b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_F16 || b_type == GGML_TYPE_Q8_1);
    GGML_ASSERT(अङ्कः_स्तम्भसङ्ख्या >= 1 && अङ्कः_स्तम्भसङ्ख्या <= mul_mat_vec_max_स्तम्भसङ्ख्या);

    यदि (b_type == GGML_TYPE_Q8_1) {
        विकल्प (a_type) {
            पक्ष GGML_TYPE_Q2_0:
            पक्ष GGML_TYPE_Q4_0:
            पक्ष GGML_TYPE_Q4_1:
            पक्ष GGML_TYPE_Q5_0:
            पक्ष GGML_TYPE_Q5_1:
            पक्ष GGML_TYPE_Q8_0:
            पक्ष GGML_TYPE_MXFP4:
            पक्ष GGML_TYPE_Q2_K:
            पक्ष GGML_TYPE_Q3_K:
            पक्ष GGML_TYPE_Q4_K:
            पक्ष GGML_TYPE_Q5_K:
            पक्ष GGML_TYPE_Q6_K:
            पक्ष GGML_TYPE_IQ1_S:
            पक्ष GGML_TYPE_IQ1_M:
            पक्ष GGML_TYPE_IQ4_XS:
                भङ्ग;
            स्वभाव:
                प्रत्यागम शून्यसूचकः;
        }
    }

    विकल्प (a_type) {
        पक्ष GGML_TYPE_F32:
        पक्ष GGML_TYPE_F16:
        पक्ष GGML_TYPE_BF16:
        पक्ष GGML_TYPE_Q1_0:
        पक्ष GGML_TYPE_Q2_0:
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q4_1:
        पक्ष GGML_TYPE_Q5_0:
        पक्ष GGML_TYPE_Q5_1:
        पक्ष GGML_TYPE_Q8_0:
        पक्ष GGML_TYPE_Q2_K:
        पक्ष GGML_TYPE_Q3_K:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q5_K:
        पक्ष GGML_TYPE_Q6_K:
        पक्ष GGML_TYPE_IQ1_S:
        पक्ष GGML_TYPE_IQ1_M:
        पक्ष GGML_TYPE_IQ2_XXS:
        पक्ष GGML_TYPE_IQ2_XS:
        पक्ष GGML_TYPE_IQ2_S:
        पक्ष GGML_TYPE_IQ3_XXS:
        पक्ष GGML_TYPE_IQ3_S:
        पक्ष GGML_TYPE_IQ4_XS:
        पक्ष GGML_TYPE_IQ4_NL:
        पक्ष GGML_TYPE_MXFP4:
        पक्ष GGML_TYPE_NVFP4:
        पक्ष GGML_TYPE_TQ1_0:
        पक्ष GGML_TYPE_TQ2_0:
            भङ्ग;
        स्वभाव:
            प्रत्यागम शून्यसूचकः;
    }

    // heuristic to choose workgroup size
    uint32_t dmmv_wg = DMMV_WG_आकार_SUBGROUP;
    यदि ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA && ctx->device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) || ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Prefer larger workgroups when M is small, to spread the work out more
        // and keep more SMs busy.
        // q6_k seems to prefer small workgroup size even for "medium" values of M.
        यदि (a_type == GGML_TYPE_Q6_K) {
            यदि (m < 4096 && k >= 1024) {
                dmmv_wg = DMMV_WG_आकार_LARGE;
            }
        } अन्यथा {
            यदि (m <= 8192 && k >= 1024) {
                dmmv_wg = DMMV_WG_आकार_LARGE;
            }
        }
    }

    यदि (b_type == GGML_TYPE_Q8_1) {
        यदि (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
            dmmv_wg = DMMV_WG_आकार_SUBGROUP;
        }
        प्रत्यागम ctx->device->pipeline_dequant_mul_mat_vec_q8_1_f32[dmmv_wg][a_type][अङ्कः_स्तम्भसङ्ख्या-1];
    }

    प्रत्यागम b_type == GGML_TYPE_F32 ? ctx->device->pipeline_dequant_mul_mat_vec_f32_f32[dmmv_wg][a_type][अङ्कः_स्तम्भसङ्ख्या-1] : ctx->device->pipeline_dequant_mul_mat_vec_f16_f32[dmmv_wg][a_type][अङ्कः_स्तम्भसङ्ख्या-1];
}

स्थिर vk_pipeline ggml_vk_get_dequantize_mul_mat_vec_id(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type, uint32_t m, uint32_t k) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec_id()");
    GGML_ASSERT(b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_Q8_1);

    यदि (b_type == GGML_TYPE_Q8_1) {
        विकल्प (a_type) {
            पक्ष GGML_TYPE_Q2_0:
            पक्ष GGML_TYPE_Q4_0:
            पक्ष GGML_TYPE_Q4_1:
            पक्ष GGML_TYPE_Q5_0:
            पक्ष GGML_TYPE_Q5_1:
            पक्ष GGML_TYPE_Q8_0:
            पक्ष GGML_TYPE_MXFP4:
            पक्ष GGML_TYPE_Q2_K:
            पक्ष GGML_TYPE_Q3_K:
            पक्ष GGML_TYPE_Q4_K:
            पक्ष GGML_TYPE_Q5_K:
            पक्ष GGML_TYPE_Q6_K:
            पक्ष GGML_TYPE_IQ1_S:
            पक्ष GGML_TYPE_IQ1_M:
            पक्ष GGML_TYPE_IQ4_XS:
                भङ्ग;
            स्वभाव:
                प्रत्यागम शून्यसूचकः;
        }
    }

    विकल्प (a_type) {
        पक्ष GGML_TYPE_F32:
        पक्ष GGML_TYPE_F16:
        पक्ष GGML_TYPE_BF16:
        पक्ष GGML_TYPE_Q1_0:
        पक्ष GGML_TYPE_Q2_0:
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q4_1:
        पक्ष GGML_TYPE_Q5_0:
        पक्ष GGML_TYPE_Q5_1:
        पक्ष GGML_TYPE_Q8_0:
        पक्ष GGML_TYPE_Q2_K:
        पक्ष GGML_TYPE_Q3_K:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q5_K:
        पक्ष GGML_TYPE_Q6_K:
        पक्ष GGML_TYPE_IQ1_S:
        पक्ष GGML_TYPE_IQ1_M:
        पक्ष GGML_TYPE_IQ2_XXS:
        पक्ष GGML_TYPE_IQ2_XS:
        पक्ष GGML_TYPE_IQ2_S:
        पक्ष GGML_TYPE_IQ3_XXS:
        पक्ष GGML_TYPE_IQ3_S:
        पक्ष GGML_TYPE_IQ4_XS:
        पक्ष GGML_TYPE_IQ4_NL:
        पक्ष GGML_TYPE_MXFP4:
        पक्ष GGML_TYPE_NVFP4:
        पक्ष GGML_TYPE_TQ1_0:
        पक्ष GGML_TYPE_TQ2_0:
            भङ्ग;
        स्वभाव:
            प्रत्यागम शून्यसूचकः;
    }

    // heuristic to choose workgroup size
    uint32_t dmmv_wg = DMMV_WG_आकार_SUBGROUP;
    यदि ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA && ctx->device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) || ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Prefer larger workgroups when M is small, to spread the work out more
        // and keep more SMs busy.
        // q6_k seems to prefer small workgroup size even for "medium" values of M.
        यदि (a_type == GGML_TYPE_Q6_K) {
            यदि (m < 4096 && k >= 1024) {
                dmmv_wg = DMMV_WG_आकार_LARGE;
            }
        } अन्यथा {
            यदि (m <= 8192 && k >= 1024) {
                dmmv_wg = DMMV_WG_आकार_LARGE;
            }
        }
    }

    यदि (b_type == GGML_TYPE_Q8_1) {
        यदि (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
            dmmv_wg = DMMV_WG_आकार_SUBGROUP;
        }
        प्रत्यागम ctx->device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[dmmv_wg][a_type];
    }

    प्रत्यागम ctx->device->pipeline_dequant_mul_mat_vec_id_f32[dmmv_wg][a_type];
}

vk_subbuffer ggml_vk_tensor_subbuffer(
    ध्रुव ggml_backend_vk_context * ctx, ध्रुव ggml_tensor * tensor, तार्किक allow_misalign) {

    vk_बफर बफर = शून्यसूचकः;
    size_t विस्थापनम् = 0;
    यदि (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, tensor->दत्त, बफर, विस्थापनम्);
    }
    यदि (!बफर) {
        स्वचालित buf_ctx = (ggml_backend_vk_बफर_context *)tensor->बफर->context;
        बफर = buf_ctx->dev_बफर;
        विस्थापनम् = vk_tensor_विस्थापनम्(tensor) + tensor->view_offs;
    }
    GGML_ASSERT(बफर != शून्यसूचकः);

    size_t size = ggml_nbytes(tensor);

    ध्रुव size_t descriptor_विस्थापनम् = ggml_vk_descriptor_विस्थापनम्(
        विस्थापनम्, ctx->device->properties.limits.min_Storage_बफर_विस्थापनम्_Alignment, ggml_type_आकार(tensor->type));
    ध्रुव size_t misalign_अष्टकानि = विस्थापनम् - descriptor_विस्थापनम्;
    // The shader must support misaligned offsets when indexing into the buffer
    GGML_ASSERT(allow_misalign || misalign_अष्टकानि == 0);
    विस्थापनम् = descriptor_विस्थापनम्;
    size += misalign_अष्टकानि;

    प्रत्यागम vk_subbuffer{बफर, विस्थापनम्, size};
}

स्थिर vk_command_बफर* ggml_vk_get_or_create_cmd_बफर(vk_device& device, vk_command_pool& pool) {
    प्रति (स्वचालित& cmd_बफर : pool.cmd_buffers) {
        यदि (!cmd_बफर.अन्तर्_use) {
            cmd_बफर.use_गणक++;
            cmd_बफर.अन्तर्_use = सत्य;
            प्रत्यागम &cmd_बफर;
        }
    }
    प्रत्यागम ggml_vk_create_cmd_बफर(device, pool);
}

स्थिर vk_submission ggml_vk_begin_submission(vk_device& device, vk_command_pool& p, तार्किक one_time = सत्य) {
    vk_submission s;
    s.बफर = ggml_vk_get_or_create_cmd_बफर(device, p);
    यदि (one_time) {
        s.बफर->buf.begin({ vk::Command_बफर_Usage_Flag_द्वयङ्काः::eOneTimeSubmit });
    } अन्यथा {
        s.बफर->buf.begin({ vk::Command_बफर_Usage_Flags{} });
    }

    प्रत्यागम s;
}

शून्य ggml_vk_cmd_label_begin(vk::Command_बफर buf, ध्रुव वर्ण * name) {
    vk::DebugUtilsLabelEXT label = {};
    label.pLabelName = name;
    label.color = std::array<भिन्न, 4>{1.0f, 1.0f, 1.0f, 1.0f};
    vk_instance.pfn_vkCmdBeginDebugUtilsLabelEXT(buf, reinterpret_cast<VkDebugUtilsLabelEXT *>(&label));
}

शून्य ggml_vk_ctx_अन्तः(vk_context& ctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_end(" << ctx << ", " << ctx->seqs.size() << ")");
    यदि (ctx->s == शून्यसूचकः) {
        प्रत्यागम;
    }

    // close open labels so this buffer is balanced; reopened in ggml_vk_ctx_begin
    यदि (vk_instance.debug_utils_support) {
        प्रति (size_t i = 0; i < ctx->debug_labels.size(); i++) {
            vk_instance.pfn_vk_Cmd_अन्तः_Debug_Utils_Label_EXT(ctx->s->बफर->buf);
        }
        // the enclosing per-command-buffer region
        vk_instance.pfn_vk_Cmd_अन्तः_Debug_Utils_Label_EXT(ctx->s->बफर->buf);
    }

    ctx->s->बफर->buf.end();
    ctx->s = शून्यसूचकः;
}

शून्य ggml_vk_ctx_begin(vk_device& device, vk_context& subctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_begin(" << device->name << ")");
    यदि (subctx->s != शून्यसूचकः) {
        ggml_vk_ctx_अन्तः(subctx);
    }

    subctx->seqs.push_back({ ggml_vk_begin_submission(device, *subctx->p) });
    subctx->s = subctx->seqs[subctx->seqs.size() - 1].दत्त();

    यदि (vk_instance.debug_utils_support) {
        // outermost region, one per command buffer, so the gaps between submits stand out
        ध्रुव std::string name = "submit " + std::to_string(device->debug_cmdbuf_idx++);
        ggml_vk_cmd_label_begin(subctx->s->बफर->buf, name.c_str());

        // reopen labels left open when the previous command buffer was submitted
        प्रति (ध्रुव std::string & label : subctx->debug_labels) {
            ggml_vk_cmd_label_begin(subctx->s->बफर->buf, label.c_str());
        }
    }
}

vk_context ggml_vk_get_compute_ctx(ggml_backend_vk_context * ctx) {
    vk_context परिणाम;
    यदि (!ctx->compute_ctx.expired()) {
        परिणाम = ctx->compute_ctx.तालकम्();
    } अन्यथा {
        परिणाम = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);

        ctx->compute_ctx = परिणाम;
        ggml_vk_ctx_begin(ctx->device, परिणाम);
    }

    यदि (ctx->device->async_use_transfer_queue && ctx->transfer_semaphore_चरमः_submitted < ctx->transfer_semaphore.value) {
        परिणाम->s->wait_semaphores.push_back(ctx->transfer_semaphore);
        ctx->transfer_semaphore_चरमः_submitted = ctx->transfer_semaphore.value;
    }

    प्रत्यागम परिणाम;
}

vk_context ggml_vk_get_transfer_ctx(ggml_backend_vk_context * ctx) {
    vk_context परिणाम;
    यदि (!ctx->transfer_ctx.expired()) {
        परिणाम = ctx->transfer_ctx.तालकम्();
    } अन्यथा {
        परिणाम = ggml_vk_create_context(ctx, ctx->transfer_cmd_pool);

        ctx->transfer_ctx = परिणाम;
        ggml_vk_ctx_begin(ctx->device, परिणाम);
    }

    प्रत्यागम परिणाम;
}

तार्किक ggml_vk_submit_transfer_ctx(ggml_backend_vk_context * ctx) {
    यदि (!ctx->device->async_use_transfer_queue || ctx->transfer_ctx.expired()) {
        प्रत्यागम असत्य;
    }

    vk_context cpy_ctx = ctx->transfer_ctx.तालकम्();
    ggml_vk_ctx_अन्तः(cpy_ctx);

    प्रति (स्वचालित& cpy : cpy_ctx->अन्तर्_memcpys) {
        स्मृतिं_प्रतिलिख(cpy.dst, cpy.src, cpy.n);
    }

    ctx->transfer_semaphore.value++;
    cpy_ctx->seqs.back().back().signal_semaphores.push_back(ctx->transfer_semaphore);

    ggml_vk_submit(cpy_ctx, {});
    ctx->transfer_ctx.reset();
    प्रत्यागम सत्य;
}

size_t ggml_vk_align_आकार(size_t विस्तारः, size_t align) {
    VK_LOG_DEBUG("ggml_vk_align_size(" << विस्तारः << ", " << align << ")");
    प्रत्यागम CEIL_DIV(विस्तारः, align) * align;
}

शून्य deferred_memcpy(शून्य * dst, ध्रुव शून्य * src, size_t size, std::vector<vk_staging_memcpy>* memcpys) {
    यदि (memcpys == शून्यसूचकः) {
        स्मृतिं_प्रतिलिख(dst, src, size);
    } अन्यथा {
        memcpys->emplace_back(dst, src, size);
    }
}

शून्य deferred_memset(शून्य * dst, uint32_t val, size_t size, std::vector<vk_staging_memset>* memsets) {
    यदि (memsets == शून्यसूचकः) {
        स्मृतिं_पूरय(dst, val, size);
    } अन्यथा {
        memsets->emplace_back(dst, val, size);
    }
}

स्थिर uint32_t ggml_vk_guess_विभज_k(ggml_backend_vk_context * ctx, uint32_t m, uint32_t n, uint32_t k, तार्किक disable_विभज_k, ध्रुव vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_vk_guess_split_k(" << m << ", " << n << ", " << k << ", " << disable_विभज_k << ")");

    यदि (disable_विभज_k) {
        प्रत्यागम 1;
    }

    uint32_t विभज_k = 1;
    यदि (ctx->device->shader_core_गणनम् != 0 && n >= pipeline->wg_denoms[1]) {
        // If k is 'large' and the SMs will fill less than halfway, use split_k.
        uint32_t m_tiles = CEIL_DIV(m, pipeline->wg_denoms[0]);
        uint32_t n_tiles = CEIL_DIV(n, pipeline->wg_denoms[1]);

        यदि (k >= 2048) {
            यदि (m_tiles * n_tiles <= ctx->device->shader_core_गणनम् / 2) {
                विभज_k = ctx->device->shader_core_गणनम् / (m_tiles * n_tiles);
            } अन्यथा यदि (m_tiles * n_tiles <= ctx->device->shader_core_गणनम् * 2 / 3) {
                विभज_k = 3;
            }
            // Cap the split at 8x. Unless k is huge this is a lot of overhead.
            विभज_k = std::min(विभज_k, 8u);

            // ggml_vk_matmul will align the splits to be a multiple of 256.
            // If this rounded up size would cause the last split to be empty,
            // then reduce the split count.
            यावत् (सत्य) {
                यदि (विभज_k == 1) {
                    भङ्ग;
                }
                uint32_t k_विभज = CEIL_DIV(k, विभज_k);
                k_विभज = ROUNDUP_POW2(k_विभज, 256);
                यदि (k_विभज * (विभज_k - 1) < k) {
                    भङ्ग;
                }
                विभज_k--;
            }
        }
    }

    प्रत्यागम विभज_k;
}

शून्य ggml_vk_matmul(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& विभज_k_बफर,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t विभज_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3,
        uint32_t padded_n, uint64_t b_col_अष्टकानि, uint64_t d_col_अष्टकानि) {
        VK_LOG_DEBUG("ggml_vk_matmul(a: (" << a.बफर->बफर << ", " << a.विस्थापनम् << ", " << a.size << "), b: (" << b.बफर->बफर << ", " << b.विस्थापनम् << ", " << b.size << "), d: (" << d.बफर->बफर << ", " << d.विस्थापनम् << ", " << d.size << "), split_k: (" << (विभज_k_बफर.बफर != शून्यसूचकः ? विभज_k_बफर.बफर->बफर : VK_NULL_HANDLE) << ", " << विभज_k_बफर.विस्थापनम् << ", " << विभज_k_बफर.size << "), m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", split_k: " << विभज_k << ", batch: " << batch << ", ne02: " << ne02 << ", ne12: " << ne12 << ", broadcast2: " << broadcast2 << ", broadcast3: " << broadcast3 << ", padded_n: " << padded_n << ")");
    यदि (विभज_k == 1) {
        // a small tile can still produce more workgroups than the device allows on the n axis:
        // split the dispatch into n-chunks, shifting the B and D column windows
        ध्रुव uint64_t max_n = (uint64_t)ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[1] * pipeline->wg_denoms[1];
        ध्रुव uint64_t n_chunks = CEIL_DIV(n, max_n);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, n_chunks * CEIL_DIV(batch, ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[2]));

        प्रति (uint64_t n_आधारः = 0; n_आधारः < n; n_आधारः += max_n) {
            ध्रुव uint32_t n_chunk = (uint32_t) std::min(n - n_आधारः, max_n);
            GGML_ASSERT(n_आधारः * b_col_अष्टकानि < b.size && n_आधारः * d_col_अष्टकानि < d.size);
            GGML_ASSERT(b.size != VK_सम्पूर्णम्_आकार && d.size != VK_सम्पूर्णम्_आकार);
            vk_subbuffer b_chunk = b;
            vk_subbuffer d_chunk = d;
            b_chunk.विस्थापनम् += n_आधारः * b_col_अष्टकानि;
            b_chunk.size   -= n_आधारः * b_col_अष्टकानि;
            d_chunk.विस्थापनम् += n_आधारः * d_col_अष्टकानि;
            d_chunk.size   -= n_आधारः * d_col_अष्टकानि;

            uint32_t आधारः_work_गणः_z = 0;
            यावत् (आधारः_work_गणः_z < batch) {
                uint32_t groups_z = std::min(batch - आधारः_work_गणः_z, ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[2]);

                ध्रुव vk_mat_mat_प्रक्षिप_constants pc = { m, n_chunk, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, आधारः_work_गणः_z, batch, k, ne02, ne12, broadcast2, broadcast3, uint32_t(padded_n - n_आधारः) };
                ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b_chunk, d_chunk }, pc, { m, n_chunk, groups_z });
                आधारः_work_गणः_z += groups_z;
            }
        }
        प्रत्यागम;
    }

    यदि (ctx->prealloc_विभज_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    GGML_ASSERT(batch_stride_d == m * n);

    // Round the split size up to a multiple of 256 (k-quant alignment)
    uint32_t k_विभज = CEIL_DIV(k, विभज_k);
    k_विभज = ROUNDUP_POW2(k_विभज, 256);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, CEIL_DIV(batch, ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[2]));

    uint32_t आधारः_work_गणः_z = 0;
    यावत् (आधारः_work_गणः_z < batch) {
        uint32_t groups_z = std::min(batch - आधारः_work_गणः_z, ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[2]);

        ध्रुव vk_mat_mat_प्रक्षिप_constants pc1 = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, आधारः_work_गणः_z, batch, k_विभज, ne02, ne12, broadcast2, broadcast3, padded_n };
        // Make sure enough workgroups get assigned for split k to work
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, विभज_k_बफर }, pc1, { (CEIL_DIV(m, pipeline->wg_denoms[0]) * pipeline->wg_denoms[0]) * विभज_k, n, groups_z });
        आधारः_work_गणः_z += groups_z;
    }
    ggml_vk_sync_buffers(ctx, subctx);
    ध्रुव std::array<uint32_t, 2> pc2 = { (uint32_t)(m * n * batch), विभज_k };
    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_matmul_विभज_k_reduce, { विभज_k_बफर, d }, pc2, { m * n * batch, 1, 1 });
    ctx->prealloc_विभज_k_need_sync = सत्य;
}

स्थिर तार्किक ggml_vk_get_mul_mat_mat_f16acc(ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec) {
    यदि (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16) प्रत्यागम असत्य;
    यदि (src1_type == GGML_TYPE_Q8_1) प्रत्यागम असत्य;
    यदि (src0_type == GGML_TYPE_F16) {
        प्रत्यागम prec == GGML_PREC_DEFAULT && ctx->device->fp16 && !(ctx->device->coopmat_support && !ctx->device->coopmat_acc_f16_support);
    }
    // quant types
    यदि (ctx->device->coopmat2) {
        प्रत्यागम prec == GGML_PREC_DEFAULT;
    }
    यदि (ctx->device->coopmat_support) {
        प्रत्यागम ctx->device->fp16 && ctx->device->coopmat_acc_f16_support && prec == GGML_PREC_DEFAULT;
    }
    प्रत्यागम ctx->device->fp16 && prec == GGML_PREC_DEFAULT;
}

स्थिर ध्रुव std::vector<vk_matmul_pipeline_pair>* ggml_vk_get_mul_mat_mat_pipeline_map(
        ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec, तार्किक mul_mat_id = असत्य) {
    तार्किक f16acc = ggml_vk_get_mul_mat_mat_f16acc(ctx, src0_type, src1_type, prec);
    vk_matmul_pipeline_कुञ्चिका कुञ्चिका{src0_type, src1_type, mul_mat_id, f16acc};
    स्वचालित it = ctx->device->pipeline_matmul.find(कुञ्चिका);
    यदि (it == ctx->device->pipeline_matmul.end() || it->second.रिक्तानि()) {
        // Try without f16acc
        यदि (f16acc) {
            कुञ्चिका.f16acc = असत्य;
            it = ctx->device->pipeline_matmul.find(कुञ्चिका);
            यदि (it != ctx->device->pipeline_matmul.end() && !it->second.रिक्तानि()) प्रत्यागम &it->second;
        }
        प्रत्यागम शून्यसूचकः;
    }
    प्रत्यागम &it->second;
}

स्थिर vk_pipeline ggml_vk_guess_matmul_pipeline_map(ggml_backend_vk_context * ctx,
        ध्रुव std::vector<vk_matmul_pipeline_pair>& configs,
        uint32_t m, uint32_t n, तार्किक aligned, तार्किक mul_mat_id) {
    स्वचालित& selector = mul_mat_id ? ctx->device->matmul_id_tile_selector : ctx->device->matmul_tile_selector;
    uint32_t idx = selector(m, n, 0, ctx->device->shader_core_गणनम्, configs);
    यदि (idx >= configs.size()) idx = (uint32_t)configs.size() - 1;
    प्रत्यागम (aligned && configs[idx].aligned) ? configs[idx].aligned : configs[idx].unaligned;
}

स्थिर uint32_t ggml_vk_guess_matmul_pipeline_align_map(ggml_backend_vk_context * ctx,
        ध्रुव std::vector<vk_matmul_pipeline_pair>& configs,
        uint32_t m, uint32_t n, तार्किक mul_mat_id) {
    स्वचालित& selector = mul_mat_id ? ctx->device->matmul_id_tile_selector : ctx->device->matmul_tile_selector;
    uint32_t idx = selector(m, n, 0, ctx->device->shader_core_गणनम्, configs);
    यदि (idx >= configs.size()) idx = (uint32_t)configs.size() - 1;
    प्रत्यागम configs[idx].align;
}

स्थिर uint64_t ggml_vk_mul_mat_id_max_tiles(uint64_t n_experts, uint64_t n_पङ्क्तिकाः, uint32_t bn) {
    // Each nonempty expert can add at most one partial tile to the combined grid.
    प्रत्यागम std::min(n_पङ्क्तिकाः, CEIL_DIV(n_पङ्क्तिकाः, bn) + std::min(n_experts, n_पङ्क्तिकाः) - 1);
}

स्थिर शून्य ggml_vk_matmul_id(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& ids, ध्रुव vk_subbuffer & expert_गणनम्_buf,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t n_as, uint32_t nei0, uint32_t nei1, uint32_t nbi1, uint32_t ne11,
        uint32_t row_ids_विस्थापनम्, uint32_t max_tiles) {
    VK_LOG_DEBUG("ggml_vk_matmul_id(a: (" << a.बफर->बफर << ", " << a.विस्थापनम् << ", " << a.size << "), b: (" << b.बफर->बफर << ", " << b.विस्थापनम् << ", " << b.size << "), d: (" << d.बफर->बफर << ", " << d.विस्थापनम् << ", " << d.size << "), ids: (" << ids.बफर->बफर << ", " << ids.विस्थापनम् << ", " << ids.size << "), expert_count: (" << expert_गणनम्_buf.बफर->बफर << ", " << expert_गणनम्_buf.विस्थापनम् << ", " << expert_गणनम्_buf.size << "), " <<
        "m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", " <<
        "batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", " <<
        "n_as: " << n_as << ", nei0: " << nei0 << ", nei1: " << nei1 << ", nbi1: " << nbi1 << ", ne11: " << ne11 << ")");
    ध्रुव vk_mat_mat_id_प्रक्षिप_constants pc = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d,
                                              nei0, nei1, nbi1, ne11, n_as, row_ids_विस्थापनम् };
    ध्रुव uint32_t tiles_y = std::min(max_tiles, ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[1]);
    ध्रुव std::array<uint32_t, 3> elements = { m, tiles_y * pipeline->wg_denoms[1], uint32_t(CEIL_DIV(max_tiles, tiles_y)) };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, d, ids, expert_गणनम्_buf }, pc, elements);
}

तार्किक ggml_vk_dim01_contiguous(ध्रुव ggml_tensor * tensor) {
    प्रत्यागम
        tensor->nb[0] == ggml_type_आकार(tensor->type) &&
        tensor->nb[1] == (tensor->nb[0]*tensor->ne[0])/ggml_blck_आकार(tensor->type) &&
        (tensor->ne[3] == 1 || tensor->nb[3] == tensor->nb[2]*tensor->ne[2]);
}

// Batch stride in elements of a tensor read in place.
स्थिर uint32_t ggml_vk_batch_stride(ध्रुव ggml_tensor * tensor) {
    प्रत्यागम (uint32_t)(tensor->nb[2] / ggml_type_आकार(tensor->type) * ggml_blck_आकार(tensor->type));
}

vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, ध्रुव ggml_tensor * src, ध्रुव ggml_tensor * dst, ggml_type गन्तव्यम्) {

    // Choose "contiguous copy" shader if src/dst are contiguous
    तार्किक contig = ggml_is_contiguous(src) && (!dst || ggml_is_contiguous(dst));

    // Use optimized "transpose" shader if src dim1 is the innermost dimension.
    तार्किक transpose = dst && src->nb[1] == ggml_type_आकार(गन्तव्यम्) && ggml_are_same_shape(dst, src);

    यदि (transpose && src->type == गन्तव्यम्) {
        यदि (ggml_type_आकार(गन्तव्यम्) == 4) {
            प्रत्यागम ctx->device->pipeline_cpy_transpose_32;
        } अन्यथा यदि (ggml_type_आकार(गन्तव्यम्) == 2) {
            प्रत्यागम ctx->device->pipeline_cpy_transpose_16;
        }
    }

    // Same, for a 0<->2 swap: src dim2 is the innermost dimension.
    तार्किक transpose02 = dst && !contig && src->nb[2] == ggml_type_आकार(गन्तव्यम्) &&
                       ggml_is_contiguous(dst) && ggml_are_same_shape(dst, src);

    यदि (transpose02 && src->type == गन्तव्यम्) {
        यदि (ggml_type_आकार(गन्तव्यम्) == 4) {
            प्रत्यागम ctx->device->pipeline_cpy_transpose_02_32;
        } अन्यथा यदि (ggml_type_आकार(गन्तव्यम्) == 2) {
            प्रत्यागम ctx->device->pipeline_cpy_transpose_02_16;
        }
    }

    यदि (src->type == GGML_TYPE_F32 && गन्तव्यम् == GGML_TYPE_F16) {
        यदि (contig) {
            प्रत्यागम ctx->device->pipeline_contig_cpy_f32_f16;
        } अन्यथा {
            प्रत्यागम ctx->device->pipeline_cpy_f32_f16;
        }
    }
    यदि (src->type == GGML_TYPE_F16 && गन्तव्यम् == GGML_TYPE_F32) {
        यदि (contig) {
            प्रत्यागम ctx->device->pipeline_contig_cpy_f16_f32;
        } अन्यथा {
            प्रत्यागम ctx->device->pipeline_cpy_f16_f32;
        }
    }
    यदि (src->type == GGML_TYPE_F32 && गन्तव्यम् == GGML_TYPE_BF16) {
        यदि (contig) {
            प्रत्यागम ctx->device->pipeline_contig_cpy_f32_bf16;
        } अन्यथा {
            प्रत्यागम ctx->device->pipeline_cpy_f32_bf16;
        }
    }
    यदि (src->type == GGML_TYPE_BF16 && गन्तव्यम् == GGML_TYPE_F32) {
        यदि (contig) {
            प्रत्यागम ctx->device->pipeline_contig_cpy_bf16_f32;
        } अन्यथा {
            प्रत्यागम ctx->device->pipeline_cpy_bf16_f32;
        }
    }
    यदि (src->type == GGML_TYPE_BF16 && गन्तव्यम् == GGML_TYPE_F16) {
        यदि (contig) {
            प्रत्यागम ctx->device->pipeline_contig_cpy_bf16_f16;
        } अन्यथा {
            प्रत्यागम ctx->device->pipeline_cpy_bf16_f16;
        }
    }
    यदि (src->type == GGML_TYPE_F32 && गन्तव्यम् == GGML_TYPE_I32) {
        यदि (contig) {
            प्रत्यागम ctx->device->pipeline_contig_cpy_f32_i32;
        } अन्यथा {
            प्रत्यागम ctx->device->pipeline_cpy_f32_i32;
        }
    }
    यदि (src->type == GGML_TYPE_I32 && गन्तव्यम् == GGML_TYPE_F32) {
        यदि (contig) {
            प्रत्यागम ctx->device->pipeline_contig_cpy_i32_f32;
        } अन्यथा {
            प्रत्यागम ctx->device->pipeline_cpy_i32_f32;
        }
    }
    यदि (src->type == GGML_TYPE_F32) {
        विकल्प (गन्तव्यम्) {
        पक्ष GGML_TYPE_Q1_0:
        पक्ष GGML_TYPE_Q2_0:
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q4_1:
        पक्ष GGML_TYPE_Q5_0:
        पक्ष GGML_TYPE_Q5_1:
        पक्ष GGML_TYPE_Q8_0:
        पक्ष GGML_TYPE_IQ4_NL:
            प्रत्यागम ctx->device->pipeline_cpy_f32_quant[गन्तव्यम्];
        स्वभाव:
            भङ्ग;
        }
    }

    यदि (गन्तव्यम् == GGML_TYPE_F32) {
        विकल्प (src->type) {
        पक्ष GGML_TYPE_Q1_0:
        पक्ष GGML_TYPE_Q2_0:
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q4_1:
        पक्ष GGML_TYPE_Q5_0:
        पक्ष GGML_TYPE_Q5_1:
        पक्ष GGML_TYPE_Q8_0:
        पक्ष GGML_TYPE_IQ4_NL:
            प्रत्यागम ctx->device->pipeline_cpy_quant_f32[src->type];
        स्वभाव:
            भङ्ग;
        }
    }

    यदि (src->type == गन्तव्यम्) {
        // Copy two or four bytes at a time, depending on block size.
        // For quantized types, we scale by block size/type size. But
        // this path is also used for bf16->bf16 for example, where the
        // type size must be exactly 2 or 4.
        GGML_ASSERT(ggml_is_quantized(गन्तव्यम्) || ggml_type_आकार(src->type) == 2 || ggml_type_आकार(src->type) == 4);
        यदि ((ggml_type_आकार(src->type) % 4) == 0) {
            यदि (contig) {
                प्रत्यागम ctx->device->pipeline_contig_cpy_32;
            } अन्यथा {
                प्रत्यागम ctx->device->pipeline_cpy_32;
            }
        } अन्यथा {
            यदि (contig) {
                प्रत्यागम ctx->device->pipeline_contig_cpy_16;
            } अन्यथा {
                प्रत्यागम ctx->device->pipeline_cpy_16;
            }
        }
    }

    std::cerr << "Missing CPY op for types: " << ggml_type_name(src->type) << " " << ggml_type_name(गन्तव्यम्) << std::पङ्क्त्यन्तः;
    GGML_ABORT("fatal error");
}

शून्य ggml_vk_cpy_गन्तव्यम्_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, ध्रुव ggml_tensor * tensor, ध्रुव vk_subbuffer & अन्तर्, ध्रुव vk_subbuffer & बहिर्गमनम्) {
    VK_LOG_DEBUG("ggml_vk_cpy_to_contiguous((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::cerr << "buffer in size=" << अन्तर्.बफर->size << ", buffer out size=" << बहिर्गमनम्.बफर->size << ")");

    ध्रुव uint32_t ne = ggml_nelements(tensor);
    std::array<uint32_t, 3> elements;

    यदि (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } अन्यथा यदि (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } अन्यथा {
        elements = { ne, 1, 1 };
    }

    vk_op_unary_प्रक्षिप_constants pc = vk_op_unary_प्रक्षिप_constants_init(tensor, tensor, ne);
    pc.nb10 = 1;
    pc.nb11 = (uint32_t)tensor->ne[0];
    pc.nb12 = (uint32_t)(tensor->ne[0] * tensor->ne[1]);
    pc.nb13 = (uint32_t)(tensor->ne[0] * tensor->ne[1] * tensor->ne[2]);
    init_pushconst_fastdiv(pc);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { अन्तर्, बहिर्गमनम् }, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
}

स्थिर शून्य ggml_vk_cpy_गन्तव्यम्_strided(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, ध्रुव ggml_tensor * tensor,
        ध्रुव vk_subbuffer & अन्तर्, ध्रुव vk_subbuffer & बहिर्गमनम्,
        uint32_t nb10, uint32_t nb11, uint32_t nb12, uint32_t nb13) {
    VK_LOG_DEBUG("ggml_vk_cpy_to_strided((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::cerr << "dst_nb=(" << nb10 << ", " << nb11 << ", " << nb12 << ", " << nb13 << "), buffer in size=" << अन्तर्.बफर->size << ", buffer out size=" << बहिर्गमनम्.बफर->size << ")");

    ध्रुव uint32_t ne = ggml_nelements(tensor);
    std::array<uint32_t, 3> elements;

    यदि (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } अन्यथा यदि (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } अन्यथा {
        elements = { ne, 1, 1 };
    }

    vk_op_unary_प्रक्षिप_constants pc = vk_op_unary_प्रक्षिप_constants_init(tensor, tensor, ne);
    pc.nb10 = nb10;
    pc.nb11 = nb11;
    pc.nb12 = nb12;
    pc.nb13 = nb13;
    init_pushconst_fastdiv(pc);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { अन्तर्, बहिर्गमनम् }, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
}

vk_pipeline ggml_vk_get_quantize_pipeline(ggml_backend_vk_context * ctx, ggml_type type) {
    विकल्प(type) {
        पक्ष GGML_TYPE_Q8_1:
            प्रत्यागम ctx->device->pipeline_quantize_q8_1_x4;
        स्वभाव:
            std::cerr << "Missing quantize pipeline for type: " << ggml_type_name(type) << std::पङ्क्त्यन्तः;
            GGML_ABORT("fatal error");
    }
}

शून्य ggml_vk_quantize_q8_1(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव vk_subbuffer & अन्तर्, ध्रुव vk_subbuffer & बहिर्गमनम्, uint32_t ne) {
    VK_LOG_DEBUG("ggml_vk_quantize_q8_1(" << "buffer in size=" << अन्तर्.बफर->size << ", buffer out size=" << बहिर्गमनम्.बफर->size << ", " << ne << ")");

    vk_pipeline pipeline = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);

    ध्रुव uint32_t अङ्कः_खण्डानि = CEIL_DIV(ne, pipeline->wg_denoms[0]);
    // clamp the number of elements to the max workgroup count. The shader will iterate over the total number of blocks.
    ध्रुव uint64_t max_elements = std::min<uint64_t>(uint64_t{ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[0]} * pipeline->wg_denoms[0], std::numeric_limits<uint32_t>::max());
    ध्रुव uint32_t elements = std::min(ne, static_cast<uint32_t>(max_elements));

    ध्रुव vk_quantize_q8_1_प्रक्षिप_constants pc = {
        ne,
        अङ्कः_खण्डानि,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { अन्तर्, बहिर्गमनम् }, pc, { elements, 1, 1 });
    ggml_vk_sync_buffers(ctx, subctx);
}

स्थिर vk_pipeline ggml_vk_get_64b_indexing_pipeline(ggml_backend_vk_context * ctx, vk_pipeline &pipeline) {
    GGML_अप्रयुक्तम्(ctx);
#यदि defined(VK_EXT_shader_64bit_indexing)
    vk_pipeline *ptr = &pipeline;
    यावत् (*ptr) {
        यदि ((*ptr)->is_64b_indexing) {
            प्रत्यागम *ptr;
        }
        ptr = &(*ptr)->अग्र;
    }
#यदि_अन्तः
    प्रत्यागम pipeline;
}

// perf-heuristic for choosing mul_mat b type, based on shader availability
स्थिर ggml_type ggml_vk_mul_mat_b_type(
        ggml_backend_vk_context * ctx, ggml_type src0_type, ध्रुव ggml_tensor * src1,
        ggml_type f16_type, ggml_prec prec, तार्किक mul_mat_id, तार्किक can_quantize_y, तार्किक prefer_f16_b) {
    स्वचालित उपलब्धम् = [&](ggml_type b) {
        प्रत्यागम ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0_type, b, prec, mul_mat_id) != शून्यसूचकः;
    };
    यदि (can_quantize_y && उपलब्धम्(GGML_TYPE_Q8_1))             प्रत्यागम GGML_TYPE_Q8_1;
    यदि (prefer_f16_b   && उपलब्धम्(f16_type))                   प्रत्यागम f16_type;
    यदि (ggml_vk_dim01_contiguous(src1) && उपलब्धम्(src1->type)) प्रत्यागम src1->type;
    यदि (src1->type == GGML_TYPE_BF16 && उपलब्धम्(GGML_TYPE_F32)) प्रत्यागम GGML_TYPE_F32;
    प्रत्यागम f16_type;
}

स्थिर शून्य ggml_vk_mul_mat_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst, तार्किक disable_विभज_k) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_q_f16((" << src0 << ", name=" << src0->name << ", type=" << ggml_type_name(src0->type) << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << ggml_type_name(src1->type) << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << ggml_type_name(dst->type) << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    ध्रुव uint64_t ne00 = src0->ne[0];
    ध्रुव uint64_t ne01 = src0->ne[1];
    ध्रुव uint64_t ne02 = src0->ne[2];
    ध्रुव uint64_t ne03 = src0->ne[3];

    ध्रुव uint64_t ne10 = src1->ne[0];
    ध्रुव uint64_t ne11 = src1->ne[1];
    ध्रुव uint64_t ne12 = src1->ne[2];
    ध्रुव uint64_t ne13 = src1->ne[3];

    ध्रुव uint64_t ne21 = dst->ne[1];
    ध्रुव uint32_t stride_d = dst->nb[1] / ggml_type_आकार(dst->type);
    ध्रुव uint32_t stride_batch_d = stride_d*ne21;

    ध्रुव uint64_t r2 = ne12 / ne02;
    ध्रुव uint64_t r3 = ne13 / ne03;

    ggml_backend_vk_बफर_context * dst_buf_ctx = (ggml_backend_vk_बफर_context *)dst->बफर->context;
    ggml_backend_vk_बफर_context * src0_buf_ctx = (ggml_backend_vk_बफर_context *)src0->बफर->context;
    ggml_backend_vk_बफर_context * src1_buf_ctx = (ggml_backend_vk_बफर_context *)src1->बफर->context;

    vk_बफर d_Qx = शून्यसूचकः;
    size_t qx_buf_विस्थापनम् = 0;
    vk_बफर d_Qy = शून्यसूचकः;
    size_t qy_buf_विस्थापनम् = 0;

    तार्किक src0_uma = असत्य;
    तार्किक src1_uma = असत्य;

    यदि (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->दत्त, d_Qx, qx_buf_विस्थापनम्);
        ggml_vk_host_get(ctx->device, src1->दत्त, d_Qy, qy_buf_विस्थापनम्);
        src0_uma = d_Qx != शून्यसूचकः;
        src1_uma = d_Qy != शून्यसूचकः;
    }

    ध्रुव तार्किक x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    // If src0 is BF16, try to use a BF16 x BF16 multiply
    ggml_type f16_type = src0->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;

    ध्रुव तार्किक is_coopmat = ctx->device->coopmat_support || ctx->device->coopmat2;
    ध्रुव तार्किक can_quantize_y = (ctx->device->integer_dot_गुणनफलम् || ctx->device->coopmat_int_support) &&
                                src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && (ne11 * ne10) % 4 == 0;
    // coopmat shaders are fast enough that memory bandwidth reduction from f16 activations helps
    ध्रुव तार्किक prefer_f16_b = is_coopmat && ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32;

    ध्रुव ggml_type y_kernel_type = ggml_vk_mul_mat_b_type(ctx, src0->type, src1, f16_type,
                                        (ggml_prec)dst->op_params[0], असत्य, can_quantize_y, prefer_f16_b);

    ध्रुव तार्किक quantize_y       = y_kernel_type == GGML_TYPE_Q8_1;
    ध्रुव तार्किक y_f32_kernel     = y_kernel_type == GGML_TYPE_F32;
    ध्रुव तार्किक y_needs_reformat = !quantize_y && (!ggml_vk_dim01_contiguous(src1) || y_kernel_type != src1->type);

    ध्रुव std::vector<vk_matmul_pipeline_pair>* mmp_map =
        ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_kernel_type, (ggml_prec)dst->op_params[0]);

    ध्रुव तार्किक qx_needs_dequant = mmp_map == शून्यसूचकः || x_non_contig;
    ध्रुव तार्किक qy_needs_dequant = y_needs_reformat;

    यदि (qx_needs_dequant) {
        // dequant src0 to f16
        mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, f16_type, y_kernel_type, (ggml_prec)dst->op_params[0]);
    }

    GGML_ASSERT(mmp_map != शून्यसूचकः);

    ध्रुव uint32_t kpad = quantize_y ? 0 : ggml_vk_align_आकार(ne10, ggml_vk_guess_matmul_pipeline_align_map(ctx, *mmp_map, ne01, ne11, असत्य));
    ध्रुव तार्किक aligned = !quantize_y && ne10 == kpad && ne01 > 8 && ne11 > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline_map(ctx, *mmp_map, ne01, ne11, aligned, असत्य);

    यदि (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_बफर_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    // Reserve extra storage in the N dimension for the Y matrix, so we can avoid bounds-checking
    uint32_t padded_n = qy_needs_dequant ? ROUNDUP_POW2(ne11, pipeline->wg_denoms[1]) : ne11;
    ध्रुव uint64_t x_ne = ggml_nelements(src0);
    // 128 elements per Q8_1 x4 block
    ध्रुव uint64_t y_ne = padded_n * ne10 * ne12 * ne13;
    ध्रुव uint64_t d_ne = ggml_nelements(dst);

    ध्रुव uint32_t विभज_k = ggml_vk_guess_विभज_k(ctx, ne01, ne11, ne10, disable_विभज_k, pipeline);

    ध्रुव uint64_t qx_sz = ggml_type_आकार(src0->type) * x_ne / ggml_blck_आकार(src0->type);
    ध्रुव uint64_t qy_sz = ggml_type_आकार(src1->type) * y_ne / ggml_blck_आकार(src1->type);
    ध्रुव uint64_t x_sz = !qx_needs_dequant ? qx_sz : परिमाण(ggml_fp16_t) * x_ne;
    ध्रुव uint64_t y_sz = quantize_y ? (ggml_vk_align_आकार(y_ne, 128) * ggml_type_आकार(GGML_TYPE_Q8_1) / ggml_blck_आकार(GGML_TYPE_Q8_1)) : (y_f32_kernel ? परिमाण(भिन्न) * y_ne : परिमाण(ggml_fp16_t) * y_ne);
    ध्रुव uint64_t d_sz = परिमाण(भिन्न) * d_ne;

    vk_pipeline गन्तव्यम्_fp16_vk_0 = शून्यसूचकः;
    vk_pipeline गन्तव्यम्_fp16_vk_1 = शून्यसूचकः;
    vk_pipeline गन्तव्यम्_q8_1 = शून्यसूचकः;

    यदि (x_non_contig) {
        गन्तव्यम्_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, शून्यसूचकः, f16_type);
    } अन्यथा {
        गन्तव्यम्_fp16_vk_0 = ggml_vk_get_गन्तव्यम्_fp16(ctx, src0->type);
    }
    यदि (y_needs_reformat) {
        गन्तव्यम्_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, शून्यसूचकः, y_kernel_type);
    } अन्यथा {
        गन्तव्यम्_fp16_vk_1 = ggml_vk_get_गन्तव्यम्_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || गन्तव्यम्_fp16_vk_0 != शून्यसूचकः);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || गन्तव्यम्_fp16_vk_1 != शून्यसूचकः);  // NOLINT

    यदि (quantize_y) {
        गन्तव्यम्_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    {
        ध्रुव uint64_t विभज_k_आकार = विभज_k > 1 ? d_sz * विभज_k : 0;
        यदि (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_बफर_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_बफर_Range) ||
                (विभज_k > 1 && विभज_k_आकार > ctx->device->properties.limits.max_Storage_बफर_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        यदि (qx_needs_dequant && ctx->prealloc_आकार_x < x_sz) {
            ctx->prealloc_आकार_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        यदि ((qy_needs_dequant || quantize_y) && ctx->prealloc_आकार_y < y_sz) {
            ctx->prealloc_आकार_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        यदि (विभज_k > 1 && ctx->prealloc_आकार_विभज_k < विभज_k_आकार) {
            ctx->prealloc_आकार_विभज_k = विभज_k_आकार;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        यदि (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, गन्तव्यम्_fp16_vk_0, 1);
        }
        यदि (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, गन्तव्यम्_fp16_vk_1, 1);
        }
        यदि (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, गन्तव्यम्_q8_1, 1);
        }
        यदि (विभज_k > 1) {
            ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_विभज_k_reduce, 1);
        }
    }

    vk_बफर d_D = dst_buf_ctx->dev_बफर;
    ध्रुव uint64_t d_buf_विस्थापनम् = vk_tensor_विस्थापनम्(dst) + dst->view_offs;
    GGML_ASSERT(d_D != शून्यसूचकः);
    GGML_ASSERT(d_D->size >= d_buf_विस्थापनम् + d_sz);
    vk_बफर d_X;
    uint64_t x_buf_विस्थापनम् = 0;
    vk_बफर d_Y;
    uint64_t y_buf_विस्थापनम् = 0;
    यदि (!src0_uma) {
        d_Qx = src0_buf_ctx->dev_बफर;
        qx_buf_विस्थापनम् = vk_tensor_विस्थापनम्(src0) + src0->view_offs;
        GGML_ASSERT(d_Qx != शून्यसूचकः);
    }
    यदि (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_बफर;
        qy_buf_विस्थापनम् = vk_tensor_विस्थापनम्(src1) + src1->view_offs;
        GGML_ASSERT(d_Qy != शून्यसूचकः);
    }
    यदि (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz);
    } अन्यथा {
        d_X = d_Qx;
        x_buf_विस्थापनम् = qx_buf_विस्थापनम्;
        GGML_ASSERT(qx_sz == x_sz);
    }
    यदि (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz);
    } अन्यथा यदि (quantize_y) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= CEIL_DIV(y_sz, 144) * 144);
    } अन्यथा {
        d_Y = d_Qy;
        y_buf_विस्थापनम् = qy_buf_विस्थापनम्;
        GGML_ASSERT(qy_sz == y_sz);
    }

    यदि (x_non_contig || qx_needs_dequant) {
        यदि (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    यदि (x_non_contig) {
        ggml_vk_cpy_गन्तव्यम्_contiguous(ctx, subctx, गन्तव्यम्_fp16_vk_0, src0, ggml_vk_subbuffer(ctx, d_Qx, qx_buf_विस्थापनम्), ggml_vk_subbuffer(ctx, d_X, 0));
    } अन्यथा यदि (qx_needs_dequant) {
        ध्रुव std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_dispatch_pipeline(ctx, subctx, गन्तव्यम्_fp16_vk_0, { vk_subbuffer{ d_Qx, qx_buf_विस्थापनम्, qx_sz }, vk_subbuffer{ d_X, 0, x_sz } }, pc, { (uint32_t)(x_ne), 1, 1});
        ggml_vk_sync_buffers(ctx, subctx);
    }
    यदि (y_needs_reformat) {
        यदि (ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् != गन्तव्यम्_fp16_vk_1.get() ||
            ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् != src1 ||
            ctx->prealloc_y_चरमः_k_padded) {
            यदि (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_गन्तव्यम्_contiguous(ctx, subctx, गन्तव्यम्_fp16_vk_1, src1, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_विस्थापनम्), ggml_vk_subbuffer(ctx, d_Y, 0));
            ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् = गन्तव्यम्_fp16_vk_1.get();
            ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् = src1;
            ctx->prealloc_y_चरमः_k_padded = असत्य;
        }
    }
    यदि (quantize_y) {
        यदि (ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् != गन्तव्यम्_q8_1.get() ||
            ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् != src1 ||
            ctx->prealloc_y_चरमः_k_padded) {
            यदि (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_विस्थापनम्), ggml_vk_subbuffer(ctx, d_Y, 0), y_ne);
            ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् = गन्तव्यम्_q8_1.get();
            ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् = src1;
            ctx->prealloc_y_चरमः_k_padded = असत्य;
        }
    }

    // The partial tiles of a quantized A rely on the bound range to read zeros past the last row,
    // so the range stays exact: the strided extent of a tensor read in place, the staged size otherwise.
    ध्रुव uint64_t x_range = qx_needs_dequant ? x_sz : ggml_nbytes(src0);
    ध्रुव uint64_t y_range = (qy_needs_dequant || quantize_y) ? y_sz : ggml_nbytes(src1);

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_batch_y = (qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1);

    यदि (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_आकार(src0->type);
    }

    यदि (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant && !quantize_y) {
        stride_batch_y = src1->nb[0] / ggml_type_आकार(src1->type);
    }

    // column stride in bytes of the B buffer: raw src1 keeps its own stride, staging buffers are tightly packed
    ध्रुव uint64_t b_col_अष्टकानि = (y_needs_reformat || qy_needs_dequant || quantize_y)
        ? (quantize_y ? (uint64_t)(ne10 / ggml_blck_आकार(GGML_TYPE_Q8_1)) * ggml_type_आकार(GGML_TYPE_Q8_1) : (uint64_t)ne10 * परिमाण(ggml_fp16_t))
        : (uint64_t)src1->nb[1];

    // compute
    ggml_vk_matmul(
        ctx, subctx, pipeline,
        { d_X, x_buf_विस्थापनम्, x_range }, { d_Y, y_buf_विस्थापनम्, y_range },
        ggml_vk_subbuffer(ctx, d_D, d_buf_विस्थापनम्), { ctx->prealloc_विभज_k, 0, d_sz * विभज_k },
        ne01, ne11, ne10,
        ne10, ne10, stride_d, stride_batch_x, stride_batch_y, stride_batch_d,
        विभज_k, ne12*ne13, ne02, ne12, r2, r3, padded_n,
        b_col_अष्टकानि, (uint64_t)dst->nb[1]
    );  // NOLINT

    यदि (x_non_contig || qx_needs_dequant) {
        ctx->prealloc_x_need_sync = सत्य;
    }
    यदि (y_needs_reformat || quantize_y) {
        ctx->prealloc_y_need_sync = सत्य;
    }
}

स्थिर तार्किक ggml_vk_should_use_mmvq(ध्रुव vk_device& device, uint32_t m, uint32_t n, uint32_t k, ggml_type src0_type) {
    यदि (device->mmvq_बहुलकम् == 1) {
        प्रत्यागम सत्य;
    } अन्यथा यदि (device->mmvq_बहुलकम् == -1) {
        प्रत्यागम असत्य;
    }

    // q6_k only has 2-byte alignment which makes it somewhat problematic,
    // using MMVQ is only a win on Intel.
    तार्किक mmvq_q6 = device->vendor_id == VK_VENDOR_ID_INTEL;
    यदि (src0_type == GGML_TYPE_Q6_K && !mmvq_q6) {
        प्रत्यागम असत्य;
    }

    // MMVQ is generally good for batches
    यदि (n > 1) {
        प्रत्यागम सत्य;
    }

    // Quantization overhead is not worth it for small k
    विकल्प (device->vendor_id) {
    पक्ष VK_VENDOR_ID_NVIDIA:
        यदि (src0_type == GGML_TYPE_Q2_0 || src0_type == GGML_TYPE_Q2_K || src0_type == GGML_TYPE_Q3_K || src0_type == GGML_TYPE_IQ1_S || src0_type == GGML_TYPE_IQ1_M) {
            प्रत्यागम सत्य;
        }

        यदि (k <= 4096) {
            प्रत्यागम असत्य;
        }

        विकल्प (src0_type) {
        पक्ष GGML_TYPE_MXFP4:
        पक्ष GGML_TYPE_Q8_0:
            प्रत्यागम device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        स्वभाव:
            प्रत्यागम सत्य;
        }
    पक्ष VK_VENDOR_ID_AMD:
        यदि (k < 2048) {
            प्रत्यागम असत्य;
        }

        विकल्प (src0_type) {
        पक्ष GGML_TYPE_Q8_0:
            प्रत्यागम device->architecture == vk_device_architecture::AMD_GCN;
        स्वभाव:
            प्रत्यागम सत्य;
        }
    पक्ष VK_VENDOR_ID_INTEL:
        यदि (device->architecture == vk_device_architecture::INTEL_XE2) {
            यदि (src0_type == GGML_TYPE_Q2_0 || src0_type == GGML_TYPE_Q2_K || src0_type == GGML_TYPE_Q3_K || src0_type == GGML_TYPE_Q6_K) {
                प्रत्यागम सत्य;
            }
        }

        यदि (device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            // Intel Windows proprietary driver MMVQ performance for !Q2/Q3/Q6 is worse than fp16,
            // see https://github.com/ggml-org/llama.cpp/issues/17628 and
            // https://github.com/ggml-org/llama.cpp/pull/23056
            प्रत्यागम असत्य;
        }

        यदि (k < 2048) {
            प्रत्यागम असत्य;
        }

        विकल्प (src0_type) {
        // From tests on A770 Linux, may need more tuning
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q5_1:
        पक्ष GGML_TYPE_IQ4_XS:
            प्रत्यागम असत्य;
        स्वभाव:
            प्रत्यागम सत्य;
        }
    पक्ष VK_VENDOR_ID_QUALCOMM:
        प्रत्यागम असत्य;
    स्वभाव:
        प्रत्यागम सत्य;
    }

    GGML_अप्रयुक्तम्(m);
}

स्थिर शून्य ggml_vk_mul_mat_vec_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx, तार्किक swap_inputs = असत्य) {
    ggml_tensor * dst = cgraph->nodes[ग्रन्थिः_idx];
    ध्रुव ggml_tensor * src0 = dst->src[swap_inputs ? 1 : 0];
    ध्रुव ggml_tensor * src1 = dst->src[swap_inputs ? 0 : 1];

    VK_LOG_DEBUG("ggml_vk_mul_mat_vec_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << ")),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    ध्रुव uint64_t ne00 = src0->ne[0];
    ध्रुव uint64_t ne01 = src0->ne[1];
    ध्रुव uint64_t ne02 = src0->ne[2];
    ध्रुव uint64_t ne03 = src0->ne[3];

    ध्रुव uint64_t ne10 = src1->ne[0];
    ध्रुव uint64_t ne11 = src1->ne[1];
    ध्रुव uint64_t ne12 = src1->ne[2];
    ध्रुव uint64_t ne13 = src1->ne[3];

    ध्रुव uint64_t ne20 = dst->ne[swap_inputs ? 1 : 0];
    ध्रुव uint64_t ne21 = dst->ne[swap_inputs ? 0 : 1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    ध्रुव uint64_t r2 = ne12 / ne02;
    ध्रुव uint64_t r3 = ne13 / ne03;

    // batch_n indicates that we need to compute a few vector results, and this assumes
    // ne12 and ne13 are 1. It overloads the batch_strides to hold the row strides.
    GGML_ASSERT(ne11 == 1 || ne12 * ne13 == 1);
    तार्किक batch_n = ne11 > 1;

    ध्रुव तार्किक x_non_contig = !ggml_vk_dim01_contiguous(src0);

    // BF16 src1 has no mul_mat_vec B-type; widen to F32
    ध्रुव तार्किक widen_bf16 = src1->type == GGML_TYPE_BF16;
    ध्रुव तार्किक y_non_contig = !ggml_vk_dim01_contiguous(src1) || widen_bf16;

    ध्रुव ggml_type y_kernel_type = widen_bf16 ? GGML_TYPE_F32 : src1->type;
    ध्रुव तार्किक f16_f32_kernel = y_kernel_type == GGML_TYPE_F32;
    तार्किक quantize_y = ctx->device->integer_dot_गुणनफलम् && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0 && ggml_vk_should_use_mmvq(ctx->device, ne01, ne11, ne10, src0->type);

    vk_pipeline गन्तव्यम्_fp16_vk_0 = शून्यसूचकः;
    vk_pipeline गन्तव्यम्_fp16_vk_1 = शून्यसूचकः;
    यदि (x_non_contig) {
        गन्तव्यम्_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, शून्यसूचकः, src0->type);
    }
    यदि (y_non_contig) {
        गन्तव्यम्_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, शून्यसूचकः, y_kernel_type);
    } अन्यथा {
        गन्तव्यम्_fp16_vk_1 = ggml_vk_get_गन्तव्यम्_fp16(ctx, src1->type);
    }

    // Check for mmq first
    vk_pipeline dmmv = quantize_y ? ggml_vk_get_dequantize_mul_mat_vec(ctx, src0->type, GGML_TYPE_Q8_1, ne11, ne20, ne00) : शून्यसूचकः;
    vk_pipeline गन्तव्यम्_q8_1 = शून्यसूचकः;

    यदि (dmmv == शून्यसूचकः) {
        // Fall back to f16 dequant mul mat
        dmmv = ggml_vk_get_dequantize_mul_mat_vec(ctx, src0->type, y_kernel_type, ne11, ne20, ne00);
        quantize_y = असत्य;
    }

    यदि (quantize_y) {
        गन्तव्यम्_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    यदि (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_बफर_Range) {
        dmmv = ggml_vk_get_64b_indexing_pipeline(ctx, dmmv);
    }

    ध्रुव तार्किक qx_needs_dequant = x_non_contig;
    ध्रुव तार्किक qy_needs_dequant = !quantize_y && ((src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig);

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT

    GGML_ASSERT(!qx_needs_dequant || गन्तव्यम्_fp16_vk_0 != शून्यसूचकः);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || गन्तव्यम्_fp16_vk_1 != शून्यसूचकः);  // NOLINT
    GGML_ASSERT(dmmv != शून्यसूचकः);

    ध्रुव uint64_t x_ne = ggml_nelements(src0);
    ध्रुव uint64_t y_ne = ggml_nelements(src1);

    ध्रुव uint64_t qx_sz = ggml_vk_align_आकार(ggml_type_आकार(src0->type) * x_ne / ggml_blck_आकार(src0->type), ctx->device->properties.limits.min_Storage_बफर_विस्थापनम्_Alignment);
    ध्रुव uint64_t x_sz = x_non_contig ? ggml_vk_align_आकार(ggml_type_आकार(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_बफर_विस्थापनम्_Alignment) : qx_sz;
    ध्रुव uint64_t y_sz = quantize_y ? (ggml_vk_align_आकार(y_ne, 128) * ggml_type_आकार(GGML_TYPE_Q8_1) / ggml_blck_आकार(GGML_TYPE_Q8_1)) :
                         (f16_f32_kernel ? परिमाण(भिन्न) * y_ne : परिमाण(ggml_fp16_t) * y_ne);

    {
        यदि (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_बफर_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_बफर_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        यदि (qx_needs_dequant && ctx->prealloc_आकार_x < x_sz) {
            ctx->prealloc_आकार_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        यदि ((qy_needs_dequant || quantize_y) && ctx->prealloc_आकार_y < y_sz) {
            ctx->prealloc_आकार_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        यदि (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, गन्तव्यम्_fp16_vk_0, 1);
        }
        यदि (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, गन्तव्यम्_fp16_vk_1, 1);
        }
        यदि (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, गन्तव्यम्_q8_1, 1);
        }
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[ग्रन्थिः_idx + ctx->अङ्कः_additional_fused_ops]);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer d_X, d_Y;

    यदि (qx_needs_dequant) {
        d_X = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    } अन्यथा {
        d_X = d_Qx;
        GGML_ASSERT(qx_sz == x_sz);
    }
    यदि (qy_needs_dequant || quantize_y) {
        d_Y = { ctx->prealloc_y, 0, ctx->prealloc_y->size };
    } अन्यथा {
        d_Y = d_Qy;
    }

    यदि (x_non_contig) {
        यदि (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        GGML_ASSERT(x_sz == ggml_vk_align_आकार(ggml_type_आकार(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_बफर_विस्थापनम्_Alignment));
        ggml_vk_cpy_गन्तव्यम्_contiguous(ctx, subctx, गन्तव्यम्_fp16_vk_0, src0, d_Qx, d_X);
    }
    यदि (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_type_आकार(y_kernel_type) * y_ne);
        यदि (ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् != गन्तव्यम्_fp16_vk_1.get() ||
            ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् != src1 ||
            ctx->prealloc_y_चरमः_k_padded) {
            यदि (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_गन्तव्यम्_contiguous(ctx, subctx, गन्तव्यम्_fp16_vk_1, src1, d_Qy, d_Y);
            ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् = गन्तव्यम्_fp16_vk_1.get();
            ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् = src1;
            ctx->prealloc_y_चरमः_k_padded = असत्य;
        }
    }
    यदि (quantize_y) {
        यदि (ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् != गन्तव्यम्_q8_1.get() ||
            ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् != src1 ||
            ctx->prealloc_y_चरमः_k_padded) {
            यदि (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, d_Qy, d_Y, y_ne);
            ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् = गन्तव्यम्_q8_1.get();
            ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् = src1;
            ctx->prealloc_y_चरमः_k_padded = असत्य;
        }
    }

    // For batch_n, the A matrix is the same for each batch, and B/D use the row stride as the batch stride
    uint32_t stride_batch_x = batch_n ? 0 : (qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0));
    uint32_t stride_batch_y = batch_n ? ne10 : ((qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1));
    uint32_t stride_batch_d = batch_n ? ne20 : (ne20*ne21);

    यदि (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_आकार(src0->type);
    }

    यदि (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[0] / ggml_type_आकार(src1->type);
    }

    ध्रुव uint32_t max_groups_x = ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    यदि (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = CEIL_DIV(groups_x, groups_z);
    }

    uint32_t fusion_flags = 0;

    vk_subbuffer d_F0 = d_D;
    यदि (ctx->अङ्कः_additional_fused_ops > 0) {
        ध्रुव ggml_tensor * सङ्कलय_2 = cgraph->nodes[ग्रन्थिः_idx + 1];
        ध्रुव ggml_tensor * bias = सङ्कलय_2->src[0] == dst ? सङ्कलय_2->src[1] : सङ्कलय_2->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    यदि (ctx->अङ्कः_additional_fused_ops == 2) {
        ध्रुव ggml_tensor * सङ्कलय_2 = cgraph->nodes[ग्रन्थिः_idx + 2];
        ध्रुव ggml_tensor * bias = सङ्कलय_2->src[0] == cgraph->nodes[ग्रन्थिः_idx + 1] ? सङ्कलय_2->src[1] : सङ्कलय_2->src[0];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    ggml_pipeline_request_descriptor_sets(ctx, dmmv, CEIL_DIV(ne12 * ne13, ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[1]));

    uint32_t आधारः_work_गणः_y = 0;
    यावत् (आधारः_work_गणः_y < ne12 * ne13) {

        uint32_t groups_y = std::min((uint32_t)(ne12 * ne13) - आधारः_work_गणः_y, ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[1]);
        ध्रुव vk_mat_vec_प्रक्षिप_constants pc = {
            (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
            stride_batch_x, stride_batch_y, stride_batch_d,
            fusion_flags, आधारः_work_गणः_y,
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
        आधारः_work_गणः_y += groups_y;
    }

    यदि (x_non_contig) {
        ctx->prealloc_x_need_sync = सत्य;
    }
    यदि (y_non_contig || quantize_y) {
        ctx->prealloc_y_need_sync = सत्य;
    }
}

स्थिर शून्य ggml_vk_mul_mat_vec_p021_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {
    ggml_tensor * dst = cgraph->nodes[ग्रन्थिः_idx];
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat_p021_f16_f32(" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]);  // NOLINT
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]);  // NOLINT
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ध्रुव uint64_t ne00 = src0->ne[0];
    ध्रुव uint64_t ne01 = src0->ne[1];
    ध्रुव uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    //const uint64_t ne10 = src1->ne[0];
    ध्रुव uint64_t ne11 = src1->ne[1];
    ध्रुव uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    GGML_ASSERT(ne11 == 1);

    // With grouped query attention there are > 1 Q matrices per K, V matrix.
    uint32_t gqa_अनुपातः = (uint32_t)ne12 / (uint32_t)ne02;
    यदि (gqa_अनुपातः > 8 || gqa_अनुपातः == 0 || ne12 != ne02 * gqa_अनुपातः) {
        gqa_अनुपातः = 1;
    }

    vk_pipeline pipeline = ctx->device->pipeline_mul_mat_vec_p021_f16_f32[gqa_अनुपातः - 1];

    यदि (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_बफर_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    {
        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[ग्रन्थिः_idx + ctx->अङ्कः_additional_fused_ops], सत्य);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1, सत्य);

    vk_subbuffer d_F0 = d_D;

    uint32_t fusion_flags = 0;

    यदि (ctx->अङ्कः_additional_fused_ops > 0) {
        ध्रुव ggml_tensor * सङ्कलय_2 = cgraph->nodes[ग्रन्थिः_idx + 1];
        ध्रुव ggml_tensor * bias = सङ्कलय_2->src[0] == dst ? सङ्कलय_2->src[1] : सङ्कलय_2->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    यदि (ctx->अङ्कः_additional_fused_ops > 1) {
        ध्रुव ggml_tensor * bias = cgraph->nodes[ग्रन्थिः_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    // compute

    vk_mat_vec_p021_प्रक्षिप_constants pc = {
        (uint32_t)ne00, (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne12,
        0, 0, fusion_flags
    };

    init_pushconst_tensor_offsets(ctx, pc, src0, src1, शून्यसूचकः, शून्यसूचकः, cgraph->nodes[ग्रन्थिः_idx + ctx->अङ्कः_additional_fused_ops]);

    uint32_t workgroups_z = (uint32_t)ne12;
    // When gqa_ratio > 1, each invocation does multiple rows and we can launch fewer workgroups
    यदि (gqa_अनुपातः > 1) {
        workgroups_z /= gqa_अनुपातः;
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

स्थिर शून्य ggml_vk_mul_mat_vec_nc_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {
    ggml_tensor * dst = cgraph->nodes[ग्रन्थिः_idx];
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat_nc_f16_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ध्रुव uint64_t ne00 = src0->ne[0];
    ध्रुव uint64_t ne01 = src0->ne[1];
    ध्रुव uint64_t ne02 = src0->ne[2];
    ध्रुव uint64_t ne03 = src0->ne[3];

    ध्रुव uint64_t nb01 = src0->nb[1];
    ध्रुव uint64_t nb02 = src0->nb[2];

    ध्रुव uint64_t nb12 = src1->nb[2];

    // const uint64_t ne10 = src1->ne[0];
    ध्रुव uint64_t ne11 = src1->ne[1];
    ध्रुव uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    ध्रुव uint32_t nb03 = (uint32_t)(src0->nb[3] / परिमाण(ggml_fp16_t));
    ध्रुव uint32_t nb13 = (uint32_t)(src1->nb[3] / परिमाण(भिन्न));
    ध्रुव uint32_t nb23 = (uint32_t)(dst->nb[3] / परिमाण(भिन्न));

    GGML_ASSERT(ne11 == 1);
    GGML_ASSERT(src0->ne[3] == src1->ne[3]); // checked in supports_op

    ध्रुव uint32_t row_stride_x = nb01 / परिमाण(ggml_fp16_t);
    ध्रुव uint32_t channel_stride_x = nb02 / परिमाण(ggml_fp16_t);
    ध्रुव uint32_t channel_stride_y = nb12 / परिमाण(भिन्न);

    vk_pipeline pipeline = ctx->device->pipeline_mul_mat_vec_nc_f16_f32;
    यदि (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_बफर_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    {
        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[ग्रन्थिः_idx + ctx->अङ्कः_additional_fused_ops], सत्य);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1, सत्य);
    vk_subbuffer d_F0 = d_D;

    uint32_t fusion_flags = 0;

    यदि (ctx->अङ्कः_additional_fused_ops > 0) {
        ध्रुव ggml_tensor * सङ्कलय_2 = cgraph->nodes[ग्रन्थिः_idx + 1];
        ध्रुव ggml_tensor * bias = सङ्कलय_2->src[0] == dst ? सङ्कलय_2->src[1] : सङ्कलय_2->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    यदि (ctx->अङ्कः_additional_fused_ops > 1) {
        ध्रुव ggml_tensor * bias = cgraph->nodes[ग्रन्थिः_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    // compute
    vk_mat_vec_nc_प्रक्षिप_constants pc = {
        (uint32_t)ne00, (uint32_t)ne01,
        row_stride_x, channel_stride_x, channel_stride_y,
        (uint32_t)(ne12 / ne02), (uint32_t)ne12,
        0, 0,
        nb03, nb13, nb23, fusion_flags
    };

    init_pushconst_tensor_offsets(ctx, pc, src0, src1, शून्यसूचकः, शून्यसूचकः, cgraph->nodes[ग्रन्थिः_idx + ctx->अङ्कः_additional_fused_ops]);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            d_Qx,
            d_Qy,
            d_D,
            d_F0,
            d_F1,
        }, pc, { (uint32_t)ne03, (uint32_t)ne01, (uint32_t)ne12 });
}

स्थिर पूर्णाङ्क ggml_vk_fwht_pipeline_idx(int64_t n) {
    विकल्प (n) {
        पक्ष 64:  प्रत्यागम 0;
        पक्ष 128: प्रत्यागम 1;
        पक्ष 256: प्रत्यागम 2;
        पक्ष 512: प्रत्यागम 3;
        स्वभाव:  प्रत्यागम -1;
    }
}

तार्किक ggml_vk_can_use_fwht(ध्रुव ggml_backend_vk_context * ctx, ध्रुव ggml_tensor * src1, ध्रुव ggml_tensor * dst) {
    यदि (ctx->अङ्कः_additional_fused_ops != 0) {
        प्रत्यागम असत्य;
    }

    यदि (ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        प्रत्यागम असत्य;
    }

    ध्रुव पूर्णाङ्क idx = ggml_vk_fwht_pipeline_idx(src1->ne[0]);
    यदि (idx < 0 || ctx->device->pipeline_fwht_f32[idx] == शून्यसूचकः) {
        प्रत्यागम असत्य;
    }

    यदि (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }

    यदि (!ggml_is_contiguous(src1)) {
        प्रत्यागम असत्य;
    }
    GGML_ASSERT(ggml_is_contiguous(dst));

    प्रत्यागम सत्य;
}

शून्य ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src, ggml_tensor * dst) {
    ध्रुव पूर्णाङ्क idx = ggml_vk_fwht_pipeline_idx(src->ne[0]);
    vk_pipeline pipeline = ctx->device->pipeline_fwht_f32[idx];

    ध्रुव uint32_t पङ्क्तिकाः_प्रत्येकम्_workgroup = 4;
    ध्रुव uint32_t n_पङ्क्तिकाः = (uint32_t)ggml_nrows(src);
    ध्रुव uint32_t max_workgroups_x = ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[0];

    ध्रुव uint32_t योग_workgroups = CEIL_DIV(n_पङ्क्तिकाः, पङ्क्तिकाः_प्रत्येकम्_workgroup);
    ध्रुव uint32_t workgroups_x = std::min(योग_workgroups, max_workgroups_x);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ध्रुव vk_subbuffer src_buf = ggml_vk_tensor_subbuffer(ctx, src, सत्य);
    ध्रुव vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, सत्य);

    vk_op_fwht_प्रक्षिप_constants pc = {
        n_पङ्क्तिकाः,
        0,
        0,
        1.0f / std::वर्गमूलम्((भिन्न)src->ne[0]),
    };
    init_pushconst_tensor_offsets(ctx, pc, src, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf }, pc, { workgroups_x, 1, 1 });
}

स्थिर uint32_t ggml_vk_nb_elem(ध्रुव ggml_tensor * t, पूर्णाङ्क i) {
    प्रत्यागम (uint32_t)(t->nb[i] / ggml_type_आकार(t->type));
}

शून्य ggml_vk_dsv4_hc_comb(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * mixes, ध्रुव ggml_tensor * scale, ध्रुव ggml_tensor * आधारः, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_comb(" << mixes << ", " << scale << ", " << आधारः << ", " << dst << ")");

    vk_pipeline pipeline = ctx->device->pipeline_dsv4_hc_comb_f32;
    GGML_ASSERT(pipeline != शून्यसूचकः);

    ध्रुव uint32_t n_tokens = (uint32_t)mixes->ne[1];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ध्रुव vk_subbuffer mixes_buf = ggml_vk_tensor_subbuffer(ctx, mixes, सत्य);
    ध्रुव vk_subbuffer scale_buf = ggml_vk_tensor_subbuffer(ctx, scale, सत्य);
    ध्रुव vk_subbuffer आधारः_buf  = ggml_vk_tensor_subbuffer(ctx, आधारः,  सत्य);
    ध्रुव vk_subbuffer dst_buf   = ggml_vk_tensor_subbuffer(ctx, dst,   सत्य);

    vk_op_dsv4_hc_comb_प्रक्षिप_constants pc = {
        n_tokens,
        ggml_vk_nb_elem(mixes, 0), ggml_vk_nb_elem(mixes, 1),
        ggml_vk_nb_elem(scale, 0),
        ggml_vk_nb_elem(आधारः,  0),
        ggml_vk_nb_elem(dst,   0), ggml_vk_nb_elem(dst, 1), ggml_vk_nb_elem(dst, 2),
        0, 0, 0, 0,
        ggml_get_op_params_f32(dst, 0),
        (uint32_t)ggml_get_op_params_i32(dst, 1),
    };
    init_pushconst_tensor_offsets(ctx, pc, mixes, scale, आधारः, शून्यसूचकः, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { mixes_buf, scale_buf, आधारः_buf, dst_buf }, pc, { n_tokens, 1, 1 });
}

शून्य ggml_vk_dsv4_hc_pre(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * x, ध्रुव ggml_tensor * weights, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_pre(" << x << ", " << weights << ", " << dst << ")");

    ध्रुव भिन्न scale = ggml_get_op_params_f32(dst, 0);
    ध्रुव तार्किक  gated = ggml_get_op_params_i32(dst, 1) != 0;

    vk_pipeline pipeline = gated ? ctx->device->pipeline_dsv4_hc_pre_gated_f32 : ctx->device->pipeline_dsv4_hc_pre_f32;
    GGML_ASSERT(pipeline != शून्यसूचकः);

    ध्रुव uint32_t n_embd   = (uint32_t)x->ne[0];
    ध्रुव uint32_t n_tokens = (uint32_t)x->ne[2];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ध्रुव vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x,       सत्य);
    ध्रुव vk_subbuffer w_buf = ggml_vk_tensor_subbuffer(ctx, weights, सत्य);
    ध्रुव vk_subbuffer d_buf = ggml_vk_tensor_subbuffer(ctx, dst,     सत्य);

    vk_op_dsv4_hc_pre_प्रक्षिप_constants pc = {
        n_embd, n_tokens,
        ggml_vk_nb_elem(x, 0), ggml_vk_nb_elem(x, 1), ggml_vk_nb_elem(x, 2),
        ggml_vk_nb_elem(weights, 0), ggml_vk_nb_elem(weights, 1), ggml_vk_nb_elem(weights, 2),
        ggml_vk_nb_elem(dst, 0), ggml_vk_nb_elem(dst, 1),
        0, 0, 0,
        scale,
    };
    init_pushconst_tensor_offsets(ctx, pc, x, weights, शून्यसूचकः, शून्यसूचकः, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, w_buf, d_buf }, pc, { n_embd, n_tokens, 1 });
}

शून्य ggml_vk_dsv4_hc_post(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * x, ध्रुव ggml_tensor * residual, ध्रुव ggml_tensor * post, ध्रुव ggml_tensor * comb, ggml_tensor * dst, ध्रुव ggml_tensor * gate_scale_अन्तर्) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_post(" << x << ", " << residual << ", " << post << ", " << comb << ", " << dst << ")");

    vk_pipeline pipeline = comb ? ctx->device->pipeline_dsv4_hc_post_f32 : ctx->device->pipeline_dsv4_hc_post_nocomb_f32;
    GGML_ASSERT(pipeline != शून्यसूचकः);

    ध्रुव uint32_t n_embd   = (uint32_t)x->ne[0];
    ध्रुव uint32_t n_tokens = (uint32_t)x->ne[1];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ध्रुव vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x,        सत्य);
    ध्रुव vk_subbuffer r_buf = ggml_vk_tensor_subbuffer(ctx, residual, सत्य);
    // with a fused gate, post is scale(sigmoid(scale(p_src))) and the shader applies it to p_src
    ध्रुव ggml_tensor * p_src = gate_scale_अन्तर् ? gate_scale_अन्तर्->src[0] : post;
    ध्रुव vk_subbuffer p_buf = ggml_vk_tensor_subbuffer(ctx, p_src,    सत्य);
    ध्रुव vk_subbuffer c_buf = comb ? ggml_vk_tensor_subbuffer(ctx, comb, सत्य) : x_buf;
    ध्रुव vk_subbuffer d_buf = ggml_vk_tensor_subbuffer(ctx, dst,      सत्य);

    vk_op_dsv4_hc_post_प्रक्षिप_constants pc = {
        n_embd, n_tokens,
        ggml_vk_nb_elem(x, 0), ggml_vk_nb_elem(x, 1),
        ggml_vk_nb_elem(residual, 0), ggml_vk_nb_elem(residual, 1), ggml_vk_nb_elem(residual, 2),
        ggml_vk_nb_elem(p_src, 0), ggml_vk_nb_elem(p_src, 1),
        comb ? ggml_vk_nb_elem(comb, 0) : 0, comb ? ggml_vk_nb_elem(comb, 1) : 0, comb ? ggml_vk_nb_elem(comb, 2) : 0,
        ggml_vk_nb_elem(dst,  0), ggml_vk_nb_elem(dst,  1), ggml_vk_nb_elem(dst,  2),
        0, 0, 0, 0, 0,
        gate_scale_अन्तर् ? 1u : 0u,
        gate_scale_अन्तर् ? ggml_get_op_params_f32(gate_scale_अन्तर्, 0) : 1.0f,
        gate_scale_अन्तर् ? ggml_get_op_params_f32(post, 0) : 1.0f,
    };
    init_pushconst_tensor_offsets(ctx, pc, x, residual, p_src, comb, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, r_buf, p_buf, c_buf, d_buf }, pc, { n_embd, n_tokens, 1 });
}

शून्य ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {
    ggml_tensor * dst = cgraph->nodes[ग्रन्थिः_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat(" << src0 << ", " << src1 << ", " << dst << ")");

    // Handle huge A matrix by splitting the M dimensions. This works well for convolution use cases
    // where the M dimension is very large.
    // Split_k doesn't work with M splitting.
    // This only supports batchsize == 1.
    ध्रुव size_t nbytes = ggml_nbytes(src0);
    ध्रुव तार्किक needs_विभज = dst->ne[2] == 1 && dst->ne[3] == 1 && nbytes > ctx->device->properties.limits.max_Storage_बफर_Range;
    यदि (needs_विभज) {
        // Choose the number of rows that can fit (and divide by two, to allow for any additional offsets)
        ध्रुव uint32_t M_विभज = ctx->device->properties.limits.max_Storage_बफर_Range / (2 * src0->nb[1]);
        uint32_t m_विस्थापनम् = 0;
        यावत् (m_विस्थापनम् < dst->ne[0]) {
            ध्रुव uint32_t cur_M_आकार = std::min(M_विभज, (uint32_t)(dst->ne[0] - m_विस्थापनम्));
            ggml_tensor dst2 = *dst;
            ggml_tensor src02 = *src0;

            dst2.view_src = dst->view_src ? dst->view_src : dst;
            src02.view_src = src0->view_src ? src0->view_src : src0;

            dst2.view_offs += m_विस्थापनम् * dst->nb[0];
            src02.view_offs += m_विस्थापनम् * src0->nb[1];
            dst2.ne[0] = cur_M_आकार;
            src02.ne[1] = cur_M_आकार;

            ggml_vk_mul_mat_q_f16(ctx, subctx, &src02, src1, &dst2, सत्य);

            m_विस्थापनम् += cur_M_आकार;
        }
    } अन्यथा यदि (ggml_vk_can_use_fwht(ctx, src1, dst)) {
        ggml_vk_fwht(ctx, subctx, src1, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && dst->ne[1] == 1 &&
        // detect 0213 permutation, and batch size of 1
        src0->nb[0] <= src0->nb[2] &&
        src0->nb[2] <= src0->nb[1] &&
        src0->nb[1] <= src0->nb[3] &&
        src1->nb[0] <= src1->nb[2] &&
        src1->nb[2] <= src1->nb[1] &&
        src1->nb[1] <= src1->nb[3] &&
        src0->ne[3] == 1 &&
        src1->ne[3] == 1 &&
        src0->ne[1] <= ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[1] &&
        src1->ne[2] <= ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[2]) {
        ggml_vk_mul_mat_vec_p021_f16_f32(ctx, subctx, cgraph, ग्रन्थिः_idx);
    } अन्यथा यदि (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && dst->ne[1] == 1 &&
               !ggml_is_permuted(src0) && !ggml_is_permuted(src1) &&
               src0->ne[3] <= ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[0] &&
               src0->ne[1] <= ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[1] &&
               src1->ne[2] <= ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[2]) {
        ggml_vk_mul_mat_vec_nc_f16_f32(ctx, subctx, cgraph, ग्रन्थिः_idx);
    // With one output row, B^T*A has the same flat output as A^T*B.
    } अन्यथा यदि (ctx->अङ्कः_additional_fused_ops == 0 &&
               (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
               (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16 || ggml_is_quantized(src1->type)) &&
               dst->ne[0] == 1 && dst->ne[1] > mul_mat_vec_max_स्तम्भसङ्ख्या &&
               src0->ne[2] == 1 && src0->ne[3] == 1 &&
               src1->ne[2] == 1 && src1->ne[3] == 1 &&
               ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
               get_misalign_अष्टकानि(ctx, src0) == 0 && get_misalign_अष्टकानि(ctx, src1) == 0 && get_misalign_अष्टकानि(ctx, dst) == 0) {
        ggml_vk_mul_mat_vec_q_f16(ctx, subctx, cgraph, ग्रन्थिः_idx, सत्य);
    // mul_mat_vec supports batching ne12*ne13 when ne11==1, or treating ne11 as the batch size (up to four)
    // when ne12 and ne13 are one.
    } अन्यथा यदि ((dst->ne[1] == 1 || (dst->ne[1] <= mul_mat_vec_max_स्तम्भसङ्ख्या && src1->ne[2] * src1->ne[3] == 1)) &&
               (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || ggml_is_quantized(src0->type))) {
        ggml_vk_mul_mat_vec_q_f16(ctx, subctx, cgraph, ग्रन्थिः_idx);
    } अन्यथा {
        ggml_vk_mul_mat_q_f16(ctx, subctx, src0, src1, dst, असत्य);
    }
}

स्थिर शून्य ggml_vk_mul_mat_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ध्रुव ggml_tensor * ids, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1=" << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ध्रुव uint64_t ne00 = src0->ne[0];
    ध्रुव uint64_t ne01 = src0->ne[1];
    ध्रुव uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    ध्रुव uint64_t ne10 = src1->ne[0];
    ध्रुव uint64_t ne11 = src1->ne[1];
    ध्रुव uint64_t ne12 = src1->ne[2];
    ध्रुव uint64_t ne13 = src1->ne[3];

    ध्रुव uint64_t nei0 = ids->ne[0];
    ध्रुव uint64_t nei1 = ids->ne[1];

    ध्रुव uint32_t nbi0 = ids->nb[0];
    ध्रुव uint32_t nbi1 = ids->nb[1];

    ध्रुव uint64_t ne20 = dst->ne[0];
    ध्रुव uint64_t ne21 = dst->ne[1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    ध्रुव uint64_t n_as = ne02;

    ggml_backend_vk_बफर_context * dst_buf_ctx = (ggml_backend_vk_बफर_context *)dst->बफर->context;
    ggml_backend_vk_बफर_context * src0_buf_ctx = (ggml_backend_vk_बफर_context *)src0->बफर->context;
    ggml_backend_vk_बफर_context * src1_buf_ctx = (ggml_backend_vk_बफर_context *)src1->बफर->context;

    vk_बफर d_Qx = शून्यसूचकः;
    size_t qx_buf_विस्थापनम् = 0;
    vk_बफर d_Qy = शून्यसूचकः;
    size_t qy_buf_विस्थापनम् = 0;

    तार्किक src0_uma = असत्य;
    तार्किक src1_uma = असत्य;

    यदि (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->दत्त, d_Qx, qx_buf_विस्थापनम्);
        ggml_vk_host_get(ctx->device, src1->दत्त, d_Qy, qy_buf_विस्थापनम्);
        src0_uma = d_Qx != शून्यसूचकः;
        src1_uma = d_Qy != शून्यसूचकः;
    }

    // Reformat and convert to fp16 if non-contiguous, or for coopmat2 for better perf
    ध्रुव तार्किक x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    // If src0 is BF16, try to use a BF16 x BF16 multiply
    ggml_type f16_type = src0->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;
#यदि defined(GGML_VULKAN_COOPMAT2_DECODE_सदिशम्_GLSLC_SUPPORT)
    // B must already be, or be convertible to, the matmul B type used by this path.
    ध्रुव तार्किक y_decode_सदिशम्_supported = ctx->device->coopmat2_decode_सदिशम् &&
                                           (f16_type != GGML_TYPE_BF16 || ctx->device->coopmat2_bf16_support) &&
                                           (src1->type == GGML_TYPE_F32 || src1->type == f16_type);
    // If B is copied to prealloc_y, we can choose a 4-element-aligned row stride.
    ध्रुव तार्किक y_decode_सदिशम्_उपयुङ्क्ते_prealloc = !ggml_vk_dim01_contiguous(src1) || src1->type != f16_type;
    // Direct B reads are safe only if row starts and the original buffer offset are 4-element aligned.
    ध्रुव तार्किक y_decode_सदिशम्_aligned =
        (ne10 % 4 == 0) &&
        (y_decode_सदिशम्_उपयुङ्क्ते_prealloc || get_misalign_अष्टकानि(ctx, src1) % (4 * ggml_type_आकार(src1->type)) == 0);
    // Stage B only when decode-vector is available and direct B reads would be misaligned.
    ध्रुव तार्किक y_decode_सदिशम्_staging = y_decode_सदिशम्_supported && !y_decode_सदिशम्_aligned;
#अन्यथा
    ध्रुव तार्किक y_decode_सदिशम्_staging = असत्य;
#यदि_अन्तः
    ध्रुव तार्किक can_quantize_y = ctx->device->integer_dot_गुणनफलम् && src1->type == GGML_TYPE_F32 &&
                                ggml_is_contiguous(src1) && (ne11 * ne10) % 4 == 0;
    // Intel coopmat1 matrix cores prefer f16 activations for quantized weights
    ध्रुव तार्किक prefer_f16_b = ctx->device->coopmat_support && !ctx->device->coopmat2 &&
                              ctx->device->vendor_id == VK_VENDOR_ID_INTEL &&
                              ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32;

    ध्रुव ggml_type y_kernel_type = ggml_vk_mul_mat_b_type(ctx, src0->type, src1, f16_type,
                                        (ggml_prec)dst->op_params[0], सत्य, can_quantize_y, prefer_f16_b);

    ध्रुव तार्किक quantize_y   = y_kernel_type == GGML_TYPE_Q8_1;
    ध्रुव तार्किक y_f32_kernel = y_kernel_type == GGML_TYPE_F32;

    ध्रुव std::vector<vk_matmul_pipeline_pair>* mmp_map =
        ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_kernel_type, (ggml_prec)dst->op_params[0], सत्य);

    ध्रुव तार्किक qx_needs_dequant = mmp_map == शून्यसूचकः || x_non_contig;

    यदि (qx_needs_dequant) {
        // dequant src0 to f16
        mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, f16_type, y_kernel_type, (ggml_prec)dst->op_params[0], सत्य);
    }

    GGML_ASSERT(mmp_map != शून्यसूचकः);

    // Coopmat2 MUL_MAT_ID BK specialization constants in ggml_vk_load_shaders are at most 64.
    ध्रुव uint32_t y_staged_row_stride = ctx->device->coopmat2 && !quantize_y ? ggml_vk_align_आकार(ne10, 64) : ne10;
    ध्रुव तार्किक y_needs_k_padding = ne10 != y_staged_row_stride;

    ध्रुव तार्किक y_needs_reformat = !quantize_y && (!ggml_vk_dim01_contiguous(src1) || y_kernel_type != src1->type ||
                                                  y_decode_सदिशम्_staging || y_needs_k_padding);
    ध्रुव तार्किक qy_needs_dequant = y_needs_reformat;

    ध्रुव uint32_t kpad = quantize_y ? 0 : ggml_vk_align_आकार(ne10, ggml_vk_guess_matmul_pipeline_align_map(ctx, *mmp_map, ne01, nei1, सत्य));
    ध्रुव तार्किक aligned = !quantize_y && ne10 == kpad && ne01 > 8 && nei1 > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline_map(ctx, *mmp_map, ne01, nei1, aligned, सत्य);

    ध्रुव uint32_t bn = pipeline->wg_denoms[1];
    ध्रुव uint64_t max_tiles = ggml_vk_mul_mat_id_max_tiles(n_as, nei0 * nei1, bn);
    ध्रुव uint64_t row_ids_विस्थापनम् = 1 + 2 * max_tiles;
    ध्रुव uint64_t row_map_शब्दाः = row_ids_विस्थापनम् + max_tiles * bn;

    यदि (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_बफर_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }
    ध्रुव uint64_t x_ne = ggml_nelements(src0);
    ध्रुव uint64_t y_ne = (uint64_t)y_staged_row_stride * ne11 * ne12 * ne13;
    ध्रुव uint64_t d_ne = ggml_nelements(dst);

    ध्रुव uint64_t qx_sz = ggml_type_आकार(src0->type) * x_ne / ggml_blck_आकार(src0->type);
    ध्रुव uint64_t qy_sz = ggml_type_आकार(src1->type) * ggml_nelements(src1) / ggml_blck_आकार(src1->type);
    ध्रुव uint64_t x_sz = !qx_needs_dequant ? qx_sz : परिमाण(ggml_fp16_t) * x_ne;
    ध्रुव uint64_t y_sz = quantize_y ? (ggml_vk_align_आकार(y_ne, 128) * ggml_type_आकार(GGML_TYPE_Q8_1) / ggml_blck_आकार(GGML_TYPE_Q8_1)) : (y_f32_kernel ? परिमाण(भिन्न) * y_ne : परिमाण(ggml_fp16_t) * y_ne);
    ध्रुव uint64_t d_sz = परिमाण(भिन्न) * d_ne;

    vk_pipeline गन्तव्यम्_fp16_vk_0 = शून्यसूचकः;
    vk_pipeline गन्तव्यम्_fp16_vk_1 = शून्यसूचकः;
    vk_pipeline गन्तव्यम्_q8_1 = शून्यसूचकः;

    स्वचालित make_y_staged_dst = [&]() {
        ggml_tensor y_staged_dst = *src1;
        y_staged_dst.type = f16_type;
        y_staged_dst.nb[0] = ggml_type_आकार(f16_type);
        y_staged_dst.nb[1] = y_staged_dst.nb[0] * y_staged_row_stride;
        y_staged_dst.nb[2] = y_staged_dst.nb[1] * ne11;
        y_staged_dst.nb[3] = y_staged_dst.nb[2] * y_staged_dst.ne[2];
        प्रत्यागम y_staged_dst;
    };

    यदि (x_non_contig) {
        गन्तव्यम्_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, शून्यसूचकः, f16_type);
    } अन्यथा {
        गन्तव्यम्_fp16_vk_0 = ggml_vk_get_गन्तव्यम्_fp16(ctx, src0->type);
    }
    यदि (y_needs_reformat) {
        ggml_tensor y_staged_dst;
        ध्रुव ggml_tensor * y_staged_dst_ptr = शून्यसूचकः;
        यदि (y_needs_k_padding) {
            y_staged_dst = make_y_staged_dst();
            y_staged_dst_ptr = &y_staged_dst;
        }

        गन्तव्यम्_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, y_staged_dst_ptr, f16_type);
    } अन्यथा {
        गन्तव्यम्_fp16_vk_1 = ggml_vk_get_गन्तव्यम्_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || गन्तव्यम्_fp16_vk_0 != शून्यसूचकः);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || गन्तव्यम्_fp16_vk_1 != शून्यसूचकः);  // NOLINT

    यदि (quantize_y) {
        गन्तव्यम्_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }
    vk_pipeline गणनम्_experts = ctx->device->pipeline_गणनम्_experts;

    ध्रुव size_t expert_दत्त_आकार = परिमाण(uint32_t) * row_map_शब्दाः;

    {
        यदि (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_बफर_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_बफर_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        यदि (qx_needs_dequant && ctx->prealloc_आकार_x < x_sz) {
            ctx->prealloc_आकार_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        यदि ((qy_needs_dequant || quantize_y) && ctx->prealloc_आकार_y < y_sz) {
            ctx->prealloc_आकार_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        यदि (ctx->prealloc_आकार_विभज_k < expert_दत्त_आकार) {
            ctx->prealloc_आकार_विभज_k = expert_दत्त_आकार;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        यदि (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, गन्तव्यम्_fp16_vk_0, 1);
        }
        यदि (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, गन्तव्यम्_fp16_vk_1, 1);
        }
        यदि (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, गन्तव्यम्_q8_1, 1);
        }
        ggml_pipeline_request_descriptor_sets(ctx, गणनम्_experts, 1);
    }

    vk_बफर d_D = dst_buf_ctx->dev_बफर;
    ध्रुव uint64_t d_buf_विस्थापनम् = vk_tensor_विस्थापनम्(dst) + dst->view_offs;
    GGML_ASSERT(d_D != शून्यसूचकः);
    vk_बफर d_X;
    uint64_t x_buf_विस्थापनम् = 0;
    vk_बफर d_Y;
    uint64_t y_buf_विस्थापनम् = 0;
    यदि (!src0_uma) {
        d_Qx = src0_buf_ctx->dev_बफर;
        qx_buf_विस्थापनम् = vk_tensor_विस्थापनम्(src0) + src0->view_offs;
        GGML_ASSERT(d_Qx != शून्यसूचकः);
    }
    यदि (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_बफर;
        qy_buf_विस्थापनम् = vk_tensor_विस्थापनम्(src1) + src1->view_offs;
        GGML_ASSERT(d_Qy != शून्यसूचकः);
    }
    यदि (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz);
    } अन्यथा {
        d_X = d_Qx;
        x_buf_विस्थापनम् = qx_buf_विस्थापनम्;
        GGML_ASSERT(qx_sz == x_sz);
    }
    यदि (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz);
    } अन्यथा यदि (quantize_y) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= CEIL_DIV(y_sz, 144) * 144);
    } अन्यथा {
        d_Y = d_Qy;
        y_buf_विस्थापनम् = qy_buf_विस्थापनम्;
        GGML_ASSERT(qy_sz == y_sz);
    }

    यदि (x_non_contig || qx_needs_dequant) {
        यदि (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }
    vk_subbuffer d_ids = ggml_vk_tensor_subbuffer(ctx, ids, सत्य);

    // Count how many times each expert is used
    vk_subbuffer expert_गणनम्_buf = { ctx->prealloc_विभज_k, 0, expert_दत्त_आकार };
    यदि (ctx->prealloc_विभज_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }
    {
        vk_op_गणनम्_experts_प्रक्षिप_constants pc = { (uint32_t)nei0,
                                           (uint32_t)nei1,
                                           (uint32_t)(nbi0 / ggml_type_आकार(ids->type)),
                                           (uint32_t)(nbi1 / ggml_type_आकार(ids->type)),
                                           (uint32_t)(get_misalign_अष्टकानि(ctx, ids) / ggml_type_आकार(ids->type)),
                                           (uint32_t)n_as,
                                           0, 0, bn, (uint32_t)row_ids_विस्थापनम् };
        init_pushconst_fastdiv(pc);
        ggml_vk_dispatch_pipeline(ctx, subctx, गणनम्_experts,
            { d_ids, expert_गणनम्_buf }, pc,
            { 1, 1, 1 });
    }

    यदि (x_non_contig) {
        ggml_vk_cpy_गन्तव्यम्_contiguous(ctx, subctx, गन्तव्यम्_fp16_vk_0, src0, ggml_vk_subbuffer(ctx, d_Qx, qx_buf_विस्थापनम्), ggml_vk_subbuffer(ctx, d_X, 0));
    } अन्यथा यदि (qx_needs_dequant) {
        ध्रुव std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_dispatch_pipeline(ctx, subctx, गन्तव्यम्_fp16_vk_0,
            { vk_subbuffer{ d_Qx, qx_buf_विस्थापनम्, qx_sz }, vk_subbuffer{ d_X, 0, x_sz } }, pc, { (uint32_t)x_ne, 1, 1});
    }
    यदि (y_needs_reformat) {
        यदि (ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् != गन्तव्यम्_fp16_vk_1.get() ||
            ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् != src1 ||
            ctx->prealloc_y_चरमः_k_padded != y_needs_k_padding) {
            यदि (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            यदि (y_needs_k_padding) {
                GGML_ASSERT(y_sz % 4 == 0);
                // Zero B padding because clamping only A can produce 0 * Inf or NaN.
                subctx->s->बफर->buf.fill_बफर(d_Y->बफर, 0, y_sz, 0);
                ggml_vk_sync_buffers(ctx, subctx);
                ध्रुव ggml_tensor y_staged_dst = make_y_staged_dst();
                ध्रुव uint32_t y_staged_dst_type_आकार = ggml_type_आकार(y_staged_dst.type);
                ggml_vk_cpy_गन्तव्यम्_strided(
                    ctx, subctx, गन्तव्यम्_fp16_vk_1, src1,
                    ggml_vk_subbuffer(ctx, d_Qy, qy_buf_विस्थापनम्), ggml_vk_subbuffer(ctx, d_Y, 0),
                    (uint32_t)(y_staged_dst.nb[0] / y_staged_dst_type_आकार),
                    (uint32_t)(y_staged_dst.nb[1] / y_staged_dst_type_आकार),
                    (uint32_t)(y_staged_dst.nb[2] / y_staged_dst_type_आकार),
                    (uint32_t)(y_staged_dst.nb[3] / y_staged_dst_type_आकार));
            } अन्यथा {
                ggml_vk_cpy_गन्तव्यम्_contiguous(ctx, subctx, गन्तव्यम्_fp16_vk_1, src1, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_विस्थापनम्), ggml_vk_subbuffer(ctx, d_Y, 0));
            }
            ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् = गन्तव्यम्_fp16_vk_1.get();
            ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् = src1;
            ctx->prealloc_y_चरमः_k_padded = y_needs_k_padding;
        }
    }
    यदि (quantize_y) {
        यदि (ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् != गन्तव्यम्_q8_1.get() ||
            ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् != src1 ||
            ctx->prealloc_y_चरमः_k_padded) {
            यदि (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_विस्थापनम्), ggml_vk_subbuffer(ctx, d_Y, 0), y_ne);
            ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् = गन्तव्यम्_q8_1.get();
            ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् = src1;
            ctx->prealloc_y_चरमः_k_padded = असत्य;
        }
    }
    ggml_vk_sync_buffers(ctx, subctx);

    ध्रुव uint64_t x_range = qx_needs_dequant ? x_sz : ggml_nbytes(src0);
    ध्रुव uint64_t y_range = (qy_needs_dequant || quantize_y) ? y_sz : ggml_nbytes(src1);

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_b_y = y_needs_k_padding ? y_staged_row_stride : ne10;
    uint32_t stride_batch_y = y_needs_k_padding ? y_staged_row_stride * ne11 : ((qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1));

    यदि (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_आकार(src0->type);
    }

    यदि (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant && !quantize_y) {
        stride_batch_y = src1->nb[0] / ggml_type_आकार(src1->type);
    }

    // compute
    ggml_vk_matmul_id(
        ctx, subctx, pipeline,
        { d_X, x_buf_विस्थापनम्, x_range }, { d_Y, y_buf_विस्थापनम्, y_range },
        { d_D, d_buf_विस्थापनम्, d_sz }, std::move(d_ids), expert_गणनम्_buf,
        ne01, ne21, ne10, ne10, stride_b_y, ne01,
        stride_batch_x, stride_batch_y, ne20*ne21,
        n_as, nei0, nei1, nbi1 / ggml_type_आकार(ids->type), ne11, row_ids_विस्थापनम्, max_tiles
    );  // NOLINT

    यदि (x_non_contig || qx_needs_dequant) {
        ctx->prealloc_x_need_sync = सत्य;
    }
    यदि (y_needs_reformat || quantize_y) {
        ctx->prealloc_y_need_sync = सत्य;
    }
    ctx->prealloc_विभज_k_need_sync = सत्य;
}

स्थिर शून्य ggml_vk_mul_mat_vec_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {
    ggml_tensor * dst = cgraph->nodes[ग्रन्थिः_idx];
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

    ध्रुव uint64_t ne00 = src0->ne[0];
    ध्रुव uint64_t ne01 = src0->ne[1];
    // const uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    ध्रुव uint64_t ne10 = src1->ne[0];
    ध्रुव uint64_t ne11 = src1->ne[1];
    ध्रुव uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    ध्रुव uint64_t nei0 = ids->ne[0];
    ध्रुव uint64_t nei1 = ids->ne[1];
    ध्रुव uint32_t nbi1 = (uint32_t)(ids->nb[1] / परिमाण(पूर्णाङ्क));

    ध्रुव uint64_t ne20 = dst->ne[0];
    ध्रुव uint64_t ne21 = dst->ne[1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    ध्रुव तार्किक x_non_contig = !ggml_vk_dim01_contiguous(src0);
    ध्रुव तार्किक y_non_contig = !ggml_vk_dim01_contiguous(src1);

    ध्रुव तार्किक f16_f32_kernel = src1->type == GGML_TYPE_F32;
    तार्किक quantize_y = ctx->device->integer_dot_गुणनफलम् && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0 && ggml_vk_should_use_mmvq(ctx->device, ne01, ne12, ne10, src0->type);

    vk_pipeline गन्तव्यम्_fp16_vk_0 = शून्यसूचकः;
    vk_pipeline गन्तव्यम्_fp16_vk_1 = शून्यसूचकः;
    यदि (x_non_contig) {
        गन्तव्यम्_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, शून्यसूचकः, src0->type);
    }
    यदि (y_non_contig) {
        गन्तव्यम्_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, शून्यसूचकः, src1->type);
    } अन्यथा {
        गन्तव्यम्_fp16_vk_1 = ggml_vk_get_गन्तव्यम्_fp16(ctx, src1->type);
    }

    // Check for mmq first
    vk_pipeline dmmv = quantize_y ? ggml_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, GGML_TYPE_Q8_1, ne20, ne00) : शून्यसूचकः;
    vk_pipeline गन्तव्यम्_q8_1 = शून्यसूचकः;

    यदि (dmmv == शून्यसूचकः) {
        // Fall back to f16 dequant mul mat
        dmmv = ggml_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, src1->type, ne20, ne00);
        quantize_y = असत्य;
    }

    यदि (quantize_y) {
        गन्तव्यम्_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    ध्रुव तार्किक qx_needs_dequant = x_non_contig;
    ध्रुव तार्किक qy_needs_dequant = !quantize_y && ((src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig);

    यदि (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_बफर_Range) {
        dmmv = ggml_vk_get_64b_indexing_pipeline(ctx, dmmv);
    }

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT
    GGML_ASSERT(!qx_needs_dequant || गन्तव्यम्_fp16_vk_0 != शून्यसूचकः);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || गन्तव्यम्_fp16_vk_1 != शून्यसूचकः);  // NOLINT
    GGML_ASSERT(dmmv != शून्यसूचकः);

    ध्रुव uint64_t x_ne = ggml_nelements(src0);
    ध्रुव uint64_t y_ne = ggml_nelements(src1);

    ध्रुव uint64_t qx_sz = ggml_vk_align_आकार(ggml_type_आकार(src0->type) * x_ne / ggml_blck_आकार(src0->type), ctx->device->properties.limits.min_Storage_बफर_विस्थापनम्_Alignment);
    ध्रुव uint64_t x_sz = x_non_contig ? ggml_vk_align_आकार(ggml_type_आकार(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_बफर_विस्थापनम्_Alignment) : qx_sz;
    ध्रुव uint64_t y_sz = quantize_y ? (ggml_vk_align_आकार(y_ne, 128) * ggml_type_आकार(GGML_TYPE_Q8_1) / ggml_blck_आकार(GGML_TYPE_Q8_1)) :
                                       (f16_f32_kernel ? परिमाण(भिन्न) * y_ne : परिमाण(ggml_fp16_t) * y_ne);

    {
        यदि (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_बफर_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_बफर_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        यदि (qx_needs_dequant && ctx->prealloc_आकार_x < x_sz) {
            ctx->prealloc_आकार_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        यदि ((qy_needs_dequant || quantize_y) && ctx->prealloc_आकार_y < y_sz) {
            ctx->prealloc_आकार_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        यदि (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, गन्तव्यम्_fp16_vk_0, 1);
        }
        यदि (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, गन्तव्यम्_fp16_vk_1, 1);
        }
        यदि (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, गन्तव्यम्_q8_1, 1);
        }
        ggml_pipeline_request_descriptor_sets(ctx, dmmv, nei1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[ग्रन्थिः_idx + ctx->अङ्कः_additional_fused_ops]);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer d_ids = ggml_vk_tensor_subbuffer(ctx, ids);
    vk_subbuffer d_F0 = d_D;
    vk_subbuffer d_X, d_Y;

    यदि (qx_needs_dequant) {
        d_X = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    } अन्यथा {
        d_X = d_Qx;
    }
    यदि (qy_needs_dequant || quantize_y) {
        d_Y = { ctx->prealloc_y, 0, ctx->prealloc_y->size };
    } अन्यथा {
        d_Y = d_Qy;
    }

    यदि (x_non_contig) {
        यदि (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    यदि (x_non_contig) {
        GGML_ASSERT(x_sz == ggml_vk_align_आकार(ggml_type_आकार(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_बफर_विस्थापनम्_Alignment));
        ggml_vk_cpy_गन्तव्यम्_contiguous(ctx, subctx, गन्तव्यम्_fp16_vk_0, src0, d_Qx, d_X);
    }
    यदि (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_type_आकार(src1->type) * y_ne);
        यदि (ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् != गन्तव्यम्_fp16_vk_1.get() ||
            ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् != src1 ||
            ctx->prealloc_y_चरमः_k_padded) {
            यदि (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_गन्तव्यम्_contiguous(ctx, subctx, गन्तव्यम्_fp16_vk_1, src1, d_Qy, d_Y);
            ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् = गन्तव्यम्_fp16_vk_1.get();
            ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् = src1;
            ctx->prealloc_y_चरमः_k_padded = असत्य;
        }
    }
    यदि (quantize_y) {
        यदि (ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् != गन्तव्यम्_q8_1.get() ||
            ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् != src1 ||
            ctx->prealloc_y_चरमः_k_padded) {
            यदि (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, d_Qy, d_Y, y_ne);
            ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् = गन्तव्यम्_q8_1.get();
            ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् = src1;
            ctx->prealloc_y_चरमः_k_padded = असत्य;
        }
    }

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_batch_y = (qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1);

    यदि (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[2] / ggml_type_आकार(src1->type);
    }

    ध्रुव uint32_t max_groups_x = ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    यदि (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = CEIL_DIV(groups_x, groups_z);
    }

    uint32_t fusion_flags = 0;

    यदि (ctx->अङ्कः_additional_fused_ops > 0) {
        ध्रुव ggml_tensor * bias = cgraph->nodes[ग्रन्थिः_idx + 1]->src[1];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);

        यदि (cgraph->nodes[ग्रन्थिः_idx + 1]->op == GGML_OP_MUL) {
            fusion_flags |= MAT_VEC_FUSION_FLAGS_SCALE0;
        } अन्यथा {
            GGML_ASSERT(cgraph->nodes[ग्रन्थिः_idx + 1]->op == GGML_OP_सङ्कलय_ID);
            fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
        }
    }

    vk_subbuffer d_F1 = d_D;
    यदि (ctx->अङ्कः_additional_fused_ops > 1) {
        ध्रुव ggml_tensor * scale = cgraph->nodes[ग्रन्थिः_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, scale);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_SCALE1;
    }

    // Loop over the batch dimension
    प्रति (uint32_t expert_i1 = 0; expert_i1 < nei1; ++expert_i1) {
        ध्रुव vk_mat_vec_id_प्रक्षिप_constants pc = {
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

    यदि (x_non_contig) {
        ctx->prealloc_x_need_sync = सत्य;
    }
    यदि (y_non_contig || quantize_y) {
        ctx->prealloc_y_need_sync = सत्य;
    }
}

स्थिर तार्किक ggml_vk_use_mul_mat_vec_id(ध्रुव ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * ids = dst->src[2];
    प्रत्यागम ids->ne[1] <= 8 && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type));
}

तार्किक ggml_vk_use_mul_mat_vec_id(ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {
    प्रत्यागम ggml_vk_use_mul_mat_vec_id(cgraph->nodes[ग्रन्थिः_idx]);
}

शून्य ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {
    ggml_tensor * dst = cgraph->nodes[ग्रन्थिः_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    ggml_tensor * src2 = dst->src[2];
    VK_LOG_DEBUG("ggml_vk_mul_mat_id(" << src0 << ", " << src1 << ", " << src2 << ", " << dst << ")");
    यदि (ggml_vk_use_mul_mat_vec_id(cgraph, ग्रन्थिः_idx)) {
        ggml_vk_mul_mat_vec_id_q_f16(ctx, subctx, cgraph, ग्रन्थिः_idx);
    } अन्यथा {
        ggml_vk_mul_mat_id_q_f16(ctx, subctx, src0, src1, src2, dst);
    }
}

तार्किक ggml_vk_flash_attn_scalar_shmem_support(ध्रुव vk_device& device, ध्रुव vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, तार्किक f32acc, ggml_type k_type, ggml_type v_type) {
    GGML_अप्रयुक्तम्(f32acc);
    // Needs to be kept up to date on shader changes
    ध्रुव uint32_t wg_आकार = params.workgroup_आकार;
    ध्रुव uint32_t Br = params.खण्डम्_पङ्क्तिकाः;
    ध्रुव uint32_t Bc = params.खण्डम्_स्तम्भसङ्ख्या;

    // BF16 uses the fp32 shader (FLOAT_TYPE=float)
    ध्रुव uint32_t float_type_आकार = (device->fp16 && k_type != GGML_TYPE_BF16) ? परिमाण(ggml_fp16_t) : परिमाण(भिन्न);

    ध्रुव तार्किक mmq = ggml_vk_fa_scalar_उपयुङ्क्ते_mmq(device, k_type, v_type);

    // tmpsh is overestimated slightly
    ध्रुव uint32_t tmpsh = wg_आकार * परिमाण(भिन्न);
    ध्रुव uint32_t tmpshv4 = wg_आकार * 4 * float_type_आकार;

    ध्रुव uint32_t masksh = Bc * (Br + 1) * float_type_आकार;
    // DATA_A_IQ4_NL is compiled into the FA shaders unconditionally, so its shared table is always allocated.
    ध्रुव uint32_t iq_shmem = 16 * float_type_आकार;

    uint32_t Qf, kvsh, kblocksh_आकार;
    यदि (mmq) {
        // block_b_cache: int32_t qs[8] + FLOAT_TYPEV2 ds
        ध्रुव uint32_t खण्डम्_b_आकार = 8 * परिमाण(int32_t) + 2 * float_type_आकार;
        Qf = Br * (hsk / 32) * खण्डम्_b_आकार;

        // kvsh uses D = HSV (K goes through kblocksh instead)
        kvsh = params.shmem_staging ? Bc * (hsv / 4 + 1) * 4 * float_type_आकार : 4 * float_type_आकार;

        // The mixed MMQ shader uses a superset block_a_cache that fits every
        // FA-supported quant: int32_t qs[8] + uint32_t qh + FLOAT_TYPEV2 dm.
        // Single-scale types leave dm.y unused; non-Q5_* leave qh unused.
        ध्रुव uint32_t खण्डम्_a_आकार = 8 * परिमाण(int32_t) + परिमाण(uint32_t) + 2 * float_type_आकार;
        kblocksh_आकार = params.shmem_staging ? Bc * (hsk / 32) * खण्डम्_a_आकार : खण्डम्_a_आकार;
    } अन्यथा {
        Qf = Br * (hsk / 4 + 1) * 4 * float_type_आकार;

        ध्रुव uint32_t D = std::max(hsk, hsv);
        kvsh = params.shmem_staging ? Bc * (D / 4 + 1) * 4 * float_type_आकार : 4 * float_type_आकार;

        kblocksh_आकार = 0;
    }

    ध्रुव uint32_t योग_आकार = tmpsh + tmpshv4 + masksh + iq_shmem + Qf + kvsh + kblocksh_आकार;
    ध्रुव तार्किक supported = योग_आकार <= device->properties.limits.max_Compute_Shared_स्मृतिः_आकार;

    VK_LOG_DEBUG("ggml_vk_flash_attn_scalar_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", mmq=" << mmq << ", total_size=" << योग_आकार << ", supported=" << supported);

    प्रत्यागम supported;
}

तार्किक ggml_vk_flash_attn_coopmat_shmem_support(ध्रुव vk_device& device, ध्रुव vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, तार्किक f32acc, ggml_type k_type, ggml_type v_type) {
    GGML_अप्रयुक्तम्(v_type);
    // Needs to be kept up to date on shader changes
    ध्रुव uint32_t Br = params.खण्डम्_पङ्क्तिकाः;
    ध्रुव uint32_t Bc = params.खण्डम्_स्तम्भसङ्ख्या;

    ध्रुव uint32_t MatBr = 16, MatBc = 16;

    ध्रुव uint32_t row_विभज = Bc / MatBc;

    ध्रुव uint32_t hsk_pad = ROUNDUP_POW2(hsk, 16);
    ध्रुव uint32_t hsv_pad = ROUNDUP_POW2(hsv, 16);

    ध्रुव uint32_t acctype = f32acc ? 4 : 2;
    ध्रुव uint32_t f16vec4 = 8;

    ध्रुव uint32_t tmpsh = (Bc / MatBc) * परिमाण(भिन्न);
    // DATA_A_IQ4_NL is compiled into the FA shaders unconditionally, so its shared table is always allocated.
    ध्रुव uint32_t iq_shmem = 16 * परिमाण(ggml_fp16_t);

    ध्रुव uint32_t qstride = hsk_pad / 4 + 2;
    ध्रुव uint32_t Qf = Br * qstride * f16vec4;

    ध्रुव uint32_t psh_stride = Br / 4 + 2;
    ध्रुव uint32_t Psh = Bc * psh_stride * f16vec4;

    ध्रुव uint32_t sfshstride = (hsk <= 128) ? (Br + 8) : Br;
    ध्रुव uint32_t sfsh = Bc * sfshstride * acctype;

    ध्रुव uint32_t kvshstride = (params.shmem_staging ? std::max(hsk_pad, hsv_pad) : MatBr) / 4 + 2;
    ध्रुव uint32_t vsh_stride = MatBc / 4 * row_विभज;
    ध्रुव uint32_t ksh = ((kvshstride >= vsh_stride) ? (Bc * kvshstride) : (Bc * vsh_stride)) * f16vec4;

    // BF16 PVMat accumulator is f32 (no bf16 accumulator support), so pvsh is vec4 (16 bytes)
    ध्रुव uint32_t pvsh_elem_आकार = (k_type == GGML_TYPE_BF16) ? 16u : f16vec4;
    ध्रुव uint32_t osh_stride = params.row_विभज * MatBr / 4;
    ध्रुव uint32_t pvsh = MatBc * osh_stride * pvsh_elem_आकार;

    ध्रुव uint32_t नतिः = Br * acctype;

    ध्रुव uint32_t योग_आकार = tmpsh + iq_shmem + Qf + Psh + sfsh + ksh + pvsh + नतिः;
    ध्रुव तार्किक supported = योग_आकार <= device->properties.limits.max_Compute_Shared_स्मृतिः_आकार;

    VK_LOG_DEBUG("ggml_vk_flash_attn_coopmat_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", f32acc=" << f32acc << ", total_size=" << योग_आकार << ", supported=" << supported);

    प्रत्यागम supported;
}

शून्य ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * q, ध्रुव ggml_tensor * k, ध्रुव ggml_tensor * v, ध्रुव ggml_tensor * आवरणम्, ध्रुव ggml_tensor * sinks, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_flash_attn((" << q << ", name=" << q->name << ", type=" << q->type << ", ne0=" << q->ne[0] << ", ne1=" << q->ne[1] << ", ne2=" << q->ne[2] << ", ne3=" << q->ne[3] << ", nb0=" << q->nb[0] << ", nb1=" << q->nb[1] << ", nb2=" << q->nb[2] << ", nb3=" << q->nb[3];
    std::cerr << "), (" << k << ", name=" << k->name << ", type=" << k->type << ", ne0=" << k->ne[0] << ", ne1=" << k->ne[1] << ", ne2=" << k->ne[2] << ", ne3=" << k->ne[3] << ", nb0=" << k->nb[0] << ", nb1=" << k->nb[1] << ", nb2=" << k->nb[2] << ", nb3=" << k->nb[3];
    std::cerr << "), (" << v << ", name=" << v->name << ", type=" << v->type << ", ne0=" << v->ne[0] << ", ne1=" << v->ne[1] << ", ne2=" << v->ne[2] << ", ne3=" << v->ne[3] << ", nb0=" << v->nb[0] << ", nb1=" << v->nb[1] << ", nb2=" << v->nb[2] << ", nb3=" << v->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    यदि (sinks) {
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

    ध्रुव uint32_t nem0 = आवरणम् ? आवरणम्->ne[0] : 0;
    ध्रुव uint32_t nem1 = आवरणम् ? आवरणम्->ne[1] : 0;
    ध्रुव uint32_t nem2 = आवरणम् ? आवरणम्->ne[2] : 0;
    ध्रुव uint32_t nem3 = आवरणम् ? आवरणम्->ne[3] : 0;

    ध्रुव uint32_t HSK = nek0;
    ध्रुव uint32_t HSV = nev0;
    uint32_t N = neq1;
    ध्रुव uint32_t KV = nek1;

    GGML_ASSERT(ne0 == HSV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_आकार(q->type));
    GGML_ASSERT(nbk0 == ggml_type_आकार(k->type));
    GGML_ASSERT(nbv0 == ggml_type_आकार(v->type));

    GGML_ASSERT(neq0 == HSK);

    GGML_ASSERT(neq1 == N);

    GGML_ASSERT(nev1 == nek1);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == परिमाण(भिन्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    प्रतिज्ञा(dst->type == GGML_TYPE_F32);
    प्रतिज्ञा(q->type == GGML_TYPE_F32);
    uint32_t gqa_अनुपातः = 1;
    uint32_t qk_अनुपातः = neq2 / nek2;
    uint32_t workgroups_x = (uint32_t)neq1;
    uint32_t workgroups_y = (uint32_t)neq2;
    uint32_t workgroups_z = (uint32_t)neq3;

    ध्रुव तार्किक f32acc = !ctx->device->fp16 || dst->op_params[3] == GGML_PREC_F32 || k->type == GGML_TYPE_BF16;

    // dequant K/V once into an f16 scratch, reordered KV layout so FA can read without a stride
    स्वचालित is_dense_kv_cache = [](ध्रुव ggml_tensor * t) {
        प्रत्यागम t->nb[0] == ggml_type_आकार(t->type) &&
               t->nb[2] == ggml_row_आकार(t->type, t->ne[0]) &&
               t->nb[1] == t->nb[2] * t->ne[2] &&
               (t->ne[3] == 1 || t->nb[3] == t->nb[1] * t->ne[1]);
    };
    ध्रुव तार्किक k_quant = k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_BF16 && k->type != GGML_TYPE_F32;
    ध्रुव तार्किक v_quant = v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_BF16 && v->type != GGML_TYPE_F32;
    ध्रुव तार्किक kv_raw = (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16) && k->type == v->type;
    // gated to AMD, except RDNA4, which regressed in benchmarks
    ध्रुव तार्किक kv_raw_strided = ctx->device->vendor_id == VK_VENDOR_ID_AMD &&
                                ctx->device->architecture != vk_device_architecture::AMD_RDNA4 &&
                                neq1 >= 256 &&
                                kv_raw &&
                                (k->nb[1] != (uint64_t)HSK * ggml_type_आकार(k->type) ||
                                 v->nb[1] != (uint64_t)HSV * ggml_type_आकार(v->type)) &&
                                (HSK % 8) == 0 && (HSV % 8) == 0;
    ध्रुव size_t kv_scratch_ts = kv_raw_strided ? ggml_type_आकार(k->type) : परिमाण(ggml_fp16_t);
    ध्रुव तार्किक use_dequant_kv = ((k_quant && v_quant) || kv_raw_strided) && neq1 >= 64 &&
                                is_dense_kv_cache(k) && is_dense_kv_cache(v) &&
                                (uint64_t)ggml_nelements(k) * kv_scratch_ts <= ctx->device->properties.limits.max_Storage_बफर_Range &&
                                (uint64_t)ggml_nelements(v) * kv_scratch_ts <= ctx->device->properties.limits.max_Storage_बफर_Range &&
                                (kv_raw_strided ||
                                 (ctx->device->pipeline_dequant_transpose[k->type] != शून्यसूचकः &&
                                  ctx->device->pipeline_dequant_transpose[v->type] != शून्यसूचकः)) &&
                                // coopmat2 path does not benefit from the f16 scratch
                                !ctx->device->coopmat2 &&
                                // Intel Xe1 regresses, see PR 25494
                                (ctx->device->vendor_id != VK_VENDOR_ID_INTEL ||
                                 (ctx->device->coopmat_support && ctx->device->architecture != vk_device_architecture::INTEL_XE1));
    ध्रुव ggml_type k_type_eff = (use_dequant_kv && !kv_raw_strided) ? GGML_TYPE_F16 : k->type;
    ध्रुव ggml_type v_type_eff = (use_dequant_kv && !kv_raw_strided) ? GGML_TYPE_F16 : v->type;

    // For scalar/coopmat1 FA, we can use the "large" size to accommodate qga.
    // For coopmat2 FA, we always use the small size (which is still pretty large for gqa).
    vk_fa_tuning_params tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, 512, KV, k_type_eff, v_type_eff, f32acc);
    ध्रुव uint32_t max_gqa = std::min(tuning_params.खण्डम्_पङ्क्तिकाः, 32u);

    यदि (N <= 8 && qk_अनुपातः > 1 && qk_अनुपातः <= max_gqa &&
        qk_अनुपातः * nek2 == neq2 && nek2 == nev2 && nem2 <= 1) {
        // grouped query attention - make the N dimension equal to gqa_ratio, reduce
        // workgroups proportionally in y dimension. The shader will detect gqa_ratio > 1
        // and change addressing calculations to index Q's dimension 2.
        gqa_अनुपातः = qk_अनुपातः;
        N = gqa_अनुपातः;
        workgroups_y /= gqa_अनुपातः;
    }

    tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, N, KV, k_type_eff, v_type_eff, f32acc);

    भिन्न scale         = 1.0f;
    भिन्न max_bias      = 0.0f;
    भिन्न logit_softcap = 0.0f;

    स्मृतिं_प्रतिलिख(&scale,         (ध्रुव भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&max_bias,      (ध्रुव भिन्न *) dst->op_params + 1, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&logit_softcap, (ध्रुव भिन्न *) dst->op_params + 2, परिमाण(भिन्न));

    यदि (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    // Sparse mask hint (op_params[4]): compact the <= n_kv_max finite positions and gather only those.
    ध्रुव int32_t n_kv_max = आवरणम् ? ggml_get_op_params_i32(dst, 4) : 0;
    स्थिर ध्रुव तार्किक disable_sparse = getenv("GGML_VK_FA_SPARSE_DISABLE") != शून्यसूचकः;
    ध्रुव तार्किक kv_f16 = k_type_eff == GGML_TYPE_F16 && v_type_eff == GGML_TYPE_F16;
    // cm2 dense is fast, so it needs a larger reduction to win.
    // With quantized K/V, sparse only breaks even around 16x (measured on RDNA3/RDNA4).
    ध्रुव int64_t min_अनुपातः = tuning_params.path == FA_COOPMAT2 ? 4 : (kv_f16 ? 2 : 16);
    // coopmat2 vector decode requires 8B strides.
    स्वचालित sparse_gather_aligned = [](ध्रुव ggml_tensor * t) {
        प्रत्यागम (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) ||
               (t->nb[1] | t->nb[2] | t->nb[3]) % (4 * परिमाण(ggml_fp16_t)) == 0;
    };
    ध्रुव तार्किक use_sparse = !disable_sparse && n_kv_max > 0 && आवरणम् &&
                            max_bias == 0.0f && logit_softcap == 0.0f &&
                            (tuning_params.path != FA_COOPMAT2 || (sparse_gather_aligned(k) && sparse_gather_aligned(v))) &&
                            nem0 == KV &&
                            (int64_t)KV >= std::max<int64_t>(4096, min_अनुपातः * (int64_t)n_kv_max) &&
                            (gqa_अनुपातः > 1 || (tuning_params.path == FA_SCALAR && N == 1));

    ध्रुव uint32_t q_stride = (uint32_t)(nbq1 / ggml_type_आकार(q->type));
    uint32_t k_stride = (uint32_t)(nbk1 / ggml_type_आकार(k->type));
    uint32_t v_stride = (uint32_t)(nbv1 / ggml_type_आकार(v->type));

    // For F32, the shader treats it as a block of size 4 (for vec4 loads)
    यदि (k->type == GGML_TYPE_F32) {
        k_stride /= 4;
    }
    यदि (v->type == GGML_TYPE_F32) {
        v_stride /= 4;
    }

    uint32_t nbk2_eff = (uint32_t)nbk2, nbk3_eff = (uint32_t)nbk3;
    uint32_t nbv2_eff = (uint32_t)nbv2, nbv3_eff = (uint32_t)nbv3;
    यदि (use_dequant_kv) {
        k_stride = HSK;
        v_stride = HSV;
        nbk2_eff = (uint32_t)((uint64_t)HSK * KV * परिमाण(ggml_fp16_t));
        nbk3_eff = (uint32_t)((uint64_t)HSK * KV * nek2 * परिमाण(ggml_fp16_t));
        nbv2_eff = (uint32_t)((uint64_t)HSV * KV * परिमाण(ggml_fp16_t));
        nbv3_eff = (uint32_t)((uint64_t)HSV * KV * nev2 * परिमाण(ggml_fp16_t));
    }
    ध्रुव uint32_t alignment = tuning_params.खण्डम्_स्तम्भसङ्ख्या;
    तार्किक aligned = (KV % alignment) == 0 &&
                   // the "aligned" shader variant will forcibly align strides, for performance
                   (q_stride & 7) == 0 && (k_stride & 7) == 0 && (v_stride & 7) == 0;

    // Need to use the coopmat2 variant that clamps loads when HSK/HSV aren't sufficiently aligned.
    यदि (((HSK | HSV) % 16) != 0 && tuning_params.path == FA_COOPMAT2) {
        aligned = असत्य;
    }

    // Only use mask opt when the mask is fairly large. This hasn't been tuned extensively.
    तार्किक use_आवरणम्_opt = आवरणम् && !use_sparse && nem1 >= 32 && nem0 * nem1 > 32768 && nem0 >= tuning_params.खण्डम्_स्तम्भसङ्ख्या * 16
                        && (ctx->device->architecture != vk_device_architecture::AMD_GCN || HSK > 256 || HSV > 256);
    vk_fa_pipeline_state fa_pipeline_state = get_fa_pipeline_state(ctx->device, tuning_params, HSK, HSV, aligned, f32acc,
                                                                   आवरणम् != शून्यसूचकः, use_आवरणम्_opt, logit_softcap != 0, use_sparse, k_type_eff, v_type_eff);

    vk_pipeline pipeline = शून्यसूचकः;

    तार्किक xe_fa_opt = असत्य;
    तार्किक fa_copy_qstate = असत्य;
    तार्किक xe_fa_supported_platform =
        (ctx->device.get()->architecture == INTEL_XE2 && ctx->device.get()->properties.deviceID != 0xFD80 && ctx->device.get()->properties.deviceID != 0xFD81) ||
        (ctx->device.get()->architecture == INTEL_XE1 && ctx->device.get()->coopmat_support && ctx->device.get()->uma);
    तार्किक xe_fa_supported_usage = neq0 % 32 == 0 && nev0 % 16 == 0 && q->nb[1] > q->nb[2] && k->nb[1] > k->nb[2] && v->nb[1] > v->nb[2] && आवरणम् != शून्यसूचकः;
    तार्किक xe_fa_supported_dtype = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 && (आवरणम् != शून्यसूचकः && आवरणम्->type == GGML_TYPE_F16);
    std::pair<vk_pipeline, vk_pipeline> xe_fa_pipeline_dual_phases = { शून्यसूचकः , शून्यसूचकः };
    vk_pipeline xe_fa_pipeline = शून्यसूचकः;
    size_t आकार_p = 0;
    size_t आकार_गणः_max = 0;

    {
        std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
        स्वचालित &pipelines = ctx->device->pipeline_flash_attn_f32_f16;
        स्वचालित it = pipelines.find(fa_pipeline_state);
        यदि (it != pipelines.end()) {
            pipeline = it->second;
        } अन्यथा {
            pipelines[fa_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
        }
    }

    प्रतिज्ञा(pipeline);
    // Compile early to initialize wg_denoms.
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    uint32_t विभज_kv = KV;
    uint32_t विभज_k = 1;

    // Intel Alchemist prefers more workgroups
    ध्रुव uint32_t shader_core_गणनम्_multiplier = (ctx->device->vendor_id == VK_VENDOR_ID_INTEL && ctx->device->architecture != INTEL_XE2) ? 2 : 1;

    // Use a placeholder core count if one isn't available. split_k is a big help for perf.
    ध्रुव uint32_t shader_core_गणनम् = ctx->device->shader_core_गणनम् ? ctx->device->shader_core_गणनम् * shader_core_गणनम्_multiplier : 16;

    ध्रुव uint32_t Br = fa_pipeline_state.Br;
    ध्रुव uint32_t Bc = fa_pipeline_state.Bc;

    GGML_ASSERT(Br == pipeline->wg_denoms[0]);
    ध्रुव uint32_t Tr = CEIL_DIV(N, Br);

    // Try to use split_k when KV is large enough to be worth the overhead.
    // Sparse: split_kv carries n_kv_max, split_k partitions its blocks for occupancy.
    यदि (use_sparse) {
        विभज_kv = (uint32_t)n_kv_max;
        ध्रुव uint32_t योग_खण्डानि = CEIL_DIV((uint32_t)n_kv_max, Bc);
        ध्रुव uint32_t आधारः_wgs = (gqa_अनुपातः > 1 ? workgroups_x : Tr) * workgroups_y * workgroups_z;
        यदि (आधारः_wgs < shader_core_गणनम् * 2) {
            विभज_k = shader_core_गणनम् * 2 / आधारः_wgs;
        }
        विभज_k = std::max(1u, std::min(विभज_k, योग_खण्डानि));
        // Match the shader's per-split block count so no split is empty.
        ध्रुव uint32_t प्रत्येकम्_खण्डानि = CEIL_DIV(योग_खण्डानि, विभज_k);
        विभज_k = CEIL_DIV(योग_खण्डानि, प्रत्येकम्_खण्डानि);
    } अन्यथा यदि (gqa_अनुपातः > 1 && workgroups_x <= Br) {
        विभज_k = shader_core_गणनम् * 2 / (workgroups_x * workgroups_y * workgroups_z);
    } अन्यथा यदि (gqa_अनुपातः <= 1) {
        uint32_t योग_wgs_no_विभज = Tr * workgroups_y * workgroups_z;
        यदि (योग_wgs_no_विभज < shader_core_गणनम् * 2) {
            विभज_k = shader_core_गणनम् * 2 / योग_wgs_no_विभज;
        }
    }

    यदि (!use_sparse && विभज_k > 1) {
        // Try to evenly split KV into split_k chunks, but it needs to be a multiple
        // of "align", so recompute split_k based on that.
        विभज_kv = ROUNDUP_POW2(std::max(1u, KV / विभज_k), alignment);
        विभज_k = CEIL_DIV(KV, विभज_kv);
        xe_fa_opt = xe_fa_supported_platform && xe_fa_supported_usage && xe_fa_supported_dtype;
        यदि (xe_fa_opt) {
            ध्रुव uint32_t विभज_p_आकार = 32;
            ध्रुव size_t max_dim = (nek1 + विभज_p_आकार - 1) / विभज_p_आकार;
            ध्रुव size_t p_dim = max_dim * विभज_p_आकार;
#यदि defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
            स्वचालित& pipelines = ctx->device->pipeline_xe_fa_decode_dual_phases;
            स्वचालित it = pipelines.find({ (uint32_t)neq0, (uint32_t)nev0, qk_अनुपातः, (uint32_t)neq1 });
            यदि (it != pipelines.end()) {
                xe_fa_pipeline_dual_phases = it->second;
            } अन्यथा {
                pipelines[{(uint32_t)neq0, (uint32_t)nev0, qk_अनुपातः, (uint32_t)neq1}] = xe_fa_pipeline_dual_phases = std::make_pair(std::make_shared<vk_pipeline_struct>(), std::make_shared<vk_pipeline_struct>());
            }
#यदि_अन्तः
            यदि (xe_fa_pipeline_dual_phases.first == शून्यसूचकः || xe_fa_pipeline_dual_phases.second == शून्यसूचकः) {
                xe_fa_opt = असत्य;
                fa_copy_qstate = असत्य;
            } अन्यथा {
                आकार_p = neq1 * neq2 * p_dim * neq3 * परिमाण(ggml_fp16_t);
                आकार_गणः_max = neq1 * neq2 * max_dim * neq3 * परिमाण(भिन्न);
                size_t अस्थायी_आकार = ggml_nelements(q) * परिमाण(ggml_fp16_t) + आकार_p + आकार_गणः_max;
                fa_copy_qstate = सत्य;
                यदि (ctx->prealloc_आकार_x < अस्थायी_आकार) {
                    ctx->prealloc_आकार_x = अस्थायी_आकार;
                    ggml_vk_preallocate_buffers(ctx, subctx);
                }

                यदि (ctx->prealloc_x_need_sync) {
                    ggml_vk_sync_buffers(ctx, subctx);
                }
            }
        }
    }

    यदि (xe_fa_opt == सत्य) {
        use_आवरणम्_opt = असत्य;
    }

    // Reserve space for split_k temporaries. For each split x batch, we need to store the O matrix (D x ne1)
    // and the per-row m and L values (ne1 rows). We store all the matrices first, followed by the rows.
    // For matrices, the order is (inner to outer) [HSV, ne1, k, ne2, ne3].
    // For L/M, the order is (inner to outer) [ne1, k, ne2, ne3].
    ध्रुव uint64_t विभज_k_आकार = विभज_k > 1 ? (HSV * ne1 * परिमाण(भिन्न) + ne1 * परिमाण(भिन्न) * 2) * विभज_k * ne2 * ne3 : 0;
    यदि (विभज_k_आकार > ctx->device->properties.limits.max_Storage_बफर_Range) {
        GGML_ABORT("Requested preallocation size is too large");
    }
    यदि (ctx->prealloc_आकार_विभज_k < विभज_k_आकार) {
        ctx->prealloc_आकार_विभज_k = विभज_k_आकार;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }

    ध्रुव uint32_t आवरणम्_opt_अङ्कः_dwords = CEIL_DIV(nem0, 16 * Bc);
    ध्रुव uint64_t आवरणम्_opt_आकार = परिमाण(uint32_t) * आवरणम्_opt_अङ्कः_dwords * CEIL_DIV(nem1, Br) * nem2 * nem3;

    vk_pipeline pipeline_fa_आवरणम्_opt = शून्यसूचकः;
    यदि (use_आवरणम्_opt) {
        {
            std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
            स्वचालित &pipelines = ctx->device->pipeline_fa_आवरणम्_opt;
            स्वचालित it = pipelines.find({Br, Bc});
            यदि (it != pipelines.end()) {
                pipeline_fa_आवरणम्_opt = it->second;
            } अन्यथा {
                pipelines[{Br, Bc}] = pipeline_fa_आवरणम्_opt = std::make_shared<vk_pipeline_struct>();
            }
        }
        प्रतिज्ञा(pipeline_fa_आवरणम्_opt);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline_fa_आवरणम्_opt, 1);

        यदि (ctx->prealloc_आकार_y < आवरणम्_opt_आकार) {
            ctx->prealloc_आकार_y = आवरणम्_opt_आकार;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        यदि (ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    // Sparse index scratch reuses prealloc_y (mutually exclusive with mask opt).
    ध्रुव uint64_t sparse_idx_आकार = use_sparse
        ? परिमाण(int32_t) * (uint64_t)n_kv_max * nem1 * nem2 * nem3
        : 0;
    vk_pipeline sparse_compact_pipeline = ctx->device->fa_sparse_compact_use_subgroups
        ? ctx->device->pipeline_fa_sparse_compact_subgroup
        : ctx->device->pipeline_fa_sparse_compact;
    यदि (use_sparse) {
        ggml_pipeline_request_descriptor_sets(ctx, sparse_compact_pipeline, 1);
        यदि (ctx->prealloc_आकार_y < sparse_idx_आकार) {
            ctx->prealloc_आकार_y = sparse_idx_आकार;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        यदि (ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    ध्रुव uint32_t n_शीर्ष_kv   = neq2;
    ध्रुव uint32_t n_शीर्ष_log2 = 1u << (uint32_t) floorf(log2f((भिन्न) n_शीर्ष_kv));
    ध्रुव भिन्न m0 = powf(2.0f, -(max_bias       ) / n_शीर्ष_log2);
    ध्रुव भिन्न m1 = powf(2.0f, -(max_bias / 2.0f) / n_शीर्ष_log2);

    vk_subbuffer q_buf = ggml_vk_tensor_subbuffer(ctx, q);
    vk_subbuffer k_buf = ggml_vk_tensor_subbuffer(ctx, k);
    vk_subbuffer v_buf = ggml_vk_tensor_subbuffer(ctx, v);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer आवरणम्_buf = आवरणम् ? ggml_vk_tensor_subbuffer(ctx, आवरणम्) : q_buf;
    vk_subbuffer sinks_buf = sinks ? ggml_vk_tensor_subbuffer(ctx, sinks) : q_buf;
    vk_subbuffer आवरणम्_opt_buf = use_आवरणम्_opt ? ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0) : q_buf;
    vk_subbuffer sparse_buf = use_sparse ? ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0) : q_buf;
    यदि (use_आवरणम्_opt || use_sparse) {
        // the mask opt bits and the sparse index list overwrite a matmul input converted into prealloc_y
        ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् = शून्यसूचकः;
        ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् = शून्यसूचकः;
    }

    यदि (use_dequant_kv) {
        ध्रुव uint64_t k_sz = (uint64_t)ggml_nelements(k) * kv_scratch_ts;
        ध्रुव uint64_t v_sz = (uint64_t)ggml_nelements(v) * kv_scratch_ts;
        यदि (ctx->prealloc_आकार_x < k_sz + v_sz) {
            ctx->prealloc_आकार_x = k_sz + v_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        vk_subbuffer k_dst = vk_subbuffer{ ctx->prealloc_x, 0,    k_sz };
        vk_subbuffer v_dst = vk_subbuffer{ ctx->prealloc_x, k_sz, v_sz };
        ध्रुव uint32_t k_nel = (uint32_t)ggml_nelements(k);
        ध्रुव uint32_t v_nel = (uint32_t)ggml_nelements(v);
        यदि (kv_raw_strided) {
            vk_pipeline cp_k = ggml_vk_get_cpy_pipeline(ctx, k, शून्यसूचकः, k->type);
            vk_pipeline cp_v = ggml_vk_get_cpy_pipeline(ctx, v, शून्यसूचकः, v->type);
            ggml_pipeline_request_descriptor_sets(ctx, cp_k, 1);
            ggml_pipeline_request_descriptor_sets(ctx, cp_v, 1);
            यदि (ctx->prealloc_x_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            स्वचालित make_pc = [](uint32_t hs, uint32_t nh, uint32_t kv, uint32_t nel) {
                vk_op_unary_प्रक्षिप_constants pc{};
                pc.ne = nel;
                pc.ne00 = hs; pc.ne01 = nh;      pc.ne02 = kv; pc.ne03 = nel / (hs * nh * kv);
                pc.nb00 = 1;  pc.nb01 = hs;      pc.nb02 = hs * nh; pc.nb03 = hs * nh * kv;
                pc.ne10 = hs; pc.ne11 = nh;      pc.ne12 = kv; pc.ne13 = pc.ne03;
                pc.nb10 = 1;  pc.nb11 = hs * kv; pc.nb12 = hs; pc.nb13 = hs * kv * nh;
                init_pushconst_fastdiv(pc);
                प्रत्यागम pc;
            };
            स्वचालित cpy_elems = [](uint32_t ne) -> std::array<uint32_t, 3> {
                यदि (ne > 262144) { प्रत्यागम { 512, 512, CEIL_DIV(ne, 262144) }; }
                यदि (ne > 512)    { प्रत्यागम { 512, CEIL_DIV(ne, 512), 1 }; }
                प्रत्यागम { ne, 1, 1 };
            };
            ggml_vk_dispatch_pipeline(ctx, subctx, cp_k, { k_buf, k_dst },
                make_pc((uint32_t)HSK, (uint32_t)nek2, (uint32_t)KV, k_nel), cpy_elems(k_nel));
            ggml_vk_dispatch_pipeline(ctx, subctx, cp_v, { v_buf, v_dst },
                make_pc((uint32_t)HSV, (uint32_t)nev2, (uint32_t)KV, v_nel), cpy_elems(v_nel));
        } अन्यथा {
            vk_pipeline tr_k = ctx->device->pipeline_dequant_transpose[k->type];
            vk_pipeline tr_v = ctx->device->pipeline_dequant_transpose[v->type];
            ggml_pipeline_request_descriptor_sets(ctx, tr_k, 1);
            ggml_pipeline_request_descriptor_sets(ctx, tr_v, 1);
            यदि (ctx->prealloc_x_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            { ध्रुव std::vector<uint32_t> pc = { (uint32_t)HSK, (uint32_t)nek2, (uint32_t)KV, 0, k_nel };
              ggml_vk_dispatch_pipeline(ctx, subctx, tr_k, { k_buf, k_dst }, pc, { k_nel, 1, 1 }); }
            { ध्रुव std::vector<uint32_t> pc = { (uint32_t)HSV, (uint32_t)nev2, (uint32_t)KV, 0, v_nel };
              ggml_vk_dispatch_pipeline(ctx, subctx, tr_v, { v_buf, v_dst }, pc, { v_nel, 1, 1 }); }
        }
        ggml_vk_sync_buffers(ctx, subctx);
        k_buf = k_dst;
        v_buf = v_dst;
    }

    uint32_t आवरणम्_n_शीर्ष_log2 = ((sinks != शून्यसूचकः) << 24) | n_शीर्ष_log2;

    यदि (use_आवरणम्_opt)
    {
        ध्रुव vk_op_flash_attn_आवरणम्_opt_प्रक्षिप_constants opt_pc = {
            nem0,
            nem1,
            nem2,
            (uint32_t)(आवरणम्->nb[1] / परिमाण(ggml_fp16_t)),
            (uint32_t)(आवरणम्->nb[2] / परिमाण(ggml_fp16_t)),
            (uint32_t)(आवरणम्->nb[3] / परिमाण(ggml_fp16_t)),
            आवरणम्_opt_अङ्कः_dwords,
            आवरणम्_opt_अङ्कः_dwords * CEIL_DIV(nem1, Br),
            आवरणम्_opt_अङ्कः_dwords * CEIL_DIV(nem1, Br) * nem2,
        };

        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline_fa_आवरणम्_opt,
                                  { आवरणम्_buf, आवरणम्_opt_buf }, opt_pc,
                                  { आवरणम्_opt_अङ्कः_dwords, CEIL_DIV(nem1, Br), nem2 * nem3 });
        ggml_vk_sync_buffers(ctx, subctx);
    }

    यदि (use_sparse)
    {
        ध्रुव vk_op_flash_attn_sparse_compact_प्रक्षिप_constants sc_pc = {
            KV,
            nem1,
            nem2,
            (uint32_t)(आवरणम्->nb[1] / परिमाण(ggml_fp16_t)),
            (uint32_t)(आवरणम्->nb[2] / परिमाण(ggml_fp16_t)),
            (uint32_t)(आवरणम्->nb[3] / परिमाण(ggml_fp16_t)),
            (uint32_t)n_kv_max,
        };

        ggml_vk_dispatch_pipeline(ctx, subctx, sparse_compact_pipeline,
                                  { आवरणम्_buf, sparse_buf }, sc_pc,
                                  { nem1, nem2, nem3 });
        ggml_vk_sync_buffers(ctx, subctx);
    }

    ध्रुव vk_flash_attn_प्रक्षिप_constants pc = { N, KV,
                                              (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
                                              (uint32_t)neq2, (uint32_t)neq3,
                                              (uint32_t)nek2, (uint32_t)nek3,
                                              (uint32_t)nev2, (uint32_t)nev3,
                                              nem1, nem2, nem3,
                                              q_stride, (uint32_t)nbq2, (uint32_t)nbq3,
                                              k_stride, nbk2_eff, nbk3_eff,
                                              v_stride, nbv2_eff, nbv3_eff,
                                              scale, max_bias, logit_softcap,
                                              आवरणम्_n_शीर्ष_log2, m0, m1,
                                              gqa_अनुपातः, विभज_kv, विभज_k };

    यदि (xe_fa_opt && विभज_k > 1) {
        स्वचालित ऊर्ध्वस्थम्_power_सम्बन्धि_2 = [&](uint32_t अन्तर्) {
            GGML_ASSERT(अन्तर् != 0);
            यदि (अन्तर् <= 1) प्रत्यागम 1u;
            uint32_t ret = अन्तर् - 1;
            ret |= ret >> 1;
            ret |= ret >> 2;
            ret |= ret >> 4;
            ret |= ret >> 8;
            ret |= ret >> 16;
            प्रत्यागम ret + 1;
        };
        स्वचालित गन्तव्यम्_fp16_vk_0 = ggml_vk_get_गन्तव्यम्_fp16(ctx, q->type);
        ध्रुव uint32_t बहिर्गमनम्_dim_प्रत्येकम्_wg = qk_अनुपातः > 16 ? 8 : 16;
        size_t x_ne = ggml_nelements(q);
        size_t अस्थायी_buf_विस्थापनम् = 0;
        uint32_t शीर्ष_stride_k = uint32_t(nbk2 / ggml_type_आकार(k->type));
        uint32_t शीर्ष_stride_v = uint32_t(nbv2 / ggml_type_आकार(v->type));
        uint32_t batch_stride_q = uint32_t(nbq3 / ggml_type_आकार(q->type));
        uint32_t batch_stride_k = uint32_t(nbk3 / ggml_type_आकार(k->type));
        uint32_t batch_stride_v = uint32_t(nbv3 / ggml_type_आकार(v->type));
        uint32_t batch_stride_m = आवरणम् ? uint32_t(आवरणम्->nb[3] / ggml_type_आकार(आवरणम्->type)) : 0u;
        uint32_t batch_stride_o = uint32_t(nb3 / ggml_type_आकार(dst->type));
        vk_fa_xe_opt_प्रक्षिप_constants pc_ph1 = { (uint32_t)nek1, (uint32_t)neq1, (uint32_t)neq2, (uint32_t)nek2, qk_अनुपातः, 1, (sinks != शून्यसूचकः) ? 1u : 0u, (uint32_t)k_stride, शीर्ष_stride_k,
            batch_stride_q, batch_stride_k, batch_stride_v, batch_stride_m, batch_stride_o, scale };
        vk_fa_xe_opt_प्रक्षिप_constants pc_ph2 = pc_ph1;
        pc_ph2.nbkv_tok = v_stride;
        pc_ph2.nbkv_शीर्ष = शीर्ष_stride_v;
        vk_subbuffer q_अस्थायी_buf = fa_copy_qstate ? ggml_vk_subbuffer(ctx, ctx->prealloc_x, अस्थायी_buf_विस्थापनम्) : q_buf;
        अस्थायी_buf_विस्थापनम् += fa_copy_qstate ? x_ne * परिमाण(ggml_fp16_t) : 0;
        vk_subbuffer p_अस्थायी_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_x, अस्थायी_buf_विस्थापनम्);
        अस्थायी_buf_विस्थापनम् += आकार_p;
        vk_subbuffer max_अस्थायी_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_x, अस्थायी_buf_विस्थापनम्);
        अस्थायी_buf_विस्थापनम् += आकार_गणः_max;
        uint32_t xe_native_sub_गणः_आकार = ctx->device.get()->architecture == INTEL_XE1 ? 8 : 16;
        uint32_t aligned_gqa_अनुपातः = ऊर्ध्वस्थम्_power_सम्बन्धि_2(qk_अनुपातः);
        uint32_t बहिर्गमनम्_प्रत्येकम्_wg_ph1 = std::min(256u / xe_native_sub_गणः_आकार, (uint32_t)neq1);
        uint32_t बहिर्गमनम्_प्रत्येकम्_wg_ph2 = std::min(std::max(16u / aligned_gqa_अनुपातः, 1u), (uint32_t)neq1);
        uint32_t ph1_wg = ((neq1 + बहिर्गमनम्_प्रत्येकम्_wg_ph1 - 1) / बहिर्गमनम्_प्रत्येकम्_wg_ph1) * nek2;
        uint32_t ph2_wg = ((neq1 + बहिर्गमनम्_प्रत्येकम्_wg_ph2 - 1) / बहिर्गमनम्_प्रत्येकम्_wg_ph2) * ne0 / बहिर्गमनम्_dim_प्रत्येकम्_wg;
        यदि (fa_copy_qstate) {
            ध्रुव std::vector<uint32_t> pc_cpy_fp16 =
            { (uint32_t)q->ne[0], (uint32_t)q->ne[1], (uint32_t)q->ne[2], (uint32_t)q->ne[3], (uint32_t)(x_ne) };
            ggml_vk_sync_buffers(ctx, subctx);
            ggml_pipeline_request_descriptor_sets(ctx, गन्तव्यम्_fp16_vk_0, 1);
            ggml_vk_dispatch_pipeline(ctx, subctx, गन्तव्यम्_fp16_vk_0, { q_buf, q_अस्थायी_buf }, pc_cpy_fp16, { (uint32_t)(x_ne), 1, 1 });
        }

        ggml_vk_sync_buffers(ctx, subctx);
        ggml_pipeline_request_descriptor_sets(ctx, xe_fa_pipeline_dual_phases.first, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, xe_fa_pipeline_dual_phases.first,
            { q_अस्थायी_buf, k_buf, आवरणम्_buf, p_अस्थायी_buf, max_अस्थायी_buf },
            pc_ph1, { (uint32_t)ph1_wg, (uint32_t)nek1, (uint32_t)neq3 });

        ggml_vk_sync_buffers(ctx, subctx);
        ggml_pipeline_request_descriptor_sets(ctx, xe_fa_pipeline_dual_phases.second, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, xe_fa_pipeline_dual_phases.second,
            { p_अस्थायी_buf, v_buf, max_अस्थायी_buf, sinks_buf, dst_buf },
            pc_ph2, { (uint32_t)ph2_wg, (uint32_t)nev2, (uint32_t)neq3 });

        ctx->prealloc_x_need_sync = सत्य;
    } अन्यथा यदि (विभज_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_flash_attn_विभज_k_reduce, 1);

        यदि (ctx->prealloc_विभज_k_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        // We reuse workgroups_x to mean the number of splits, so we need to
        // cancel out the divide by wg_denoms[0].
        uint32_t dispatch_x;
        यदि (gqa_अनुपातः > 1) {
            workgroups_x *= pipeline->wg_denoms[0];
            dispatch_x = विभज_k * workgroups_x;
        } अन्यथा {
            dispatch_x = Tr * विभज_k * pipeline->wg_denoms[0];
        }

        vk_subbuffer विभज_k_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_विभज_k, 0);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                    {q_buf, k_buf, v_buf, आवरणम्_buf, sinks_buf, विभज_k_buf, आवरणम्_opt_buf, sparse_buf},
                                    pc, { dispatch_x, workgroups_y, workgroups_z });

        ggml_vk_sync_buffers(ctx, subctx);
        ध्रुव vk_op_flash_attn_विभज_k_reduce_प्रक्षिप_constants pc2 = { HSV, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3, विभज_k, (sinks != शून्यसूचकः) };
        ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_flash_attn_विभज_k_reduce,
                                    {विभज_k_buf, sinks_buf, dst_buf},
                                    pc2, { (uint32_t)ne1, HSV, (uint32_t)(ne2 * ne3) });
        ctx->prealloc_विभज_k_need_sync = सत्य;
    } अन्यथा {
        यदि (gqa_अनुपातः > 1) {
            // When using gqa, we want one actual workgroup per batch, so cancel out wg_denoms
            workgroups_x *= pipeline->wg_denoms[0];
        }
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                    {q_buf, k_buf, v_buf, आवरणम्_buf, sinks_buf, dst_buf, आवरणम्_opt_buf, sparse_buf},
                                    pc, { workgroups_x, workgroups_y, workgroups_z });
    }

    यदि (use_dequant_kv) {
        ctx->prealloc_x_need_sync = सत्य;
    }
    यदि (use_आवरणम्_opt || use_sparse) {
        ctx->prealloc_y_need_sync = सत्य;
    }
}

स्थिर vk_conv_shapes ggml_vk_conv_select_shape(ggml_backend_vk_context * ctx, uint32_t K, uint32_t NPQ) {
    स्वचालित n_tiles = [&](vk_conv_shapes s) {
        प्रत्यागम CEIL_DIV(K, vk_conv_खण्डम्_sizes[s].K)
            * CEIL_DIV(NPQ, vk_conv_खण्डम्_sizes[s].NPQ);
    };

    // We can't query number of shader cores on Intel, use 32 as a placeholder
    // so small convolutions will still choose a smaller tile.
    ध्रुव uint32_t shader_core_गणनम् = ctx->device->shader_core_गणनम् > 0 ? ctx->device->shader_core_गणनम् : 32;

    // 128x128 isn't used with cm1 due to shared memory size; fall through to a smaller tile.
    तार्किक allow_128x128 = सत्य;
#यदि defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    यदि (!ctx->device->coopmat2 && ctx->device->coopmat_support && ctx->device->coopmat_support_16x16x16_f16acc) {
        allow_128x128 = असत्य;
    }
#यदि_अन्तः

    यदि (allow_128x128 && K > 64 && n_tiles(CONV_SHAPE_128x128) >= shader_core_गणनम् * 2) {
        प्रत्यागम CONV_SHAPE_128x128;
    } अन्यथा यदि (K <= 32 && n_tiles(CONV_SHAPE_32x256) >= shader_core_गणनम् * 2) {
        प्रत्यागम CONV_SHAPE_32x256;
    } अन्यथा यदि (K <= 64 && n_tiles(CONV_SHAPE_64x128) >= shader_core_गणनम् * 2) {
        प्रत्यागम CONV_SHAPE_64x128;
    } अन्यथा यदि (!allow_128x128 && K > 64 && n_tiles(CONV_SHAPE_64x128) >= shader_core_गणनम् * 2) {
        // cm1 fallback for large K when 128x128 isn't available
        प्रत्यागम CONV_SHAPE_64x128;
    } अन्यथा {
        प्रत्यागम CONV_SHAPE_64x32;
    }
}

स्थिर vk_pipeline ggml_vk_op_get_pipeline(ggml_backend_vk_context * ctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ध्रुव ggml_tensor * src2, ध्रुव ggml_tensor * dst, ggml_op op) {
    विकल्प (op) {
    पक्ष GGML_OP_GET_पङ्क्तिकाः:
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        यदि (src0->type == GGML_TYPE_I32) {
            // i32 src only supports i32 result
            GGML_ASSERT(dst->type == GGML_TYPE_I32);
            प्रत्यागम ctx->device->pipeline_get_पङ्क्तिकाः[src0->type];
        }
        यदि (dst->type == GGML_TYPE_F16) {
            प्रत्यागम ctx->device->pipeline_get_पङ्क्तिकाः[src0->type];
        }
        यदि (dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_get_पङ्क्तिकाः_f32[src0->type];
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_GET_पङ्क्तिकाः_पृष्ठभागः:
        यदि (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_get_पङ्क्तिकाः_पृष्ठभागः_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_ACC:
        यदि (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_acc_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_समुच्चयः:
        यदि (src0->type == src1->type && src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32)) {
            प्रत्यागम ctx->device->pipeline_समुच्चयः_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_सङ्कलय:
    पक्ष GGML_OP_SUB:
    पक्ष GGML_OP_MUL:
    पक्ष GGML_OP_DIV:
        यदि ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16)) {
            प्रत्यागम शून्यसूचकः;
        }
        विकल्प (op) {
        पक्ष GGML_OP_सङ्कलय:
        {
            यदि (ctx->अङ्कः_additional_fused_ops > 0) {
                यदि (ctx->do_सङ्कलय_rms_partials) {
                    प्रत्यागम ctx->device->pipeline_multi_सङ्कलय_rms[ctx->अङ्कः_additional_fused_ops];
                } अन्यथा {
                    प्रत्यागम ctx->device->pipeline_multi_सङ्कलय[ctx->अङ्कः_additional_fused_ops];
                }
            }
            यदि (ctx->do_सङ्कलय_rms_partials) {
                स्वचालित pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_सङ्कलय_rms_norepeat : ctx->device->pipeline_सङ्कलय_rms;
                प्रत्यागम pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
            } अन्यथा {
                स्वचालित pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_सङ्कलय_norepeat : ctx->device->pipeline_सङ्कलय;
                प्रत्यागम pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
            }
        }
        पक्ष GGML_OP_SUB:
        {
            स्वचालित pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_sub_norepeat : ctx->device->pipeline_sub;
            प्रत्यागम pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        पक्ष GGML_OP_MUL:
        {
            स्वचालित pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_mul_norepeat : ctx->device->pipeline_mul;
            प्रत्यागम pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        पक्ष GGML_OP_DIV:
        {
            स्वचालित pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_div_norepeat : ctx->device->pipeline_div;
            प्रत्यागम pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        स्वभाव:
            भङ्ग;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_सङ्कलय_ID:
        यदि (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_सङ्कलय_id_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_बहिर्गमनम्_PROD:
        यदि (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_बहिर्गमनम्_prod_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_CONCAT: {
        यदि (!ggml_vk_concat_supported(src0, src1, dst)) {
            प्रत्यागम शून्यसूचकः;
        }
        विकल्प (ggml_vk_concat_unit_आकार(src0->type)) {
        पक्ष 1:
            प्रत्यागम ctx->device->pipeline_concat_i8;
        पक्ष 2:
            प्रत्यागम ctx->device->pipeline_concat_i16;
        पक्ष 4:
            प्रत्यागम ctx->device->pipeline_concat_i32;
        पक्ष 8:
            प्रत्यागम ctx->device->pipeline_concat_i64;
        स्वभाव:
            प्रत्यागम शून्यसूचकः;
        }
    }
    पक्ष GGML_OP_UPSCALE:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            uint32_t mode = (ggml_get_op_params_i32(dst, 0) & (0xFF | GGML_SCALE_FLAG_ANTIALIAS));
            विकल्प (mode) {
                पक्ष GGML_SCALE_बहुलकम्_निकटतमः:
                    प्रत्यागम ctx->device->pipeline_upscale_निकटतमः_f32;
                पक्ष GGML_SCALE_बहुलकम्_BILINEAR:
                    प्रत्यागम ctx->device->pipeline_upscale_bilinear_f32;
                पक्ष GGML_SCALE_बहुलकम्_BICUBIC:
                    प्रत्यागम ctx->device->pipeline_upscale_bicubic_f32;
                पक्ष GGML_SCALE_बहुलकम्_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS:
                    प्रत्यागम ctx->device->pipeline_upscale_bilinear_antialias_f32;
                स्वभाव:
                    प्रत्यागम शून्यसूचकः;
            }
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_SCALE:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_scale_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_SQR:
        यदि (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            प्रत्यागम ctx->device->pipeline_sqr[dst->type == GGML_TYPE_F16];
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_SQRT:
        यदि (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            प्रत्यागम ctx->device->pipeline_sqrt[dst->type == GGML_TYPE_F16];
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_SIN:
        यदि (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            प्रत्यागम ctx->device->pipeline_sin[dst->type == GGML_TYPE_F16];
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_COS:
        यदि (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            प्रत्यागम ctx->device->pipeline_cos[dst->type == GGML_TYPE_F16];
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_LOG:
        यदि (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            प्रत्यागम ctx->device->pipeline_log[dst->type == GGML_TYPE_F16];
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_TRI:
        यदि (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            प्रत्यागम ctx->device->pipeline_tri[dst->type == GGML_TYPE_F16];
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_DIAG:
        यदि (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            प्रत्यागम ctx->device->pipeline_diag[dst->type == GGML_TYPE_F16];
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_CLAMP:
        यदि (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            प्रत्यागम ctx->device->pipeline_clamp[dst->type == GGML_TYPE_F16];
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_PAD:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_pad_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_PAD_REFLECT_1D:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_pad_reflect_1d_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_ROLL:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_roll_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_REPEAT:
        यदि (ggml_type_आकार(src0->type) == परिमाण(भिन्न) && ggml_type_आकार(dst->type) == परिमाण(भिन्न)) {
            प्रत्यागम ctx->device->pipeline_repeat_i32;
        }
        यदि (ggml_type_आकार(src0->type) == 2 && ggml_type_आकार(dst->type) == 2) {
            प्रत्यागम ctx->device->pipeline_repeat_i16;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_REPEAT_पृष्ठभागः:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_repeat_पृष्ठभागः_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_CPY:
    पक्ष GGML_OP_CONT:
    पक्ष GGML_OP_DUP:
        प्रत्यागम ggml_vk_get_cpy_pipeline(ctx, src0, dst, dst->type);
    पक्ष GGML_OP_समुच्चयः_पङ्क्तिकाः:
        {
            यदि (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
                प्रत्यागम शून्यसूचकः;
            }
            ध्रुव पूर्णाङ्क src_idx = src0->type == GGML_TYPE_F16;
            यदि (src1->type == GGML_TYPE_I64) {
                प्रत्यागम ctx->device->pipeline_समुच्चयः_पङ्क्तिकाः_i64[src_idx][dst->type];
            } अन्यथा यदि (src1->type == GGML_TYPE_I32) {
                प्रत्यागम ctx->device->pipeline_समुच्चयः_पङ्क्तिकाः_i32[src_idx][dst->type];
            }
            प्रत्यागम शून्यसूचकः;
        }
    पक्ष GGML_OP_SILU_पृष्ठभागः:
        यदि (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_silu_पृष्ठभागः_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_NORM:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_norm_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_गणः_NORM:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_गणः_norm_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_RMS_NORM:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            यदि (ctx->do_सङ्कलय_rms_partials) {
                प्रत्यागम ctx->fused_rms_norm_बहुलकम् == RMS_NORM_MUL ? ctx->device->pipeline_rms_norm_mul_partials_f32 : ctx->device->pipeline_rms_norm_partials_f32;
            }
            प्रत्यागम ctx->fused_rms_norm_बहुलकम् == RMS_NORM_MUL ? ctx->device->pipeline_rms_norm_mul_f32 : ctx->device->pipeline_rms_norm_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_RMS_NORM_पृष्ठभागः:
        यदि (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_rms_norm_पृष्ठभागः_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_L2_NORM:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_l2_norm_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_UNARY:
        यदि ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            प्रत्यागम शून्यसूचकः;
        }

        विकल्प (ggml_get_unary_op(dst)) {
            पक्ष GGML_UNARY_OP_EXP:
                प्रत्यागम ctx->device->pipeline_exp[dst->type == GGML_TYPE_F16];
            पक्ष GGML_UNARY_OP_EXPM1:
                प्रत्यागम ctx->device->pipeline_expm1[dst->type == GGML_TYPE_F16];
            पक्ष GGML_UNARY_OP_ELU:
                प्रत्यागम ctx->device->pipeline_elu[dst->type == GGML_TYPE_F16];
            पक्ष GGML_UNARY_OP_SILU:
                प्रत्यागम ctx->device->pipeline_silu[dst->type == GGML_TYPE_F16];
            पक्ष GGML_UNARY_OP_GELU:
                प्रत्यागम ctx->device->pipeline_gelu[dst->type == GGML_TYPE_F16];
            पक्ष GGML_UNARY_OP_GELU_ERF:
                प्रत्यागम ctx->device->pipeline_gelu_erf[dst->type == GGML_TYPE_F16];
            पक्ष GGML_UNARY_OP_GELU_शीघ्र:
                प्रत्यागम ctx->device->pipeline_gelu_शीघ्र[dst->type == GGML_TYPE_F16];
            पक्ष GGML_UNARY_OP_RELU:
                प्रत्यागम ctx->device->pipeline_relu[dst->type == GGML_TYPE_F16];
            पक्ष GGML_UNARY_OP_XIELU:
                प्रत्यागम ctx->device->pipeline_xielu[dst->type == GGML_TYPE_F16];
            पक्ष GGML_UNARY_OP_NEG:
                प्रत्यागम ctx->device->pipeline_neg[dst->type == GGML_TYPE_F16];
            पक्ष GGML_UNARY_OP_TANH:
                प्रत्यागम ctx->device->pipeline_tanh[dst->type == GGML_TYPE_F16];
            पक्ष GGML_UNARY_OP_SIGMOID:
                प्रत्यागम ctx->device->pipeline_sigmoid[dst->type == GGML_TYPE_F16];
            पक्ष GGML_UNARY_OP_HARDSIGMOID:
                प्रत्यागम ctx->device->pipeline_hardsigmoid[dst->type == GGML_TYPE_F16];
            पक्ष GGML_UNARY_OP_HARDSWISH:
                प्रत्यागम ctx->device->pipeline_hardswish[dst->type == GGML_TYPE_F16];
            पक्ष GGML_UNARY_OP_ABS:
                प्रत्यागम ctx->device->pipeline_abs[dst->type == GGML_TYPE_F16];
            पक्ष GGML_UNARY_OP_SOFTPLUS:
                प्रत्यागम ctx->device->pipeline_softplus[dst->type == GGML_TYPE_F16];
            पक्ष GGML_UNARY_OP_चरणः:
                प्रत्यागम ctx->device->pipeline_चरणः[dst->type == GGML_TYPE_F16];
            पक्ष GGML_UNARY_OP_आवर्तनम्:
                प्रत्यागम ctx->device->pipeline_आवर्तनम्[dst->type == GGML_TYPE_F16];
            पक्ष GGML_UNARY_OP_CEIL:
                प्रत्यागम ctx->device->pipeline_ceil[dst->type == GGML_TYPE_F16];
            पक्ष GGML_UNARY_OP_FLOOR:
                प्रत्यागम ctx->device->pipeline_floor[dst->type == GGML_TYPE_F16];
            पक्ष GGML_UNARY_OP_TRUNC:
                प्रत्यागम ctx->device->pipeline_trunc[dst->type == GGML_TYPE_F16];
            पक्ष GGML_UNARY_OP_SGN:
                प्रत्यागम ctx->device->pipeline_sgn[dst->type == GGML_TYPE_F16];
            स्वभाव:
                भङ्ग;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_GLU:
        यदि ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            प्रत्यागम शून्यसूचकः;
        }

        विकल्प (ggml_get_glu_op(dst)) {
            पक्ष GGML_GLU_OP_GEGLU:
                प्रत्यागम ctx->device->pipeline_geglu[dst->type == GGML_TYPE_F16];
            पक्ष GGML_GLU_OP_REGLU:
                प्रत्यागम ctx->device->pipeline_reglu[dst->type == GGML_TYPE_F16];
            पक्ष GGML_GLU_OP_SWIGLU:
                प्रत्यागम ctx->device->pipeline_swiglu[dst->type == GGML_TYPE_F16];
            पक्ष GGML_GLU_OP_SWIGLU_OAI:
                प्रत्यागम ctx->device->pipeline_swiglu_oai[dst->type == GGML_TYPE_F16];
            पक्ष GGML_GLU_OP_SWIGLU_CLAMP:
                प्रत्यागम ctx->device->pipeline_swiglu_clamp[dst->type == GGML_TYPE_F16];
            पक्ष GGML_GLU_OP_GEGLU_ERF:
                प्रत्यागम ctx->device->pipeline_geglu_erf[dst->type == GGML_TYPE_F16];
            पक्ष GGML_GLU_OP_GEGLU_शीघ्र:
                प्रत्यागम ctx->device->pipeline_geglu_शीघ्र[dst->type == GGML_TYPE_F16];
            स्वभाव:
                भङ्ग;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_DIAG_आवरणम्_INF:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_diag_आवरणम्_inf_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_SOFT_MAX:
        GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        GGML_ASSERT(!src2 || src2->type == GGML_TYPE_F32);

        यदि (ctx->अङ्कः_additional_fused_ops) {
            uint32_t idx = (uint32_t)ceilf(log2f(भिन्न(dst->ne[0])));
            GGML_ASSERT(idx < अङ्कः_topk_moe_pipelines);
            // use n_experts from push constant if it's not equal to the power of two spec constant
            तार्किक use_प्रक्षिप = dst->ne[0] != (1u << idx);
            प्रत्यागम ctx->device->pipeline_topk_moe[idx][use_प्रक्षिप];
        }

        यदि (src0->type == GGML_TYPE_F32 && (src1 == शून्यसूचकः || src1->type == GGML_TYPE_F32) && dst->type == GGML_TYPE_F32) {
            प्रत्यागम src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_wg512 : ctx->device->pipeline_soft_max_f32;
        }
        यदि (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_f16_wg512 : ctx->device->pipeline_soft_max_f32_f16;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_SOFT_MAX_पृष्ठभागः:
        यदि (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_soft_max_पृष्ठभागः_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_ROPE:
    पक्ष GGML_OP_ROPE_पृष्ठभागः:
        {
            ध्रुव ggml_tensor *rope = ctx->अङ्कः_additional_fused_ops == 2 ? dst->src[0]->src[0] : dst;
            ध्रुव पूर्णाङ्क mode = ((ध्रुव int32_t *) rope->op_params)[2];
            ध्रुव तार्किक is_neox = mode & GGML_ROPE_TYPE_NEOX;
            ध्रुव तार्किक is_mrope = mode & GGML_ROPE_TYPE_MROPE;
            ध्रुव तार्किक is_vision = mode == GGML_ROPE_TYPE_VISION;

            यदि (is_neox) {
                यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    प्रत्यागम ctx->device->pipeline_rope_neox_f32;
                }
                यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    प्रत्यागम ctx->device->pipeline_rope_neox_f32_f16;
                }
                यदि (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    प्रत्यागम ctx->device->pipeline_rope_neox_f16;
                }
            } अन्यथा यदि (is_mrope && !is_vision) {
                यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    प्रत्यागम ctx->device->pipeline_rope_multi_f32;
                }
                यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    प्रत्यागम ctx->device->pipeline_rope_multi_f32_f16;
                }
                यदि (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    प्रत्यागम ctx->device->pipeline_rope_multi_f16;
                }
            } अन्यथा यदि (is_vision) {
                यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    प्रत्यागम ctx->device->pipeline_rope_vision_f32;
                }
                यदि (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    प्रत्यागम ctx->device->pipeline_rope_vision_f16;
                }
            } अन्यथा {
                यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    प्रत्यागम ctx->device->pipeline_rope_norm_f32;
                }
                यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    प्रत्यागम ctx->device->pipeline_rope_norm_f32_f16;
                }
                यदि (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    प्रत्यागम ctx->device->pipeline_rope_norm_f16;
                }
            }
            प्रत्यागम शून्यसूचकः;
        }
    पक्ष GGML_OP_योग:
    पक्ष GGML_OP_योग_पङ्क्तिकाः:
    पक्ष GGML_OP_माध्यम्:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_योग_पङ्क्तिकाः_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_CROSS_ENTROPY_LOSS:
        यदि (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम src0->ne[0] > 1024 ? ctx->device->pipeline_cross_entropy_loss_f32_wg512 : ctx->device->pipeline_cross_entropy_loss_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_CROSS_ENTROPY_LOSS_पृष्ठभागः:
        // src0 is the scalar grad; src1 is logits
        यदि (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2 && src2->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम src1->ne[0] > 1024 ? ctx->device->pipeline_cross_entropy_loss_पृष्ठभागः_f32_wg512 : ctx->device->pipeline_cross_entropy_loss_पृष्ठभागः_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_CUMSUM:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            यदि (src0->ne[0] <= 512) {
                प्रत्यागम ctx->device->pipeline_cumsum_small_f32;
            } अन्यथा {
                प्रत्यागम ctx->device->pipeline_cumsum_f32;
            }
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_SOLVE_TRI:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {

            vk_solve_tri_pipeline_state solve_tri_pipeline_state(src0->ne[0], src1->ne[0]);

            vk_pipeline pipeline = शून्यसूचकः;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                स्वचालित it = ctx->device->pipeline_solve_tri_f32.find(solve_tri_pipeline_state);
                यदि (it != ctx->device->pipeline_solve_tri_f32.end()) {
                    pipeline = it->second;
                } अन्यथा {
                    ctx->device->pipeline_solve_tri_f32[solve_tri_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            प्रत्यागम pipeline;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_ARGMAX:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) {
            प्रत्यागम ctx->device->pipeline_argmax_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_गणनम्_EQUAL:
        यदि (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I64) {
            प्रत्यागम ctx->device->pipeline_गणनम्_equal_i32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_IM2COL:
        यदि (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_im2col_f32;
        }
        यदि (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            प्रत्यागम ctx->device->pipeline_im2col_f32_f16;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_IM2COL_3D:
        यदि (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_im2col_3d_f32;
        }
        यदि (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            प्रत्यागम ctx->device->pipeline_im2col_3d_f32_f16;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_TIMESTEP_EMBEDDING:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_timestep_embedding_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_CONV_TRANSPOSE_1D:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_conv_transpose_1d_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_COL2IM_1D:
        विकल्प (src0->type) {
            पक्ष GGML_TYPE_F32:  प्रत्यागम ctx->device->pipeline_col2im_1d_f32;
            पक्ष GGML_TYPE_F16:  प्रत्यागम ctx->device->pipeline_col2im_1d_f16;
            पक्ष GGML_TYPE_BF16: प्रत्यागम ctx->device->pipeline_col2im_1d_bf16;
            स्वभाव:             प्रत्यागम शून्यसूचकः;
        }
    पक्ष GGML_OP_POOL_1D:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_pool1d_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_POOL_2D:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            // Wide windows use neighboring subgroup lanes to read neighboring inputs.
            यदि (ctx->device->pipeline_pool2d_subgroup_f32 && dst->op_params[1] >= int32_t(ctx->device->subgroup_आकार)) {
                प्रत्यागम ctx->device->pipeline_pool2d_subgroup_f32;
            }
            प्रत्यागम ctx->device->pipeline_pool2d_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_RWKV_WKV6:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_rwkv_wkv6_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_RWKV_WKV7:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_rwkv_wkv7_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_GATED_रैखिकम्_ATTN:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_gated_रैखिकम्_attn_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_LIGHTNING_INDEXER:
        // only the k type selects a pipeline, the other types are fixed by ggml_lightning_indexer()
        यदि (ggml_vk_lightning_indexer_k_type_supported(src1->type)) {
            प्रत्यागम ctx->device->pipeline_lightning_indexer_f32[src1->type];
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_GATED_DELTA_NET:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ध्रुव uint32_t S_v = dst->src[2]->ne[0];
            ध्रुव uint32_t kda = (dst->src[3]->ne[0] == (int64_t)S_v) ? 1 : 0;
            uint32_t si;
            विकल्प (S_v) {
                पक्ष 16:  si = 0; भङ्ग;
                पक्ष 32:  si = 1; भङ्ग;
                पक्ष 64:  si = 2; भङ्ग;
                पक्ष 128: si = 3; भङ्ग;
                स्वभाव: प्रत्यागम शून्यसूचकः;
            }
            प्रत्यागम ctx->device->pipeline_gated_delta_net[si][kda];
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_SSM_अवलोकनम्:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ध्रुव uint32_t d_state = src0->ne[0];
            यदि (d_state == 128) {
                प्रत्यागम ctx->device->pipeline_ssm_अवलोकनम्_f32_d128;
            } अन्यथा यदि (d_state == 256) {
                प्रत्यागम ctx->device->pipeline_ssm_अवलोकनम्_f32_d256;
            }
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_SSM_CONV:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            विकल्प (ctx->अङ्कः_additional_fused_ops) {
                पक्ष 0:  प्रत्यागम ctx->device->pipeline_ssm_conv_f32;
                पक्ष 1:  प्रत्यागम ctx->device->pipeline_ssm_conv_silu_f32;
                पक्ष 2:  प्रत्यागम ctx->device->pipeline_ssm_conv_bias_silu_f32;
                स्वभाव: प्रत्यागम शून्यसूचकः;
            }
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_OPT_चरणः_ADAMW:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_opt_चरणः_adamw_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_OPT_चरणः_SGD:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_opt_चरणः_sgd_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_LEAKY_RELU:
        यदि (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            प्रत्यागम ctx->device->pipeline_leaky_relu[dst->type == GGML_TYPE_F16];
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_CONV_2D:
    पक्ष GGML_OP_CONV_TRANSPOSE_2D:
        यदि (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            uint32_t K = dst->ne[2]; // Cout
            uint32_t NPQ = dst->ne[3] * dst->ne[1] * dst->ne[0]; // N * OH * OW
            vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, K, NPQ);

            तार्किक transpose = dst->op == GGML_OP_CONV_TRANSPOSE_2D;
            uint32_t KW = (uint32_t)src0->ne[0];
            uint32_t KH = (uint32_t)src0->ne[1];
            uint32_t s0 = (uint32_t)(ggml_get_op_params_i32(dst, 0));
            uint32_t s1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 1) : s0;
            uint32_t p0 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 2) : 0;
            uint32_t p1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 3) : 0;
            uint32_t d0 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 4) : 1;
            uint32_t d1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 5) : 1;

            // tile-aligned shapes let the shader skip bounds checks
            ध्रुव uint32_t Cin = (uint32_t)src1->ne[2];
            ध्रुव uint32_t CRS = Cin * KW * KH;
            ध्रुव uint32_t BS_K   = vk_conv_खण्डम्_sizes[shape].K;
            ध्रुव uint32_t BS_CRS = vk_conv_खण्डम्_sizes[shape].CRS;
            ध्रुव uint32_t BS_NPQ = vk_conv_खण्डम्_sizes[shape].NPQ;
            ध्रुव uint32_t aligned = ((K   % BS_K   == 0) &&
                                      (CRS % BS_CRS == 0) &&
                                      (NPQ % BS_NPQ == 0)) ? 1u : 0u;

            vk_conv2d_pipeline_state conv2d_pipeline_state(s0, s1, p0, p1, d0, d1, KW, KH, aligned);

            std::map<vk_conv2d_pipeline_state, vk_pipeline> *pipelines = शून्यसूचकः;
            यदि (op == GGML_OP_CONV_2D) {
                यदि (src0->type == GGML_TYPE_F32) {
                    pipelines = &ctx->device->pipeline_conv2d_f32[shape];
                } अन्यथा यदि (src0->type == GGML_TYPE_F16) {
                    pipelines = &ctx->device->pipeline_conv2d_f16_f32[shape];
                }
            } अन्यथा यदि (op == GGML_OP_CONV_TRANSPOSE_2D) {
                यदि (src0->type == GGML_TYPE_F32) {
                    pipelines = &ctx->device->pipeline_conv_transpose_2d_f32[shape];
                } अन्यथा यदि (src0->type == GGML_TYPE_F16) {
                    pipelines = &ctx->device->pipeline_conv_transpose_2d_f16_f32[shape];
                }
            }

            vk_pipeline pipeline = शून्यसूचकः;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                स्वचालित it = pipelines->find(conv2d_pipeline_state);
                यदि (it != pipelines->end()) {
                    pipeline = it->second;
                } अन्यथा {
                    (*pipelines)[conv2d_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            प्रत्यागम pipeline;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_CONV_2D_DW:
        यदि (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            यदि (ggml_is_contiguous(src1)) {
                प्रत्यागम ctx->device->pipeline_conv2d_dw_whcn_f32;
            } अन्यथा यदि (ggml_is_contiguous_channels(src1)) {
                प्रत्यागम ctx->device->pipeline_conv2d_dw_cwhn_f32;
            }
        } अन्यथा यदि (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            यदि (ggml_is_contiguous(src1)) {
                प्रत्यागम ctx->device->pipeline_conv2d_dw_whcn_f16_f32;
            } अन्यथा यदि (ggml_is_contiguous_channels(src1)) {
                प्रत्यागम ctx->device->pipeline_conv2d_dw_cwhn_f16_f32;
            }
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_CONV_3D:
        यदि (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ध्रुव uint32_t OC = (uint32_t)ggml_get_op_params_i32(dst, 11);
            ध्रुव uint32_t IC = (uint32_t)ggml_get_op_params_i32(dst, 9);
            ध्रुव uint32_t N  = (uint32_t)ggml_get_op_params_i32(dst, 10);
            ध्रुव uint32_t NPQ = N * dst->ne[2] * dst->ne[1] * dst->ne[0];
            ध्रुव vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, OC, NPQ);

            ध्रुव uint32_t KW = (uint32_t)src0->ne[0];
            ध्रुव uint32_t KH = (uint32_t)src0->ne[1];
            ध्रुव uint32_t KD = (uint32_t)src0->ne[2];
            ध्रुव uint32_t s0 = (uint32_t)ggml_get_op_params_i32(dst, 0);
            ध्रुव uint32_t s1 = (uint32_t)ggml_get_op_params_i32(dst, 1);
            ध्रुव uint32_t s2 = (uint32_t)ggml_get_op_params_i32(dst, 2);
            ध्रुव uint32_t p0 = (uint32_t)ggml_get_op_params_i32(dst, 3);
            ध्रुव uint32_t p1 = (uint32_t)ggml_get_op_params_i32(dst, 4);
            ध्रुव uint32_t p2 = (uint32_t)ggml_get_op_params_i32(dst, 5);
            ध्रुव uint32_t d0 = (uint32_t)ggml_get_op_params_i32(dst, 6);
            ध्रुव uint32_t d1 = (uint32_t)ggml_get_op_params_i32(dst, 7);
            ध्रुव uint32_t d2 = (uint32_t)ggml_get_op_params_i32(dst, 8);

            ध्रुव uint32_t CRS = IC * KW * KH * KD;
            ध्रुव uint32_t BS_K   = vk_conv_खण्डम्_sizes[shape].K;
            ध्रुव uint32_t BS_CRS = vk_conv_खण्डम्_sizes[shape].CRS;
            ध्रुव uint32_t BS_NPQ = vk_conv_खण्डम्_sizes[shape].NPQ;
            ध्रुव uint32_t aligned = ((OC  % BS_K   == 0) &&
                                      (CRS % BS_CRS == 0) &&
                                      (NPQ % BS_NPQ == 0)) ? 1u : 0u;

            vk_conv3d_pipeline_state conv3d_pipeline_state(s0, s1, s2, p0, p1, p2, d0, d1, d2, KW, KH, KD, aligned);

            std::map<vk_conv3d_pipeline_state, vk_pipeline> *pipelines = शून्यसूचकः;
            यदि (src0->type == GGML_TYPE_F32) {
                pipelines = &ctx->device->pipeline_conv3d_f32[shape];
            } अन्यथा यदि (src0->type == GGML_TYPE_F16) {
                pipelines = &ctx->device->pipeline_conv3d_f16_f32[shape];
            } अन्यथा {
                प्रत्यागम शून्यसूचकः;
            }

            vk_pipeline pipeline = शून्यसूचकः;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                स्वचालित it = pipelines->find(conv3d_pipeline_state);
                यदि (it != pipelines->end()) {
                    pipeline = it->second;
                } अन्यथा {
                    (*pipelines)[conv3d_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            प्रत्यागम pipeline;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_ADD1:
        यदि (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            प्रत्यागम ctx->device->pipeline_add1_f16_f16;
        }
        यदि (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            प्रत्यागम ctx->device->pipeline_add1_f16_f32;
        }
        यदि (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_add1_f32_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_ARANGE:
        यदि (dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_arange_f32;
        }
        प्रत्यागम शून्यसूचकः;
    पक्ष GGML_OP_FILL:
        यदि (dst->type == GGML_TYPE_F32) {
            प्रत्यागम ctx->device->pipeline_fill_f32;
        }
        यदि (dst->type == GGML_TYPE_F16) {
            प्रत्यागम ctx->device->pipeline_fill_f16;
        }
        प्रत्यागम शून्यसूचकः;
    स्वभाव:
        प्रत्यागम शून्यसूचकः;
    }

    GGML_अप्रयुक्तम्(src2);
}

प्रतिरूपम्<प्रकारनाम PC>
स्थिर शून्य ggml_vk_op_f32(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ध्रुव ggml_tensor * src2, ध्रुव ggml_tensor * src3, ggml_tensor * dst, ggml_op op, PC&& pc, vk_pipeline pipeline_override = शून्यसूचकः) {
    VK_LOG_DEBUG("ggml_vk_op_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    यदि (src1 != शून्यसूचकः) {
        std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    }
    यदि (src2 != शून्यसूचकः) {
        std::cerr << "), (" << src2 << ", name=" << src2->name << ", type=" << src2->type << ", ne0=" << src2->ne[0] << ", ne1=" << src2->ne[1] << ", ne2=" << src2->ne[2] << ", ne3=" << src2->ne[3] << ", nb0=" << src2->nb[0] << ", nb1=" << src2->nb[1] << ", nb2=" << src2->nb[2] << ", nb3=" << src2->nb[3];
    }
    यदि (src3 != शून्यसूचकः) {
        std::cerr << "), (" << src3 << ", name=" << src3->name << ", type=" << src3->type << ", ne0=" << src3->ne[0] << ", ne1=" << src3->ne[1] << ", ne2=" << src3->ne[2] << ", ne3=" << src3->ne[3] << ", nb0=" << src3->nb[0] << ", nb1=" << src3->nb[1] << ", nb2=" << src3->nb[2] << ", nb3=" << src3->nb[3];
    }
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << ggml_op_name(op) << ")");
    GGML_ASSERT(op == GGML_OP_GET_पङ्क्तिकाः || op == GGML_OP_CPY || op == GGML_OP_CONCAT || (!ggml_is_quantized(src0->type) && (src1 == शून्यसूचकः || !ggml_is_quantized(src1->type))));  // NOLINT
    GGML_ASSERT(dst->बफर != शून्यसूचकः);
    ध्रुव uint64_t ne00 = src0->ne[0];
    ध्रुव uint64_t ne01 = src0->ne[1];
    ध्रुव uint64_t ne02 = src0->ne[2];
    ध्रुव uint64_t ne03 = src0->ne[3];

    ध्रुव तार्किक use_src1 = src1 != शून्यसूचकः;
    ध्रुव uint64_t ne10 = use_src1 ? src1->ne[0] : 0;
    ध्रुव uint64_t ne11 = use_src1 ? src1->ne[1] : 0;
    ध्रुव uint64_t ne12 = use_src1 ? src1->ne[2] : 0;
    ध्रुव uint64_t ne13 = use_src1 ? src1->ne[3] : 0;

    ध्रुव तार्किक use_src2 = src2 != शून्यसूचकः;
    ध्रुव तार्किक use_src3 = src3 != शून्यसूचकः;

    init_pushconst_fastdiv(pc);

    vk_pipeline pipeline;
    यदि (pipeline_override) {
        pipeline = pipeline_override;
    } अन्यथा {
        pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, op);
    }

    यदि (pipeline == शून्यसूचकः) {
        std::cerr << "ggml_vulkan: Error: Missing op: " << ggml_op_name(op) << " for " << ggml_type_name(src0->type);
        यदि (src1 != शून्यसूचकः) {
            std::cerr << " and " << ggml_type_name(src1->type);
        }
        std::cerr << " to " << ggml_type_name(dst->type) << std::पङ्क्त्यन्तः;
        GGML_ABORT("fatal error");
    }

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0, सत्य);
    vk_subbuffer src1_buf = use_src1 ? ggml_vk_tensor_subbuffer(ctx, src1, सत्य) : vk_subbuffer{};
    vk_subbuffer src2_buf = use_src2 ? ggml_vk_tensor_subbuffer(ctx, src2, सत्य) : vk_subbuffer{};
    vk_subbuffer src3_buf = use_src3 ? ggml_vk_tensor_subbuffer(ctx, src3, सत्य) : vk_subbuffer{};
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, सत्य);

    // Compute misalignment offset for descriptors and store it in in push constants.
    init_pushconst_tensor_offsets(ctx, pc, src0, src1, src2, src3, dst);

    std::array<uint32_t, 3> elements;

    विकल्प (op) {
    पक्ष GGML_OP_NORM:
    पक्ष GGML_OP_RMS_NORM_पृष्ठभागः:
    पक्ष GGML_OP_L2_NORM:
    पक्ष GGML_OP_SOFT_MAX:
    पक्ष GGML_OP_SOFT_MAX_पृष्ठभागः:
    पक्ष GGML_OP_योग_पङ्क्तिकाः:
    पक्ष GGML_OP_CUMSUM:
    पक्ष GGML_OP_माध्यम्:
    पक्ष GGML_OP_ARGMAX:
        {
            ध्रुव uint32_t nr = ggml_nrows(src0);
            यदि (nr > 262144) {
                elements = { 512, 512, CEIL_DIV(nr, 262144) };
            } अन्यथा यदि (nr > 512) {
                elements = { 512, CEIL_DIV(nr, 512), 1 };
            } अन्यथा {
                elements = { nr, 1, 1 };
            }
        } भङ्ग;
    पक्ष GGML_OP_SOLVE_TRI:
        {
            uint32_t nr = (uint32_t)(ne02 * ne03);
            यदि (nr > 262144) {
                elements = { 512, 512, CEIL_DIV(nr, 262144) };
            } अन्यथा यदि (nr > 512) {
                elements = { 512, CEIL_DIV(nr, 512), 1 };
            } अन्यथा {
                elements = { nr, 1, 1 };
            }
        }
        भङ्ग;
    पक्ष GGML_OP_RMS_NORM:
        यदि (ctx->do_सङ्कलय_rms_partials) {
            // Run one element per thread, 128 threads per workgroup
            elements = { (uint32_t)CEIL_DIV(ne00, 128), 1, 1 };
        } अन्यथा {
            elements = { (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne03 };
            elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[1]);
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[2]);
        }
        भङ्ग;

    पक्ष GGML_OP_योग:
        // We use GGML_OP_SUM_ROWS with 1 row.
        elements = { 1, 1, 1 };
        भङ्ग;
    पक्ष GGML_OP_गणः_NORM:
        {
            ध्रुव uint32_t अङ्कः_groups = dst->op_params[0];
            elements = { अङ्कः_groups * (uint32_t)src0->ne[3], 1, 1 };
        } भङ्ग;
    पक्ष GGML_OP_DIAG_आवरणम्_INF:
        elements = { (uint32_t)ggml_nrows(src0), (uint32_t)ne00, 1 };
        भङ्ग;
    पक्ष GGML_OP_ROPE:
    पक्ष GGML_OP_ROPE_पृष्ठभागः:
        {
            uint32_t nrows = (uint32_t)ggml_nrows(src0);
            uint32_t z = 1;
            यदि (nrows > ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[0]) {
                z = CEIL_DIV(nrows, 32768);
                nrows = 32768;
            }
            elements = { nrows, (uint32_t)ne00, z };

        } भङ्ग;
    पक्ष GGML_OP_GET_पङ्क्तिकाः:
        elements = { (uint32_t)ne00, (uint32_t)ne10, (uint32_t)(ne11 * ne12) };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[1]);
        elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[2]);
        भङ्ग;
    पक्ष GGML_OP_GET_पङ्क्तिकाः_पृष्ठभागः:
        elements = { (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], 1 };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[1]);
        भङ्ग;
    पक्ष GGML_OP_ARGSORT:
        GGML_ASSERT(0);
        भङ्ग;
    पक्ष GGML_OP_IM2COL:
        {
            ध्रुव तार्किक is_2D = dst->op_params[6] == 1;

            ध्रुव uint32_t IC = src1->ne[is_2D ? 2 : 1];

            ध्रुव uint32_t KH = is_2D ? src0->ne[1] : 1;
            ध्रुव uint32_t KW =         src0->ne[0];

            ध्रुव uint32_t OH = is_2D ? dst->ne[2] : 1;
            ध्रुव uint32_t OW =         dst->ne[1];

            ध्रुव uint32_t batch = src1->ne[is_2D ? 3 : 2];

            ध्रुव uint32_t CHW = IC * KH * KW;
            // Cap X workgroups to limit concurrent IC channel reads.
            // The shader loops over X to cover the full CHW dimension.
            // AMD prefers a lower limit
            ध्रुव uint32_t min_cap = ctx->device->vendor_id == VK_VENDOR_ID_AMD ? 512u : 4096u;
            ध्रुव uint32_t x_elements = std::min(CHW, std::max(min_cap, OW * KH * KW));
            elements = { x_elements, OW, OH * batch };
            elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[1]);
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[2]);
        } भङ्ग;
    पक्ष GGML_OP_IM2COL_3D:
        {
            ध्रुव uint32_t IC = ((ध्रुव uint32_t *)(dst->op_params))[9];

            ध्रुव uint32_t N  = ne13 / IC;

            ध्रुव uint32_t KD = ne02;
            ध्रुव uint32_t KH = ne01;
            ध्रुव uint32_t KW = ne00;

            ध्रुव uint32_t OD = dst->ne[3] / N;
            ध्रुव uint32_t OH = dst->ne[2];
            ध्रुव uint32_t OW = dst->ne[1];

            ध्रुव uint32_t IC_KD_KH_KW = IC*KD*KH*KW;
            ध्रुव uint32_t N_OD_OH = N*OD*OH;

            elements = { IC_KD_KH_KW, OW, N_OD_OH };
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[2]);
        } भङ्ग;
    पक्ष GGML_OP_TIMESTEP_EMBEDDING:
        {
            ध्रुव uint32_t dim = dst->op_params[0];
            uint32_t half_ceil = (dim + 1) / 2;
            elements = { half_ceil, (uint32_t)src0->ne[0], 1 };
        } भङ्ग;
    पक्ष GGML_OP_CONV_TRANSPOSE_1D:
        {
            elements = {uint32_t(src0->ne[1]), 1, 1}; // parallelize in {Cout, 1, 1}
        } भङ्ग;
    पक्ष GGML_OP_COL2IM_1D:
        {
            elements = { uint32_t(dst->ne[0]), uint32_t(dst->ne[1]), 1 };
        } भङ्ग;
    पक्ष GGML_OP_POOL_1D:
        {
            ध्रुव uint32_t N = dst->ne[3] * dst->ne[2];
            ध्रुव uint32_t OC = dst->ne[1];
            ध्रुव uint32_t OL = dst->ne[0];
            elements = { N * OC * OL, 1, 1};
        } भङ्ग;
    पक्ष GGML_OP_POOL_2D:
        {
            ध्रुव uint32_t outputs = ggml_nelements(dst);
            ध्रुव uint32_t outputs_प्रत्येकम्_workgroup = pipeline == ctx->device->pipeline_pool2d_subgroup_f32 ? 128 / ctx->device->subgroup_आकार : 512;
            ध्रुव uint32_t workgroups = CEIL_DIV(outputs, outputs_प्रत्येकम्_workgroup);
            ध्रुव uint32_t groups_y = CEIL_DIV(workgroups, ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[0]);
            ध्रुव uint32_t groups_x = CEIL_DIV(workgroups, groups_y);
            elements = { groups_x, groups_y, 1 };
        } भङ्ग;
    पक्ष GGML_OP_CONV_2D:
    पक्ष GGML_OP_CONV_TRANSPOSE_2D:
        यदि constexpr (std::is_same_v<PC, vk_op_conv2d_प्रक्षिप_constants>) {
            ध्रुव uint32_t NPQ = pc.N * pc.OH * pc.OW;
            ध्रुव vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, pc.Cout, NPQ);
            ध्रुव uint32_t NPQ_खण्डानि = CEIL_DIV(NPQ, vk_conv_खण्डम्_sizes[shape].NPQ);

            elements = { pc.Cout, NPQ_खण्डानि, 1 };
            यदि (elements[1] > 512) {
                elements[2] = CEIL_DIV(elements[1], 512);
                elements[1] = 512;
            }
        } अन्यथा {
            GGML_ABORT("invalid push constant type for CONV_2D");
        }
        भङ्ग;
    पक्ष GGML_OP_CONV_3D:
        यदि constexpr (std::is_same_v<PC, vk_op_conv3d_प्रक्षिप_constants>) {
            ध्रुव uint32_t NPQ = pc.N * pc.OD * pc.OH * pc.OW;
            ध्रुव vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, pc.OC, NPQ);
            ध्रुव uint32_t NPQ_खण्डानि = CEIL_DIV(NPQ, vk_conv_खण्डम्_sizes[shape].NPQ);

            elements = { pc.OC, NPQ_खण्डानि, 1 };
            यदि (elements[1] > 512) {
                elements[2] = CEIL_DIV(elements[1], 512);
                elements[1] = 512;
            }
        } अन्यथा {
            GGML_ABORT("invalid push constant type for CONV_3D");
        }
        भङ्ग;
    पक्ष GGML_OP_सङ्कलय:
    पक्ष GGML_OP_SUB:
    पक्ष GGML_OP_DIV:
    पक्ष GGML_OP_MUL:
    पक्ष GGML_OP_ADD1:
    पक्ष GGML_OP_बहिर्गमनम्_PROD:
    पक्ष GGML_OP_ARANGE:
    पक्ष GGML_OP_FILL:
    पक्ष GGML_OP_SCALE:
    पक्ष GGML_OP_SQR:
    पक्ष GGML_OP_SQRT:
    पक्ष GGML_OP_SIN:
    पक्ष GGML_OP_COS:
    पक्ष GGML_OP_LOG:
    पक्ष GGML_OP_TRI:
    पक्ष GGML_OP_DIAG:
    पक्ष GGML_OP_CLAMP:
    पक्ष GGML_OP_LEAKY_RELU:
    पक्ष GGML_OP_PAD:
    पक्ष GGML_OP_PAD_REFLECT_1D:
    पक्ष GGML_OP_ROLL:
    पक्ष GGML_OP_REPEAT:
    पक्ष GGML_OP_REPEAT_पृष्ठभागः:
    पक्ष GGML_OP_CPY:
    पक्ष GGML_OP_CONCAT:
    पक्ष GGML_OP_UPSCALE:
    पक्ष GGML_OP_UNARY:
    पक्ष GGML_OP_GLU:
    पक्ष GGML_OP_CONV_2D_DW:
        {
            uint32_t ne = ggml_nelements(dst);
            यदि (op == GGML_OP_CPY && ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                // Convert from number of logical elements to 2- or 4-byte units.
                ne /= ggml_blck_आकार(src0->type);
                यदि ((ggml_type_आकार(src0->type) % 4) == 0) {
                    ne *= ggml_type_आकार(src0->type) / 4;
                } अन्यथा {
                    ne *= ggml_type_आकार(src0->type) / 2;
                }
            }
            यदि (op == GGML_OP_CONCAT && ggml_is_quantized(dst->type)) {
                ne = ne / ggml_blck_आकार(dst->type) * ggml_type_आकार(dst->type) / ggml_vk_concat_unit_आकार(dst->type);
            }
            // copy_to_quant has block size of 32, and each thread does QUANT_K elements.
            // Splitting into 512x512xZ wouldn't work well since each workgroup does 1024 elements.
            // So divide by block size here before splitting into 512x512 groups.
            यदि (op == GGML_OP_CPY && !ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                ne = CEIL_DIV(ne, ggml_blck_आकार(dst->type));
            }
            यदि (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } अन्यथा यदि (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } अन्यथा {
                elements = { ne, 1, 1 };
            }

            यदि (pipeline == ctx->device->pipeline_cpy_transpose_02_32 ||
                pipeline == ctx->device->pipeline_cpy_transpose_02_16) {
                // 32x32 tiles over dims 0 and 2; dim1 and dim3 are the batch
                elements[0] = (uint32_t)CEIL_DIV(dst->ne[0], 32);
                elements[1] = (uint32_t)CEIL_DIV(dst->ne[2], 32);
                elements[2] = (uint32_t)(dst->ne[1]*dst->ne[3]);
                elements[0] = std::min(elements[0], ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[0]);
                elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[1]);
                elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[2]);
            } अन्यथा यदि (pipeline == ctx->device->pipeline_cpy_transpose_32 ||
                pipeline == ctx->device->pipeline_cpy_transpose_16) {
                // 32x32 tiles
                elements[0] = (uint32_t)CEIL_DIV(dst->ne[0], 32);
                elements[1] = (uint32_t)CEIL_DIV(dst->ne[1], 32);
                elements[2] = (uint32_t)(dst->ne[2]*dst->ne[3]);
                elements[0] = std::min(elements[0], ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[0]);
                elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[1]);
                elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[2]);
            }
        } भङ्ग;
    पक्ष GGML_OP_सङ्कलय_ID:
        {
            elements = { (uint32_t)ne01, (uint32_t)ne02, 1 };
        } भङ्ग;
    पक्ष GGML_OP_समुच्चयः_पङ्क्तिकाः:
        {
            uint32_t ne = ggml_nelements(src0);
            यदि (ggml_is_quantized(dst->type)) {
                // quants run 32 threads each doing QUANT_K elements
                ne = CEIL_DIV(ne, 32 * ggml_blck_आकार(dst->type));
            } अन्यथा {
                // scalar types do one element per thread, running 512 threads
                ne = CEIL_DIV(ne, 512);
            }
            यदि (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } अन्यथा यदि (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } अन्यथा {
                elements = { ne, 1, 1 };
            }
        }
        भङ्ग;
    पक्ष GGML_OP_SSM_CONV:
        {
            ध्रुव uint32_t nr  = src0->ne[1];
            ध्रुव uint32_t n_t = dst->ne[1];
            ध्रुव uint32_t n_s = dst->ne[2];
            elements = { nr, n_t, n_s };
        }
        भङ्ग;
    स्वभाव:
        elements = { (uint32_t)ggml_nelements(src0), 1, 1 };
        भङ्ग;
    }

    यदि (op == GGML_OP_सङ्कलय || op == GGML_OP_RMS_NORM) {
        vk_subbuffer a_buf = src0_buf;
        यदि (ctx->do_सङ्कलय_rms_partials) {
            a_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_सङ्कलय_rms_partials, ctx->prealloc_आकार_सङ्कलय_rms_partials_विस्थापनम्);
        }
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            { src0_buf, src1_buf, dst_buf, a_buf }, pc, elements);
    } अन्यथा यदि (op == GGML_OP_GLU) {
        // Empty src1 is possible in glu, but the shader needs a buffer
        vk_subbuffer subbuf1 = use_src1 ? src1_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc, elements);
    } अन्यथा यदि (op == GGML_OP_SOFT_MAX) {
        // Empty src1 and src2 is possible in soft_max, but the shader needs a buffer
        vk_subbuffer subbuf1 = use_src1 ? src1_buf : src0_buf;
        vk_subbuffer subbuf2 = use_src2 ? src2_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, subbuf2, dst_buf }, pc, elements);
    } अन्यथा यदि (op == GGML_OP_ROPE || op == GGML_OP_ROPE_पृष्ठभागः) {
        // Empty src2 and src3 is possible in rope, but the shader needs a buffer
        vk_subbuffer subbuf2 = use_src2 ? src2_buf : src0_buf;
        vk_subbuffer subbuf3 = use_src3 ? src3_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, subbuf2, dst_buf, subbuf3 }, pc, elements);
    } अन्यथा यदि (op == GGML_OP_IM2COL || op == GGML_OP_IM2COL_3D) {
        यदि (ctx->device->shader_int64 && ctx->device->बफर_device_स्थानाङ्कः) {
            // buffer device address path doesn't use dst buffer
            dst_buf.size = 1;
        }
        // im2col uses only src1 and dst buffers
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src1_buf, dst_buf }, pc, elements);
    } अन्यथा यदि (op == GGML_OP_गणनम्_EQUAL) {
        // count_equal assumes that destination buffer is initialized with zeroes
        ggml_vk_बफर_memset_async(subctx, dst_buf.बफर, dst_buf.विस्थापनम्, 0, dst_buf.size);
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, dst_buf }, pc, elements);
    } अन्यथा यदि (op == GGML_OP_OPT_चरणः_SGD) {
        // OPT_STEP_SGD works on src0, it does not need dst
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf }, pc, elements);
    } अन्यथा यदि (use_src3) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf, src3_buf, dst_buf }, pc, elements);
    } अन्यथा यदि (use_src2) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf, dst_buf }, pc, elements);
    } अन्यथा यदि (use_src1) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, dst_buf }, pc, elements);
    } अन्यथा {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, dst_buf }, pc, elements);
    }
}

शून्य ggml_vk_get_पङ्क्तिकाः(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    ध्रुव uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    ध्रुव uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    ध्रुव uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_द्विमान_प्रक्षिप_constants>(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_GET_पङ्क्तिकाः, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t) dst->nb[1] /  dst_type_आकार, (uint32_t) dst->nb[2] /  dst_type_आकार, (uint32_t) dst->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, 0,
    });
}

शून्य ggml_vk_get_पङ्क्तिकाः_पृष्ठभागः(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    ध्रुव uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    ध्रुव uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    ध्रुव uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_द्विमान_प्रक्षिप_constants>(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_GET_पङ्क्तिकाः_पृष्ठभागः, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2], (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2], (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], (uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t) dst->nb[1] /  dst_type_आकार, (uint32_t) dst->nb[2] /  dst_type_आकार, (uint32_t) dst->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, 0,
    });
}

शून्य ggml_vk_acc(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    ध्रुव uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    ध्रुव uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    ध्रुव uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    पूर्णाङ्क nb1 = dst->op_params[0] / src0_type_आकार; // 4 bytes of float32
    पूर्णाङ्क nb2 = dst->op_params[1] / src0_type_आकार; // 4 bytes of float32
    पूर्णाङ्क nb3 = dst->op_params[2] / src0_type_आकार; // 4 bytes of float32
    पूर्णाङ्क विस्थापनम् = dst->op_params[3] / src0_type_आकार; // offset in bytes

    ggml_vk_op_f32<vk_op_द्विमान_प्रक्षिप_constants>(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, dst->op, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)nb1, (uint32_t)nb2, (uint32_t)nb3,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t)nb1, (uint32_t)nb2, (uint32_t)nb3,
        0,
        0.0f, 0.0f, विस्थापनम्,
    });
}

शून्य ggml_vk_multi_सङ्कलय(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {
    ध्रुव ggml_tensor *first_ग्रन्थिः = cgraph->nodes[ग्रन्थिः_idx];
    ध्रुव ggml_tensor *dst = cgraph->nodes[ग्रन्थिः_idx + ctx->अङ्कः_additional_fused_ops];

    // Make a list of all the tensors used by the op.
    // Last element of the list is the dest tensor.
    ध्रुव ggml_tensor *tensors[MAX_PARAMETER_गणनम्];
    uint32_t अङ्कः_srcs = ctx->अङ्कः_additional_fused_ops + 2;
    uint32_t अङ्कः_tensors = अङ्कः_srcs + 1;
    GGML_ASSERT(अङ्कः_tensors + ctx->do_सङ्कलय_rms_partials <= MAX_PARAMETER_गणनम्);

    tensors[0] = first_ग्रन्थिः->src[0];
    tensors[1] = first_ग्रन्थिः->src[1];
    प्रति (int32_t i = 0; i < ctx->अङ्कः_additional_fused_ops; ++i) {
        // check whether the previous result is src[0] or src[1]
        यदि (cgraph->nodes[ग्रन्थिः_idx + i] == cgraph->nodes[ग्रन्थिः_idx + i + 1]->src[0]) {
            tensors[i+2] = cgraph->nodes[ग्रन्थिः_idx + i + 1]->src[1];
        } अन्यथा {
            tensors[i+2] = cgraph->nodes[ग्रन्थिः_idx + i + 1]->src[0];
        }
    }
    tensors[अङ्कः_srcs] = dst;

    vk_op_multi_सङ्कलय_प्रक्षिप_constants pc;
    pc.ne20 = (uint32_t)dst->ne[0];
    pc.ne21 = (uint32_t)dst->ne[1];
    pc.ne22 = (uint32_t)dst->ne[2];
    pc.ne23 = (uint32_t)dst->ne[3];

    प्रति (uint32_t i = 0; i < अङ्कः_tensors; ++i) {
        ध्रुव ggml_tensor *t = tensors[i];
        pc.nb[i][0] = (uint32_t)t->nb[0] / परिमाण(भिन्न);
        pc.nb[i][1] = (uint32_t)t->nb[1] / परिमाण(भिन्न);
        pc.nb[i][2] = (uint32_t)t->nb[2] / परिमाण(भिन्न);
        pc.nb[i][3] = (uint32_t)t->nb[3] / परिमाण(भिन्न);
    }
    pc.rms_partials = ctx->do_सङ्कलय_rms_partials;

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, tensors[0], tensors[1], शून्यसूचकः, dst, dst->op);

    यदि (pipeline == शून्यसूचकः) {
        std::cerr << "ggml_vulkan: Error: Missing multi_add";
        GGML_ABORT("fatal error");
    }

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ggml_backend_vk_बफर_context * buf_ctx[MAX_PARAMETER_गणनम्];
    vk_बफर buf[MAX_PARAMETER_गणनम्];
    size_t विस्थापनम्[MAX_PARAMETER_गणनम्];
    तार्किक uma[MAX_PARAMETER_गणनम्];

    प्रति (uint32_t i = 0; i < अङ्कः_tensors; ++i) {
        buf_ctx[i] = (ggml_backend_vk_बफर_context *)tensors[i]->बफर->context;
        buf[i] = शून्यसूचकः;
        विस्थापनम्[i] = 0;
        uma[i] = असत्य;

        यदि (ctx->device->uma) {
            ggml_vk_host_get(ctx->device, tensors[i]->दत्त, buf[i], विस्थापनम्[i]);
            uma[i] = buf[i] != शून्यसूचकः;
        }
        यदि (!uma[i]) {
            buf[i] = buf_ctx[i]->dev_बफर;
            विस्थापनम्[i] = vk_tensor_विस्थापनम्(tensors[i]) + tensors[i]->view_offs;
        }
        GGML_ASSERT(buf[i] != शून्यसूचकः);
    }
    // If any remaining descriptors are unused, just point them at src[0]
    प्रति (uint32_t i = अङ्कः_tensors; i < MAX_PARAMETER_गणनम्; ++i) {
        buf[i] = buf[0];
        विस्थापनम्[i] = 0;
    }
    यदि (ctx->do_सङ्कलय_rms_partials) {
        buf[अङ्कः_tensors] = ctx->prealloc_सङ्कलय_rms_partials;
        विस्थापनम्[अङ्कः_tensors] = ctx->prealloc_आकार_सङ्कलय_rms_partials_विस्थापनम्;
    }

    std::array<uint32_t, 3> elements;

    uint32_t ne = ggml_nelements(dst);
    यदि (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } अन्यथा यदि (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } अन्यथा {
        elements = { ne, 1, 1 };
    }

    static_assert(MAX_PARAMETER_गणनम् == 12);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            ggml_vk_subbuffer(ctx, buf[0], विस्थापनम्[0]),
            ggml_vk_subbuffer(ctx, buf[1], विस्थापनम्[1]),
            ggml_vk_subbuffer(ctx, buf[2], विस्थापनम्[2]),
            ggml_vk_subbuffer(ctx, buf[3], विस्थापनम्[3]),
            ggml_vk_subbuffer(ctx, buf[4], विस्थापनम्[4]),
            ggml_vk_subbuffer(ctx, buf[5], विस्थापनम्[5]),
            ggml_vk_subbuffer(ctx, buf[6], विस्थापनम्[6]),
            ggml_vk_subbuffer(ctx, buf[7], विस्थापनम्[7]),
            ggml_vk_subbuffer(ctx, buf[8], विस्थापनम्[8]),
            ggml_vk_subbuffer(ctx, buf[9], विस्थापनम्[9]),
            ggml_vk_subbuffer(ctx, buf[10], विस्थापनम्[10]),
            ggml_vk_subbuffer(ctx, buf[11], विस्थापनम्[11]),
        }, pc, elements);
}

शून्य ggml_vk_सङ्कलय(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    ध्रुव uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    ध्रुव uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    ध्रुव uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_द्विमान_प्रक्षिप_constants>(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_सङ्कलय, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t) dst->nb[1] /  dst_type_आकार, (uint32_t) dst->nb[2] /  dst_type_आकार, (uint32_t) dst->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, ctx->do_सङ्कलय_rms_partials,
    });
}

शून्य ggml_vk_बहिर्गमनम्_prod(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    ध्रुव uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    ध्रुव uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    ध्रुव uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_द्विमान_प्रक्षिप_constants>(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_बहिर्गमनम्_PROD, {
        (uint32_t)ggml_nelements(dst),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3],
        (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3],
        (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3],
        (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t) dst->nb[1] /  dst_type_आकार, (uint32_t) dst->nb[2] /  dst_type_आकार, (uint32_t) dst->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, 0,
    });
}

शून्य ggml_vk_sub(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    ध्रुव uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    ध्रुव uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    ध्रुव uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_द्विमान_प्रक्षिप_constants>(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_SUB, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t) dst->nb[1] /  dst_type_आकार, (uint32_t) dst->nb[2] /  dst_type_आकार, (uint32_t) dst->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, 0,
    });
}

शून्य ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    ध्रुव uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    ध्रुव uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    ध्रुव uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_द्विमान_प्रक्षिप_constants>(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_MUL, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t) dst->nb[1] /  dst_type_आकार, (uint32_t) dst->nb[2] /  dst_type_आकार, (uint32_t) dst->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, 0,
    });
}

पूर्णाङ्क ggml_vk_unary_mul_op_सूचकांक(ggml_unary_op op) {
    विकल्प (op) {
        पक्ष GGML_UNARY_OP_GELU:     प्रत्यागम 0;
        पक्ष GGML_UNARY_OP_SIGMOID:  प्रत्यागम 1;
        पक्ष GGML_UNARY_OP_SILU:     प्रत्यागम 2;
        पक्ष GGML_UNARY_OP_SOFTPLUS: प्रत्यागम 3;
        स्वभाव:                     प्रत्यागम -1;
    }
}

शून्य ggml_vk_unary_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {
    ध्रुव ggml_tensor * unary = cgraph->nodes[ग्रन्थिः_idx];
    ggml_tensor * mul = cgraph->nodes[ग्रन्थिः_idx + 1];

    // unary on src1 that tiles into src0
    ध्रुव तार्किक op_on_b = mul->src[1] == unary &&
                         !ggml_are_same_shape(unary->src[0], mul->src[0]) &&
                         ggml_can_repeat(unary, mul->src[0]);

    ध्रुव ggml_tensor * src0 = op_on_b ? mul->src[0] : unary->src[0];
    ध्रुव ggml_tensor * src1 = op_on_b ? unary->src[0] :
        ((mul->src[0] == unary) ? mul->src[1] : mul->src[0]);

    ध्रुव तार्किक f16 = src0->type == GGML_TYPE_F16;
    ध्रुव तार्किक norepeat = ggml_are_same_shape(src0, src1);
    ध्रुव पूर्णाङ्क oi = ggml_vk_unary_mul_op_सूचकांक(ggml_get_unary_op(unary));
    यदि (oi < 0) {
        GGML_ABORT("fatal error");
    }
    vk_pipeline pipeline = ctx->device->pipeline_unary_mul[oi][f16][norepeat][op_on_b];

    ध्रुव uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    ध्रुव uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    ध्रुव uint32_t dst_type_आकार = ggml_type_आकार(mul->type);

    ggml_vk_op_f32<vk_op_द्विमान_प्रक्षिप_constants>(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, mul, GGML_OP_UNARY, {
        (uint32_t)ggml_nelements(op_on_b ? mul : src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) mul->ne[0], (uint32_t) mul->ne[1], (uint32_t) mul->ne[2],(uint32_t) mul->ne[3], (uint32_t) mul->nb[0] /  dst_type_आकार, (uint32_t) mul->nb[1] /  dst_type_आकार, (uint32_t) mul->nb[2] /  dst_type_आकार, (uint32_t) mul->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, 0,
    }, pipeline);
}

शून्य ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    ध्रुव uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    ध्रुव uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    ध्रुव uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_द्विमान_प्रक्षिप_constants>(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_DIV, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t) dst->nb[1] /  dst_type_आकार, (uint32_t) dst->nb[2] /  dst_type_आकार, (uint32_t) dst->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, 0,
    });
}

शून्य ggml_vk_सङ्कलय_id(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ध्रुव ggml_tensor * src2, ggml_tensor * dst) {
    ध्रुव uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    ध्रुव uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    ध्रुव uint32_t src2_type_आकार = ggml_type_आकार(src2->type);

    ggml_vk_op_f32<vk_op_सङ्कलय_id_प्रक्षिप_constants>(ctx, subctx, src0, src1, src2, शून्यसूचकः, dst, GGML_OP_सङ्कलय_ID, {
        (uint32_t)dst->ne[0],
        (uint32_t)dst->ne[1],
        (uint32_t)src0->nb[1] / src0_type_आकार,
        (uint32_t)src0->nb[2] / src0_type_आकार,
        (uint32_t)src1->nb[1] / src1_type_आकार,
        (uint32_t)src2->nb[1] / src2_type_आकार,
    });
}

स्थिर शून्य ggml_vk_op_f32_wkv(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, ध्रुव vk_op_rwkv_wkv6_प्रक्षिप_constants&& pc, पूर्णाङ्क संस्करणम्_2) {
    GGML_ASSERT(संस्करणम्_2 == 6 || संस्करणम्_2 == 7);
    पूर्णाङ्क अङ्कः_srcs = संस्करणम्_2 == 6 ? 6 : 7;

    प्रति (पूर्णाङ्क i = 0; i < अङ्कः_srcs; i++) {
        GGML_ASSERT(!ggml_is_quantized(dst->src[i]->type));
    }

    GGML_ASSERT(dst->बफर != शून्यसूचकः);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != शून्यसूचकः);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[7] = {};
    प्रति (पूर्णाङ्क i = 0; i < अङ्कः_srcs; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    std::array<uint32_t, 3> elements = {
        (uint32_t)(pc.B * pc.H),
        1,
        1
    };

    यदि (संस्करणम्_2 == 6) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], dst_buf},
            pc, elements);
    } अन्यथा यदि (संस्करणम्_2 == 7) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], src_buf[6], dst_buf},
            pc, elements);
    } अन्यथा {
        // shouldn't happen
        GGML_ASSERT(असत्य);
    }
}

शून्य ggml_vk_rwkv_wkv6(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ध्रुव size_t seq_दीर्घता = dst->src[0]->ne[2];
    ध्रुव size_t n_embed = dst->ne[0];
    ध्रुव size_t n_heads = dst->src[0]->ne[1];
    ध्रुव size_t n_seqs = dst->src[5]->ne[1];

    ggml_vk_op_f32_wkv(
        ctx, subctx, dst,
        {
            (uint32_t)n_seqs,
            (uint32_t)seq_दीर्घता,
            (uint32_t)n_embed,
            (uint32_t)n_heads,
        },
        6
    );
}

शून्य ggml_vk_rwkv_wkv7(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ध्रुव size_t seq_दीर्घता = dst->src[0]->ne[2];
    ध्रुव size_t n_embed = dst->ne[0];
    ध्रुव size_t n_heads = dst->src[0]->ne[1];
    ध्रुव size_t n_seqs = dst->src[6]->ne[1];

    ggml_vk_op_f32_wkv(
        ctx, subctx, dst,
        {
            (uint32_t)n_seqs,
            (uint32_t)seq_दीर्घता,
            (uint32_t)n_embed,
            (uint32_t)n_heads,
        },
        7
    );
}

शून्य ggml_vk_gated_रैखिकम्_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ध्रुव size_t seq_दीर्घता = dst->src[0]->ne[2];
    ध्रुव size_t n_embed    = dst->ne[0];
    ध्रुव size_t n_heads    = dst->src[0]->ne[1];
    ध्रुव size_t n_seqs     = dst->src[4]->ne[1];

    भिन्न scale;
    स्मृतिं_प्रतिलिख(&scale, dst->op_params, परिमाण(भिन्न));

    GGML_ASSERT(dst->बफर != शून्यसूचकः);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != शून्यसूचकः);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[5] = {};
    प्रति (पूर्णाङ्क i = 0; i < 5; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    ध्रुव vk_op_gated_रैखिकम्_attn_प्रक्षिप_constants pc = {
        (uint32_t)n_seqs,
        (uint32_t)seq_दीर्घता,
        (uint32_t)n_embed,
        (uint32_t)n_heads,
        scale,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], dst_buf},
        pc, { (uint32_t)(n_seqs * n_heads), 1, 1 });
}

शून्य ggml_vk_lightning_indexer(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ध्रुव ggml_tensor * q = dst->src[0];
    ध्रुव ggml_tensor * k = dst->src[1];
    ध्रुव ggml_tensor * w = dst->src[2];
    ध्रुव ggml_tensor * m = dst->src[3];

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, q, k, w, dst, dst->op);
    GGML_ASSERT(pipeline != शून्यसूचकः);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ध्रुव uint32_t n_kv      = k->ne[2];
    ध्रुव uint32_t n_heads   = q->ne[1];
    ध्रुव uint32_t n_tokens  = q->ne[2];
    ध्रुव uint32_t n_streams = q->ne[3];
    ध्रुव uint32_t n_masks   = m->ne[3];

    // one workgroup per tile of 64 keys and 8 tokens, see lightning_indexer.comp
    ध्रुव uint32_t n_tiles_kv = CEIL_DIV(n_kv, 64);
    ध्रुव uint32_t n_tiles_t  = CEIL_DIV(n_tokens, 8);

    // q, w and dst are f32 and m is f16, so their strides are passed in elements;
    // k may be quantized, so its strides stay in bytes
    ध्रुव uint32_t q_nb1 = q->nb[1] / परिमाण(भिन्न);
    ध्रुव uint32_t q_nb2 = q->nb[2] / परिमाण(भिन्न);
    ध्रुव uint32_t q_nb3 = q->nb[3] / परिमाण(भिन्न);
    ध्रुव uint32_t k_nb2 = k->nb[2];
    ध्रुव uint32_t k_nb3 = k->nb[3];
    ध्रुव uint32_t w_nb1 = w->nb[1] / परिमाण(भिन्न);
    ध्रुव uint32_t w_nb3 = w->nb[3] / परिमाण(भिन्न);
    ध्रुव uint32_t m_nb1 = m->nb[1] / परिमाण(ggml_fp16_t);
    ध्रुव uint32_t m_nb3 = m->nb[3] / परिमाण(ggml_fp16_t);
    ध्रुव uint32_t d_nb1 = dst->nb[1] / परिमाण(भिन्न);
    ध्रुव uint32_t d_nb3 = dst->nb[3] / परिमाण(भिन्न);

    ध्रुव vk_op_lightning_indexer_प्रक्षिप_constants pc = {
        n_kv, n_heads, n_tokens, n_masks,
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

शून्य ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src_q     = dst->src[0];
    ध्रुव ggml_tensor * src_v     = dst->src[2];
    ध्रुव ggml_tensor * src_beta  = dst->src[4];

    GGML_ASSERT(dst->बफर != शून्यसूचकः);

    ध्रुव uint32_t S_v      = (uint32_t)src_v->ne[0];
    ध्रुव uint32_t H        = (uint32_t)src_v->ne[1];
    ध्रुव uint32_t n_tokens = (uint32_t)src_v->ne[2];
    ध्रुव uint32_t n_seqs   = (uint32_t)src_v->ne[3];

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ध्रुव uint32_t K = (uint32_t)ggml_get_op_params_i32(dst, 0);

    ध्रुव uint32_t s_off = S_v * H * n_tokens * n_seqs;

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != शून्यसूचकः);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[6] = {};
    प्रति (पूर्णाङ्क i = 0; i < 6; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    ध्रुव uint32_t sq1 = (uint32_t)(src_q->nb[1] / परिमाण(भिन्न));
    ध्रुव uint32_t sq2 = (uint32_t)(src_q->nb[2] / परिमाण(भिन्न));
    ध्रुव uint32_t sq3 = (uint32_t)(src_q->nb[3] / परिमाण(भिन्न));
    ध्रुव uint32_t sv1 = (uint32_t)(src_v->nb[1] / परिमाण(भिन्न));
    ध्रुव uint32_t sv2 = (uint32_t)(src_v->nb[2] / परिमाण(भिन्न));
    ध्रुव uint32_t sv3 = (uint32_t)(src_v->nb[3] / परिमाण(भिन्न));
    ध्रुव uint32_t sb1 = (uint32_t)(src_beta->nb[1] / परिमाण(भिन्न));
    ध्रुव uint32_t sb2 = (uint32_t)(src_beta->nb[2] / परिमाण(भिन्न));
    ध्रुव uint32_t sb3 = (uint32_t)(src_beta->nb[3] / परिमाण(भिन्न));

    ध्रुव uint32_t neq1 = (uint32_t)src_q->ne[1];
    ध्रुव uint32_t rq3  = (uint32_t)(src_v->ne[3] / src_q->ne[3]);

    ध्रुव भिन्न scale = 1.0f / sqrtf((भिन्न)S_v);
    ध्रुव vk_op_gated_delta_net_प्रक्षिप_constants pc = {
        H, n_tokens, n_seqs, s_off,
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

शून्य ggml_vk_ssm_अवलोकनम्(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    ध्रुव ggml_tensor * src2 = dst->src[2];
    ध्रुव ggml_tensor * src3 = dst->src[3];
    ध्रुव ggml_tensor * src4 = dst->src[4];
    ध्रुव ggml_tensor * src5 = dst->src[5];

    GGML_ASSERT(dst->बफर != शून्यसूचकः);

    ध्रुव uint32_t शीर्ष_dim = src0->ne[1];
    ध्रुव uint32_t n_शीर्ष = src1->ne[1];
    ध्रुव uint32_t n_गणः = src4->ne[1];
    ध्रुव uint32_t n_tok = src1->ne[2];
    ध्रुव uint32_t n_seq = src1->ne[3];

    तार्किक is_mamba2 = (src3->nb[1] == परिमाण(भिन्न));
    GGML_ASSERT(is_mamba2);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, dst->op);
    GGML_ASSERT(pipeline != शून्यसूचकः);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ध्रुव int64_t s_off = ggml_nelements(src1) * परिमाण(भिन्न);

    ध्रुव vk_op_ssm_अवलोकनम्_प्रक्षिप_constants pc = {
        (uint32_t)src0->nb[2], (uint32_t)src0->nb[3],
        (uint32_t)src1->nb[2], (uint32_t)src1->nb[3],
        (uint32_t)src2->nb[1], (uint32_t)src2->nb[2],
        (uint32_t)src3->nb[1],
        (uint32_t)src4->nb[2], (uint32_t)src4->nb[3],
        (uint32_t)src5->nb[2], (uint32_t)src5->nb[3],
        (uint32_t)s_off,
        n_शीर्ष, शीर्ष_dim, n_गणः, n_tok,
        n_seq, (uint32_t) ggml_get_op_params_i32(dst, 0)
    };

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[7] = {};
    प्रति (पूर्णाङ्क i = 0; i < 7 && dst->src[i] != शून्यसूचकः; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    std::array<uint32_t, 3> elements;

    ध्रुव uint32_t d_state = src0->ne[0];
    uint32_t अङ्कः_subgroups = d_state / ctx->device->subgroup_आकार;
    ध्रुव uint32_t अङ्कः_workgroups_x = CEIL_DIV(n_शीर्ष * शीर्ष_dim, अङ्कः_subgroups);
    ध्रुव uint32_t अङ्कः_workgroups_y = n_seq;
    elements = { अङ्कः_workgroups_x, अङ्कः_workgroups_y, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], src_buf[6], dst_buf},
        pc, elements);
}

शून्य ggml_vk_ssm_conv(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {
    ggml_tensor * conv = cgraph->nodes[ग्रन्थिः_idx];
    ध्रुव ggml_tensor * src0 = conv->src[0];
    ध्रुव ggml_tensor * src1 = conv->src[1];

    // Pick the destination tensor (last node in the fused chain) and the optional bias.
    // Fusion modes: 0 = ssm_conv, 1 = ssm_conv+silu, 2 = ssm_conv+add(bias)+silu.
    ggml_tensor * dst = conv;
    ध्रुव ggml_tensor * bias = शून्यसूचकः;

    यदि (ctx->अङ्कः_additional_fused_ops == 1) {
        dst = cgraph->nodes[ग्रन्थिः_idx + 1]; // silu
    } अन्यथा यदि (ctx->अङ्कः_additional_fused_ops == 2) {
        ggml_tensor * सङ्कलय_2 = cgraph->nodes[ग्रन्थिः_idx + 1];
        bias = (सङ्कलय_2->src[0] == conv) ? सङ्कलय_2->src[1] : सङ्कलय_2->src[0];
        dst = cgraph->nodes[ग्रन्थिः_idx + 2]; // silu
    }

    // The shader always declares 4 bindings; bind src0 as a dummy when bias isn't fused.
    ध्रुव ggml_tensor * src2 = bias ? bias : src0;

    ggml_vk_op_f32<vk_op_ssm_conv_प्रक्षिप_constants>(ctx, subctx, src0, src1, src2, शून्यसूचकः, dst, GGML_OP_SSM_CONV, {
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

स्थिर शून्य ggml_vk_op_f32_opt_चरणः_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, ध्रुव vk_op_प्रक्षिप_constants&& pc) {
    ध्रुव ggml_tensor * x = dst->src[0];
    ध्रुव ggml_tensor * g = dst->src[1];
    ध्रुव ggml_tensor * gm = dst->src[2];
    ध्रुव ggml_tensor * gv = dst->src[3];
    ध्रुव ggml_tensor * p = dst->src[4];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(g->type == GGML_TYPE_F32);
    GGML_ASSERT(gm->type == GGML_TYPE_F32);
    GGML_ASSERT(gv->type == GGML_TYPE_F32);
    GGML_ASSERT(p->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->बफर != शून्यसूचकः);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(gm));
    GGML_ASSERT(ggml_is_contiguous(gv));
    GGML_ASSERT(ggml_is_contiguous(p));
    GGML_ASSERT(ggml_are_same_shape(x, g));
    GGML_ASSERT(ggml_are_same_shape(x, gm));
    GGML_ASSERT(ggml_are_same_shape(x, gv));
    GGML_ASSERT(ggml_nelements(p) == 7);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, g, gm, gv, dst, GGML_OP_OPT_चरणः_ADAMW);
    GGML_ASSERT(pipeline != शून्यसूचकः);

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

शून्य ggml_vk_opt_चरणः_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ध्रुव size_t n = ggml_nelements(dst->src[0]);

    ggml_vk_op_f32_opt_चरणः_adamw(
        ctx, subctx, dst,
        { (uint32_t)n, 0, 0.0f, 0.0f, 0.0f, 0.0f }
    );
}

शून्य ggml_vk_opt_चरणः_sgd(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ध्रुव ggml_tensor * src2, ggml_tensor * dst) {
    ध्रुव size_t n = ggml_nelements(dst->src[0]);

    ggml_vk_op_f32<vk_op_प्रक्षिप_constants>(ctx, subctx, src0, src1, src2, शून्यसूचकः, dst, GGML_OP_OPT_चरणः_SGD, { (uint32_t)n, 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

शून्य ggml_vk_concat(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    पूर्णाङ्क * op_params = (पूर्णाङ्क *)dst->op_params;

    ध्रुव uint32_t unit_आकार = ggml_vk_concat_unit_आकार(dst->type);
    ध्रुव uint32_t units_प्रत्येकम्_खण्डम् = ggml_type_आकार(dst->type) / unit_आकार;
    ध्रुव uint32_t खण्डम्_आकार_3 = ggml_blck_आकार(dst->type);
    ध्रुव तार्किक quantized = ggml_is_quantized(dst->type);

    // Address dimension 0 in packed storage units; higher strides may be noncontiguous.
    ध्रुव uint32_t ne00 = src0->ne[0] / खण्डम्_आकार_3 * units_प्रत्येकम्_खण्डम्;
    ध्रुव uint32_t ne10 = src1->ne[0] / खण्डम्_आकार_3 * units_प्रत्येकम्_खण्डम्;
    ध्रुव uint32_t ne20 =  dst->ne[0] / खण्डम्_आकार_3 * units_प्रत्येकम्_खण्डम्;
    ध्रुव uint32_t nb00 = quantized ? 1 : src0->nb[0] / unit_आकार;
    ध्रुव uint32_t nb10 = quantized ? 1 : src1->nb[0] / unit_आकार;
    ध्रुव uint32_t nb20 = quantized ? 1 :  dst->nb[0] / unit_आकार;

    vk_op_concat_प्रक्षिप_constants pc {{
        ne20 * (uint32_t)dst->ne[1] * (uint32_t)dst->ne[2] * (uint32_t)dst->ne[3],
        ne00, (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], nb00, (uint32_t)src0->nb[1] / unit_आकार, (uint32_t)src0->nb[2] / unit_आकार, (uint32_t)src0->nb[3] / unit_आकार,
        ne10, (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], nb10, (uint32_t)src1->nb[1] / unit_आकार, (uint32_t)src1->nb[2] / unit_आकार, (uint32_t)src1->nb[3] / unit_आकार,
        ne20, (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], nb20, (uint32_t) dst->nb[1] / unit_आकार, (uint32_t) dst->nb[2] / unit_आकार, (uint32_t) dst->nb[3] / unit_आकार,
        0,
        0.0f, 0.0f, op_params[0],
    }};
    ggml_vk_op_f32<vk_op_concat_प्रक्षिप_constants>(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_CONCAT, std::move(pc));
}

शून्य ggml_vk_upscale(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    ध्रुव uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    ध्रुव uint32_t mode = (uint32_t)ggml_get_op_params_i32(dst, 0);

    GGML_TENSOR_UNARY_OP_LOCALS

    भिन्न sf0 = (भिन्न)ne0 / ne00;
    भिन्न sf1 = (भिन्न)ne1 / ne01;
    भिन्न sf2 = (भिन्न)ne2 / ne02;
    भिन्न sf3 = (भिन्न)ne3 / ne03;
    भिन्न pixel_विस्थापनम् = 0.5f;

    यदि (mode & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = ne0 > 1 && ne00 > 1 ? (भिन्न)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (भिन्न)(ne1 - 1) / (ne01 - 1) : sf1;
        pixel_विस्थापनम् = 0.0f;
    }

    ggml_vk_op_f32<vk_op_upscale_प्रक्षिप_constants>(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_UPSCALE, {
        (uint32_t)ggml_nelements(dst), 0, 0,
        (uint32_t)ne00, (uint32_t)ne01,
        (uint32_t)nb00 / src0_type_आकार, (uint32_t)nb01 / src0_type_आकार, (uint32_t)nb02 / src0_type_आकार, (uint32_t)nb03 / src0_type_आकार,
        (uint32_t)ne0, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
        sf0, sf1, sf2, sf3, pixel_विस्थापनम्
    });
}

शून्य ggml_vk_scale(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_प्रक्षिप_constants p = vk_op_unary_प्रक्षिप_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);
    p.param2 = ggml_get_op_params_f32(dst, 1);

    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_SCALE, std::move(p));
}

शून्य ggml_vk_sqr(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_SQR, vk_op_unary_प्रक्षिप_constants_init(src0, dst));
}

शून्य ggml_vk_sqrt(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_SQRT, vk_op_unary_प्रक्षिप_constants_init(src0, dst));
}

शून्य ggml_vk_add1(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    ध्रुव uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    ध्रुव uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    ध्रुव uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_द्विमान_प्रक्षिप_constants>(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_ADD1, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t) dst->nb[1] /  dst_type_आकार, (uint32_t) dst->nb[2] /  dst_type_आकार, (uint32_t) dst->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, 0,
    });
}

शून्य ggml_vk_arange(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_arange(dst=" << dst << ", ne=" << ggml_nelements(dst) << ")");

    vk_op_प्रक्षिप_constants pc = {
        (uint32_t)ggml_nelements(dst),
        1,
        ggml_get_op_params_f32(dst, 0),
        ggml_get_op_params_f32(dst, 2),
        0.0f, 0.0f,
    };

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_ARANGE);
    GGML_ASSERT(pipeline != शून्यसूचकः);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, असत्य);

    std::array<uint32_t, 3> elements = { (uint32_t)ggml_nelements(dst), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { dst_buf }, pc, elements);
}

शून्य ggml_vk_fill(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_fill(dst=" << dst << ", ne=" << ggml_nelements(dst) << ")");
    ध्रुव uint64_t n = ggml_nelements(dst);
    GGML_ASSERT(n > 0);

    vk_op_प्रक्षिप_constants pc = {
        (uint32_t)n,
        1,
        ggml_get_op_params_f32(dst, 0),
        0.0f,
        0.0f, 0.0f,
    };

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_FILL);
    GGML_ASSERT(pipeline != शून्यसूचकः);

    // Split the task distribution to 2D to avoid exceeding maxComputeWorkGroupCount
    ध्रुव uint32_t योग_wg = CEIL_DIV(n, pipeline->wg_denoms[0]);
    ध्रुव uint32_t wg_x = std::min(योग_wg, ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[0]);
    ध्रुव uint32_t wg_y = CEIL_DIV(योग_wg, wg_x);
    GGML_ASSERT(wg_y <= ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[1]);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, असत्य);

    std::array<uint32_t, 3> elements = { wg_x * pipeline->wg_denoms[0], wg_y * pipeline->wg_denoms[1], 1 };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { dst_buf }, pc, elements);
}

शून्य ggml_vk_sin(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_SIN, vk_op_unary_प्रक्षिप_constants_init(src0, dst));
}

शून्य ggml_vk_cos(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_COS, vk_op_unary_प्रक्षिप_constants_init(src0, dst));
}

शून्य ggml_vk_log(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_LOG, vk_op_unary_प्रक्षिप_constants_init(src0, dst));
}

शून्य ggml_vk_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_प्रक्षिप_constants p = vk_op_unary_प्रक्षिप_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);

    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_TRI, std::move(p));
}

शून्य ggml_vk_diag(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_प्रक्षिप_constants p = vk_op_unary_प्रक्षिप_constants_init(src0, dst, ggml_nelements(dst));

    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_DIAG, std::move(p));
}

शून्य ggml_vk_clamp(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_प्रक्षिप_constants p = vk_op_unary_प्रक्षिप_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);
    p.param2 = ggml_get_op_params_f32(dst, 1);

    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_CLAMP, std::move(p));
}

शून्य ggml_vk_pad(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_pad_प्रक्षिप_constants p = vk_op_pad_प्रक्षिप_constants_init(src0, dst);
    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_PAD, std::move(p));
}

शून्य ggml_vk_pad_reflect_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    ध्रुव uint32_t p0 = (uint32_t)dst->op_params[0];
    ध्रुव uint32_t p1 = (uint32_t)dst->op_params[1];

    vk_op_unary_प्रक्षिप_constants p = vk_op_unary_प्रक्षिप_constants_init(src0, dst, ggml_nelements(dst));
    स्मृतिं_प्रतिलिख(&p.param1, &p0, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&p.param2, &p1, परिमाण(भिन्न));

    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_PAD_REFLECT_1D, std::move(p));
}

शून्य ggml_vk_roll(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    ध्रुव int32_t s0 = ggml_get_op_params_i32(dst, 0);
    ध्रुव int32_t s1 = ggml_get_op_params_i32(dst, 1);
    ध्रुव int32_t s2 = ggml_get_op_params_i32(dst, 2);
    ध्रुव int32_t s3 = ggml_get_op_params_i32(dst, 3);
    ध्रुव uint32_t s01_packed = ((s0 + 0x8000) << 16) | (s1 + 0x8000);
    ध्रुव uint32_t s23_packed = ((s2 + 0x8000) << 16) | (s3 + 0x8000);

    vk_op_unary_प्रक्षिप_constants p = vk_op_unary_प्रक्षिप_constants_init(src0, dst);
    स्मृतिं_प्रतिलिख(&p.param1, &s01_packed, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&p.param2, &s23_packed, परिमाण(भिन्न));

    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_ROLL, std::move(p));
}

शून्य ggml_vk_repeat(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_प्रक्षिप_constants p = vk_op_unary_प्रक्षिप_constants_init(src0, dst, ggml_nelements(dst));
    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_REPEAT, std::move(p));
}

शून्य ggml_vk_repeat_पृष्ठभागः(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_प्रक्षिप_constants p = vk_op_unary_प्रक्षिप_constants_init(src0, dst, ggml_nelements(dst));
    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_REPEAT_पृष्ठभागः, std::move(p));
}

शून्य ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ne = (uint32_t)ggml_nelements(src0);
    यदि (ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
        // Convert from number of logical elements to 2- or 4-byte units.
        ne /= ggml_blck_आकार(src0->type);
        यदि ((ggml_type_आकार(src0->type) % 4) == 0) {
            ne *= ggml_type_आकार(src0->type) / 4;
        } अन्यथा {
            ne *= ggml_type_आकार(src0->type) / 2;
        }
    }

    vk_op_unary_प्रक्षिप_constants p = vk_op_unary_प्रक्षिप_constants_init(src0, dst, ne);
    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_CPY, std::move(p));
}

शून्य ggml_vk_समुच्चयः_पङ्क्तिकाः(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    ध्रुव uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    ध्रुव uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    ध्रुव uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    // Skip empty skip_rows operations. For most ops the empty check at the start
    // of ggml_vk_build_graph is sufficient, but set_rows can have a nonempty dst
    // with empty srcs.
    यदि (ggml_is_रिक्तानि(src0) || ggml_is_रिक्तानि(src1)) {
        प्रत्यागम;
    }

    ggml_vk_op_f32<vk_op_द्विमान_प्रक्षिप_constants>(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_समुच्चयः_पङ्क्तिकाः, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t) dst->nb[1] /  dst_type_आकार, (uint32_t) dst->nb[2] /  dst_type_आकार, (uint32_t) dst->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, 0,
    });
}

शून्य ggml_vk_silu_पृष्ठभागः(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_प्रक्षिप_constants>(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_SILU_पृष्ठभागः, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

शून्य ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    भिन्न * op_params = (भिन्न *)dst->op_params;
    vk_op_unary_प्रक्षिप_constants p = vk_op_unary_प्रक्षिप_constants_init(src0, dst);
    p.param1 = op_params[0];

    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_NORM, std::move(p));
}

शून्य ggml_vk_गणः_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    ध्रुव पूर्णाङ्क * int_op_params = (ध्रुव पूर्णाङ्क *)dst->op_params;
    ध्रुव भिन्न * float_op_params = (ध्रुव भिन्न *)dst->op_params;

    ध्रुव uint32_t अङ्कः_groups = int_op_params[0];
    ध्रुव भिन्न eps = float_op_params[1];
    ध्रुव uint32_t गणः_आकार = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + अङ्कः_groups - 1) / अङ्कः_groups);

    ggml_vk_op_f32<vk_op_प्रक्षिप_constants>(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_गणः_NORM, { गणः_आकार, 0, eps, 0.0f, 0.0f, 0.0f });
}

स्थिर uint32_t ggml_vk_rms_अङ्कः_partials(ggml_backend_vk_context * ctx, ध्रुव ggml_tensor *ग्रन्थिः) {
    ध्रुव uint32_t ne = (uint32_t)ग्रन्थिः->ne[0];
    ध्रुव uint32_t denom = ctx->device->pipeline_सङ्कलय_rms[0][0][0]->wg_denoms[0];
    ध्रुव uint32_t अङ्कः_partials = CEIL_DIV(ne, denom);
    प्रत्यागम अङ्कः_partials;
}

uint32_t ggml_vk_rms_partials_आकार(ggml_backend_vk_context * ctx, ध्रुव ggml_tensor *ग्रन्थिः) {
    ध्रुव uint32_t अङ्कः_partials = ggml_vk_rms_अङ्कः_partials(ctx, ग्रन्थिः);
    ध्रुव uint32_t अङ्कः_अष्टकानि = ROUNDUP_POW2(अङ्कः_partials * परिमाण(uint32_t), ctx->device->partials_binding_alignment);
    प्रत्यागम अङ्कः_अष्टकानि;
}

स्थिर vk_op_rope_प्रक्षिप_constants ggml_vk_make_rope_constants(ध्रुव ggml_tensor *dst, ध्रुव ggml_tensor *src0, ध्रुव तार्किक has_ff, तार्किक backprop, ध्रुव uint32_t समुच्चयः_पङ्क्तिकाः_stride) {
    ध्रुव पूर्णाङ्क n_dims        = ((ध्रुव int32_t *) dst->op_params)[1];
    ध्रुव पूर्णाङ्क mode          = ((ध्रुव int32_t *) dst->op_params)[2];
    ध्रुव पूर्णाङ्क n_offs        = ((ध्रुव int32_t *) dst->op_params)[15];
    // const int n_ctx         = ((const int32_t *) dst->op_params)[3];
    ध्रुव पूर्णाङ्क n_ctx_orig    = ((ध्रुव int32_t *) dst->op_params)[4];
    ध्रुव भिन्न freq_आधारः   = ((ध्रुव भिन्न *)   dst->op_params)[5];
    ध्रुव भिन्न freq_scale  = ((ध्रुव भिन्न *)   dst->op_params)[6];
    ध्रुव भिन्न ext_गुणकः  = ((ध्रुव भिन्न *)   dst->op_params)[7];
    ध्रुव भिन्न attn_गुणकः = ((ध्रुव भिन्न *)   dst->op_params)[8];
    ध्रुव भिन्न beta_fast   = ((ध्रुव भिन्न *)   dst->op_params)[9];
    ध्रुव भिन्न beta_slow   = ((ध्रुव भिन्न *)   dst->op_params)[10];
    पूर्णाङ्क sections[4] {};
    यदि (mode & GGML_ROPE_TYPE_MROPE) {
        स्मृतिं_प्रतिलिख(sections, (ध्रुव int32_t *) dst->op_params + 11, परिमाण(पूर्णाङ्क)*4);
    }

    ध्रुव तार्किक is_imrope = mode == GGML_ROPE_TYPE_IMROPE;

    भिन्न corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_आधारः, beta_fast, beta_slow, corr_dims);

    ध्रुव भिन्न theta_scale = powf(freq_आधारः, -2.0f/n_dims);

    uint32_t nb01 = src0->nb[1] / ggml_type_आकार(src0->type);
    uint32_t nb02 = src0->nb[2] / ggml_type_आकार(src0->type);
    uint32_t nb03 = src0->nb[3] / ggml_type_आकार(src0->type);

    uint32_t nb11 = dst->nb[1] / ggml_type_आकार(dst->type);
    uint32_t nb12 = dst->nb[2] / ggml_type_आकार(dst->type);
    uint32_t nb13 = dst->nb[3] / ggml_type_आकार(dst->type);

    vk_op_rope_प्रक्षिप_constants rope {
        (uint32_t)mode, (uint32_t)ggml_nrows(src0), (uint32_t)n_dims, (uint32_t)n_offs, freq_scale,
        freq_आधारः, ext_गुणकः, attn_गुणकः, {corr_dims[0], corr_dims[1]}, theta_scale, has_ff,
        { sections[0], sections[1], sections[2], sections[3] }, is_imrope, backprop, समुच्चयः_पङ्क्तिकाः_stride,

        (uint32_t)src0->ne[0],
        (uint32_t)src0->ne[1],
        (uint32_t)src0->ne[2],
        nb01, nb02, nb03,
        nb11, nb12, nb13,
        0, 0, // a_offset, d_offset filled in by init_pushconst_tensor_offsets
    };

    प्रत्यागम rope;
}

स्थिर शून्य ggml_vk_rms_norm_finish(ggml_backend_vk_context * ctx, ध्रुव ggml_tensor * src0) {
    यदि (ctx->do_सङ्कलय_rms_partials_विस्थापनम्_calculation) {
        ctx->prealloc_आकार_सङ्कलय_rms_partials_विस्थापनम् += ggml_vk_rms_partials_आकार(ctx, src0);
        ctx->do_सङ्कलय_rms_partials = असत्य;
        ctx->do_सङ्कलय_rms_partials_विस्थापनम्_calculation = असत्य;
    }
}

शून्य ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx, भिन्न * op_params) {
    ggml_tensor * rms = cgraph->nodes[ग्रन्थिः_idx];
    ध्रुव ggml_tensor * src0 = rms->src[0];

    यदि (ctx->fused_rms_norm_बहुलकम् == RMS_NORM_VIEW_समुच्चयः_पङ्क्तिकाः) {
        GGML_ASSERT(ctx->अङ्कः_additional_fused_ops == 2);
        ggml_tensor * समुच्चयः_पङ्क्तिकाः = cgraph->nodes[ग्रन्थिः_idx + 2];
        ध्रुव ggml_tensor * indices = समुच्चयः_पङ्क्तिकाः->src[1];
        vk_op_द्विमान_प्रक्षिप_constants pc = ggml_vk_rms_norm_प्रक्षिप_constants(src0, src0, समुच्चयः_पङ्क्तिकाः, op_params[0], 0);
        init_pushconst_tensor_offsets(ctx, pc, src0, src0, शून्यसूचकः, शून्यसूचकः, समुच्चयः_पङ्क्तिकाः);

        vk_pipeline pipeline = समुच्चयः_पङ्क्तिकाः->type == GGML_TYPE_F16 ?
            ctx->device->pipeline_rms_norm_समुच्चयः_पङ्क्तिकाः_f32_f16 : ctx->device->pipeline_rms_norm_समुच्चयः_पङ्क्तिकाः_f32_f32;
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {
                ggml_vk_tensor_subbuffer(ctx, src0, सत्य),
                ggml_vk_tensor_subbuffer(ctx, src0, सत्य),
                ggml_vk_tensor_subbuffer(ctx, समुच्चयः_पङ्क्तिकाः, सत्य),
                ggml_vk_tensor_subbuffer(ctx, indices),
            }, pc, {
                (uint32_t)src0->ne[1],
                std::min((uint32_t)src0->ne[2], ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[1]),
                std::min((uint32_t)src0->ne[3], ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[2]),
            });
        ggml_vk_rms_norm_finish(ctx, src0);
        प्रत्यागम;
    }

    यदि (ctx->fused_rms_norm_बहुलकम् == RMS_NORM_MUL_सङ्कलय || ctx->fused_rms_norm_बहुलकम् == RMS_NORM_MUL_सङ्कलय_MUL) {
        ggml_tensor * mul = cgraph->nodes[ग्रन्थिः_idx + 1];
        ggml_tensor * सङ्कलय_2 = cgraph->nodes[ग्रन्थिः_idx + 2];
        ध्रुव ggml_tensor * weight = mul->src[0] == rms ? mul->src[1] : mul->src[0];
        ध्रुव ggml_tensor * residual = सङ्कलय_2->src[0] == mul ? सङ्कलय_2->src[1] : सङ्कलय_2->src[0];
        ध्रुव तार्किक do_post_multiply = ctx->fused_rms_norm_बहुलकम् == RMS_NORM_MUL_सङ्कलय_MUL;
        GGML_ASSERT(ctx->अङ्कः_additional_fused_ops == (do_post_multiply ? 3 : 2));
        ggml_tensor * dst = do_post_multiply ? cgraph->nodes[ग्रन्थिः_idx + 3] : सङ्कलय_2;
        ध्रुव ggml_tensor * post_scale = do_post_multiply ?
            (dst->src[0] == सङ्कलय_2 ? dst->src[1] : dst->src[0]) : src0;

        ध्रुव uint32_t अङ्कः_partials = ctx->do_सङ्कलय_rms_partials ? ggml_vk_rms_अङ्कः_partials(ctx, dst) : 0;
        vk_op_द्विमान_प्रक्षिप_constants pc = ggml_vk_rms_norm_प्रक्षिप_constants(src0, weight, dst, op_params[0], अङ्कः_partials);
        init_pushconst_tensor_offsets(ctx, pc, src0, weight, residual, post_scale, dst);

        vk_pipeline pipeline;
        यदि (ctx->do_सङ्कलय_rms_partials) {
            pipeline = do_post_multiply ?
                ctx->device->pipeline_rms_norm_mul_सङ्कलय_mul_partials_f32 : ctx->device->pipeline_rms_norm_mul_सङ्कलय_partials_f32;
        } अन्यथा {
            pipeline = do_post_multiply ?
                ctx->device->pipeline_rms_norm_mul_सङ्कलय_mul_f32 : ctx->device->pipeline_rms_norm_mul_सङ्कलय_f32;
        }
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        यदि (ctx->do_सङ्कलय_rms_partials) {
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                {
                    ggml_vk_tensor_subbuffer(ctx, src0, सत्य),
                    ggml_vk_tensor_subbuffer(ctx, weight, सत्य),
                    ggml_vk_tensor_subbuffer(ctx, dst, सत्य),
                    ggml_vk_subbuffer(ctx, ctx->prealloc_सङ्कलय_rms_partials, ctx->prealloc_आकार_सङ्कलय_rms_partials_विस्थापनम्),
                    ggml_vk_tensor_subbuffer(ctx, residual),
                    ggml_vk_tensor_subbuffer(ctx, post_scale),
                }, pc, { (uint32_t)CEIL_DIV(src0->ne[0], 128), 1, 1 });
        } अन्यथा {
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                {
                    ggml_vk_tensor_subbuffer(ctx, src0, सत्य),
                    ggml_vk_tensor_subbuffer(ctx, weight, सत्य),
                    ggml_vk_tensor_subbuffer(ctx, dst, सत्य),
                    ggml_vk_tensor_subbuffer(ctx, residual),
                    ggml_vk_tensor_subbuffer(ctx, post_scale),
                }, pc, {
                    (uint32_t)src0->ne[1],
                    std::min((uint32_t)src0->ne[2], ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[1]),
                    std::min((uint32_t)src0->ne[3], ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[2]),
                });
        }
        ggml_vk_rms_norm_finish(ctx, src0);
        प्रत्यागम;
    }

    ggml_tensor * dst;
    ध्रुव ggml_tensor * src1;

    यदि (ctx->fused_rms_norm_बहुलकम् != RMS_NORM_गणनम्) {
        ggml_tensor * mul = cgraph->nodes[ग्रन्थिः_idx + 1];
        dst = mul;
        src1 = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    } अन्यथा {
        dst = rms;
        src1 = src0;
    }

    ध्रुव uint32_t अङ्कः_partials = ctx->do_सङ्कलय_rms_partials ? ggml_vk_rms_अङ्कः_partials(ctx, dst) : 0;
    vk_op_द्विमान_प्रक्षिप_constants bin = ggml_vk_rms_norm_प्रक्षिप_constants(src0, src1, dst, op_params[0], अङ्कः_partials);

    यदि (ctx->fused_rms_norm_बहुलकम् == RMS_NORM_MUL_ROPE ||
        ctx->fused_rms_norm_बहुलकम् == RMS_NORM_MUL_ROPE_VIEW_समुच्चयः_पङ्क्तिकाः) {
        स्थिर constexpr uint32_t max_tensors = 7;
        ध्रुव ggml_tensor *tensors[max_tensors] {};

        ggml_tensor *rms = cgraph->nodes[ग्रन्थिः_idx + 0];
        ggml_tensor *mul = cgraph->nodes[ग्रन्थिः_idx + 1];
        ggml_tensor *rope = cgraph->nodes[ग्रन्थिः_idx + 2];

        ggml_tensor *other_src = mul->src[0] == rms ? mul->src[1] : mul->src[0];

        तार्किक do_समुच्चयः_पङ्क्तिकाः = ctx->fused_rms_norm_बहुलकम् == RMS_NORM_MUL_ROPE_VIEW_समुच्चयः_पङ्क्तिकाः;
        GGML_ASSERT(ctx->अङ्कः_additional_fused_ops == (do_समुच्चयः_पङ्क्तिकाः ? 4 : 2));

        tensors[0] = rms->src[0];
        tensors[1] = other_src;
        tensors[2] = mul;
        tensors[3] = rope->src[1]; // pos
        tensors[4] = rope->src[2]; // ff
        tensors[5] = cgraph->nodes[ग्रन्थिः_idx + ctx->अङ्कः_additional_fused_ops]; // dst
        tensors[6] = do_समुच्चयः_पङ्क्तिकाः ? tensors[5]->src[1] : शून्यसूचकः;
        ध्रुव uint32_t समुच्चयः_पङ्क्तिकाः_stride = do_समुच्चयः_पङ्क्तिकाः ? tensors[5]->nb[1] / ggml_type_आकार(tensors[5]->type) : 0;

        vk_op_rms_norm_mul_rope_प्रक्षिप_constants pc;
        pc.bin = bin;
        pc.rope = ggml_vk_make_rope_constants(rope, rope->src[0], tensors[4] != शून्यसूचकः, असत्य, समुच्चयः_पङ्क्तिकाः_stride);

        vk_pipeline pipeline = tensors[5]->type == GGML_TYPE_F16 ? ctx->device->pipeline_rms_norm_mul_rope_f32_f16 : ctx->device->pipeline_rms_norm_mul_rope_f32_f32;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

        ggml_backend_vk_बफर_context * buf_ctx[max_tensors];
        vk_बफर buf[max_tensors];
        size_t विस्थापनम्[max_tensors];
        तार्किक uma[max_tensors];

        प्रति (uint32_t i = 0; i < max_tensors; ++i) {
            यदि (!tensors[i]) {
                // If any remaining descriptors are unused, just point them at src[0]
                buf[i] = buf[0];
                विस्थापनम्[i] = 0;
                अनुवर्त;
            }
            buf_ctx[i] = (ggml_backend_vk_बफर_context *)tensors[i]->बफर->context;
            buf[i] = शून्यसूचकः;
            विस्थापनम्[i] = 0;
            uma[i] = असत्य;

            यदि (ctx->device->uma) {
                ggml_vk_host_get(ctx->device, tensors[i]->दत्त, buf[i], विस्थापनम्[i]);
                uma[i] = buf[i] != शून्यसूचकः;
            }
            यदि (!uma[i]) {
                buf[i] = buf_ctx[i]->dev_बफर;
                विस्थापनम्[i] = vk_tensor_विस्थापनम्(tensors[i]) + tensors[i]->view_offs;
            }
            GGML_ASSERT(buf[i] != शून्यसूचकः);
        }

        // a_offset is unused (the fused path reads from shared memory), but the rope/set_rows dst can be misaligned.
        // Round the binding offset down to the storage buffer alignment; the in-element shift goes in pc.rope.d_offset.
        pc.rope.d_विस्थापनम् = get_misalign_अष्टकानि(ctx, tensors[5]) / ggml_type_आकार(tensors[5]->type);
        विस्थापनम्[5] &= ~(size_t(ctx->device->properties.limits.min_Storage_बफर_विस्थापनम्_Alignment) - 1);

        std::array<uint32_t, 3> elements;
        elements = { (uint32_t)rms->src[0]->ne[1], (uint32_t)rms->src[0]->ne[2], (uint32_t)rms->src[0]->ne[3] };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[1]);
        elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[2]);

        static_assert(max_tensors == 7);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {
                ggml_vk_subbuffer(ctx, buf[0], विस्थापनम्[0]),
                ggml_vk_subbuffer(ctx, buf[1], विस्थापनम्[1]),
                ggml_vk_subbuffer(ctx, buf[2], विस्थापनम्[2]),
                ggml_vk_subbuffer(ctx, buf[3], विस्थापनम्[3]),
                ggml_vk_subbuffer(ctx, buf[4], विस्थापनम्[4]),
                ggml_vk_subbuffer(ctx, buf[5], विस्थापनम्[5]),
                ggml_vk_subbuffer(ctx, buf[6], विस्थापनम्[6]),
            }, pc, elements);
    } अन्यथा {
        GGML_ASSERT(ctx->fused_rms_norm_बहुलकम् == RMS_NORM_MUL || ctx->fused_rms_norm_बहुलकम् == RMS_NORM_गणनम्);
        ggml_vk_op_f32<vk_op_द्विमान_प्रक्षिप_constants>(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_RMS_NORM, std::move(bin));
    }

    ggml_vk_rms_norm_finish(ctx, src0);
}

शून्य ggml_vk_rms_norm_पृष्ठभागः(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    भिन्न * op_params = (भिन्न *)dst->op_params;
    ggml_vk_op_f32<vk_op_प्रक्षिप_constants>(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_RMS_NORM_पृष्ठभागः, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], 0.0f, 0.0f, 0.0f });
}

शून्य ggml_vk_l2_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    ध्रुव भिन्न * op_params = (ध्रुव भिन्न *)dst->op_params;
    vk_op_unary_प्रक्षिप_constants p = vk_op_unary_प्रक्षिप_constants_init(src0, dst);
    p.param1 = op_params[0];
    ggml_vk_op_f32<vk_op_unary_प्रक्षिप_constants>(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_L2_NORM, std::move(p));
}

शून्य ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_UNARY, vk_op_unary_प्रक्षिप_constants_init(src0, dst));
}

शून्य ggml_vk_xielu(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    भिन्न * op_params = (भिन्न *)dst->op_params;
    vk_op_unary_प्रक्षिप_constants p = vk_op_unary_प्रक्षिप_constants_init(src0, dst);
    p.param1 = op_params[1];
    p.param2 = op_params[2];
    p.param3 = op_params[3];
    p.param4 = op_params[4];
    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_UNARY, std::move(p));
}

शून्य ggml_vk_glu(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    ध्रुव भिन्न * op_params_f = (ध्रुव भिन्न *)dst->op_params;

    ध्रुव तार्किक swapped = (तार्किक)dst->op_params[1];
    ध्रुव तार्किक विभज = src1 != शून्यसूचकः;
    ध्रुव भिन्न alpha = op_params_f[2];
    ध्रुव भिन्न सीमा = op_params_f[3];

    यदि (!विभज) {
        GGML_ASSERT(src0->ne[0] / 2 == dst->ne[0]);
    } अन्यथा {
        GGML_ASSERT(src0->ne[0] == src1->ne[0]);
        GGML_ASSERT(src0->ne[0] == dst->ne[0]);
        GGML_ASSERT(src0->type == src1->type);
    }

    ध्रुव uint32_t mode = विभज ? 2 : (swapped ? 1 : 0);
    ध्रुव uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    ध्रुव uint32_t src1_type_आकार = विभज ? ggml_type_आकार(src1->type) : src0_type_आकार;
    ध्रुव uint32_t dst_type_आकार  = ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_glu_प्रक्षिप_constants>(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_GLU,
        {
            (uint32_t)ggml_nelements(dst),
            (uint32_t)src0->ne[0],
            (uint32_t)dst->ne[0],
            mode,
            alpha,
            सीमा,
            (uint32_t)(src0->nb[0] / src0_type_आकार),
            (uint32_t)(src0->nb[1] / src0_type_आकार),
            (uint32_t)(src0->nb[2] / src0_type_आकार),
            (uint32_t)(src0->nb[3] / src0_type_आकार),
            (uint32_t)((विभज ? src1->nb[0] : src0->nb[0]) / src1_type_आकार),
            (uint32_t)((विभज ? src1->nb[1] : src0->nb[1]) / src1_type_आकार),
            (uint32_t)((विभज ? src1->nb[2] : src0->nb[2]) / src1_type_आकार),
            (uint32_t)((विभज ? src1->nb[3] : src0->nb[3]) / src1_type_आकार),
            (uint32_t)(dst->nb[0] / dst_type_आकार),
            (uint32_t)(dst->nb[1] / dst_type_आकार),
            (uint32_t)(dst->nb[2] / dst_type_आकार),
            (uint32_t)(dst->nb[3] / dst_type_आकार),
            (uint32_t)dst->ne[1],
            (uint32_t)dst->ne[2],
            0,
            0, 0, 0, 0, 0, 0,
        });
}

शून्य ggml_vk_diag_आवरणम्_inf(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    int32_t * op_params = (int32_t *)dst->op_params;
    ggml_vk_op_f32<vk_op_diag_आवरणम्_प्रक्षिप_constants>(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_DIAG_आवरणम्_INF, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0] });
}

शून्य ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ध्रुव ggml_tensor * src2, ggml_tensor * dst) {
    भिन्न * op_params = (भिन्न *)dst->op_params;

    भिन्न scale = op_params[0];
    भिन्न max_bias = op_params[1];

    ध्रुव uint32_t ncols =   (uint32_t)src0->ne[0];
    ध्रुव uint32_t nrows_x = (uint32_t)ggml_nrows(src0);
    ध्रुव uint32_t nrows_y = (uint32_t)src0->ne[1];

    ध्रुव uint32_t ne12 = src1 ? (uint32_t)(src1->ne[2]) : 0u;
    ध्रुव uint32_t ne13 = src1 ? (uint32_t)(src1->ne[3]) : 0u;
    ध्रुव uint32_t nb11 = src1 ? (uint32_t)(src1->nb[1] / src1->nb[0]) : 0u;
    ध्रुव uint32_t nb12 = src1 ? (uint32_t)(src1->nb[2] / src1->nb[0]) : 0u;
    ध्रुव uint32_t nb13 = src1 ? (uint32_t)(src1->nb[3] / src1->nb[0]) : 0u;

    ध्रुव uint32_t n_शीर्ष_kv   = src0->ne[2];
    ध्रुव uint32_t n_शीर्ष_log2 = 1u << (uint32_t) floorf(log2f((भिन्न) n_शीर्ष_kv));

    ध्रुव भिन्न m0 = powf(2.0f, -(max_bias       ) / n_शीर्ष_log2);
    ध्रुव भिन्न m1 = powf(2.0f, -(max_bias / 2.0f) / n_शीर्ष_log2);

    vk_op_soft_max_प्रक्षिप_constants pc {
        ncols,
        src1 != शून्यसूचकः ? nrows_y : (uint32_t)0,
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
        ne12, ne13,
        nb11, nb12, nb13,
        scale, max_bias,
        m0, m1,
        n_शीर्ष_log2,
        nrows_x,
        src2 != शून्यसूचकः
    };

    यदि (ncols <= 16384) {
        ggml_vk_op_f32<vk_op_soft_max_प्रक्षिप_constants>(ctx, subctx, src0, src1, src2, शून्यसूचकः, dst, GGML_OP_SOFT_MAX, std::move(pc));
    } अन्यथा {

        vk_subbuffer buf_a = ggml_vk_tensor_subbuffer(ctx, src0);
        vk_subbuffer buf_b = src1 ? ggml_vk_tensor_subbuffer(ctx, src1) : buf_a;
        vk_subbuffer buf_c = src2 ? ggml_vk_tensor_subbuffer(ctx, src2) : buf_a;
        vk_subbuffer buf_d = ggml_vk_tensor_subbuffer(ctx, dst);

        uint32_t elems_प्रत्येकम्_wg = 128 * 4;
        uint32_t अङ्कः_wgs = CEIL_DIV(ncols, elems_प्रत्येकम्_wg);
        size_t tmp_आकार = अङ्कः_wgs * nrows_x * परिमाण(भिन्न);

        यदि (ctx->prealloc_आकार_x < tmp_आकार) {
            ctx->prealloc_आकार_x = tmp_आकार;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        यदि (ctx->prealloc_आकार_y < tmp_आकार) {
            ctx->prealloc_आकार_y = tmp_आकार;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        यदि (ctx->prealloc_x_need_sync || ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        vk_subbuffer buf_x = { ctx->prealloc_x, 0, tmp_आकार };
        vk_subbuffer buf_y = { ctx->prealloc_y, 0, tmp_आकार };

        // the partial results overwrite a matmul input converted into prealloc_y
        ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् = शून्यसूचकः;
        ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् = शून्यसूचकः;

        std::array<uint32_t, 3> elements = { अङ्कः_wgs, nrows_x, 1 };

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

        ctx->prealloc_x_need_sync = सत्य;
        ctx->prealloc_y_need_sync = सत्य;
    }
}

शून्य ggml_vk_soft_max_पृष्ठभागः(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    भिन्न * op_params = (भिन्न *)dst->op_params;
    ggml_vk_op_f32<vk_op_प्रक्षिप_constants>(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_SOFT_MAX_पृष्ठभागः, { (uint32_t)src0->ne[0], (uint32_t)ggml_nrows(src0), op_params[0], op_params[1], 0.0f, 0.0f });
}

शून्य ggml_vk_topk_moe(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {
    topk_moe_बहुलकम् mode = ctx->fused_topk_moe_बहुलकम्;
    ध्रुव तार्किक has_bias = mode == TOPK_MOE_SIGMOID_NORM_BIAS || mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS;
    ggml_tensor * logits = cgraph->nodes[ग्रन्थिः_idx + 0]->src[0];
    ggml_tensor * bias = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? cgraph->nodes[ग्रन्थिः_idx + 2]->src[1] :
                         mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? cgraph->nodes[ग्रन्थिः_idx + 3]->src[1] :
                                                                   logits;
    ggml_tensor * weights = cgraph->nodes[ग्रन्थिः_idx + ctx->अङ्कः_additional_fused_ops];
    ggml_tensor * ids = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? cgraph->nodes[ग्रन्थिः_idx + 4] :
                        mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? cgraph->nodes[ग्रन्थिः_idx + 5] :
                        mode == TOPK_MOE_LATE_SOFTMAX             ? cgraph->nodes[ग्रन्थिः_idx + 1] :
                                                                   cgraph->nodes[ग्रन्थिः_idx + 3];

    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(bias->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ध्रुव पूर्णाङ्क n_experts = logits->ne[0];
    ध्रुव पूर्णाङ्क n_पङ्क्तिकाः    = logits->ne[1];
    ध्रुव पूर्णाङ्क n_expert_प्रयुक्तम् = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_आकार(ids->type) == (size_t) n_experts);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, cgraph->nodes[ग्रन्थिः_idx], GGML_OP_SOFT_MAX);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer logits_buf = ggml_vk_tensor_subbuffer(ctx, logits);
    vk_subbuffer bias_buf = ggml_vk_tensor_subbuffer(ctx, bias);
    vk_subbuffer weights_buf = ggml_vk_tensor_subbuffer(ctx, weights);
    vk_subbuffer ids_buf = ggml_vk_tensor_subbuffer(ctx, ids);

    vk_op_topk_moe_प्रक्षिप_constants pc {};
    pc.n_पङ्क्तिकाः = n_पङ्क्तिकाः;
    pc.n_experts_प्रक्षिप = n_experts;
    pc.n_expert_प्रयुक्तम् = n_expert_प्रयुक्तम्;
    pc.clamp_min = -std::numeric_limits<भिन्न>::infinity();
    pc.clamp_max = std::numeric_limits<भिन्न>::infinity();
    यदि (mode == TOPK_MOE_EARLY_SOFTMAX_NORM) {
        ggml_tensor * clamp = cgraph->nodes[ग्रन्थिः_idx + 7];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }
    यदि (mode == TOPK_MOE_SIGMOID_NORM_BIAS) {
        ggml_tensor * clamp = cgraph->nodes[ग्रन्थिः_idx + 8];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }
    यदि (mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS) {
        ggml_tensor * clamp = cgraph->nodes[ग्रन्थिः_idx + 9];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }

#परिभाषय GATING_FUNC_SOFTMAX 0
#परिभाषय GATING_FUNC_SIGMOID 1
#परिभाषय GATING_FUNC_SOFTMAX_WEIGHT 2
#परिभाषय GATING_FUNC_SQRT_SOFTPLUS 3

    pc.gating_func = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? GATING_FUNC_SIGMOID :
                     mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? GATING_FUNC_SQRT_SOFTPLUS :
                     mode == TOPK_MOE_LATE_SOFTMAX             ? GATING_FUNC_SOFTMAX_WEIGHT :
                                                                 GATING_FUNC_SOFTMAX;
    pc.has_bias = has_bias;
    pc.with_norm = mode == TOPK_MOE_EARLY_SOFTMAX_NORM || has_bias;
    यदि (ctx->fused_topk_moe_scale) {
        GGML_ASSERT(weights->op == GGML_OP_SCALE);
        pc.output_scale = ggml_get_op_params_f32(weights, 0);
        pc.output_bias = ggml_get_op_params_f32(weights, 1);
    } अन्यथा {
        pc.output_scale = 1.0f;
        pc.output_bias = 0.0f;
    }

    GGML_ASSERT(n_expert_प्रयुक्तम् <= n_experts);

    ध्रुव uint32_t पङ्क्तिकाः_प्रत्येकम्_खण्डम् = 4;
    std::array<uint32_t, 3> elements = { CEIL_DIV(n_पङ्क्तिकाः, पङ्क्तिकाः_प्रत्येकम्_खण्डम्), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, {logits_buf, bias_buf, weights_buf, ids_buf}, pc, elements);
}

शून्य ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx, तार्किक backprop) {
    ggml_tensor * dst = cgraph->nodes[ग्रन्थिः_idx];
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    ध्रुव ggml_tensor * src2 = dst->src[2];
    ध्रुव ggml_tensor * src3 = शून्यसूचकः;
    ध्रुव पूर्णाङ्क n_dims        = ((int32_t *) dst->op_params)[1];
    ध्रुव पूर्णाङ्क mode          = ((int32_t *) dst->op_params)[2];
    // const int n_ctx         = ((int32_t *) dst->op_params)[3];
    ध्रुव पूर्णाङ्क n_ctx_orig    = ((int32_t *) dst->op_params)[4];
    ध्रुव भिन्न freq_आधारः   = ((भिन्न *)   dst->op_params)[5];
    ध्रुव भिन्न beta_fast   = ((भिन्न *)   dst->op_params)[9];
    ध्रुव भिन्न beta_slow   = ((भिन्न *)   dst->op_params)[10];
    पूर्णाङ्क sections[4] {};
    यदि (mode & GGML_ROPE_TYPE_MROPE) {
        स्मृतिं_प्रतिलिख(sections, (int32_t *) dst->op_params + 11, परिमाण(पूर्णाङ्क)*4);
    }

    भिन्न corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_आधारः, beta_fast, beta_slow, corr_dims);

    uint32_t समुच्चयः_पङ्क्तिकाः_stride = 0;
    // Fused rope + view + set_rows passes the set_rows destination stride in set_rows_stride
    // and overrides the dst and sets src3=row_indices
    यदि (ctx->अङ्कः_additional_fused_ops > 0) {
        समुच्चयः_पङ्क्तिकाः_stride = cgraph->nodes[ग्रन्थिः_idx + 2]->nb[1] / ggml_type_आकार(cgraph->nodes[ग्रन्थिः_idx + 2]->type);
        src3 = cgraph->nodes[ग्रन्थिः_idx + 2]->src[1];
        dst = cgraph->nodes[ग्रन्थिः_idx + 2];
    }

    ggml_vk_op_f32<vk_op_rope_प्रक्षिप_constants>(ctx, subctx, src0, src1, src2, src3, dst, GGML_OP_ROPE,
        ggml_vk_make_rope_constants(cgraph->nodes[ग्रन्थिः_idx], src0, src2 != शून्यसूचकः, backprop, समुच्चयः_पङ्क्तिकाः_stride));
}

शून्य ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    ध्रुव uint32_t * op_params = (ध्रुव uint32_t *)dst->op_params;

    uint32_t ncols = src0->ne[0];
    uint32_t nrows = ggml_nrows(src0);

    uint32_t ncols_pad_log2 = (uint32_t)ceilf(log2f(भिन्न(ncols)));
    uint32_t ncolsp2 = 1 << ncols_pad_log2;

    vk_op_argsort_प्रक्षिप_constants pc { ncols, ncolsp2, ncols_pad_log2, nrows, op_params[0], 0, 0, 0, 0, };

    // Pick the largest workgroup size <= ncolsp2
    uint32_t pipeline_idx = std::min(ncols_pad_log2, अङ्कः_argsort_pipelines - 1);

    uint32_t max_wg_log2 = std::min(ctx->device->max_workgroup_आकार_log2, अङ्कः_argsort_pipelines - 1);

    // Use the "small" argsort shader if the whole sort can be done by a single workgroup.
    तार्किक use_small = ncols_pad_log2 <= max_wg_log2 &&
                     ctx->device->pipeline_argsort_f32[pipeline_idx] != शून्यसूचकः;

    vk_pipeline pipeline = use_small ? ctx->device->pipeline_argsort_f32[pipeline_idx]
                                     : ctx->device->pipeline_argsort_large_f32[pipeline_idx];

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer subbuf1 = dst_buf;

    // Reserve space for ivec2 per element, with rows padded to a power of two
    यदि (!use_small) {
        ध्रुव size_t x_sz = size_t{ncolsp2} * nrows * 2 * परिमाण(पूर्णाङ्क);

        यदि (ctx->prealloc_आकार_x < x_sz) {
            ctx->prealloc_आकार_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        यदि (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
        subbuf1 = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    }

    std::array<uint32_t, 3> elements;

    elements[0] = ncolsp2;
    elements[1] = std::min((uint32_t)ggml_nrows(src0), ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[1]);
    elements[2] = 1;

    // First dispatch initializes tmp_idx and does the first N passes where
    // there is only communication between threads in the same workgroup.
    {
        vk_op_argsort_प्रक्षिप_constants pc2 = pc;
        pc2.outer_start = 0;
        pc2.outer_अन्तः = std::min(ncols_pad_log2, max_wg_log2);
        pc2.inner_start = 0;
        pc2.inner_अन्तः = 100;
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc2, elements);
    }
    यदि (!use_small) {
        ggml_vk_sync_buffers(ctx, subctx);
        // Loop over outer/inner passes, synchronizing between each pass.
        प्रति (uint32_t outer = max_wg_log2; outer < ncols_pad_log2; ++outer) {
            प्रति (uint32_t inner = 0; inner < outer + 1; ++inner) {
                vk_op_argsort_प्रक्षिप_constants pc2 = pc;
                pc2.outer_start = outer;
                pc2.outer_अन्तः = outer + 1;
                pc2.inner_start = inner;
                pc2.inner_अन्तः = inner + 1;
                // When the inner idx is large enough, there's only communication
                // within a workgroup. So the remaining inner iterations can all
                // run in the same dispatch.
                यदि (outer - inner < pipeline_idx) {
                    pc2.inner_अन्तः = 100;
                    inner = outer;
                    pipeline = ctx->device->pipeline_argsort_large_f32[pipeline_idx];
                } अन्यथा {
                    // Smaller workgroup empirically seems to perform better
                    pipeline = ctx->device->pipeline_argsort_large_f32[pipeline_idx - 2];
                }
                ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
                ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc2, elements);
                ggml_vk_sync_buffers(ctx, subctx);
            }
        }
        ctx->prealloc_x_need_sync = सत्य;
    }
}

शून्य ggml_vk_topk(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ncols = src0->ne[0];
    uint32_t nrows = ggml_nrows(src0);
    uint32_t k = dst->ne[0];

    // tournament path is faster where it fits; use radix-select only past its k limit
    ध्रुव uint32_t k_min_pipeline = std::max((uint32_t) log2f(भिन्न(k)) + 1, ctx->device->subgroup_आकार_log2);
    यदि (k_min_pipeline >= अङ्कः_topk_pipelines || ctx->device->pipeline_topk_f32[k_min_pipeline] == शून्यसूचकः) {
        vk_pipeline pipeline = ctx->device->pipeline_topk_radix_f32;
        GGML_ASSERT(pipeline != शून्यसूचकः);

        यदि (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        vk_op_topk_radix_प्रक्षिप_constants pc { ncols, k, nrows, 0, 0, 0 };
        std::array<uint32_t, 3> elements {
            pipeline->wg_denoms[0],
            std::min(nrows, ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[1]),
            1,
        };
        // the non-QSA path only uses bindings 0/1; bind valid buffers for the unused QSA slots
        vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
        vk_subbuffer dst_buf  = ggml_vk_tensor_subbuffer(ctx, dst);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            { src0_buf, dst_buf, src0_buf, src0_buf, src0_buf }, pc, elements);
        प्रत्यागम;
    }

    vk_op_topk_प्रक्षिप_constants pc { ncols, ncols, ncols, k, nrows, 0, 0 };

    यदि (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    std::array<uint32_t, 3> elements;
    elements[1] = std::min(nrows, ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[1]);
    elements[2] = 1;

    uint32_t अङ्कः_elements = ncols;

    // Each iteration reduces a workgroup's worth of elements down to the K
    // largest elements. Repeat until we have the top K elements.
    // Need to do at least one iteration to write out the results.
    तार्किक समाप्तम्_one_iter = असत्य;
    uint32_t dbl_buf_सूचकांक = 0;
    size_t dbl_buf_आकार;
    यावत् (अङ्कः_elements > k || !समाप्तम्_one_iter) {

        // Prefer going as small as num_topk_pipelines - 3 for perf reasons.
        // But if K is larger, then we need a larger workgroup
        uint32_t max_pipeline = अङ्कः_topk_pipelines - 1;
        uint32_t preferred_pipeline = std::max(अङ्कः_topk_pipelines - 3, (uint32_t)log2f(भिन्न(k)) + 2);
        max_pipeline = std::min(preferred_pipeline, max_pipeline);
        uint32_t min_pipeline = (uint32_t)log2f(भिन्न(k)) + 1;
        // require full subgroup
        min_pipeline = std::max(min_pipeline, ctx->device->subgroup_आकार_log2);

        uint32_t pipeline_idx = (uint32_t)ceilf(log2f(भिन्न(अङ्कः_elements)));
        pipeline_idx = std::min(pipeline_idx, max_pipeline);
        pipeline_idx = std::max(pipeline_idx, min_pipeline);

        यदि (अङ्कः_elements > (1u << pipeline_idx)) {
            // If we could finish on this loop iteration (i.e. a single workgroup)
            // then do so. It's better than the overhead of another pass.
            प्रति (uint32_t i = pipeline_idx; i < अङ्कः_topk_pipelines; ++i) {
                यदि (अङ्कः_elements <= (1u << i)) {
                    pipeline_idx = i;
                    भङ्ग;
                }
            }
        }

        vk_pipeline pipeline = ctx->device->pipeline_topk_f32[pipeline_idx];
        // If the device doesn't support a pipeline this large, use smaller
        यावत् (!pipeline) {
            pipeline_idx--;
            GGML_ASSERT(pipeline_idx >= min_pipeline);
            pipeline = ctx->device->pipeline_topk_f32[pipeline_idx];
        }

        vk_op_topk_प्रक्षिप_constants pc2 = pc;
        pc2.ncols_निवेशः = अङ्कः_elements;

        // Number of elements remaining after this pass
        uint32_t अङ्कः_dst_elements = (अङ्कः_elements / pipeline->wg_denoms[0]) * k + std::min(k, अङ्कः_elements % pipeline->wg_denoms[0]);

        pc2.ncols_output = अङ्कः_dst_elements;

        यदि (!समाप्तम्_one_iter) {
            // Reserve space for ivec2 per element, double buffered
            // K per workgroup per row
            dbl_buf_आकार = अङ्कः_dst_elements * nrows * 2 * परिमाण(पूर्णाङ्क);
            dbl_buf_आकार = ROUNDUP_POW2(dbl_buf_आकार, ctx->device->properties.limits.min_Storage_बफर_विस्थापनम्_Alignment);
            ध्रुव size_t x_sz = dbl_buf_आकार * 2;

            यदि (ctx->prealloc_आकार_x < x_sz) {
                ctx->prealloc_आकार_x = x_sz;
                ggml_vk_preallocate_buffers(ctx, subctx);
            }
        }

        vk_subbuffer src_buf;
        vk_subbuffer dst_buf;

        यदि (अङ्कः_elements == ncols) {
            pc2.first_pass = 1;
            src_buf = ggml_vk_tensor_subbuffer(ctx, src0);
        } अन्यथा {
            src_buf = { ctx->prealloc_x, dbl_buf_सूचकांक * dbl_buf_आकार, dbl_buf_आकार };
        }
        यदि (अङ्कः_dst_elements == k) {
            pc2.चरमः_pass = 1;
            dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
        } अन्यथा {
            dst_buf = { ctx->prealloc_x, (dbl_buf_सूचकांक ^ 1) * dbl_buf_आकार, dbl_buf_आकार };
        }

        elements[0] = अङ्कः_elements;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf }, pc2, elements);
        अङ्कः_elements = अङ्कः_dst_elements;
        dbl_buf_सूचकांक ^= 1;
        यदि (अङ्कः_elements > k) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
        समाप्तम्_one_iter = सत्य;
    }
    ctx->prealloc_x_need_sync = सत्य;
}

शून्य ggml_vk_topk_qsa(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {
    ध्रुव ggml_tensor * get_पङ्क्तिकाः = cgraph->nodes[ग्रन्थिः_idx + 0];
    ध्रुव ggml_tensor * सङ्कलय_2      = cgraph->nodes[ग्रन्थिः_idx + ctx->अङ्कः_additional_fused_ops - 1];
    ggml_tensor *       शिखरम्_k    = cgraph->nodes[ग्रन्थिः_idx + ctx->अङ्कः_additional_fused_ops];

    ध्रुव ggml_tensor * scores   = get_पङ्क्तिकाः->src[0]; // [n_tps, n_blocks, n_stream]
    ध्रुव ggml_tensor * cell_blk = get_पङ्क्तिकाः->src[1]; // [n_kv, n_stream]

    // raw f16 mask: follow the reshape/cpy chain back to the materialized input
    ध्रुव ggml_tensor * आवरणम् = सङ्कलय_2->src[1];
    यावत् (आवरणम्->op == GGML_OP_RESHAPE || आवरणम्->op == GGML_OP_CPY) {
        आवरणम् = आवरणम्->src[0];
    }

    ध्रुव uint32_t n_tps    = scores->ne[0];
    ध्रुव uint32_t n_खण्डानि = scores->ne[1];
    ध्रुव uint32_t n_प्रवाहः = scores->ne[2];
    ध्रुव uint32_t n_kv     = cell_blk->ne[0];
    ध्रुव uint32_t विस्तारः    = शिखरम्_k->ne[0];
    ध्रुव uint32_t nrows    = n_tps * n_प्रवाहः;

    vk_pipeline pipeline = ctx->device->pipeline_topk_radix_qsa;
    GGML_ASSERT(pipeline != शून्यसूचकः);

    // scratch holds the gathered+masked input, materialized once and reused across passes
    ध्रुव size_t scratch_आकार = size_t{ n_kv } * nrows * परिमाण(भिन्न);
    यदि (ctx->prealloc_आकार_x < scratch_आकार) {
        ctx->prealloc_आकार_x = scratch_आकार;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    यदि (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_op_topk_radix_प्रक्षिप_constants pc { n_kv, विस्तारः, nrows, n_tps, n_खण्डानि, n_प्रवाहः };
    std::array<uint32_t, 3> elements {
        pipeline->wg_denoms[0],
        std::min(nrows, ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[1]),
        1,
    };
    vk_subbuffer scratch_buf { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        { ggml_vk_tensor_subbuffer(ctx, scores), ggml_vk_tensor_subbuffer(ctx, शिखरम्_k),
          ggml_vk_tensor_subbuffer(ctx, cell_blk), ggml_vk_tensor_subbuffer(ctx, आवरणम्),
          scratch_buf }, pc, elements);
    ctx->prealloc_x_need_sync = सत्य;
}

शून्य ggml_vk_योग(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_योग_पङ्क्तिकाः_प्रक्षिप_constants p = vk_op_योग_पङ्क्तिकाः_प्रक्षिप_constants_init(src0, dst, ggml_nelements(src0));
    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_योग, p);
}

शून्य ggml_vk_योग_पङ्क्तिकाः(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_योग_पङ्क्तिकाः_प्रक्षिप_constants p = vk_op_योग_पङ्क्तिकाः_प्रक्षिप_constants_init(src0, dst, src0->ne[0]);
    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_योग_पङ्क्तिकाः, p);
}

शून्य ggml_vk_माध्यम्(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_योग_पङ्क्तिकाः_प्रक्षिप_constants p = vk_op_योग_पङ्क्तिकाः_प्रक्षिप_constants_init(src0, dst, src0->ne[0]);
    p.weight = 1.0f / (भिन्न)src0->ne[0];
    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_माध्यम्, p);
}

शून्य ggml_vk_cumsum(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_योग_पङ्क्तिकाः_प्रक्षिप_constants pc = vk_op_योग_पङ्क्तिकाः_प्रक्षिप_constants_init(src0, dst, src0->ne[0]);
    // Use the single pass shader when the rows are small or there are enough rows to fill the GPU.
    // For fewer, larger rows, use the multipass shader to spread each row across SMs.
    यदि (dst->ne[0] <= 4096 || ggml_nrows(dst) >= ctx->device->shader_core_गणनम्) {
        ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_CUMSUM, pc);
        प्रत्यागम;
    }

    // First pass computes partial sums within a block, and stores the last partial
    // to the temp buffer. Second pass sums the block partials from the temp buffer
    // and adds that to the result of the first pass.
    vk_pipeline pipeline1 = ctx->device->pipeline_cumsum_multipass1_f32;
    vk_pipeline pipeline2 = ctx->device->pipeline_cumsum_multipass2_f32;
    GGML_ASSERT(pipeline1 != शून्यसूचकः && pipeline2 != शून्यसूचकः);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline1, 1);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline2, 1);

    std::array<uint32_t, 3> elements;

    elements[0] = dst->ne[0];
    elements[1] = (uint32_t)ggml_nrows(dst);
    elements[2] = 1;

    size_t अस्थायी_आकार = परिमाण(भिन्न) * elements[0] * ggml_nrows(dst);

    यदि (ctx->prealloc_आकार_विभज_k < अस्थायी_आकार) {
        ctx->prealloc_आकार_विभज_k = अस्थायी_आकार;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }

    vk_subbuffer src_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer अस्थायी_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_विभज_k, 0);

    यदि (ctx->prealloc_विभज_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline1, {src_buf, dst_buf, अस्थायी_buf}, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline2, {src_buf, dst_buf, अस्थायी_buf}, pc, elements);

    ctx->prealloc_विभज_k_need_sync = सत्य;
}

स्थिर std::array<uint32_t, 3> ggml_vk_nrows_elements(uint32_t nr) {
    यदि (nr > 262144) {
        प्रत्यागम { 512, 512, CEIL_DIV(nr, 262144) };
    }
    यदि (nr > 512) {
        प्रत्यागम { 512, CEIL_DIV(nr, 512), 1 };
    }
    प्रत्यागम { nr, 1, 1 };
}

शून्य ggml_vk_cross_entropy_loss(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    ध्रुव uint32_t nclasses = (uint32_t)src0->ne[0];
    ध्रुव uint32_t nrows    = (uint32_t)ggml_nrows(src0);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, शून्यसूचकः, dst, GGML_OP_CROSS_ENTROPY_LOSS);
    GGML_ASSERT(pipeline != शून्यसूचकः);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_योग_पङ्क्तिकाः_f32, 1);

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer src1_buf = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer dst_buf  = ggml_vk_tensor_subbuffer(ctx, dst, सत्य);

    ध्रुव vk_op_प्रक्षिप_constants pc = { nclasses, nrows, 0.0f, 0.0f, 0.0f, 0.0f };

    ध्रुव size_t tmp_आकार = (size_t)nrows * परिमाण(भिन्न);
    यदि (ctx->prealloc_आकार_x < tmp_आकार) {
        ctx->prealloc_आकार_x = tmp_आकार;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    यदि (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_subbuffer tmp_buf = { ctx->prealloc_x, 0, tmp_आकार };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, tmp_buf }, pc, ggml_vk_nrows_elements(nrows));
    ggml_vk_sync_buffers(ctx, subctx);

    vk_op_योग_पङ्क्तिकाः_प्रक्षिप_constants sp = {};
    sp.n_स्तम्भसङ्ख्या = nrows;
    sp.ne01 = 1;
    sp.ne02 = 1;
    sp.weight = 1.0f;
    init_pushconst_fastdiv(sp);
    sp.misalign_offsets = get_misalign_अष्टकानि(ctx, dst) / ggml_type_आकार(dst->type);

    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_योग_पङ्क्तिकाः_f32, { tmp_buf, dst_buf }, sp, { 1, 1, 1 });
    ctx->prealloc_x_need_sync = सत्य;
}

शून्य ggml_vk_cross_entropy_loss_पृष्ठभागः(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ध्रुव ggml_tensor * grad   = dst->src[0];
    ध्रुव ggml_tensor * logits = dst->src[1];
    ध्रुव ggml_tensor * labels = dst->src[2];

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

    ध्रुव uint32_t nclasses = (uint32_t)logits->ne[0];
    ध्रुव uint32_t nrows    = (uint32_t)ggml_nrows(logits);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, grad, logits, labels, dst, GGML_OP_CROSS_ENTROPY_LOSS_पृष्ठभागः);
    GGML_ASSERT(pipeline != शून्यसूचकः);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer grad_buf   = ggml_vk_tensor_subbuffer(ctx, grad);
    vk_subbuffer logits_buf = ggml_vk_tensor_subbuffer(ctx, logits);
    vk_subbuffer labels_buf = ggml_vk_tensor_subbuffer(ctx, labels);
    vk_subbuffer dst_buf    = ggml_vk_tensor_subbuffer(ctx, dst);

    ध्रुव vk_op_प्रक्षिप_constants pc = { nclasses, nrows, 0.0f, 0.0f, 0.0f, 0.0f };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { grad_buf, logits_buf, labels_buf, dst_buf }, pc, ggml_vk_nrows_elements(nrows));
}

शून्य ggml_vk_argmax(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_प्रक्षिप_constants>(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_ARGMAX, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], 0.0f, 0.0f, 0.0f, 0.0f });
}

शून्य ggml_vk_गणनम्_equal(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_प्रक्षिप_constants>(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_गणनम्_EQUAL, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

शून्य ggml_vk_solve_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    ध्रुव uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    ध्रुव uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    ध्रुव uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_द्विमान_प्रक्षिप_constants>(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_SOLVE_TRI, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t) dst->nb[1] /  dst_type_आकार, (uint32_t) dst->nb[2] /  dst_type_आकार, (uint32_t) dst->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, 0,
    });
}

शून्य ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    ध्रुव int32_t s0 = dst->op_params[0];
    ध्रुव int32_t s1 = dst->op_params[1];
    ध्रुव int32_t p0 = dst->op_params[2];
    ध्रुव int32_t p1 = dst->op_params[3];
    ध्रुव int32_t d0 = dst->op_params[4];
    ध्रुव int32_t d1 = dst->op_params[5];

    ध्रुव तार्किक is_2D = dst->op_params[6] == 1;

    ध्रुव uint32_t IC = src1->ne[is_2D ? 2 : 1];
    ध्रुव uint32_t IH = is_2D ? src1->ne[1] : 1;
    ध्रुव uint32_t IW =         src1->ne[0];

    ध्रुव uint32_t KH = is_2D ? src0->ne[1] : 1;
    ध्रुव uint32_t KW =         src0->ne[0];

    ध्रुव uint32_t OH = is_2D ? dst->ne[2] : 1;
    ध्रुव uint32_t OW =         dst->ne[1];

    ध्रुव uint32_t विस्थापनम्_delta = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    ध्रुव uint32_t batch_विस्थापनम् = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    ध्रुव uint32_t batch = src1->ne[is_2D ? 3 : 2];

    ध्रुव ggml_backend_vk_बफर_context * d_buf_ctx = (ggml_backend_vk_बफर_context *)dst->बफर->context;
    ध्रुव vk_बफर d_buf = d_buf_ctx->dev_बफर;

    ध्रुव vk::Device_स्थानाङ्कः dst_addr = d_buf->bda_addr + vk_tensor_विस्थापनम्(dst) + dst->view_offs;

    ggml_vk_op_f32<vk_op_im2col_प्रक्षिप_constants>(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_IM2COL, {
        dst_addr,
        batch_विस्थापनम्, विस्थापनम्_delta,
        IC, IW, IH, OW, OH, KW, KH,
        OH * batch,
        IC * KH * KW,
        s0, s1, p0, p1, d0, d1, batch * IC
    });
}

शून्य ggml_vk_im2col_3d(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव int32_t s0 = ((ध्रुव int32_t *)(dst->op_params))[0];
    ध्रुव int32_t s1 = ((ध्रुव int32_t *)(dst->op_params))[1];
    ध्रुव int32_t s2 = ((ध्रुव int32_t *)(dst->op_params))[2];
    ध्रुव int32_t p0 = ((ध्रुव int32_t *)(dst->op_params))[3];
    ध्रुव int32_t p1 = ((ध्रुव int32_t *)(dst->op_params))[4];
    ध्रुव int32_t p2 = ((ध्रुव int32_t *)(dst->op_params))[5];
    ध्रुव int32_t d0 = ((ध्रुव int32_t *)(dst->op_params))[6];
    ध्रुव int32_t d1 = ((ध्रुव int32_t *)(dst->op_params))[7];
    ध्रुव int32_t d2 = ((ध्रुव int32_t *)(dst->op_params))[8];
    ध्रुव int32_t IC = ((ध्रुव int32_t *)(dst->op_params))[9];

    ध्रुव int64_t N  = ne13 / IC;
    ध्रुव int64_t ID = ne12;
    ध्रुव int64_t IH = ne11;
    ध्रुव int64_t IW = ne10;

    ध्रुव int64_t KD = ne02;
    ध्रुव int64_t KH = ne01;
    ध्रुव int64_t KW = ne00;

    ध्रुव int64_t OD = ne3 / N;
    ध्रुव int64_t OH = ne2;
    ध्रुव int64_t OW = ne1;

    ध्रुव ggml_backend_vk_बफर_context * d_buf_ctx = (ggml_backend_vk_बफर_context *)dst->बफर->context;
    ध्रुव vk_बफर d_buf = d_buf_ctx->dev_बफर;

    ध्रुव vk::Device_स्थानाङ्कः dst_addr = d_buf->bda_addr + vk_tensor_विस्थापनम्(dst) + dst->view_offs;

    vk_op_im2col_3d_प्रक्षिप_constants pc {};

    pc.dst_addr = dst_addr;
    pc.nb10 = nb10 / ggml_type_आकार(src1->type);
    pc.nb11 = nb11 / ggml_type_आकार(src1->type);
    pc.nb12 = nb12 / ggml_type_आकार(src1->type);
    pc.nb13 = nb13 / ggml_type_आकार(src1->type);
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

    ggml_vk_op_f32<vk_op_im2col_3d_प्रक्षिप_constants>(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_IM2COL_3D, std::move(pc));
}

शून्य ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    ध्रुव uint32_t dim = dst->op_params[0];
    ध्रुव uint32_t max_period = dst->op_params[1];
    ध्रुव uint32_t nb1 = dst->nb[1] / ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_timestep_embedding_प्रक्षिप_constants>(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_TIMESTEP_EMBEDDING, {
        nb1, dim, max_period,
    });
}

शून्य ggml_vk_conv_transpose_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    // src0: (K, Cout, Cin, 1) -- kernel
    // src1: (L, Cin, 1, 1) -- input
    // dst: (*, Cout, 1, 1)

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विमान_OP_LOCALS

    GGML_ASSERT(nb00 == परिमाण(भिन्न));
    GGML_ASSERT(nb10 == परिमाण(भिन्न));

    ध्रुव int32_t s0 = dst->op_params[0];

    vk_op_conv_transpose_1d_प्रक्षिप_constants p{};
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

    ggml_vk_op_f32(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_CONV_TRANSPOSE_1D, std::move(p));
}

शून्य ggml_vk_col2im_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    // src0: [K_OC, T_in] columns from matmul
    // dst:  [T_out, OC]

    ध्रुव int32_t stride = dst->op_params[0];
    ध्रुव int32_t oc     = dst->op_params[1];
    ध्रुव int32_t p0     = dst->op_params[2];

    ध्रुव uint32_t K_OC  = static_cast<uint32_t>(src0->ne[0]);
    ध्रुव uint32_t T_अन्तर्  = static_cast<uint32_t>(src0->ne[1]);
    ध्रुव uint32_t T_बहिर्गमनम् = static_cast<uint32_t>(dst->ne[0]);
    ध्रुव uint32_t OC    = static_cast<uint32_t>(oc);
    ध्रुव uint32_t K     = K_OC / OC;

    vk_op_col2im_1d_प्रक्षिप_constants p{};
    p.T_बहिर्गमनम्  = T_बहिर्गमनम्;
    p.OC     = OC;
    p.K_OC   = K_OC;
    p.T_अन्तर्   = T_अन्तर्;
    p.K      = K;
    p.stride = stride;
    p.p0     = p0;

    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_COL2IM_1D, std::move(p));
}

शून्य ggml_vk_snake_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {
    ध्रुव ggml_tensor * mul0 = cgraph->nodes[ग्रन्थिः_idx + 0];
    ध्रुव ggml_tensor * sqr  = cgraph->nodes[ग्रन्थिः_idx + 2];
    ध्रुव ggml_tensor * mul1 = cgraph->nodes[ग्रन्थिः_idx + 3];
    ggml_tensor *       सङ्कलय_2  = cgraph->nodes[ग्रन्थिः_idx + 4];

    // x carries the full activation shape, a is the broadcast operand
    ध्रुव ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ध्रुव ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    ध्रुव ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    vk_pipeline pipeline = शून्यसूचकः;
    विकल्प (x->type) {
        पक्ष GGML_TYPE_F32:  pipeline = ctx->device->pipeline_snake_f32;  भङ्ग;
        पक्ष GGML_TYPE_F16:  pipeline = ctx->device->pipeline_snake_f16;  भङ्ग;
        पक्ष GGML_TYPE_BF16: pipeline = ctx->device->pipeline_snake_bf16; भङ्ग;
        स्वभाव:             GGML_ABORT("unsupported type");
    }
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer x_buf     = ggml_vk_tensor_subbuffer(ctx, x);
    vk_subbuffer a_buf     = ggml_vk_tensor_subbuffer(ctx, a);
    vk_subbuffer inv_b_buf = ggml_vk_tensor_subbuffer(ctx, inv_b);
    vk_subbuffer dst_buf   = ggml_vk_tensor_subbuffer(ctx, सङ्कलय_2);

    vk_op_snake_प्रक्षिप_constants pc{};
    pc.ne0 = static_cast<uint32_t>(x->ne[0]);
    pc.ne1 = static_cast<uint32_t>(x->ne[1]);

    std::array<uint32_t, 3> elements = { pc.ne0, pc.ne1, 1 };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, a_buf, inv_b_buf, dst_buf }, pc, elements);
}

शून्य ggml_vk_pool_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t op = static_cast<uint32_t>(dst->op_params[0]);
    ध्रुव int32_t k0 = dst->op_params[1];
    ध्रुव int32_t s0 = dst->op_params[2];
    ध्रुव int32_t p0 = dst->op_params[3];

    ध्रुव uint32_t IL = src0->ne[0];

    ध्रुव uint32_t N = dst->ne[3] * dst->ne[2];

    ध्रुव uint32_t OC = dst->ne[1];
    ध्रुव uint32_t OL = dst->ne[0];

    ध्रुव uint32_t parallel_elements = N * OC * OL;

    ggml_vk_op_f32<vk_op_pool1d_प्रक्षिप_constants>(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_POOL_1D, {
        IL, OL, OC,
        parallel_elements,
        op,
        k0, s0, p0,
    });
}

शून्य ggml_vk_pool_2d(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t op = static_cast<uint32_t>(dst->op_params[0]);
    ध्रुव int32_t k1 = dst->op_params[1];
    ध्रुव int32_t k0 = dst->op_params[2];
    ध्रुव int32_t s1 = dst->op_params[3];
    ध्रुव int32_t s0 = dst->op_params[4];
    ध्रुव int32_t p1 = dst->op_params[5];
    ध्रुव int32_t p0 = dst->op_params[6];

    ध्रुव uint32_t IH = src0->ne[1];
    ध्रुव uint32_t IW = src0->ne[0];

    ध्रुव uint32_t N = dst->ne[3];

    ध्रुव uint32_t OC = dst->ne[2];
    ध्रुव uint32_t OH = dst->ne[1];
    ध्रुव uint32_t OW = dst->ne[0];

    ध्रुव uint32_t parallel_elements = N * OC * OH * OW;

    ggml_vk_op_f32<vk_op_pool2d_प्रक्षिप_constants>(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_POOL_2D, {
        IW, IH, OW, OH, OC,
        parallel_elements,
        op,
        k0, k1, s0, s1, p0, p1,
    });
}

शून्य ggml_vk_conv_2d(ggml_backend_vk_context * ctx, vk_context & subctx, ध्रुव ggml_tensor * src0,
                            ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विमान_OP_LOCALS
    GGML_ASSERT(nb00 == परिमाण(भिन्न) || nb00 == परिमाण(ggml_fp16_t));
    GGML_ASSERT(nb10 == परिमाण(भिन्न));
    GGML_ASSERT(nb0 == परिमाण(भिन्न));

    तार्किक transpose = dst->op == GGML_OP_CONV_TRANSPOSE_2D;

    vk_op_conv2d_प्रक्षिप_constants p{};
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

    ggml_vk_op_f32(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, dst->op, std::move(p));
}

शून्य ggml_vk_conv_3d(ggml_backend_vk_context * ctx, vk_context & subctx, ध्रुव ggml_tensor * src0,
                            ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विमान_OP_LOCALS
    GGML_ASSERT(nb00 == परिमाण(भिन्न) || nb00 == परिमाण(ggml_fp16_t));
    GGML_ASSERT(nb10 == परिमाण(भिन्न));
    GGML_ASSERT(nb0 == परिमाण(भिन्न));

    vk_op_conv3d_प्रक्षिप_constants p{};
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

    ggml_vk_op_f32(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_CONV_3D, std::move(p));
}

शून्य ggml_vk_conv_2d_dw(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    vk_op_conv2d_dw_प्रक्षिप_constants p{};
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

    ggml_vk_op_f32(ctx, subctx, src0, src1, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_CONV_2D_DW, std::move(p));
}

शून्य ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst) {
    ध्रुव भिन्न * op_params = (ध्रुव भिन्न *)dst->op_params;
    vk_op_unary_प्रक्षिप_constants p = vk_op_unary_प्रक्षिप_constants_init(src0, dst);
    p.param1 = op_params[0];

    ggml_vk_op_f32(ctx, subctx, src0, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, dst, GGML_OP_LEAKY_RELU, std::move(p));
}

शून्य ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx) {
#यदि defined(GGML_VULKAN_सततम्_TESTS)
    ध्रुव std::vector<size_t> vals {
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
    ध्रुव size_t अङ्कः_it = 100;

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अङ्कः_it, 1, 0, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अङ्कः_it, 1, 1, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अङ्कः_it, 1, 2, GGML_TYPE_Q4_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अङ्कः_it, 1, 0, GGML_TYPE_Q4_0, सत्य);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अङ्कः_it, 1, 1, GGML_TYPE_Q4_0, सत्य);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अङ्कः_it, 1, 2, GGML_TYPE_Q4_0, सत्य);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अङ्कः_it, 1, 0, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अङ्कः_it, 1, 1, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अङ्कः_it, 1, 2, GGML_TYPE_Q8_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अङ्कः_it, 1, 0, GGML_TYPE_Q8_0, सत्य);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अङ्कः_it, 1, 1, GGML_TYPE_Q8_0, सत्य);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अङ्कः_it, 1, 2, GGML_TYPE_Q8_0, सत्य);

    abort();

    प्रति (size_t i = 0; i < vals.size(); i += 3) {
        ggml_vk_test_matmul<ggml_fp16_t, भिन्न>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 1, 0);
        ggml_vk_test_matmul<ggml_fp16_t, भिन्न>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 1, 1);
        ggml_vk_test_matmul<ggml_fp16_t, भिन्न>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 1, 2);
        std::cerr << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, भिन्न>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 2, 0);
        ggml_vk_test_matmul<ggml_fp16_t, भिन्न>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 2, 1);
        ggml_vk_test_matmul<ggml_fp16_t, भिन्न>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 2, 2);
        std::cerr << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, भिन्न>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 4, 0);
        ggml_vk_test_matmul<ggml_fp16_t, भिन्न>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 4, 1);
        ggml_vk_test_matmul<ggml_fp16_t, भिन्न>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 4, 2);
        std::cerr << '\n' << std::पङ्क्त्यन्तः;

        यदि (vals[i + 2] % 32 == 0) {
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 1, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 1, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 1, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 2, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 2, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 2, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 4, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 4, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 4, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n' << std::पङ्क्त्यन्तः;
        }

        यदि (vals[i + 2] % 256 == 0) {
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 1, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 1, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 1, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 2, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 2, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 2, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 4, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 4, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अङ्कः_it, 4, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n' << std::पङ्क्त्यन्तः;
        }
    }

    GGML_ABORT("fatal error");
#यदि_अन्तः

    यदि (subctx) {
        // Submit and wait for any pending work before reallocating the buffers
        ggml_vk_ctx_अन्तः(subctx);
        ggml_vk_submit(subctx, {});
        ctx->submit_pending = सत्य;
        ggml_vk_synchronize(ctx);
        GGML_ASSERT(ctx->compute_ctx.expired());
        ggml_vk_ctx_begin(ctx->device, subctx);
        ctx->compute_ctx = subctx;
    }

    यदि (ctx->prealloc_x == शून्यसूचकः || (ctx->prealloc_आकार_x > 0 && ctx->prealloc_x->size < ctx->prealloc_आकार_x)) {
        VK_LOG_स्मृतिः("ggml_vk_preallocate_buffers(x_size: " << ctx->prealloc_आकार_x << ")");
        // Resize buffer
        यदि (ctx->prealloc_x != शून्यसूचकः) {
            ggml_vk_destroy_बफर(ctx->prealloc_x);
        }
        ctx->prealloc_x = ggml_vk_create_बफर_device(ctx->device, ctx->prealloc_आकार_x);
    }
    यदि (ctx->prealloc_y == शून्यसूचकः || (ctx->prealloc_आकार_y > 0 && ctx->prealloc_y->size < ctx->prealloc_आकार_y)) {
        VK_LOG_स्मृतिः("ggml_vk_preallocate_buffers(y_size: " << ctx->prealloc_आकार_y << ")");
        // Resize buffer
        यदि (ctx->prealloc_y != शून्यसूचकः) {
            ggml_vk_destroy_बफर(ctx->prealloc_y);
        }
        ctx->prealloc_y = ggml_vk_create_बफर_device(ctx->device, ctx->prealloc_आकार_y);
        ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् = शून्यसूचकः;
        ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् = शून्यसूचकः;
        ctx->prealloc_y_चरमः_k_padded = असत्य;
    }
    यदि (ctx->prealloc_विभज_k == शून्यसूचकः || (ctx->prealloc_आकार_विभज_k > 0 && ctx->prealloc_विभज_k->size < ctx->prealloc_आकार_विभज_k)) {
        VK_LOG_स्मृतिः("ggml_vk_preallocate_buffers(split_k_size: " << ctx->prealloc_आकार_विभज_k << ")");
        // Resize buffer
        यदि (ctx->prealloc_विभज_k != शून्यसूचकः) {
            ggml_vk_destroy_बफर(ctx->prealloc_विभज_k);
        }
        ctx->prealloc_विभज_k = ggml_vk_create_बफर_device(ctx->device, ctx->prealloc_आकार_विभज_k);
    }
    यदि (ctx->prealloc_सङ्कलय_rms_partials == शून्यसूचकः || (ctx->prealloc_आकार_सङ्कलय_rms_partials > 0 && ctx->prealloc_सङ्कलय_rms_partials->size < ctx->prealloc_आकार_सङ्कलय_rms_partials)) {
        VK_LOG_स्मृतिः("ggml_vk_preallocate_buffers(add_partials_size: " << ctx->prealloc_सङ्कलय_rms_partials << ")");
        // Resize buffer
        यदि (ctx->prealloc_सङ्कलय_rms_partials != शून्यसूचकः) {
            ggml_vk_destroy_बफर(ctx->prealloc_सङ्कलय_rms_partials);
        }
        ctx->prealloc_सङ्कलय_rms_partials = ggml_vk_create_बफर_device(ctx->device, ctx->prealloc_आकार_सङ्कलय_rms_partials);
    }
}

तार्किक ggml_vk_build_graph(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx, ggml_tensor *ग्रन्थिः_begin, पूर्णाङ्क ग्रन्थिः_idx_begin, तार्किक चरमः_ग्रन्थिः, तार्किक almost_ready, तार्किक submit){
    ggml_tensor * ग्रन्थिः = cgraph->nodes[ग्रन्थिः_idx];
    यदि (ggml_is_रिक्तानि(ग्रन्थिः) || ggml_op_is_रिक्तानि(ग्रन्थिः->op) || !ग्रन्थिः->बफर) {
        प्रत्यागम असत्य;
    }
    यदि ((ग्रन्थिः->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        प्रत्यागम असत्य;
    }

    VK_LOG_DEBUG("ggml_vk_build_graph(" << ग्रन्थिः << ", " << ggml_op_name(ग्रन्थिः->op) << ")");
    ctx->semaphore_idx = 0;

    ggml_tensor * src0 = ग्रन्थिः->src[0];
    ggml_tensor * src1 = ग्रन्थिः->src[1];
    ggml_tensor * src2 = ग्रन्थिः->src[2];
    ggml_tensor * src3 = ग्रन्थिः->src[3];

    यदि (ग्रन्थिः->op == GGML_OP_सङ्कलय) {
        पूर्णाङ्क अग्र_ग्रन्थिः_idx = ग्रन्थिः_idx + 1 + ctx->अङ्कः_additional_fused_ops;
        यदि (अग्र_ग्रन्थिः_idx < cgraph->n_nodes &&
            cgraph->nodes[अग्र_ग्रन्थिः_idx]->op == GGML_OP_RMS_NORM &&
            cgraph->nodes[अग्र_ग्रन्थिः_idx]->src[0] == cgraph->nodes[अग्र_ग्रन्थिः_idx - 1] &&
            ggml_nrows(cgraph->nodes[अग्र_ग्रन्थिः_idx]) == 1 &&
            ctx->device->सङ्कलय_rms_fusion) {
            uint32_t size = ggml_vk_rms_partials_आकार(ctx, cgraph->nodes[ग्रन्थिः_idx]);
            ctx->do_सङ्कलय_rms_partials_विस्थापनम्_calculation = सत्य;
            यदि (ctx->prealloc_आकार_सङ्कलय_rms_partials_विस्थापनम् + size <= ctx->prealloc_आकार_सङ्कलय_rms_partials) {
                ctx->do_सङ्कलय_rms_partials = सत्य;
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
        तार्किक need_sync = असत्य;

        // Check whether "node" requires synchronization. The node requires synchronization if it
        // overlaps in memory with another unsynchronized node and at least one of them is a write.
        // Destination nodes are checked against both the written/read lists. Source nodes are only
        // checked against the written list. Two nodes overlap in memory if they come from the same
        // buffer and the tensor or view ranges overlap.
        स्वचालित ध्रुव &overlaps_unsynced = [&](ध्रुव ggml_tensor *ग्रन्थिः, ध्रुव std::vector<ध्रुव ggml_tensor *> &unsynced_nodes) -> तार्किक {
            यदि (unsynced_nodes.size() == 0) {
                प्रत्यागम असत्य;
            }
            स्वचालित n_आधारः = vk_tensor_विस्थापनम्(ग्रन्थिः) + ग्रन्थिः->view_offs;
            स्वचालित n_आकार = ggml_nbytes(ग्रन्थिः);
            ggml_backend_vk_बफर_context * a_buf_ctx = (ggml_backend_vk_बफर_context *)ग्रन्थिः->बफर->context;
            vk_बफर a_buf = a_buf_ctx->dev_बफर;
            प्रति (स्वचालित &other : unsynced_nodes) {
                ggml_backend_vk_बफर_context * o_buf_ctx = (ggml_backend_vk_बफर_context *)other->बफर->context;
                vk_बफर o_buf = o_buf_ctx->dev_बफर;
                यदि (a_buf == o_buf) {
                    स्वचालित o_आधारः = vk_tensor_विस्थापनम्(other) + other->view_offs;
                    स्वचालित o_आकार = ggml_nbytes(other);

                    यदि ((o_आधारः <= n_आधारः && n_आधारः < o_आधारः + o_आकार) ||
                        (n_आधारः <= o_आधारः && o_आधारः < n_आधारः + n_आकार)) {
                        प्रत्यागम सत्य;
                    }
                }
            }
            प्रत्यागम असत्य;
        };

        // For all fused ops, check if the destination node or any of the source
        // nodes require synchronization.
        प्रति (int32_t i = 0; i < ctx->अङ्कः_additional_fused_ops + 1 && !need_sync; ++i) {
            ध्रुव ggml_tensor *cur_ग्रन्थिः = cgraph->nodes[ग्रन्थिः_idx + i];
            // If the node actually writes to memory, then check if it needs to sync
            यदि (ctx->fused_ops_write_आवरणम् & (1 << i)) {
                यदि (overlaps_unsynced(cur_ग्रन्थिः, ctx->unsynced_nodes_read) || overlaps_unsynced(cur_ग्रन्थिः, ctx->unsynced_nodes_written)) {
                    need_sync = सत्य;
                    भङ्ग;
                }
            }
            प्रति (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                यदि (!cur_ग्रन्थिः->src[j]) {
                    अनुवर्त;
                }
                यदि (overlaps_unsynced(cur_ग्रन्थिः->src[j], ctx->unsynced_nodes_written)) {
                    need_sync = सत्य;
                    भङ्ग;
                }
            }
        }

        यदि (need_sync) {
            यदि (vk_enable_sync_logger) {
                std::cerr <<  "sync" << std::पङ्क्त्यन्तः;
            }
            ctx->unsynced_nodes_written.clear();
            ctx->unsynced_nodes_read.clear();
            ggml_vk_sync_buffers(ctx, compute_ctx);

            यदि (vk_perf_logger_enabled && vk_perf_logger_concurrent) {
                ctx->query_ग्रन्थिः_idx[ctx->query_idx] = ग्रन्थिः_idx;
                compute_ctx->s->बफर->buf.writeTimestamp(vk::Pipeline_Stage_Flag_द्वयङ्काः::eAllCommands, ctx->query_pool, ctx->query_idx++);
                ggml_vk_sync_buffers(ctx, compute_ctx);
            }
        }
        // Add all fused nodes to the unsynchronized lists.
        प्रति (int32_t i = 0; i < ctx->अङ्कः_additional_fused_ops + 1; ++i) {
            ध्रुव ggml_tensor *cur_ग्रन्थिः = cgraph->nodes[ग्रन्थिः_idx + i];
            // Multiple outputs could be written, e.g. in topk_moe. Add them all to the list.
            यदि (ctx->fused_ops_write_आवरणम् & (1 << i)) {
                ctx->unsynced_nodes_written.push_back(cur_ग्रन्थिः);
            }
            प्रति (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                यदि (!cur_ग्रन्थिः->src[j]) {
                    अनुवर्त;
                }
                ctx->unsynced_nodes_read.push_back(cur_ग्रन्थिः->src[j]);
            }
        }
    }
    यदि (vk_enable_sync_logger) {
        प्रति (पूर्णाङ्क i = 0; i < ctx->अङ्कः_additional_fused_ops + 1; ++i) {
            स्वचालित *n = cgraph->nodes[ग्रन्थिः_idx + i];
            std::cerr << ग्रन्थिः_idx + i << " " << ggml_op_name(n->op) << " " <<  n->name;
            यदि (n->op == GGML_OP_GLU) {
                std::cerr << " " << ggml_glu_op_name(ggml_get_glu_op(n)) << " " << (n->src[1] ? "split" : "single") << " ";
            }
            यदि (n->op == GGML_OP_ROPE) {
                ध्रुव पूर्णाङ्क mode = ((ध्रुव int32_t *) n->op_params)[2];
                std::cerr << " rope mode: " << mode;
            }
            std::cerr << std::पङ्क्त्यन्तः;
        }
    }

    // closed explicitly below, and by the destructor on the early returns
    ggml_vk_debug_label dbg(compute_ctx, cgraph, ग्रन्थिः_idx, ctx->अङ्कः_additional_fused_ops);

    विकल्प (ग्रन्थिः->op) {
    पक्ष GGML_OP_REPEAT:
        ggml_vk_repeat(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_REPEAT_पृष्ठभागः:
        ggml_vk_repeat_पृष्ठभागः(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_ACC:
    पक्ष GGML_OP_समुच्चयः:
        ggml_vk_acc(ctx, compute_ctx, src0, src1, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_GET_पङ्क्तिकाः:
        यदि (ctx->fused_topk_qsa) {
            ggml_vk_topk_qsa(ctx, compute_ctx, cgraph, ग्रन्थिः_idx);
        } अन्यथा {
            ggml_vk_get_पङ्क्तिकाः(ctx, compute_ctx, src0, src1, ग्रन्थिः);
        }

        भङ्ग;
    पक्ष GGML_OP_GET_पङ्क्तिकाः_पृष्ठभागः:
        ggml_vk_get_पङ्क्तिकाः_पृष्ठभागः(ctx, compute_ctx, src0, src1, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_सङ्कलय:
        यदि (ctx->अङ्कः_additional_fused_ops) {
            ggml_vk_multi_सङ्कलय(ctx, compute_ctx, cgraph, ग्रन्थिः_idx);
        } अन्यथा {
            ggml_vk_सङ्कलय(ctx, compute_ctx, src0, src1, ग्रन्थिः);
        }
        भङ्ग;
    पक्ष GGML_OP_बहिर्गमनम्_PROD:
        ggml_vk_बहिर्गमनम्_prod(ctx, compute_ctx, src0, src1, ग्रन्थिः);
        भङ्ग;
    पक्ष GGML_OP_SUB:
        ggml_vk_sub(ctx, compute_ctx, src0, src1, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_MUL:
        यदि (ctx->अङ्कः_additional_fused_ops) {
            ggml_vk_snake_dispatch_fused(ctx, compute_ctx, cgraph, ग्रन्थिः_idx);
        } अन्यथा {
            ggml_vk_mul(ctx, compute_ctx, src0, src1, ग्रन्थिः);
        }

        भङ्ग;
    पक्ष GGML_OP_DIV:
        ggml_vk_div(ctx, compute_ctx, src0, src1, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_सङ्कलय_ID:
        ggml_vk_सङ्कलय_id(ctx, compute_ctx, src0, src1, src2, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_CONCAT:
        ggml_vk_concat(ctx, compute_ctx, src0, src1, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_UPSCALE:
        ggml_vk_upscale(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_ADD1:
        ggml_vk_add1(ctx, compute_ctx, src0, src1, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_ARANGE:
        ggml_vk_arange(ctx, compute_ctx, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_FILL:
        ggml_vk_fill(ctx, compute_ctx, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_SCALE:
        यदि (ctx->fused_hc_post_gate) {
            ggml_tensor * hc_post = cgraph->nodes[ग्रन्थिः_idx + ctx->अङ्कः_additional_fused_ops];
            ggml_vk_dsv4_hc_post(ctx, compute_ctx, hc_post->src[0], hc_post->src[1], hc_post->src[2], hc_post->src[3], hc_post, ग्रन्थिः);
        } अन्यथा {
            ggml_vk_scale(ctx, compute_ctx, src0, ग्रन्थिः);
        }

        भङ्ग;
    पक्ष GGML_OP_SQR:
        ggml_vk_sqr(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_SQRT:
        ggml_vk_sqrt(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_SIN:
        ggml_vk_sin(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_COS:
        ggml_vk_cos(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_LOG:
        ggml_vk_log(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_TRI:
        ggml_vk_tri(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_DIAG:
        ggml_vk_diag(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_CLAMP:
        ggml_vk_clamp(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_PAD:
        ggml_vk_pad(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_PAD_REFLECT_1D:
        ggml_vk_pad_reflect_1d(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_ROLL:
        ggml_vk_roll(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_CPY:
    पक्ष GGML_OP_CONT:
    पक्ष GGML_OP_DUP:
        ggml_vk_cpy(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_समुच्चयः_पङ्क्तिकाः:
        ggml_vk_समुच्चयः_पङ्क्तिकाः(ctx, compute_ctx, src0, src1, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_SILU_पृष्ठभागः:
        ggml_vk_silu_पृष्ठभागः(ctx, compute_ctx, src0, src1, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_NORM:
        ggml_vk_norm(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_गणः_NORM:
        ggml_vk_गणः_norm(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_RMS_NORM:
        ggml_vk_rms_norm(ctx, compute_ctx, cgraph, ग्रन्थिः_idx, (भिन्न *)ग्रन्थिः->op_params);
        भङ्ग;
    पक्ष GGML_OP_RMS_NORM_पृष्ठभागः:
        ggml_vk_rms_norm_पृष्ठभागः(ctx, compute_ctx, src0, src1, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_L2_NORM:
        ggml_vk_l2_norm(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_UNARY:
        यदि (ctx->fused_topk_moe_बहुलकम् != TOPK_MOE_गणनम्) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, ग्रन्थिः_idx);
            भङ्ग;
        }
        यदि (ctx->अङ्कः_additional_fused_ops) {
            ggml_vk_unary_mul(ctx, compute_ctx, cgraph, ग्रन्थिः_idx);
            भङ्ग;
        }

        विकल्प (ggml_get_unary_op(ग्रन्थिः)) {
        पक्ष GGML_UNARY_OP_ELU:
        पक्ष GGML_UNARY_OP_EXP:
        पक्ष GGML_UNARY_OP_EXPM1:
        पक्ष GGML_UNARY_OP_SILU:
        पक्ष GGML_UNARY_OP_GELU:
        पक्ष GGML_UNARY_OP_GELU_ERF:
        पक्ष GGML_UNARY_OP_GELU_शीघ्र:
        पक्ष GGML_UNARY_OP_RELU:
        पक्ष GGML_UNARY_OP_NEG:
        पक्ष GGML_UNARY_OP_TANH:
        पक्ष GGML_UNARY_OP_SIGMOID:
        पक्ष GGML_UNARY_OP_HARDSIGMOID:
        पक्ष GGML_UNARY_OP_HARDSWISH:
        पक्ष GGML_UNARY_OP_ABS:
        पक्ष GGML_UNARY_OP_SOFTPLUS:
        पक्ष GGML_UNARY_OP_चरणः:
        पक्ष GGML_UNARY_OP_आवर्तनम्:
        पक्ष GGML_UNARY_OP_CEIL:
        पक्ष GGML_UNARY_OP_FLOOR:
        पक्ष GGML_UNARY_OP_TRUNC:
        पक्ष GGML_UNARY_OP_SGN:
            ggml_vk_unary(ctx, compute_ctx, src0, ग्रन्थिः);
            भङ्ग;
        पक्ष GGML_UNARY_OP_XIELU:
            ggml_vk_xielu(ctx, compute_ctx, src0, ग्रन्थिः);
            भङ्ग;
        स्वभाव:
            प्रत्यागम असत्य;
        }
        भङ्ग;
    पक्ष GGML_OP_GLU:
        विकल्प (ggml_get_glu_op(ग्रन्थिः)) {
        पक्ष GGML_GLU_OP_GEGLU:
        पक्ष GGML_GLU_OP_REGLU:
        पक्ष GGML_GLU_OP_SWIGLU:
        पक्ष GGML_GLU_OP_SWIGLU_OAI:
        पक्ष GGML_GLU_OP_GEGLU_ERF:
        पक्ष GGML_GLU_OP_GEGLU_शीघ्र:
        पक्ष GGML_GLU_OP_SWIGLU_CLAMP:
            ggml_vk_glu(ctx, compute_ctx, src0, src1, ग्रन्थिः);
            भङ्ग;
        स्वभाव:
            प्रत्यागम असत्य;
        }
        भङ्ग;
    पक्ष GGML_OP_DIAG_आवरणम्_INF:
        ggml_vk_diag_आवरणम्_inf(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_SOFT_MAX:
        यदि (ctx->fused_topk_moe_बहुलकम् != TOPK_MOE_गणनम्) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, ग्रन्थिः_idx);
        } अन्यथा {
            ggml_vk_soft_max(ctx, compute_ctx, src0, src1, src2, ग्रन्थिः);
        }

        भङ्ग;
    पक्ष GGML_OP_SOFT_MAX_पृष्ठभागः:
        ggml_vk_soft_max_पृष्ठभागः(ctx, compute_ctx, src0, src1, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_ROPE:
        ggml_vk_rope(ctx, compute_ctx, cgraph, ग्रन्थिः_idx, असत्य);

        भङ्ग;
    पक्ष GGML_OP_ROPE_पृष्ठभागः:
        ggml_vk_rope(ctx, compute_ctx, cgraph, ग्रन्थिः_idx, सत्य);

        भङ्ग;
    पक्ष GGML_OP_ARGSORT:
        यदि (ctx->fused_topk_moe_बहुलकम् != TOPK_MOE_गणनम्) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, ग्रन्थिः_idx);
        } अन्यथा {
            ggml_vk_argsort(ctx, compute_ctx, src0, ग्रन्थिः);
        }

        भङ्ग;
    पक्ष GGML_OP_शिखरम्_K:
        ggml_vk_topk(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_योग:
        ggml_vk_योग(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_योग_पङ्क्तिकाः:
        ggml_vk_योग_पङ्क्तिकाः(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_CUMSUM:
        ggml_vk_cumsum(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_DSV4_HC_COMB:
        ggml_vk_dsv4_hc_comb(ctx, compute_ctx, src0, src1, src2, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_DSV4_HC_PRE:
        ggml_vk_dsv4_hc_pre(ctx, compute_ctx, src0, src1, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_DSV4_HC_POST:
        ggml_vk_dsv4_hc_post(ctx, compute_ctx, src0, src1, src2, src3, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_माध्यम्:
        ggml_vk_माध्यम्(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_ARGMAX:
        ggml_vk_argmax(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_CROSS_ENTROPY_LOSS:
        ggml_vk_cross_entropy_loss(ctx, compute_ctx, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_CROSS_ENTROPY_LOSS_पृष्ठभागः:
        ggml_vk_cross_entropy_loss_पृष्ठभागः(ctx, compute_ctx, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_गणनम्_EQUAL:
        ggml_vk_गणनम्_equal(ctx, compute_ctx, src0, src1, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_SOLVE_TRI:
        ggml_vk_solve_tri(ctx, compute_ctx, src0, src1, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_IM2COL:
        ggml_vk_im2col(ctx, compute_ctx, src0, src1, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_IM2COL_3D:
        ggml_vk_im2col_3d(ctx, compute_ctx, src0, src1, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_TIMESTEP_EMBEDDING:
        ggml_vk_timestep_embedding(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_COL2IM_1D:
        ggml_vk_col2im_1d(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_CONV_TRANSPOSE_1D:
        ggml_vk_conv_transpose_1d(ctx, compute_ctx, src0, src1, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_POOL_1D:
        ggml_vk_pool_1d(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_POOL_2D:
        ggml_vk_pool_2d(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_CONV_2D:
    पक्ष GGML_OP_CONV_TRANSPOSE_2D:
        ggml_vk_conv_2d(ctx, compute_ctx, src0, src1, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_CONV_3D:
        ggml_vk_conv_3d(ctx, compute_ctx, src0, src1, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_CONV_2D_DW:
        ggml_vk_conv_2d_dw(ctx, compute_ctx, src0, src1, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_LEAKY_RELU:
        ggml_vk_leaky_relu(ctx, compute_ctx, src0, ग्रन्थिः);

        भङ्ग;
    पक्ष GGML_OP_MUL_MAT:
        ggml_vk_mul_mat(ctx, compute_ctx, cgraph, ग्रन्थिः_idx);

        भङ्ग;
    पक्ष GGML_OP_MUL_MAT_ID:
        ggml_vk_mul_mat_id(ctx, compute_ctx, cgraph, ग्रन्थिः_idx);

        भङ्ग;

    पक्ष GGML_OP_FLASH_ATTN_EXT:
        ggml_vk_flash_attn(ctx, compute_ctx, src0, src1, src2, src3, ग्रन्थिः->src[4], ग्रन्थिः);

        भङ्ग;

    पक्ष GGML_OP_RWKV_WKV6:
        ggml_vk_rwkv_wkv6(ctx, compute_ctx, ग्रन्थिः);

        भङ्ग;

    पक्ष GGML_OP_RWKV_WKV7:
        ggml_vk_rwkv_wkv7(ctx, compute_ctx, ग्रन्थिः);

        भङ्ग;

    पक्ष GGML_OP_GATED_रैखिकम्_ATTN:
        ggml_vk_gated_रैखिकम्_attn(ctx, compute_ctx, ग्रन्थिः);

        भङ्ग;

    पक्ष GGML_OP_LIGHTNING_INDEXER:
        ggml_vk_lightning_indexer(ctx, compute_ctx, ग्रन्थिः);

        भङ्ग;

    पक्ष GGML_OP_GATED_DELTA_NET:
        ggml_vk_gated_delta_net(ctx, compute_ctx, ग्रन्थिः);

        भङ्ग;

    पक्ष GGML_OP_SSM_अवलोकनम्:
        ggml_vk_ssm_अवलोकनम्(ctx, compute_ctx, ग्रन्थिः);

        भङ्ग;

    पक्ष GGML_OP_SSM_CONV:
        ggml_vk_ssm_conv(ctx, compute_ctx, cgraph, ग्रन्थिः_idx);

        भङ्ग;

    पक्ष GGML_OP_OPT_चरणः_ADAMW:
        ggml_vk_opt_चरणः_adamw(ctx, compute_ctx, ग्रन्थिः);

        भङ्ग;

    पक्ष GGML_OP_OPT_चरणः_SGD:
        ggml_vk_opt_चरणः_sgd(ctx, compute_ctx, src0, src1, src2, ग्रन्थिः);

        भङ्ग;
    स्वभाव:
        प्रत्यागम असत्य;
    }

    // the submit path below can end the command buffer, so close the region first
    dbg.close();

    ctx->tensor_ctxs[ग्रन्थिः_idx] = compute_ctx;

#यदि defined(GGML_VULKAN_CHECK_RESULTS)
    // Force context reset on each node so that each tensor ends up in its own context
    // and can be run and compared to its CPU equivalent separately
    चरमः_ग्रन्थिः = सत्य;
#यदि_अन्तः

    यदि (submit || चरमः_ग्रन्थिः) {
        ggml_vk_ctx_अन्तः(compute_ctx);

        // TODO probably it'd be better to pass a exit_node flag to ggml_vk_compute_forward
        यदि (चरमः_ग्रन्थिः) {
            compute_ctx->exit_tensor_idx = ग्रन्थिः_idx_begin;
        }
        अन्यथा {
            compute_ctx->exit_tensor_idx = -1;
        }

        ctx->compute_ctx.reset();

        ggml_vk_compute_forward(ctx, cgraph, ग्रन्थिः_begin, ग्रन्थिः_idx_begin, almost_ready);
    }
    प्रत्यागम सत्य;
}

शून्य ggml_vk_compute_forward(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ggml_tensor * tensor, पूर्णाङ्क tensor_idx, तार्किक almost_ready) {
    GGML_अप्रयुक्तम्(cgraph);
    GGML_अप्रयुक्तम्(tensor);

    VK_LOG_DEBUG("ggml_vk_compute_forward(" << tensor << ", name=" << tensor->name << ", op=" << ggml_op_name(tensor->op) << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << ", view_src=" << tensor->view_src << ", view_offs=" << tensor->view_offs << ")");

    vk_context subctx = ctx->tensor_ctxs[tensor_idx].तालकम्();

    // Only run if ctx hasn't been submitted yet
    यदि (!subctx->seqs.रिक्तानि()) {
#यदि_परिभाषितम् GGML_VULKAN_CHECK_RESULTS
        ggml_vk_check_results_0(ctx, cgraph, tensor_idx);
#यदि_अन्तः

        // Do staging buffer copies
        प्रति (स्वचालित& cpy : subctx->अन्तर्_memcpys) {
            स्मृतिं_प्रतिलिख(cpy.dst, cpy.src, cpy.n);
        }

        प्रति (स्वचालित& mset : subctx->memsets) {
            स्मृतिं_पूरय(mset.dst, mset.val, mset.n);
        }

        यदि (ctx->device->serialize_submissions) {
            ggml_vk_submit(subctx, ctx->fence);
        } अन्यथा यदि (almost_ready && !ctx->almost_ready_fence_pending) {
            ggml_vk_submit(subctx, ctx->almost_ready_fence);
            ctx->almost_ready_fence_pending = सत्य;
        } अन्यथा {
            ggml_vk_submit(subctx, {});
        }
        ctx->submit_pending = सत्य;

#यदि_परिभाषितम् GGML_VULKAN_CHECK_RESULTS
        ggml_vk_synchronize(ctx);
        ggml_vk_check_results_1(ctx, cgraph, tensor_idx);
#यदि_अन्तः
    }

    यदि (tensor_idx == subctx->exit_tensor_idx) {
        // Do staging buffer copies
        प्रति (स्वचालित& cpy : subctx->बहिर्गमनम्_memcpys) {
            स्मृतिं_प्रतिलिख(cpy.dst, cpy.src, cpy.n);
        }
        subctx->अन्तर्_memcpys.clear();
        subctx->बहिर्गमनम्_memcpys.clear();
        subctx->memsets.clear();
    }
}

शून्य ggml_vk_graph_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_graph_cleanup()");
    ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् = {};
    ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् = शून्यसूचकः;
    ctx->prealloc_y_चरमः_k_padded = असत्य;

    ctx->unsynced_nodes_written.clear();
    ctx->unsynced_nodes_read.clear();
    ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_विभज_k_need_sync = असत्य;

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);
    यदि (ctx->device->async_use_transfer_queue) {
        ggml_vk_command_pool_cleanup(ctx->device, ctx->transfer_cmd_pool);
    }

    प्रति (size_t i = 0; i < ctx->gc.semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.semaphores[i].s });
    }
    ctx->gc.semaphores.clear();

    प्रति (size_t i = 0; i < ctx->gc.tl_semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.tl_semaphores[i].s });
    }
    ctx->gc.tl_semaphores.clear();
    ctx->semaphore_idx = 0;

    ctx->event_idx = 0;

    प्रति (स्वचालित& event : ctx->gc.events) {
        ctx->device->device.resetEvent(event);
    }

    ctx->tensor_ctxs.clear();
    ctx->gc.contexts.clear();
    ctx->pipeline_descriptor_समुच्चयः_requirements = 0;
    ctx->descriptor_समुच्चयः_idx = 0;
}

शून्य ggml_vk_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_cleanup(" << ctx->name << ")");
    // discard any unsubmitted command buffers
    ctx->compute_ctx.reset();
    // wait for any pending command buffers to finish
    ggml_vk_synchronize(ctx);

    ggml_vk_graph_cleanup(ctx);

    ggml_vk_destroy_बफर(ctx->prealloc_x);
    ggml_vk_destroy_बफर(ctx->prealloc_y);
    ggml_vk_destroy_बफर(ctx->prealloc_विभज_k);
    ggml_vk_destroy_बफर(ctx->prealloc_सङ्कलय_rms_partials);
    ggml_vk_destroy_बफर(ctx->sync_staging);

    ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् = शून्यसूचकः;
    ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् = शून्यसूचकः;
    ctx->prealloc_y_चरमः_k_padded = असत्य;

    ctx->prealloc_आकार_x = 0;
    ctx->prealloc_आकार_y = 0;
    ctx->prealloc_आकार_विभज_k = 0;

    प्रति (स्वचालित& event : ctx->gc.events) {
        ctx->device->device.destroyEvent(event);
    }
    ctx->gc.events.clear();

    ctx->device->device.destroyFence(ctx->fence);
    ctx->device->device.destroyFence(ctx->almost_ready_fence);

    प्रति (स्वचालित& pool : ctx->descriptor_pools) {
        ctx->device->device.destroyDescriptorPool(pool);
    }
    ctx->descriptor_pools.clear();
    ctx->descriptor_sets.clear();
    ctx->descriptor_समुच्चयः_bindings.clear();

    ctx->compute_cmd_pool.destroy(ctx->device->device);
    यदि (ctx->device->async_use_transfer_queue) {
        ctx->device->device.destroySemaphore(ctx->transfer_semaphore.s);

        ctx->transfer_cmd_pool.destroy(ctx->device->device);
    }
    यदि (vk_perf_logger_enabled) {
        ctx->perf_logger->print_timings(सत्य);
    }
}

पूर्णाङ्क ggml_vk_get_device_गणनम्() {
    ggml_vk_instance_init();

    प्रत्यागम vk_instance.device_indices.size();
}

शून्य ggml_vk_get_device_description(पूर्णाङ्क device, वर्ण * description, size_t description_आकार) {
    ggml_vk_instance_init();

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    vk::PhysicalDeviceProperties props;
    devices[device].getProperties(&props);

    snprintf(description, description_आकार, "%s", props.deviceName.दत्त());
}

तार्किक ggml_backend_बफर_is_vk(ggml_backend_बफर_t बफर) {
    प्रत्यागम बफर->buft->iface.get_name == ggml_backend_vk_बफर_type_name;
}

शून्य ggml_backend_vk_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    VK_LOG_स्मृतिः("ggml_backend_vk_buffer_free_buffer()");
    ggml_backend_vk_बफर_context * ctx = (ggml_backend_vk_बफर_context *)बफर->context;
    ggml_vk_destroy_बफर(ctx->dev_बफर);
    विलोपय ctx;
}

शून्य * ggml_backend_vk_बफर_get_आधारः(ggml_backend_बफर_t बफर) {
    प्रत्यागम vk_ptr_आधारः;

    अप्रयुक्तम्(बफर);
}

गणना ggml_स्थितिः ggml_backend_vk_बफर_init_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_init_tensor(" << बफर << " (" << बफर->context << "), " << tensor << ")");
    यदि (tensor->view_src != शून्यसूचकः) {
        GGML_ASSERT(tensor->view_src->बफर->buft == बफर->buft);
    }
    प्रत्यागम GGML_स्थितिः_SUCCESS;
}

शून्य ggml_backend_vk_बफर_memset_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, uint8_t value, size_t विस्थापनम्, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_memset_tensor(" << बफर << ", " << tensor << ", " << value << ", " << विस्थापनम् << ", " << size << ")");
    ggml_backend_vk_बफर_context * buf_ctx = (ggml_backend_vk_बफर_context *)बफर->context;
    vk_बफर buf = buf_ctx->dev_बफर;

    यदि (size == 0) {
        प्रत्यागम;
    }

    uint32_t val32 = (uint32_t)value * 0x01010101;
    ggml_vk_बफर_memset(buf, vk_tensor_विस्थापनम्(tensor) + tensor->view_offs + विस्थापनम्, val32, size);
}

शून्य ggml_backend_vk_बफर_समुच्चयः_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, ध्रुव शून्य * दत्त, size_t विस्थापनम्, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor(" << बफर << ", " << tensor << ", " << दत्त << ", " << विस्थापनम् << ", " << size << ")");
    ggml_backend_vk_बफर_context * buf_ctx = (ggml_backend_vk_बफर_context *)बफर->context;
    vk_बफर buf = buf_ctx->dev_बफर;

    यदि (size == 0) {
        प्रत्यागम;
    }

    ggml_vk_बफर_write(buf, vk_tensor_विस्थापनम्(tensor) + tensor->view_offs + विस्थापनम्, दत्त, size);
}

शून्य ggml_backend_vk_बफर_समुच्चयः_tensor_2d(ggml_backend_बफर_t बफर, ggml_tensor * tensor, ध्रुव शून्य * दत्त, size_t विस्थापनम्,
                                                 size_t size, size_t n_copies, size_t stride_tensor, size_t stride_दत्त) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor_2d(" << बफर << ", " << tensor << ", " << दत्त << ", " << विस्थापनम् << ", " << size << ", " <<
                 n_copies << ", " << stride_tensor << ", " << stride_दत्त << ")");
    ggml_backend_vk_बफर_context * buf_ctx = (ggml_backend_vk_बफर_context *)बफर->context;
    vk_बफर buf = buf_ctx->dev_बफर;

    यदि (size == 0) {
        प्रत्यागम;
    }

    ggml_vk_बफर_write_2d(buf, vk_tensor_विस्थापनम्(tensor) + tensor->view_offs + विस्थापनम्, दत्त, stride_दत्त, stride_tensor, size, n_copies);
}

शून्य ggml_backend_vk_बफर_get_tensor(ggml_backend_बफर_t बफर, ध्रुव ggml_tensor * tensor, शून्य * दत्त, size_t विस्थापनम्, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor(" << बफर << ", " << tensor << ", " << दत्त << ", " << विस्थापनम् << ", " << size << ")");
    ggml_backend_vk_बफर_context * buf_ctx = (ggml_backend_vk_बफर_context *)बफर->context;

    यदि (size == 0) {
        प्रत्यागम;
    }

    vk_बफर buf = buf_ctx->dev_बफर;

    ggml_vk_बफर_read(buf, vk_tensor_विस्थापनम्(tensor) + tensor->view_offs + विस्थापनम्, दत्त, size);
}

शून्य ggml_backend_vk_बफर_get_tensor_2d(ggml_backend_बफर_t बफर, ध्रुव ggml_tensor * tensor, शून्य * दत्त, size_t विस्थापनम्,
                                                 size_t size, size_t n_copies, size_t stride_tensor, size_t stride_दत्त) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor_2d(" << बफर << ", " << tensor << ", " << दत्त << ", " << विस्थापनम् << ", " << size << ", " <<
                 n_copies << ", " << stride_tensor << ", " << stride_दत्त << ")");
    ggml_backend_vk_बफर_context * buf_ctx = (ggml_backend_vk_बफर_context *)बफर->context;

    यदि (size == 0) {
        प्रत्यागम;
    }

    vk_बफर buf = buf_ctx->dev_बफर;

    ggml_vk_बफर_read_2d(buf, vk_tensor_विस्थापनम्(tensor) + tensor->view_offs + विस्थापनम्, दत्त, stride_tensor, stride_दत्त, size, n_copies);
}

तार्किक ggml_backend_vk_बफर_cpy_tensor(ggml_backend_बफर_t बफर, ध्रुव ggml_tensor * src, ggml_tensor * dst) {
    यदि (ggml_nbytes(src) == 0) {
        प्रत्यागम सत्य;
    }

    यदि (ggml_backend_बफर_is_vk(src->बफर)) {
        ggml_backend_vk_बफर_context * src_buf_ctx = (ggml_backend_vk_बफर_context *)src->बफर->context;
        ggml_backend_vk_बफर_context * dst_buf_ctx = (ggml_backend_vk_बफर_context *)dst->बफर->context;

        vk_बफर src_buf = src_buf_ctx->dev_बफर;
        vk_बफर dst_buf = dst_buf_ctx->dev_बफर;

        ggml_vk_बफर_copy(dst_buf, vk_tensor_विस्थापनम्(dst) + dst->view_offs, src_buf, vk_tensor_विस्थापनम्(src) + src->view_offs, ggml_nbytes(src));

        प्रत्यागम सत्य;
    }
    प्रत्यागम असत्य;

    अप्रयुक्तम्(बफर);
}

शून्य ggml_backend_vk_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    ggml_backend_vk_बफर_context * ctx = (ggml_backend_vk_बफर_context *)बफर->context;

    ggml_vk_बफर_memset(ctx->dev_बफर, 0, value, बफर->size);
}

ध्रुव वर्ण * ggml_backend_vk_बफर_type_name(ggml_backend_बफर_type_t buft) {
    ggml_backend_vk_बफर_type_context * ctx = (ggml_backend_vk_बफर_type_context *)buft->context;

    प्रत्यागम ctx->name.c_str();
}

ggml_backend_बफर_t ggml_backend_vk_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, size_t size) {
    VK_LOG_स्मृतिः("ggml_backend_vk_buffer_type_alloc_buffer(" << size << ")");
    ggml_backend_vk_बफर_type_context * ctx = (ggml_backend_vk_बफर_type_context *) buft->context;

    vk_बफर dev_बफर = शून्यसूचकः;
    यत्न {
        dev_बफर = ggml_vk_create_बफर_device(ctx->device, size);
    } गृहाण (ध्रुव vk::SystemError& e) {
        प्रत्यागम शून्यसूचकः;
    }

    ggml_backend_vk_बफर_context * bufctx = नूतन ggml_backend_vk_बफर_context(ctx->device, std::move(dev_बफर), ctx->name);

    प्रत्यागम ggml_backend_बफर_init(buft, ggml_backend_vk_बफर_interface, bufctx, size);
}

size_t ggml_backend_vk_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    ggml_backend_vk_बफर_type_context * ctx = (ggml_backend_vk_बफर_type_context *) buft->context;
    प्रत्यागम ctx->device->properties.limits.min_Storage_बफर_विस्थापनम्_Alignment;
}

size_t ggml_backend_vk_बफर_type_get_max_आकार(ggml_backend_बफर_type_t buft) {
    ggml_backend_vk_बफर_type_context * ctx = (ggml_backend_vk_बफर_type_context *) buft->context;
    प्रत्यागम ctx->device->suballocation_खण्डम्_आकार;
}

size_t ggml_backend_vk_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft, ध्रुव ggml_tensor * tensor) {
    प्रत्यागम ggml_nbytes(tensor);

    अप्रयुक्तम्(buft);
}

ggml_backend_बफर_type_t ggml_backend_vk_बफर_type(size_t dev_अङ्कः) {
    ggml_vk_instance_init();

    VK_LOG_DEBUG("ggml_backend_vk_buffer_type(" << dev_अङ्कः << ")");

    vk_device dev = ggml_vk_get_device(dev_अङ्कः);

    प्रत्यागम &dev->बफर_type;
}

स्थिर ध्रुव वर्ण * ggml_backend_vk_host_बफर_type_name(ggml_backend_बफर_type_t buft) {
    प्रत्यागम GGML_VK_NAME "_Host";

    अप्रयुक्तम्(buft);
}

स्थिर शून्य ggml_backend_vk_host_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    VK_LOG_स्मृतिः("ggml_backend_vk_host_buffer_free_buffer()");
    ggml_vk_host_free(बफर->context);
}

स्थिर ggml_backend_बफर_t ggml_backend_vk_host_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, size_t size) {
    VK_LOG_स्मृतिः("ggml_backend_vk_host_buffer_type_alloc_buffer(" << size << ")");

    size += 32;  // Behave like the CPU buffer type
    शून्य * ptr = ggml_vk_host_malloc(size);
    यदि (ptr == शून्यसूचकः) {
        GGML_LOG_WARN("ggml_vulkan: Failed to allocate pinned memory\n");
        // fallback to cpu buffer
        प्रत्यागम ggml_backend_buft_alloc_बफर(ggml_backend_cpu_बफर_type(), size);
    }

    ggml_backend_बफर_t बफर = ggml_backend_cpu_बफर_from_ptr(ptr, size);
    बफर->buft = buft;
    बफर->iface.free_बफर = ggml_backend_vk_host_बफर_free_बफर;

    प्रत्यागम बफर;

    अप्रयुक्तम्(buft);
}

स्थिर size_t ggml_backend_vk_host_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    प्रत्यागम vk_instance.devices[0]->properties.limits.min_स्मृतिः_Map_Alignment;

    अप्रयुक्तम्(buft);
}

स्थिर size_t ggml_backend_vk_host_बफर_type_get_max_आकार(ggml_backend_बफर_type_t buft) {
    प्रत्यागम vk_instance.devices[0]->suballocation_खण्डम्_आकार;

    अप्रयुक्तम्(buft);
}

ggml_backend_बफर_type_t ggml_backend_vk_host_बफर_type() {
    स्थिर संरचना ggml_backend_बफर_type ggml_backend_vk_बफर_type_host = {
        /* .iface    = */ {
            /* .get_name            = */ ggml_backend_vk_host_बफर_type_name,
            /* .alloc_buffer        = */ ggml_backend_vk_host_बफर_type_alloc_बफर,
            /* .alloc_buffer_n      = */ शून्यसूचकः,
            /* .get_alignment       = */ ggml_backend_vk_host_बफर_type_get_alignment,
            /* .get_max_size        = */ ggml_backend_vk_host_बफर_type_get_max_आकार,
            /* .get_alloc_size      = */ ggml_backend_cpu_बफर_type()->iface.get_alloc_आकार,
            /* .get_alloc_size_n    = */ रिक्त,
            /* .is_host             = */ ggml_backend_cpu_बफर_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), 0),
        /* .context  = */ शून्यसूचकः,
    };

    // Make sure device 0 is initialized
    ggml_vk_instance_init();
    ggml_vk_get_device(0);

    प्रत्यागम &ggml_backend_vk_बफर_type_host;
}

स्थिर ध्रुव वर्ण * ggml_backend_vk_name(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    प्रत्यागम ctx->name.c_str();
}

शून्य ggml_backend_vk_free(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    VK_LOG_DEBUG("ggml_backend_vk_free(" << ctx->name << ")");

    ggml_vk_cleanup(ctx);

    विलोपय ctx;
    विलोपय backend;
}

स्थिर ggml_backend_बफर_type_t ggml_backend_vk_get_default_बफर_type(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    प्रत्यागम &ctx->device->बफर_type;
}

स्थिर शून्य ggml_backend_vk_समुच्चयः_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, ध्रुव शून्य * दत्त, size_t विस्थापनम्,
                                                size_t size, size_t n_copies, size_t stride_tensor, size_t stride_दत्त) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->बफर->buft == ggml_backend_vk_get_default_बफर_type(backend) || tensor->बफर->buft == ggml_backend_vk_host_बफर_type()) && "unsupported buffer type");

    यदि (size == 0) {
        प्रत्यागम;
    }

    ggml_backend_vk_बफर_context * buf_ctx = (ggml_backend_vk_बफर_context *)tensor->बफर->context;

    vk_context cpy_ctx;

    यदि (ctx->device->async_use_transfer_queue) {
        cpy_ctx = ggml_vk_get_transfer_ctx(ctx);
    } अन्यथा {
        cpy_ctx = ggml_vk_get_compute_ctx(ctx);
    }

    vk_बफर buf = buf_ctx->dev_बफर;

    स्वचालित dst_विस्थापनम् = vk_tensor_विस्थापनम्(tensor) + tensor->view_offs + विस्थापनम्;

    तार्किक ret = ggml_vk_बफर_write_2d_async(cpy_ctx, buf, dst_विस्थापनम्, दत्त, stride_दत्त, stride_tensor, size, n_copies);

    यदि (!ret) {
        ध्रुव size_t staging_आकार = size * n_copies;
        ggml_vk_ensure_sync_staging_बफर(ctx, staging_आकार);
        ggml_vk_sync_buffers(शून्यसूचकः, cpy_ctx);

        std::vector<vk::बफर_Copy> slices(1);
        यदि (size == stride_tensor) {
            slices[0].src_विस्थापनम्_2 = 0;
            slices[0].dst_विस्थापनम्_2 = dst_विस्थापनम्;
            slices[0].size = staging_आकार;
        } अन्यथा {
            slices.resize(n_copies);
            प्रति (size_t i = 0; i < n_copies; i++) {
                slices[i].src_विस्थापनम्_2 = i * size;
                slices[i].dst_विस्थापनम्_2 = dst_विस्थापनम् + i * stride_tensor;
                slices[i].size = size;
            }
        }

        cpy_ctx->s->बफर->buf.copy_बफर(ctx->sync_staging->बफर, buf->बफर, slices);

        यदि (size == stride_दत्त) {
            deferred_memcpy(ctx->sync_staging->ptr, दत्त, staging_आकार, &cpy_ctx->अन्तर्_memcpys);
        } अन्यथा {
            प्रति (size_t i = 0; i < n_copies; i++) {
                deferred_memcpy((uint8_t *)ctx->sync_staging->ptr + i * size, (ध्रुव uint8_t *)दत्त + i * stride_दत्त, size, &cpy_ctx->अन्तर्_memcpys);
            }
        }
        ggml_vk_synchronize(ctx);
    }
}

स्थिर शून्य ggml_backend_vk_समुच्चयः_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ध्रुव शून्य * दत्त, size_t विस्थापनम्, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_async(" << size << ")");
    ggml_backend_vk_समुच्चयः_tensor_2d_async(backend, tensor, दत्त, विस्थापनम्, size, 1, size, size);
}

स्थिर शून्य ggml_backend_vk_get_tensor_2d_async(ggml_backend_t backend, ध्रुव ggml_tensor * tensor, शून्य * दत्त, size_t विस्थापनम्,
                                                size_t size, size_t n_copies, size_t stride_tensor, size_t stride_दत्त) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->बफर->buft == ggml_backend_vk_get_default_बफर_type(backend) || tensor->बफर->buft == ggml_backend_vk_host_बफर_type()) && "unsupported buffer type");

    यदि (size == 0) {
        प्रत्यागम;
    }

    ggml_backend_vk_बफर_context * buf_ctx = (ggml_backend_vk_बफर_context *)tensor->बफर->context;

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    vk_बफर buf = buf_ctx->dev_बफर;

    स्वचालित src_विस्थापनम् = vk_tensor_विस्थापनम्(tensor) + tensor->view_offs + विस्थापनम्;
    तार्किक ret = ggml_vk_बफर_read_2d_async(compute_ctx, buf, src_विस्थापनम्, दत्त, stride_tensor, stride_दत्त, size, n_copies);

    यदि (!ret) {
        ध्रुव size_t staging_आकार = size * n_copies;
        ggml_vk_ensure_sync_staging_बफर(ctx, staging_आकार);
        ggml_vk_sync_buffers(शून्यसूचकः, compute_ctx);

        std::vector<vk::बफर_Copy> slices(1);
        यदि (size == stride_tensor) {
            slices[0].src_विस्थापनम्_2 = src_विस्थापनम्;
            slices[0].dst_विस्थापनम्_2 = 0;
            slices[0].size = staging_आकार;
        } अन्यथा {
            slices.resize(n_copies);
            प्रति (size_t i = 0; i < n_copies; i++) {
                slices[i].src_विस्थापनम्_2 = src_विस्थापनम् + i * stride_tensor;
                slices[i].dst_विस्थापनम्_2 = i * size;
                slices[i].size = size;
            }
        }

        compute_ctx->s->बफर->buf.copy_बफर(buf->बफर, ctx->sync_staging->बफर, slices);

        यदि (size == stride_दत्त) {
            deferred_memcpy(दत्त, ctx->sync_staging->ptr, staging_आकार, &compute_ctx->बहिर्गमनम्_memcpys);
        } अन्यथा {
            प्रति (size_t i = 0; i < n_copies; i++) {
                deferred_memcpy((uint8_t *)दत्त + i * stride_दत्त, (ध्रुव uint8_t *)ctx->sync_staging->ptr + i * size, size, &compute_ctx->बहिर्गमनम्_memcpys);
            }
        }
        ggml_vk_synchronize(ctx);
    }
}

स्थिर शून्य ggml_backend_vk_get_tensor_async(ggml_backend_t backend, ध्रुव ggml_tensor * tensor, शून्य * दत्त, size_t विस्थापनम्, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_async(" << size << ")");
    ggml_backend_vk_get_tensor_2d_async(backend, tensor, दत्त, विस्थापनम्, size, 1, size, size);
}

स्थिर तार्किक ggml_backend_vk_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ध्रुव ggml_tensor * src, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_backend_vk_cpy_tensor_async(" << src << " -> " << dst << ", size=" << ggml_nbytes(src) << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend_dst->context;

    // Skip zero-size tensors
    यदि (ggml_nbytes(src) == 0) {
        प्रत्यागम सत्य;
    }

    यदि (dst->बफर->buft != ggml_backend_vk_get_default_बफर_type(backend_dst)) {
        प्रत्यागम असत्य;
    }

    ggml_backend_vk_बफर_context * dst_buf_ctx = (ggml_backend_vk_बफर_context *)dst->बफर->context;
    vk_बफर dst_buf = dst_buf_ctx->dev_बफर;

    यदि (ggml_backend_बफर_is_vk(src->बफर)) {
        ggml_backend_vk_बफर_context * src_buf_ctx = (ggml_backend_vk_बफर_context *)src->बफर->context;

        // Async copy only works within the same device
        यदि (src_buf_ctx->dev_बफर->device != dst_buf->device) {
            प्रत्यागम असत्य;
        }

        vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

        ggml_vk_बफर_copy_async(compute_ctx, dst_buf, vk_tensor_विस्थापनम्(dst) + dst->view_offs,
                                   src_buf_ctx->dev_बफर, vk_tensor_विस्थापनम्(src) + src->view_offs,
                                   ggml_nbytes(src));
        प्रत्यागम सत्य;
    }

    यदि (ggml_backend_बफर_is_host(src->बफर)) {
        vk_बफर pinned_buf = शून्यसूचकः;
        size_t pinned_विस्थापनम् = 0;
        ggml_vk_host_get(ctx->device, src->दत्त, pinned_buf, pinned_विस्थापनम्);
        यदि (pinned_buf == शून्यसूचकः) {
            प्रत्यागम असत्य;
        }

        // If the backend is idle, use a CPU copy to avoid GPU synchronization overhead.
        स्थिर constexpr size_t max_cpu_copy_आकार = 128 * 1024;
        ध्रुव तार्किक src_backend_synchronous = backend_src->iface.synchronize == शून्यसूचकः;
        ध्रुव तार्किक transfer_idle = !ctx->device->async_use_transfer_queue ||
                                   ctx->transfer_semaphore_चरमः_submitted == ctx->transfer_semaphore.value;
        ध्रुव तार्किक backend_idle = ctx->compute_ctx.expired() && ctx->transfer_ctx.expired() &&
                                  !ctx->submit_pending && !ctx->almost_ready_fence_pending && transfer_idle;
        ध्रुव तार्किक dst_host_coherent =
            (dst_buf->स्मृतिः_property_flags & (vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCoherent)) ==
            (vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCoherent);

        यदि ((backend_src == backend_dst || src_backend_synchronous) && backend_idle && dst_host_coherent && ggml_nbytes(src) <= max_cpu_copy_आकार) {
            ggml_vk_बफर_write(dst_buf, vk_tensor_विस्थापनम्(dst) + dst->view_offs, src->दत्त, ggml_nbytes(src));
            प्रत्यागम सत्य;
        }

        vk_context cpy_ctx;
        यदि (ctx->device->async_use_transfer_queue) {
            cpy_ctx = ggml_vk_get_transfer_ctx(ctx);
        } अन्यथा {
            cpy_ctx = ggml_vk_get_compute_ctx(ctx);
        }

        प्रत्यागम ggml_vk_बफर_write_async(cpy_ctx, dst_buf,
                                          vk_tensor_विस्थापनम्(dst) + dst->view_offs,
                                          src->दत्त, ggml_nbytes(src));
    }

    प्रत्यागम असत्य;
}

शून्य ggml_vk_synchronize(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_synchronize()");

    तार्किक do_transfer = !ctx->compute_ctx.expired();

    यदि (ggml_vk_submit_transfer_ctx(ctx)) {
        ctx->submit_pending = सत्य;
    }

    vk_context compute_ctx;
    vk_command_बफर* cmd_buf = शून्यसूचकः;
    यदि (do_transfer) {
        compute_ctx = ctx->compute_ctx.तालकम्();
        यदि (compute_ctx->s) {
            cmd_buf = compute_ctx->s->बफर;
        }

        ggml_vk_ctx_अन्तः(compute_ctx);

        प्रति (स्वचालित& cpy : compute_ctx->अन्तर्_memcpys) {
            स्मृतिं_प्रतिलिख(cpy.dst, cpy.src, cpy.n);
        }

        यदि (ctx->device->serialize_submissions) {
            ggml_vk_submit(compute_ctx, ctx->fence);
            VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, सत्य, UINT64_MAX), "synchronize waitForFences", ctx->device);
            ctx->device->device.resetFences({ ctx->fence });
        } अन्यथा {
            ggml_vk_submit(compute_ctx, {});
        }
        ctx->submit_pending = सत्य;
    }

    यदि (ctx->submit_pending) {
        यदि (ctx->device->serialize_submissions) {
            ctx->submit_pending = असत्य;
        } अन्यथा यदि (ctx->device->async_use_transfer_queue && ctx->transfer_semaphore_चरमः_submitted < ctx->transfer_semaphore.value) {
            vk::TimelineSemaphoreSubmitInfo tl_info{
                1, &ctx->transfer_semaphore.value,
                0, शून्यसूचकः,
            };
            vk::PipelineStageFlags stage = ctx->device->transfer_queue->stage_flags;
            vk::SubmitInfo si{
                1, &ctx->transfer_semaphore.s, &stage,
                0, शून्यसूचकः,
                0, शून्यसूचकः,
            };
            si.समुच्चयः_PNext(&tl_info);
            ctx->device->compute_queue->handle->submit({ si }, ctx->fence);
            ctx->transfer_semaphore_चरमः_submitted = ctx->transfer_semaphore.value;
        } अन्यथा {
            ctx->device->compute_queue->handle->submit({}, ctx->fence);
        }
        यदि (!ctx->device->serialize_submissions) {
            ggml_vk_wait_for_fence(ctx);
        }
        ctx->submit_pending = असत्य;
        यदि (cmd_buf) {
            cmd_buf->अन्तर्_use = असत्य;
            cmd_buf->buf.reset();
        }
    }

    यदि (do_transfer) {
        प्रति (स्वचालित& cpy : compute_ctx->बहिर्गमनम्_memcpys) {
            स्मृतिं_प्रतिलिख(cpy.dst, cpy.src, cpy.n);
        }
        ctx->compute_ctx.reset();
    }
}

स्थिर शून्य ggml_backend_vk_synchronize(ggml_backend_t backend) {
    VK_LOG_DEBUG("ggml_backend_vk_synchronize()");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ggml_vk_synchronize(ctx);

    ggml_vk_graph_cleanup(ctx);
}

तार्किक ggml_vk_is_रिक्तानि(ggml_tensor * ग्रन्थिः) {
    प्रत्यागम ggml_is_रिक्तानि(ग्रन्थिः) || ग्रन्थिः->op == GGML_OP_NONE || ग्रन्थिः->op == GGML_OP_RESHAPE || ग्रन्थिः->op == GGML_OP_TRANSPOSE || ग्रन्थिः->op == GGML_OP_VIEW || ग्रन्थिः->op == GGML_OP_PERMUTE;
}

स्थिर तार्किक ggml_vk_can_fuse_unary_mul(ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क unary_idx, पूर्णाङ्क mul_idx) {
    ध्रुव ggml_tensor * unary = cgraph->nodes[unary_idx];
    ध्रुव ggml_tensor * mul = cgraph->nodes[mul_idx];

    यदि (ggml_vk_unary_mul_op_सूचकांक(ggml_get_unary_op(unary)) < 0) {
        प्रत्यागम असत्य;
    }
    यदि (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
        प्रत्यागम असत्य;
    }
    यदि (unary->type != mul->type) {
        प्रत्यागम असत्य;
    }
    यदि (mul->src[0] != unary && mul->src[1] != unary) {
        प्रत्यागम असत्य;
    }
    ध्रुव ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
    यदि (other == शून्यसूचकः || other->type != unary->type) {
        प्रत्यागम असत्य;
    }
    यदि (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0])) {
        प्रत्यागम असत्य;
    }
    // fastmod needs src to tile into dst
    यदि (mul->src[0] == unary) {
        प्रत्यागम ggml_can_repeat(other, unary);
    }
    प्रत्यागम ggml_can_repeat(unary, mul->src[0]);
}

स्थिर तार्किक ggml_vk_can_fuse_unary_mul_pair(ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {
    ध्रुव गणना ggml_op ops[]    = { GGML_OP_UNARY, GGML_OP_MUL };
    ध्रुव पूर्णाङ्क           outputs[] = { ग्रन्थिः_idx + 1 };
    प्रत्यागम ggml_can_fuse_subgraph(cgraph, ग्रन्थिः_idx, 2, ops, outputs, 1) &&
           ggml_vk_can_fuse_unary_mul(cgraph, ग्रन्थिः_idx, ग्रन्थिः_idx + 1);
}

स्थिर तार्किक ggml_vk_can_fuse_hc_post_gate(ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {
    ध्रुव ggml_tensor * scale_अन्तर्  = cgraph->nodes[ग्रन्थिः_idx];
    ध्रुव ggml_tensor * sigmoid   = cgraph->nodes[ग्रन्थिः_idx + 1];
    ध्रुव ggml_tensor * scale_बहिर्गमनम् = cgraph->nodes[ग्रन्थिः_idx + 2];

    // the shader folds scale -> sigmoid -> scale; a bias on either scale is not handled
    प्रत्यागम ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID &&
           ggml_get_op_params_f32(scale_अन्तर्, 1) == 0.0f &&
           ggml_get_op_params_f32(scale_बहिर्गमनम्, 1) == 0.0f &&
           scale_अन्तर्->src[0]->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(scale_अन्तर्->src[0], scale_बहिर्गमनम्);
}

तार्किक ggml_vk_can_fuse(ध्रुव ggml_backend_vk_context * ctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx, std::initializer_list<गणना ggml_op> ops) {
    यदि (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL) {
        प्रत्यागम ggml_vk_can_fuse_unary_mul_pair(cgraph, ग्रन्थिः_idx);
    }

    यदि (!ggml_can_fuse(cgraph, ग्रन्थिः_idx, ops)) {
        प्रत्यागम असत्य;
    }

    यदि ((ops.size() == 2 || ops.size() == 3 || ops.size() == 4) &&
        ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        ध्रुव ggml_tensor *rms_norm = cgraph->nodes[ग्रन्थिः_idx];
        ध्रुव ggml_tensor *mul = cgraph->nodes[ग्रन्थिः_idx + 1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        // rms_norm only supports f32
        यदि (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            प्रत्यागम असत्य;
        }
        // if rms_norm is the B operand, then we don't handle broadcast
        यदि (rms_norm == mul->src[1] &&
            !ggml_are_same_shape(mul->src[0], rms_norm)) {
            प्रत्यागम असत्य;
        }
        // rms_norm shader assumes contiguous rows
        यदि (!ggml_is_contiguous_पङ्क्तिकाः(mul->src[0]) || !ggml_is_contiguous_पङ्क्तिकाः(mul->src[1])) {
            प्रत्यागम असत्य;
        }

        यदि (ops.size() >= 3 && ops.begin()[2] == GGML_OP_सङ्कलय) {
            ध्रुव ggml_tensor *सङ्कलय_2 = cgraph->nodes[ग्रन्थिः_idx + 2];
            ध्रुव ggml_tensor *residual = सङ्कलय_2->src[0] == mul ? सङ्कलय_2->src[1] : सङ्कलय_2->src[0];
            यदि (सङ्कलय_2->src[0] != mul && सङ्कलय_2->src[1] != mul) {
                प्रत्यागम असत्य;
            }
            यदि (residual->type != GGML_TYPE_F32 || सङ्कलय_2->type != GGML_TYPE_F32 ||
                !ggml_are_same_shape(सङ्कलय_2, residual) || !ggml_is_contiguous(residual) ||
                !ggml_is_contiguous(सङ्कलय_2) || get_misalign_अष्टकानि(ctx, residual) != 0) {
                प्रत्यागम असत्य;
            }

            ध्रुव ggml_tensor *dst = सङ्कलय_2;
            यदि (ops.size() == 4) {
                यदि (ops.begin()[3] != GGML_OP_MUL) {
                    प्रत्यागम असत्य;
                }

                ध्रुव ggml_tensor *post_mul = cgraph->nodes[ग्रन्थिः_idx + 3];
                ध्रुव ggml_tensor *scale = post_mul->src[0] == सङ्कलय_2 ? post_mul->src[1] : post_mul->src[0];
                यदि (post_mul->src[0] != सङ्कलय_2 && post_mul->src[1] != सङ्कलय_2) {
                    प्रत्यागम असत्य;
                }
                // The shader reads data_e[0], so the final multiply must use a scalar.
                यदि (scale->type != GGML_TYPE_F32 || post_mul->type != GGML_TYPE_F32 ||
                    ggml_nelements(scale) != 1 || !ggml_is_contiguous(post_mul) ||
                    get_misalign_अष्टकानि(ctx, scale) != 0) {
                    प्रत्यागम असत्य;
                }
                dst = post_mul;
            }

            यदि (get_misalign_अष्टकानि(ctx, dst) != 0) {
                प्रत्यागम असत्य;
            }
        }
    }

    स्वचालित ध्रुव &mm_सङ्कलय_ok = [&](ध्रुव ggml_tensor *mul, ध्रुव ggml_tensor *सङ्कलय_2) {
        ध्रुव ggml_tensor *bias = सङ्कलय_2->src[0] == mul ? सङ्कलय_2->src[1] : सङ्कलय_2->src[0];

        // mat-vec only
        यदि (ggml_nrows(mul) != 1) {
            प्रत्यागम असत्य;
        }
        // shaders assume the types match
        यदि (mul->type != bias->type) {
            प्रत्यागम असत्य;
        }
        // shaders reuse the D shape for bias
        यदि (!ggml_are_same_shape(mul, bias) ||
            !ggml_are_same_stride(mul, bias)) {
            प्रत्यागम असत्य;
        }
        // unaligned bias isn't handled
        यदि (get_misalign_अष्टकानि(ctx, bias) != 0) {
            प्रत्यागम असत्य;
        }
        प्रत्यागम सत्य;
    };

    यदि ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_सङ्कलय) {
        // additional constraints specific to this fusion
        ध्रुव ggml_tensor *mul = cgraph->nodes[ग्रन्थिः_idx];
        ध्रुव ggml_tensor *सङ्कलय_2 = cgraph->nodes[ग्रन्थिः_idx + 1];

        यदि (!mm_सङ्कलय_ok(mul, सङ्कलय_2)) {
            प्रत्यागम असत्य;
        }
        यदि (ops.size() == 3) {
            यदि (ops.begin()[2] != GGML_OP_सङ्कलय) {
                प्रत्यागम असत्य;
            }
            यदि (!mm_सङ्कलय_ok(सङ्कलय_2, cgraph->nodes[ग्रन्थिः_idx + 2])) {
                प्रत्यागम असत्य;
            }
        }
    }

    स्वचालित ध्रुव &mmid_mul_ok = [&](ध्रुव ggml_tensor *mmid, ध्रुव ggml_tensor *mul) {
        ध्रुव ggml_tensor *scale = mul->src[1];

        यदि (mmid != mul->src[0]) {
            प्रत्यागम असत्य;
        }
        // mat-vec only
        यदि (!ggml_vk_use_mul_mat_vec_id(cgraph, ग्रन्थिः_idx)) {
            प्रत्यागम असत्य;
        }
        // shaders assume the types match
        यदि (mmid->type != scale->type) {
            प्रत्यागम असत्य;
        }
        // shaders assume the bias is contiguous
        यदि (!ggml_is_contiguous(scale)) {
            प्रत्यागम असत्य;
        }
        // unaligned bias isn't handled
        यदि (get_misalign_अष्टकानि(ctx, scale) != 0) {
            प्रत्यागम असत्य;
        }
        // shader only indexes by expert index
        यदि (scale->ne[0] != 1 ||
            scale->ne[1] != mul->ne[1] ||
            scale->ne[2] != 1 ||
            scale->ne[3] != 1) {
            प्रत्यागम असत्य;
        }
        प्रत्यागम सत्य;
    };

    यदि ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_सङ्कलय_ID) {
        // additional constraints specific to this fusion
        ध्रुव ggml_tensor *mul = cgraph->nodes[ग्रन्थिः_idx];
        ध्रुव ggml_tensor *सङ्कलय_2 = cgraph->nodes[ग्रन्थिः_idx + 1];
        ध्रुव ggml_tensor *bias = सङ्कलय_2->src[1];

        यदि (mul != सङ्कलय_2->src[0]) {
            प्रत्यागम असत्य;
        }
        // mat-vec only
        यदि (!ggml_vk_use_mul_mat_vec_id(cgraph, ग्रन्थिः_idx)) {
            प्रत्यागम असत्य;
        }
        // shaders assume the types match
        यदि (mul->type != bias->type) {
            प्रत्यागम असत्य;
        }
        // shaders assume the bias is contiguous
        यदि (!ggml_is_contiguous(bias)) {
            प्रत्यागम असत्य;
        }
        // the ID tensor must be the same for mul_mat_id and add_id
        यदि (mul->src[2] != सङ्कलय_2->src[2]) {
            प्रत्यागम असत्य;
        }
        // unaligned bias isn't handled
        यदि (get_misalign_अष्टकानि(ctx, bias) != 0) {
            प्रत्यागम असत्य;
        }

        यदि (ops.size() == 3) {
            यदि (ops.begin()[2] != GGML_OP_MUL) {
                प्रत्यागम असत्य;
            }
            ध्रुव ggml_tensor *mul = cgraph->nodes[ग्रन्थिः_idx + 2];
            प्रत्यागम mmid_mul_ok(सङ्कलय_2, mul);
        }
    }

    यदि (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        ध्रुव ggml_tensor *mmid = cgraph->nodes[ग्रन्थिः_idx];
        ध्रुव ggml_tensor *mul = cgraph->nodes[ग्रन्थिः_idx + 1];

        यदि (!mmid_mul_ok(mmid, mul)) {
            प्रत्यागम असत्य;
        }
    }

    प्रत्यागम सत्य;
}

तार्किक ggml_vk_can_fuse_ssm_conv(ध्रुव ggml_backend_vk_context * ctx, ध्रुव संरचना ggml_cgraph * cgraph,
                                      पूर्णाङ्क ग्रन्थिः_idx, पूर्णाङ्क अङ्कः_extra) {
    ध्रुव ggml_tensor * conv = cgraph->nodes[ग्रन्थिः_idx];
    यदि (conv->op != GGML_OP_SSM_CONV) {
        प्रत्यागम असत्य;
    }

    ध्रुव ggml_tensor * silu = शून्यसूचकः;
    ध्रुव ggml_tensor * bias = शून्यसूचकः;

    यदि (अङ्कः_extra == 1) {
        यदि (!ggml_can_fuse(cgraph, ग्रन्थिः_idx, { GGML_OP_SSM_CONV, GGML_OP_UNARY })) {
            प्रत्यागम असत्य;
        }
        silu = cgraph->nodes[ग्रन्थिः_idx + 1];
    } अन्यथा यदि (अङ्कः_extra == 2) {
        यदि (!ggml_can_fuse(cgraph, ग्रन्थिः_idx, { GGML_OP_SSM_CONV, GGML_OP_सङ्कलय, GGML_OP_UNARY })) {
            प्रत्यागम असत्य;
        }
        ध्रुव ggml_tensor * सङ्कलय_2 = cgraph->nodes[ग्रन्थिः_idx + 1];
        silu = cgraph->nodes[ग्रन्थिः_idx + 2];
        bias = (सङ्कलय_2->src[0] == conv) ? सङ्कलय_2->src[1] : सङ्कलय_2->src[0];

        यदि (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            प्रत्यागम असत्य;
        }
        // bias must be channel-wise (one element per channel of the conv output)
        यदि (ggml_nelements(bias) != conv->ne[0] || bias->ne[0] != conv->ne[0]) {
            प्रत्यागम असत्य;
        }
        यदि (सङ्कलय_2->type != GGML_TYPE_F32) {
            प्रत्यागम असत्य;
        }
        // The shader doesn't apply per-tensor offsets, so reject misaligned bias.
        यदि (get_misalign_अष्टकानि(ctx, bias) != 0) {
            प्रत्यागम असत्य;
        }
    } अन्यथा {
        प्रत्यागम असत्य;
    }

    यदि (ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU) {
        प्रत्यागम असत्य;
    }
    यदि (conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }
    // The shader writes to the fused dst using its own strides, but the push constants don't
    // carry a per-tensor offset, so the binding must be naturally aligned.
    यदि (get_misalign_अष्टकानि(ctx, silu) != 0) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम सत्य;
}

तार्किक ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, ध्रुव संरचना ggml_cgraph * cgraph,
                                      पूर्णाङ्क ग्रन्थिः_idx, topk_moe_बहुलकम् mode) {

    ध्रुव ggml_tensor * softmax;
    ध्रुव ggml_tensor * weights;
    ध्रुव ggml_tensor * get_पङ्क्तिकाः;
    ध्रुव ggml_tensor * argsort;

    विकल्प (mode) {
    पक्ष TOPK_MOE_EARLY_SOFTMAX_NORM:
        softmax = cgraph->nodes[ग्रन्थिः_idx + 0];
        weights = cgraph->nodes[ग्रन्थिः_idx + 9];
        get_पङ्क्तिकाः = cgraph->nodes[ग्रन्थिः_idx + 4];
        argsort = cgraph->nodes[ग्रन्थिः_idx + 2];
        भङ्ग;
    पक्ष TOPK_MOE_SIGMOID_NORM_BIAS:
        softmax = cgraph->nodes[ग्रन्थिः_idx + 0]; // really sigmoid
        weights = cgraph->nodes[ग्रन्थिः_idx + 10];
        get_पङ्क्तिकाः = cgraph->nodes[ग्रन्थिः_idx + 5];
        argsort = cgraph->nodes[ग्रन्थिः_idx + 3];
        यदि (ggml_get_unary_op(softmax) != GGML_UNARY_OP_SIGMOID) {
            प्रत्यागम असत्य;
        }
        // bias is expected to be 1D
        यदि (ggml_nrows(cgraph->nodes[ग्रन्थिः_idx + 2]->src[1]) != 1 ||
            !ggml_is_contiguous(cgraph->nodes[ग्रन्थिः_idx + 2]->src[1])) {
            प्रत्यागम असत्य;
        }
        // sigmoid fusion seems to generate infinities on moltenvk
        यदि (ctx->device->driver_id == vk::DriverId::eMoltenvk) {
            प्रत्यागम असत्य;
        }
        भङ्ग;
    पक्ष TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS:
        softmax = cgraph->nodes[ग्रन्थिः_idx + 0]; // really softplus
        weights = cgraph->nodes[ग्रन्थिः_idx + 11];
        get_पङ्क्तिकाः = cgraph->nodes[ग्रन्थिः_idx + 6];
        argsort = cgraph->nodes[ग्रन्थिः_idx + 4];
        यदि (ggml_get_unary_op(softmax) != GGML_UNARY_OP_SOFTPLUS) {
            प्रत्यागम असत्य;
        }
        // bias is expected to be 1D
        यदि (ggml_nrows(cgraph->nodes[ग्रन्थिः_idx + 3]->src[1]) != 1 ||
            !ggml_is_contiguous(cgraph->nodes[ग्रन्थिः_idx + 3]->src[1])) {
            प्रत्यागम असत्य;
        }
        भङ्ग;
    पक्ष TOPK_MOE_EARLY_SOFTMAX:
        softmax = cgraph->nodes[ग्रन्थिः_idx + 0];
        weights = cgraph->nodes[ग्रन्थिः_idx + 4];
        get_पङ्क्तिकाः = cgraph->nodes[ग्रन्थिः_idx + 4];
        argsort = cgraph->nodes[ग्रन्थिः_idx + 2];
        भङ्ग;
    पक्ष TOPK_MOE_LATE_SOFTMAX:
        softmax = cgraph->nodes[ग्रन्थिः_idx + 4];
        weights = cgraph->nodes[ग्रन्थिः_idx + 5];
        get_पङ्क्तिकाः = cgraph->nodes[ग्रन्थिः_idx + 2];
        argsort = cgraph->nodes[ग्रन्थिः_idx + 0];
        भङ्ग;
    स्वभाव:
        प्रत्यागम असत्य;
    }

    ggml_tensor * probs = get_पङ्क्तिकाः->src[0];
    यदि (probs->op != GGML_OP_RESHAPE) {
        प्रत्यागम असत्य;
    }
    probs = probs->src[0];
    ggml_tensor * selection_probs = argsort->src[0];

    यदि (probs != selection_probs &&
        mode != TOPK_MOE_SIGMOID_NORM_BIAS &&
        mode != TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS) {
        प्रत्यागम असत्य;
    }

    यदि (!ggml_is_contiguous(softmax->src[0]) || !ggml_is_contiguous(weights)) {
        प्रत्यागम असत्य;
    }

    यदि (softmax->op == GGML_OP_SOFT_MAX) {
        ध्रुव भिन्न * op_params = (ध्रुव भिन्न *)softmax->op_params;

        भिन्न scale = op_params[0];
        भिन्न max_bias = op_params[1];

        यदि (scale != 1.0f || max_bias != 0.0f) {
            प्रत्यागम असत्य;
        }

        // don't fuse when masks or sinks are present
        यदि (softmax->src[1] || softmax->src[2]) {
            प्रत्यागम असत्य;
        }
    }

    ध्रुव पूर्णाङ्क n_expert = softmax->ne[0];
    यदि (n_expert > (1 << (अङ्कः_topk_moe_pipelines-1))) {
        प्रत्यागम असत्य;
    }

    यदि (!ctx->device->subgroup_arithmetic ||
        !ctx->device->subgroup_shuffle ||
        !ctx->device->subgroup_require_पूर्णम्_support ||
        ctx->device->disable_fusion) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;
}

स्थिर तार्किक ggml_vk_मेलनम्_ops(ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx,
                              ध्रुव std::initializer_list<ggml_op> & ops) {
    यदि (ग्रन्थिः_idx + (पूर्णाङ्क) ops.size() > cgraph->n_nodes) {
        प्रत्यागम असत्य;
    }
    प्रति (size_t j = 0; j < ops.size(); ++j) {
        ध्रुव ggml_tensor * ग्रन्थिः = cgraph->nodes[ग्रन्थिः_idx + j];
        यदि (ग्रन्थिः->op != ops.begin()[j] ||
            (ग्रन्थिः->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
            (ग्रन्थिः->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            प्रत्यागम असत्य;
        }
    }
    प्रत्यागम सत्य;
}

तार्किक ggml_vk_can_fuse_topk_qsa(ggml_backend_vk_context * ctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {
    यदि (ctx->device->disable_fusion || !ctx->device->pipeline_topk_radix_qsa) {
        प्रत्यागम असत्य;
    }

    ध्रुव पूर्णाङ्क n_ops = topk_qsa_आदर्शः.size();
    यदि (!ggml_vk_मेलनम्_ops(cgraph, ग्रन्थिः_idx, topk_qsa_आदर्शः) ||
        !ggml_check_धाराः(cgraph, ग्रन्थिः_idx, topk_qsa_धाराः)) {
        प्रत्यागम असत्य;
    }

    // elided nodes must be single-use (cpy counts its own src[1] self-reference)
    प्रति (पूर्णाङ्क j = 0; j < n_ops - 1; ++j) {
        ध्रुव ggml_tensor * ग्रन्थिः = cgraph->nodes[ग्रन्थिः_idx + j];
        ध्रुव int32_t want = ग्रन्थिः->op == GGML_OP_CPY ? 2 : 1;
        यदि (ggml_ग्रन्थिः_get_use_गणनम्(cgraph, ग्रन्थिः_idx + j) != want) {
            प्रत्यागम असत्य;
        }
    }

    ध्रुव ggml_tensor * get_पङ्क्तिकाः = cgraph->nodes[ग्रन्थिः_idx + 0];
    ध्रुव ggml_tensor * सङ्कलय_2      = cgraph->nodes[ग्रन्थिः_idx + n_ops - 2];
    ध्रुव ggml_tensor * शिखरम्_k    = cgraph->nodes[ग्रन्थिः_idx + n_ops - 1];

    ध्रुव ggml_tensor * scores   = get_पङ्क्तिकाः->src[0]; // [n_tps, n_blocks, n_stream]
    ध्रुव ggml_tensor * cell_blk = get_पङ्क्तिकाः->src[1]; // [n_kv, n_stream]
    ध्रुव ggml_tensor * expanded = सङ्कलय_2->src[0];      // [n_kv, n_tps, n_stream]

    // raw mask: follow the reshape/cpy chain back to the materialized f16 input
    ध्रुव ggml_tensor * आवरणम् = सङ्कलय_2->src[1];
    यावत् (आवरणम् && (आवरणम्->op == GGML_OP_RESHAPE || आवरणम्->op == GGML_OP_CPY)) {
        आवरणम् = आवरणम्->src[0];
    }
    यदि (!आवरणम् || आवरणम्->type != GGML_TYPE_F16) {
        प्रत्यागम असत्य;
    }

    यदि (scores->type != GGML_TYPE_F32 || cell_blk->type != GGML_TYPE_I32 || शिखरम्_k->type != GGML_TYPE_I32) {
        प्रत्यागम असत्य;
    }
    यदि (!ggml_is_contiguous(scores) || !ggml_is_contiguous(cell_blk) || !ggml_is_contiguous(आवरणम्) ||
        !ggml_is_contiguous(expanded) || !ggml_is_contiguous(शिखरम्_k)) {
        प्रत्यागम असत्य;
    }

    ध्रुव int64_t n_tps    = scores->ne[0];
    ध्रुव int64_t n_खण्डानि = scores->ne[1];
    ध्रुव int64_t n_प्रवाहः = scores->ne[2];
    ध्रुव int64_t n_kv     = cell_blk->ne[0];
    ध्रुव int64_t विस्तारः    = शिखरम्_k->ne[0];

    // pin the indexer layout the shader's addressing assumes
    यदि (scores->ne[3] != 1 || cell_blk->ne[1] != n_प्रवाहः || ggml_nrows(cell_blk) != n_प्रवाहः ||
        ggml_nelements(आवरणम्) != n_kv * n_tps * n_प्रवाहः ||
        expanded->ne[0] != n_kv || expanded->ne[1] != n_tps || expanded->ne[2] != n_प्रवाहः ||
        शिखरम्_k->ne[1] != n_tps || शिखरम्_k->ne[2] != n_प्रवाहः || शिखरम्_k->ne[3] != 1 ||
        n_खण्डानि <= 0 || n_kv <= 0 || विस्तारः <= 0 || विस्तारः > n_kv) {
        प्रत्यागम असत्य;
    }

    // only worth it in the radix regime; small k uses the faster tournament unfused
    ध्रुव uint32_t k_min_pipeline = std::max((uint32_t) log2f(भिन्न(विस्तारः)) + 1, ctx->device->subgroup_आकार_log2);
    यदि (k_min_pipeline < अङ्कः_topk_pipelines && ctx->device->pipeline_topk_f32[k_min_pipeline]) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम सत्य;
}

तार्किक ggml_vk_can_fuse_rope_समुच्चयः_पङ्क्तिकाः(ggml_backend_vk_context * ctx, ध्रुव संरचना ggml_cgraph * cgraph,
                                           पूर्णाङ्क ग्रन्थिः_idx) {
    ध्रुव ggml_tensor *rope = cgraph->nodes[ग्रन्थिः_idx + 0];
    ध्रुव ggml_tensor *view = cgraph->nodes[ग्रन्थिः_idx + 1];
    ध्रुव ggml_tensor *समुच्चयः_पङ्क्तिकाः = cgraph->nodes[ग्रन्थिः_idx + 2];

    // The set_rows epilogue uses one index per ne2 slice and does not encode ne3.
    यदि (rope->src[0]->ne[3] != 1) {
        प्रत्यागम असत्य;
    }

    यदि (समुच्चयः_पङ्क्तिकाः->type != GGML_TYPE_F32 && समुच्चयः_पङ्क्तिकाः->type != GGML_TYPE_F16) {
        प्रत्यागम असत्य;
    }

    // The shader reads each aligned I64 index as a uvec2 and uses its low 32 bits.
    यदि (समुच्चयः_पङ्क्तिकाः->src[1]->type != GGML_TYPE_I64 || !ggml_is_contiguous(समुच्चयः_पङ्क्तिकाः->src[1]) ||
        समुच्चयः_पङ्क्तिकाः->nb[0] != ggml_type_आकार(समुच्चयः_पङ्क्तिकाः->type) || get_misalign_अष्टकानि(ctx, समुच्चयः_पङ्क्तिकाः->src[1]) != 0) {
        प्रत्यागम असत्य;
    }

    // SET_ROWS consumes one flattened [ne0*ne1] row for each ne2 slice.
    यदि (!ggml_is_contiguous(view) ||
        view->ne[0] != rope->ne[0] * rope->ne[1] || view->ne[1] != rope->ne[2] ||
        view->ne[2] != 1 || view->ne[3] != 1 ||
        ggml_nelements(समुच्चयः_पङ्क्तिकाः->src[1]) != rope->ne[2]) {
        प्रत्यागम असत्य;
    }

    // Only norm/neox/mrope/imrope shaders have the fusion code
    ध्रुव पूर्णाङ्क mode = ((ध्रुव int32_t *) rope->op_params)[2];
    यदि (mode != GGML_ROPE_TYPE_प्रसामान्यम् && mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE && mode != GGML_ROPE_TYPE_IMROPE) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;
}

तार्किक ggml_vk_can_fuse_rms_norm_समुच्चयः_पङ्क्तिकाः(ggml_backend_vk_context * ctx, ध्रुव संरचना ggml_cgraph * cgraph,
                                               पूर्णाङ्क ग्रन्थिः_idx) {
    ध्रुव ggml_tensor * rms = cgraph->nodes[ग्रन्थिः_idx];
    ध्रुव ggml_tensor * view = cgraph->nodes[ग्रन्थिः_idx + 1];
    ध्रुव ggml_tensor * समुच्चयः_पङ्क्तिकाः = cgraph->nodes[ग्रन्थिः_idx + 2];

    // The RMS kernel reads F32 and writes directly to the F32 or F16 SET_ROWS destination.
    यदि (rms->src[0]->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 ||
        (समुच्चयः_पङ्क्तिकाः->type != GGML_TYPE_F32 && समुच्चयः_पङ्क्तिकाः->type != GGML_TYPE_F16) ||
        समुच्चयः_पङ्क्तिकाः->src[1]->type != GGML_TYPE_I64 || !ggml_is_contiguous(समुच्चयः_पङ्क्तिकाः->src[1]) ||
        समुच्चयः_पङ्क्तिकाः->nb[0] != ggml_type_आकार(समुच्चयः_पङ्क्तिकाः->type) || get_misalign_अष्टकानि(ctx, समुच्चयः_पङ्क्तिकाः->src[1]) != 0) {
        प्रत्यागम असत्य;
    }
    // As with the ROPE epilogue, each ne2 slice supplies one flattened row and ne3 is not encoded.
    यदि (rms->ne[3] != 1 || !ggml_is_contiguous(rms->src[0]) || !ggml_is_contiguous(view)) {
        प्रत्यागम असत्य;
    }
    यदि (view->ne[0] != rms->ne[0] * rms->ne[1] || view->ne[1] != rms->ne[2] ||
        view->ne[2] != 1 || view->ne[3] != 1 ||
        ggml_nelements(समुच्चयः_पङ्क्तिकाः->src[1]) != rms->ne[2]) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;
}

तार्किक ggml_vk_can_fuse_snake(ggml_backend_vk_context * ctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {
    GGML_अप्रयुक्तम्(ctx);
    यदि (!ggml_can_fuse(cgraph, ग्रन्थिः_idx, snake_आदर्शः)) {
        प्रत्यागम असत्य;
    }

    ध्रुव ggml_tensor * mul0     = cgraph->nodes[ग्रन्थिः_idx + 0];
    ध्रुव ggml_tensor * sin_ग्रन्थिः = cgraph->nodes[ग्रन्थिः_idx + 1];
    ध्रुव ggml_tensor * sqr      = cgraph->nodes[ग्रन्थिः_idx + 2];
    ध्रुव ggml_tensor * mul1     = cgraph->nodes[ग्रन्थिः_idx + 3];
    ध्रुव ggml_tensor * सङ्कलय_2      = cgraph->nodes[ग्रन्थिः_idx + 4];

    ध्रुव ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ध्रुव ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    ध्रुव ggml_tensor * inv_b    = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];
    ध्रुव ggml_tensor * x_अन्तर्_सङ्कलय = (सङ्कलय_2->src[0] == mul1) ? सङ्कलय_2->src[1] : सङ्कलय_2->src[0];

    यदि (x_अन्तर्_सङ्कलय != x) {
        प्रत्यागम असत्य;
    }
    यदि (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_F16 && x->type != GGML_TYPE_BF16) {
        प्रत्यागम असत्य;
    }
    // Shader bindings: data_a is A_TYPE so it follows x's precision, while
    // data_b and data_c are hardcoded float, so the broadcast operands must
    // be F32 regardless of x's type.
    यदि (a->type     != GGML_TYPE_F32) प्रत्यागम असत्य;
    यदि (inv_b->type != GGML_TYPE_F32) प्रत्यागम असत्य;
    // Chain intermediates and output share x's precision (single A_TYPE / D_TYPE pipeline).
    यदि (mul0->type     != x->type) प्रत्यागम असत्य;
    यदि (sin_ग्रन्थिः->type != x->type) प्रत्यागम असत्य;
    यदि (sqr->type      != x->type) प्रत्यागम असत्य;
    यदि (mul1->type     != x->type) प्रत्यागम असत्य;
    यदि (सङ्कलय_2->type      != x->type) प्रत्यागम असत्य;
    यदि (!ggml_are_same_shape(a, inv_b)) {
        प्रत्यागम असत्य;
    }
    यदि (a->ne[0] != 1 || a->ne[1] != x->ne[1]) {
        प्रत्यागम असत्य;
    }
    // Dispatch is 2D over (ne0, ne1), so x and add must be 2D and a / inv_b
    // must collapse to [1, C, 1, 1]. Higher dims are not handled by the shader.
    यदि (x->ne[2]     != 1 || x->ne[3]     != 1) प्रत्यागम असत्य;
    यदि (सङ्कलय_2->ne[2]   != 1 || सङ्कलय_2->ne[3]   != 1) प्रत्यागम असत्य;
    यदि (a->ne[2]     != 1 || a->ne[3]     != 1) प्रत्यागम असत्य;
    यदि (inv_b->ne[2] != 1 || inv_b->ne[3] != 1) प्रत्यागम असत्य;
    // Shader uses idx = i0 + i1 * ne0 and reads data_b[i1] / data_c[i1],
    // so every operand must be contiguous.
    यदि (!ggml_is_contiguous(x) || !ggml_is_contiguous(सङ्कलय_2) ||
        !ggml_is_contiguous(a) || !ggml_is_contiguous(inv_b)) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम सत्य;
}

तार्किक ggml_vk_tensors_overlap(ध्रुव ggml_tensor * a, ध्रुव ggml_tensor * b, तार्किक elementwise) {
    ggml_backend_vk_बफर_context * a_buf_ctx = (ggml_backend_vk_बफर_context *)a->बफर->context;
    vk_बफर a_buf = a_buf_ctx->dev_बफर;
    ggml_backend_vk_बफर_context * b_buf_ctx = (ggml_backend_vk_बफर_context *)b->बफर->context;
    vk_बफर b_buf = b_buf_ctx->dev_बफर;
    यदि (a_buf == b_buf) {
        स्वचालित a_आधारः = vk_tensor_विस्थापनम्(a) + a->view_offs;
        स्वचालित a_आकार = ggml_nbytes(a);
        स्वचालित b_आधारः = vk_tensor_विस्थापनम्(b) + b->view_offs;
        स्वचालित b_आकार = ggml_nbytes(b);

        यदि (elementwise && a_आधारः == b_आधारः && a_आकार == b_आकार) {
            प्रत्यागम असत्य;
        }

        यदि ((b_आधारः <= a_आधारः && a_आधारः < b_आधारः + b_आकार) ||
            (a_आधारः <= b_आधारः && b_आधारः < a_आधारः + a_आकार)) {
            प्रत्यागम सत्य;
        }
    }
    प्रत्यागम असत्य;
}

तार्किक ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, ध्रुव संरचना ggml_cgraph * cgraph,
                                               पूर्णाङ्क ग्रन्थिः_idx) {
    ध्रुव ggml_tensor *rms = cgraph->nodes[ग्रन्थिः_idx + 0];
    ध्रुव ggml_tensor *mul = cgraph->nodes[ग्रन्थिः_idx + 1];
    ध्रुव ggml_tensor *rope = cgraph->nodes[ग्रन्थिः_idx + 2];

    ध्रुव पूर्णाङ्क mode = ((ध्रुव int32_t *) rope->op_params)[2];

    // noncontig tensors aren't tested, and don't seem common in practice
    यदि (!ggml_is_contiguous(rms) ||
        !ggml_is_contiguous(mul) ||
        !ggml_is_contiguous(rope)) {
        प्रत्यागम असत्य;
    }

    // only norm/neox are handled in the shader
    यदि (mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_प्रसामान्यम्) {
        प्रत्यागम असत्य;
    }

    // shared memory size for passing data from mul->rope
    यदि (mul->ne[0] > 1024) {
        प्रत्यागम असत्य;
    }

    // conditions for pipeline creation
    यदि (परिमाण(vk_op_rms_norm_mul_rope_प्रक्षिप_constants) > ctx->device->properties.limits.max_प्रक्षिप_Constants_आकार) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;
}

uint32_t ggml_vk_fuse_multi_सङ्कलय(ggml_backend_vk_context * ctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {

    ध्रुव ggml_tensor *first_ग्रन्थिः = cgraph->nodes[ग्रन्थिः_idx];
    यदि (first_ग्रन्थिः->op != GGML_OP_सङ्कलय) {
        प्रत्यागम 0;
    }

    यदि (!ctx->device->multi_सङ्कलय) {
        प्रत्यागम 0;
    }

    int32_t अङ्कः_adds = 1;
    यावत् (ग्रन्थिः_idx + अङ्कः_adds < cgraph->n_nodes &&
           cgraph->nodes[ग्रन्थिः_idx + अङ्कः_adds]->op == GGML_OP_सङ्कलय &&
           अङ्कः_adds < MAX_FUSED_ADDS) {
        अङ्कः_adds++;
    }

    // The shader currently requires same shapes (but different strides are allowed),
    // everything f32, and no misalignment
    प्रति (int32_t i = 0; i < अङ्कः_adds; ++i) {
        ध्रुव ggml_tensor *अग्र_ग्रन्थिः = cgraph->nodes[ग्रन्थिः_idx + i];
        यदि (!ggml_are_same_shape(first_ग्रन्थिः, अग्र_ग्रन्थिः->src[0]) ||
            !ggml_are_same_shape(first_ग्रन्थिः, अग्र_ग्रन्थिः->src[1]) ||
            अग्र_ग्रन्थिः->type != GGML_TYPE_F32 ||
            अग्र_ग्रन्थिः->src[0]->type != GGML_TYPE_F32 ||
            अग्र_ग्रन्थिः->src[1]->type != GGML_TYPE_F32 ||
            get_misalign_अष्टकानि(ctx, अग्र_ग्रन्थिः) ||
            get_misalign_अष्टकानि(ctx, अग्र_ग्रन्थिः->src[0]) ||
            get_misalign_अष्टकानि(ctx, अग्र_ग्रन्थिः->src[1])) {
            अङ्कः_adds = i;
        }
    }

    // Verify we can fuse these
    ggml_op adds[MAX_FUSED_ADDS];
    प्रति (int32_t i = 0; i < अङ्कः_adds; ++i) {
        adds[i] = GGML_OP_सङ्कलय;
    }

    // decrease num_adds if they can't all be fused
    यावत् (अङ्कः_adds > 1 && !ggml_can_fuse(cgraph, ग्रन्थिः_idx, adds, अङ्कः_adds)) {
        अङ्कः_adds--;
    }

    // a single add is not "fused", so just return zero
    यदि (अङ्कः_adds == 1) {
        प्रत्यागम 0;
    }
    प्रत्यागम अङ्कः_adds;
}

स्थिर int32_t find_first_समुच्चयः(uint32_t x) {
    int32_t ret = 0;
    यदि (!x) {
        प्रत्यागम -1;
    }
    यावत् (!(x & 1)) {
        x >>= 1;
        ret++;
    }
    प्रत्यागम ret;
}

स्थिर ggml_स्थितिः ggml_backend_vk_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    VK_LOG_DEBUG("ggml_backend_vk_graph_compute(" << cgraph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ctx->device->diag_cgraph = शून्यसूचकः;
    ctx->device->diag_पूर्व_start = -1;
    ctx->device->diag_पूर्व_अन्तः = -1;

    यदि (vk_instance.debug_utils_support) {
        ctx->device->debug_cmdbuf_idx = 0;
    }

    // queue scope, so it encloses every submit this evaluation makes.
    // closed when the function returns
    ggml_vk_debug_label queue_dbg(ctx->device->compute_queue->handle.get(), "ggml_backend_vk_graph_compute");

    ctx->prealloc_आकार_सङ्कलय_rms_partials_विस्थापनम् = 0;
    ctx->do_सङ्कलय_rms_partials = असत्य;
    ctx->do_सङ्कलय_rms_partials_विस्थापनम्_calculation = असत्य;

    पूर्णाङ्क चरमः_ग्रन्थिः = cgraph->n_nodes - 1;

    // If the last op in the cgraph isn't backend GPU, the command buffer doesn't get closed properly
    यावत् (चरमः_ग्रन्थिः > 0 && (ggml_vk_is_रिक्तानि(cgraph->nodes[चरमः_ग्रन्थिः]) || ((cgraph->nodes[चरमः_ग्रन्थिः]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0))) {
        चरमः_ग्रन्थिः -= 1;
    }

    // Reserve tensor context space for all nodes
    ctx->tensor_ctxs.resize(cgraph->n_nodes);

    तार्किक first_ग्रन्थिः_अन्तर्_batch = सत्य; // true if next node will be first node in a batch
    पूर्णाङ्क submit_ग्रन्थिः_idx = 0; // index to first node in a batch

    ggml_vk_submit_transfer_ctx(ctx);

    vk_context compute_ctx;
    यदि (vk_perf_logger_enabled) {
        // allocate/resize the query pool
        यदि (ctx->अङ्कः_queries < cgraph->n_nodes + 1) {
            यदि (ctx->query_pool) {
                ctx->device->device.destroyQueryPool(ctx->query_pool);
            }
            vk::QueryPoolCreateInfo query_create_info;
            query_create_info.queryType = vk::QueryType::eTimestamp;
            query_create_info.query_गणनम् = cgraph->n_nodes + 100;
            ctx->query_pool = ctx->device->device.createQueryPool(query_create_info);
            ctx->अङ्कः_queries = query_create_info.query_गणनम्;
            ctx->query_fusion_नामानि.resize(ctx->अङ्कः_queries);
            ctx->query_fusion_ग्रन्थिः_गणनम्.resize(ctx->अङ्कः_queries);
            ctx->query_nodes.resize(ctx->अङ्कः_queries);
            ctx->query_ग्रन्थिः_idx.resize(ctx->अङ्कः_queries);
        }

        ctx->device->device.resetQueryPool(ctx->query_pool, 0, cgraph->n_nodes+1);
        std::fill(ctx->query_fusion_नामानि.begin(), ctx->query_fusion_नामानि.end(), शून्यसूचकः);
        std::fill(ctx->query_fusion_ग्रन्थिः_गणनम्.begin(), ctx->query_fusion_ग्रन्थिः_गणनम्.end(), 0);
        std::fill(ctx->query_nodes.begin(), ctx->query_nodes.end(), शून्यसूचकः);
        std::fill(ctx->query_ग्रन्थिः_idx.begin(), ctx->query_ग्रन्थिः_idx.end(), 0);

        GGML_ASSERT(ctx->compute_ctx.expired());
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        ctx->query_idx = 0;
        compute_ctx->s->बफर->buf.writeTimestamp(vk::Pipeline_Stage_Flag_द्वयङ्काः::eAllCommands, ctx->query_pool, ctx->query_idx++);
        ggml_vk_sync_buffers(ctx, compute_ctx);
    }

    ctx->prealloc_y_चरमः_pipeline_प्रयुक्तम् = शून्यसूचकः;
    ctx->prealloc_y_चरमः_tensor_प्रयुक्तम् = शून्यसूचकः;
    ctx->prealloc_y_चरमः_k_padded = असत्य;

    यदि (ctx->prealloc_आकार_सङ्कलय_rms_partials) {
        ggml_vk_preallocate_buffers(ctx, शून्यसूचकः);
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        // initialize partial sums to zero.
        ggml_vk_बफर_memset_async(compute_ctx, ctx->prealloc_सङ्कलय_rms_partials, 0, 0, ctx->prealloc_आकार_सङ्कलय_rms_partials);
        ggml_vk_sync_buffers(ctx, compute_ctx);
    }

    // Submit after enough work has accumulated, to overlap CPU cmdbuffer generation with GPU execution.
    // Estimate the amount of compute work using flops, and submit every 200 GFLOP
    // (and scaled down based on total graph flops, so smaller models submit earlier).
    // Also submit at least every 100 nodes, in case there are workloads without heavy compute.
    uint32_t submitted_nodes = 0;
    uint32_t submit_गणनम् = 0;
    uint64_t batch_flops = 0;
    uint64_t योग_flops = 0;
    uint64_t flops_cap = 200'000'000'000ULL;

    // On weaker AMD GPUs larger submissions can hit a driver timeout, submit more often to avoid this
    यदि (ctx->device->vendor_id == VK_VENDOR_ID_AMD && ctx->device->shader_core_गणनम् > 0) {
        यदि (ctx->device->architecture == AMD_GCN && ctx->device->shader_core_गणनम् < 32) {
            flops_cap = 500'000'000ULL * ctx->device->shader_core_गणनम्;
        } अन्यथा यदि (ctx->device->architecture != AMD_GCN && ctx->device->shader_core_गणनम् < 24) {
            flops_cap = 2'000'000'000ULL * ctx->device->shader_core_गणनम्;
        }
    }
    uint64_t flops_प्रत्येकम्_submit = std::min(flops_cap, ctx->चरमः_योग_flops / 40u);

    स्वचालित ध्रुव submit_after = [&](पूर्णाङ्क start, पूर्णाङ्क end) {
        यदि (ctx->device->serialize_submissions) {
            यत्न {
                स्वचालित res = ctx->device->device.waitForFences({ ctx->fence }, सत्य, UINT64_MAX);
                यदि (res != vk::परिणाम_2::eSuccess) {
                    GGML_LOG_ERROR("ggml_vulkan: waitForFences error during serialized submission\n");
                    क्षिप vk::SystemError(vk::make_error_code(res), "ggml_vulkan: waitForFences during serialized submission");
                }
            } गृहाण (vk::Device_नष्टम्_Error &) {
                ggml_vk_print_device_fault_info(ctx->device);
                GGML_LOG_ERROR("ggml_vulkan: device lost on %s waiting for submission (nodes %d to %d):\n",
                        ctx->device->name.c_str(), start, end);
                ggml_vk_print_ग्रन्थिः_list(cgraph, start, end);
                क्षिप;
            }
            ctx->device->device.resetFences({ ctx->fence });
            ctx->submit_pending = असत्य;
            ctx->device->diag_cgraph = cgraph;
            ctx->device->diag_पूर्व_start = start;
            ctx->device->diag_पूर्व_अन्तः = end;
        }
        first_ग्रन्थिः_अन्तर्_batch = सत्य;
        submitted_nodes = 0;
        batch_flops = 0;
        यदि (submit_गणनम् < 3) {
            flops_प्रत्येकम्_submit *= 2;
        }
        submit_गणनम्++;
    };

    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
        यदि (first_ग्रन्थिः_अन्तर्_batch) {
            submit_ग्रन्थिः_idx = i;
        }

        {
            स्वचालित ग्रन्थिः_flops = ggml_vk_get_ग्रन्थिः_flops(cgraph->nodes[i]);
            योग_flops += ग्रन्थिः_flops;

            // Flush the current batch before recording a node that would push it over the flop threshold
            यदि (flops_प्रत्येकम्_submit != 0 && submitted_nodes > 0 && batch_flops + ग्रन्थिः_flops >= flops_प्रत्येकम्_submit) {
                vk_context flush_ctx = ggml_vk_get_compute_ctx(ctx);
                ggml_vk_ctx_अन्तः(flush_ctx);
                flush_ctx->exit_tensor_idx = -1;
                ctx->compute_ctx.reset();
                ggml_vk_compute_forward(ctx, cgraph, cgraph->nodes[submit_ग्रन्थिः_idx], submit_ग्रन्थिः_idx, असत्य);
                submit_after(submit_ग्रन्थिः_idx, i - 1);
                submit_ग्रन्थिः_idx = i;
            }

            batch_flops += ग्रन्थिः_flops;
        }

        // op_srcs_fused_elementwise indicates whether an op's srcs all contribute to
        // the fused result in an elementwise-way. This affects whether the memory for
        // the src is allowed to overlap the memory for the destination.
        // The array is sized to handle the largest fusion (asserted later).
        तार्किक op_srcs_fused_elementwise[13];

        ctx->fused_topk_moe_बहुलकम् = TOPK_MOE_गणनम्;
        ctx->fused_topk_moe_scale = असत्य;
        ctx->fused_topk_qsa = असत्य;
        ctx->fused_hc_post_gate = असत्य;
        ctx->fused_rms_norm_बहुलकम् = RMS_NORM_गणनम्;
        ध्रुव वर्ण *fusion_string {};
        यदि (!ctx->device->disable_fusion) {
            uint32_t अङ्कः_adds = ggml_vk_fuse_multi_सङ्कलय(ctx, cgraph, i);
            यदि (अङ्कः_adds) {
                ctx->अङ्कः_additional_fused_ops = अङ्कः_adds - 1;
                fusion_string = "MULTI_ADD";
                std::fill_n(op_srcs_fused_elementwise, ctx->अङ्कः_additional_fused_ops + 1, सत्य);
            } अन्यथा यदि (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय })) {
                ctx->अङ्कः_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ADD_ADD";
                op_srcs_fused_elementwise[0] = असत्य;
                op_srcs_fused_elementwise[1] = सत्य;
                op_srcs_fused_elementwise[2] = सत्य;
            } अन्यथा यदि (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_सङ्कलय })) {
                ctx->अङ्कः_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ADD";
                op_srcs_fused_elementwise[0] = असत्य;
                op_srcs_fused_elementwise[1] = सत्य;
            } अन्यथा यदि (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_सङ्कलय_ID, GGML_OP_MUL })) {
                ctx->अङ्कः_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ID_ADD_ID_MUL";
                op_srcs_fused_elementwise[0] = असत्य;
                op_srcs_fused_elementwise[1] = सत्य;
                op_srcs_fused_elementwise[2] = सत्य;
            } अन्यथा यदि (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_सङ्कलय_ID })) {
                ctx->अङ्कः_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_ADD_ID";
                op_srcs_fused_elementwise[0] = असत्य;
                op_srcs_fused_elementwise[1] = सत्य;
            } अन्यथा यदि (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_MUL })) {
                ctx->अङ्कः_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_MUL";
                op_srcs_fused_elementwise[0] = असत्य;
                op_srcs_fused_elementwise[1] = सत्य;
            } अन्यथा यदि (ggml_can_fuse_subgraph(cgraph, i, rms_norm_mul_rope_view_समुच्चयः_पङ्क्तिकाः_आदर्शः, { i + 4 }) &&
                       ggml_check_धाराः(cgraph, i, rms_norm_mul_rope_view_समुच्चयः_पङ्क्तिकाः_धाराः) &&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i) &&
                       ggml_vk_can_fuse_rope_समुच्चयः_पङ्क्तिकाः(ctx, cgraph, i + 2)) {
                ctx->अङ्कः_additional_fused_ops = 4;
                ctx->fused_rms_norm_बहुलकम् = RMS_NORM_MUL_ROPE_VIEW_समुच्चयः_पङ्क्तिकाः;
                fusion_string = "RMS_NORM_MUL_ROPE_VIEW_SET_ROWS";
                op_srcs_fused_elementwise[0] = असत्य;
                op_srcs_fused_elementwise[1] = असत्य;
                op_srcs_fused_elementwise[2] = असत्य;
                op_srcs_fused_elementwise[3] = असत्य;
                op_srcs_fused_elementwise[4] = असत्य;
            } अन्यथा यदि (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }) &&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i)) {
                ctx->अङ्कः_additional_fused_ops = 2;
                ctx->fused_rms_norm_बहुलकम् = RMS_NORM_MUL_ROPE;
                fusion_string = "RMS_NORM_MUL_ROPE";
                // rope is approximately elementwise - whole rows are done by a single workgroup and it's row-wise
                op_srcs_fused_elementwise[0] = असत्य;
                op_srcs_fused_elementwise[1] = सत्य;
                op_srcs_fused_elementwise[2] = सत्य;
            } अन्यथा यदि (ggml_can_fuse_subgraph(cgraph, i, hc_post_gate_आदर्शः, { i + 3 }) &&
                       ggml_check_धाराः(cgraph, i, hc_post_gate_धाराः) &&
                       ggml_vk_can_fuse_hc_post_gate(cgraph, i)) {
                ctx->अङ्कः_additional_fused_ops = hc_post_gate_आदर्शः.size() - 1;
                ctx->fused_hc_post_gate = सत्य;
                fusion_string = "HC_POST_GATE";
                std::fill_n(op_srcs_fused_elementwise, ctx->अङ्कः_additional_fused_ops + 1, असत्य);
            } अन्यथा यदि (ggml_vk_can_fuse(ctx, cgraph, i, rms_norm_mul_सङ्कलय_mul_आदर्शः)) {
                ctx->अङ्कः_additional_fused_ops = 3;
                ctx->fused_rms_norm_बहुलकम् = RMS_NORM_MUL_सङ्कलय_MUL;
                fusion_string = "RMS_NORM_MUL_ADD_MUL";
                std::fill_n(op_srcs_fused_elementwise, 4, सत्य);
            } अन्यथा यदि (ggml_vk_can_fuse(ctx, cgraph, i, rms_norm_mul_सङ्कलय_आदर्शः)) {
                ctx->अङ्कः_additional_fused_ops = 2;
                ctx->fused_rms_norm_बहुलकम् = RMS_NORM_MUL_सङ्कलय;
                fusion_string = "RMS_NORM_MUL_ADD";
                std::fill_n(op_srcs_fused_elementwise, 3, सत्य);
            } अन्यथा यदि (ggml_can_fuse_subgraph(cgraph, i, rms_norm_view_समुच्चयः_पङ्क्तिकाः_आदर्शः, { i + 2 }) &&
                       ggml_check_धाराः(cgraph, i, rms_norm_view_समुच्चयः_पङ्क्तिकाः_धाराः) &&
                       ggml_vk_can_fuse_rms_norm_समुच्चयः_पङ्क्तिकाः(ctx, cgraph, i)) {
                ctx->अङ्कः_additional_fused_ops = 2;
                ctx->fused_rms_norm_बहुलकम् = RMS_NORM_VIEW_समुच्चयः_पङ्क्तिकाः;
                fusion_string = "RMS_NORM_VIEW_SET_ROWS";
                std::fill_n(op_srcs_fused_elementwise, 3, असत्य);
            } अन्यथा यदि (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                ctx->अङ्कः_additional_fused_ops = 1;
                ctx->fused_rms_norm_बहुलकम् = RMS_NORM_MUL;
                fusion_string = "RMS_NORM_MUL";
                // rms_norm is not elementwise, but whole rows must be consumed and the scale factor computed before
                // they are overwritten, and one workgroup per row. So close enough.
                op_srcs_fused_elementwise[0] = सत्य;
                op_srcs_fused_elementwise[1] = सत्य;
            } अन्यथा यदि (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL })) {
                ctx->अङ्कः_additional_fused_ops = 1;
                विकल्प (ggml_get_unary_op(cgraph->nodes[i])) {
                    पक्ष GGML_UNARY_OP_GELU:     fusion_string = "GELU_MUL";     भङ्ग;
                    पक्ष GGML_UNARY_OP_SIGMOID:  fusion_string = "SIGMOID_MUL";  भङ्ग;
                    पक्ष GGML_UNARY_OP_SILU:     fusion_string = "SILU_MUL";     भङ्ग;
                    स्वभाव:                     fusion_string = "SOFTPLUS_MUL"; भङ्ग;
                }
                op_srcs_fused_elementwise[0] = सत्य;
                op_srcs_fused_elementwise[1] = सत्य;
            } अन्यथा यदि (ggml_vk_can_fuse_ssm_conv(ctx, cgraph, i, 2)) {
                ctx->अङ्कः_additional_fused_ops = 2;
                fusion_string = "SSM_CONV_BIAS_SILU";
                // ssm_conv reads multiple input tokens per output, so it's not elementwise w.r.t. its srcs.
                // The downstream add and silu are elementwise on the conv output.
                op_srcs_fused_elementwise[0] = असत्य;
                op_srcs_fused_elementwise[1] = सत्य;
                op_srcs_fused_elementwise[2] = सत्य;
            } अन्यथा यदि (ggml_vk_can_fuse_ssm_conv(ctx, cgraph, i, 1)) {
                ctx->अङ्कः_additional_fused_ops = 1;
                fusion_string = "SSM_CONV_SILU";
                op_srcs_fused_elementwise[0] = असत्य;
                op_srcs_fused_elementwise[1] = सत्य;
            } अन्यथा यदि (ggml_can_fuse_subgraph(cgraph, i, rope_view_समुच्चयः_पङ्क्तिकाः_आदर्शः, { i + 2 }) &&
                       ggml_check_धाराः(cgraph, i, rope_view_समुच्चयः_पङ्क्तिकाः_धाराः) &&
                       ggml_vk_can_fuse_rope_समुच्चयः_पङ्क्तिकाः(ctx, cgraph, i)) {
                ctx->अङ्कः_additional_fused_ops = 2;
                fusion_string = "ROPE_VIEW_SET_ROWS";
                op_srcs_fused_elementwise[0] = असत्य;
                op_srcs_fused_elementwise[1] = असत्य;
                op_srcs_fused_elementwise[2] = असत्य;
            } अन्यथा यदि (ggml_vk_can_fuse_snake(ctx, cgraph, i)) {
                ctx->अङ्कः_additional_fused_ops = 4;
                fusion_string = "SNAKE";
                // elementwise=true: snake.comp is safe under exact aliasing because each
                // thread reads data_x[idx] into a register before writing data_d[idx]
                // with a data dependency on that register. The overlap check still
                // rejects partial overlaps (different base or size).
                std::fill_n(op_srcs_fused_elementwise, 5, सत्य);
            } अन्यथा यदि (ggml_vk_can_fuse_topk_qsa(ctx, cgraph, i)) {
                ctx->अङ्कः_additional_fused_ops = topk_qsa_आदर्शः.size() - 1;
                ctx->fused_topk_qsa = सत्य;
                fusion_string = "TOPK_QSA";
                std::fill_n(op_srcs_fused_elementwise, ctx->अङ्कः_additional_fused_ops + 1, असत्य);
            } अन्यथा यदि (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax_norm, { i + 3, i + 9 }) &&
                       ggml_check_धाराः(cgraph, i, topk_moe_early_softmax_norm_धाराः) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX_NORM)) {
                ctx->अङ्कः_additional_fused_ops = topk_moe_early_softmax_norm.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_आवरणम् |= 1 << 3;
                ctx->fused_topk_moe_बहुलकम् = TOPK_MOE_EARLY_SOFTMAX_NORM;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX_NORM";
                std::fill_n(op_srcs_fused_elementwise, ctx->अङ्कः_additional_fused_ops + 1, असत्य);
            } अन्यथा यदि (ggml_can_fuse_subgraph(cgraph, i, topk_moe_sigmoid_norm_bias, { i + 4, i + 10 }) &&
                       ggml_check_धाराः(cgraph, i, topk_moe_sigmoid_norm_bias_धाराः) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_SIGMOID_NORM_BIAS)) {
                ctx->अङ्कः_additional_fused_ops = topk_moe_sigmoid_norm_bias.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_आवरणम् |= 1 << 4;
                ctx->fused_topk_moe_बहुलकम् = TOPK_MOE_SIGMOID_NORM_BIAS;
                fusion_string = "TOPK_MOE_SIGMOID_NORM_BIAS";
                std::fill_n(op_srcs_fused_elementwise, ctx->अङ्कः_additional_fused_ops + 1, असत्य);
            } अन्यथा यदि (ggml_can_fuse_subgraph(cgraph, i, topk_moe_sqrt_softplus_norm_bias, { i + 5, i + 11 }) &&
                       ggml_check_धाराः(cgraph, i, topk_moe_sqrt_softplus_norm_bias_धाराः) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS)) {
                ctx->अङ्कः_additional_fused_ops = topk_moe_sqrt_softplus_norm_bias.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_आवरणम् |= 1 << 5;
                ctx->fused_topk_moe_बहुलकम् = TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS;
                fusion_string = "TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS";
                std::fill_n(op_srcs_fused_elementwise, ctx->अङ्कः_additional_fused_ops + 1, असत्य);
            } अन्यथा यदि (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax, { i + 3, i + 4 }) &&
                       ggml_check_धाराः(cgraph, i, topk_moe_early_softmax_धाराः) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX)) {
                ctx->अङ्कः_additional_fused_ops = topk_moe_early_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_आवरणम् |= 1 << 3;
                ctx->fused_topk_moe_बहुलकम् = TOPK_MOE_EARLY_SOFTMAX;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX";
                std::fill_n(op_srcs_fused_elementwise, ctx->अङ्कः_additional_fused_ops + 1, असत्य);
            } अन्यथा यदि (ggml_can_fuse_subgraph(cgraph, i, topk_moe_late_softmax, { i + 1, i + 5 }) &&
                       ggml_check_धाराः(cgraph, i, topk_moe_late_softmax_धाराः) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_LATE_SOFTMAX)) {
                ctx->अङ्कः_additional_fused_ops = topk_moe_late_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_आवरणम् |= 1 << 1;
                ctx->fused_topk_moe_बहुलकम् = TOPK_MOE_LATE_SOFTMAX;
                fusion_string = "TOPK_MOE_LATE_SOFTMAX";
                std::fill_n(op_srcs_fused_elementwise, ctx->अङ्कः_additional_fused_ops + 1, असत्य);
            }
            यदि (ctx->fused_topk_moe_बहुलकम् != TOPK_MOE_गणनम्) {
                // Look for an additional scale op to fuse - occurs in deepseek2 and nemotron3 nano.
                यदि (ggml_can_fuse_subgraph(cgraph, i + ctx->अङ्कः_additional_fused_ops - 1, { GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE }, { i + ctx->अङ्कः_additional_fused_ops + 1 }) ||
                    ggml_can_fuse_subgraph(cgraph, i + ctx->अङ्कः_additional_fused_ops, { GGML_OP_GET_पङ्क्तिकाः, GGML_OP_SCALE }, { i + ctx->अङ्कः_additional_fused_ops + 1 })) {
                    ctx->fused_topk_moe_scale = सत्य;
                    ctx->अङ्कः_additional_fused_ops++;
                    op_srcs_fused_elementwise[ctx->अङ्कः_additional_fused_ops] = असत्य;
                }
            }
        }
        GGML_ASSERT(ctx->अङ्कः_additional_fused_ops < (पूर्णाङ्क)(परिमाण(op_srcs_fused_elementwise) / परिमाण(op_srcs_fused_elementwise[0])));
        ctx->fused_ops_write_आवरणम् |= 1 << ctx->अङ्कः_additional_fused_ops;

        // Check whether fusion would overwrite src operands while they're still in use.
        // If so, disable fusion.
        यदि (ctx->अङ्कः_additional_fused_ops) {
            // There are up to two output nodes - topk_moe has two.
            uint32_t द्वयङ्काः = ctx->fused_ops_write_आवरणम् & ~(1 << ctx->अङ्कः_additional_fused_ops);
            ggml_tensor *output_nodes[2] {};
            output_nodes[0] = cgraph->nodes[i + ctx->अङ्कः_additional_fused_ops];
            यदि (द्वयङ्काः) {
                पूर्णाङ्क output_idx = find_first_समुच्चयः(द्वयङ्काः);
                GGML_ASSERT(द्वयङ्काः == (1u << output_idx));
                output_nodes[1] = cgraph->nodes[i + output_idx];
            }

            तार्किक need_disable = असत्य;

            प्रति (पूर्णाङ्क j = 0; j < 2; ++j) {
                ggml_tensor *dst = output_nodes[j];
                यदि (!dst) {
                    अनुवर्त;
                }
                // Loop over all srcs of all nodes in the fusion. If the src overlaps
                // the destination and the src is not an intermediate node that's being
                // elided, then disable fusion.
                प्रति (पूर्णाङ्क k = 0; k <= ctx->अङ्कः_additional_fused_ops; ++k) {
                    प्रति (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
                        ggml_tensor *src = cgraph->nodes[i + k]->src[s];
                        यदि (!src || src->op == GGML_OP_NONE) {
                            अनुवर्त;
                        }
                        यदि (ggml_vk_tensors_overlap(src, dst, op_srcs_fused_elementwise[k])) {
                            तार्किक प्राप्तम् = असत्य;
                            प्रति (पूर्णाङ्क n = 0; n < k; ++n) {
                                यदि (cgraph->nodes[i + n] == src) {
                                    प्राप्तम् = सत्य;
                                    भङ्ग;
                                }
                            }
                            यदि (!प्राप्तम्) {
                                need_disable = सत्य;
                            }
                        }
                    }
                }
            }
            यदि (need_disable) {
                ctx->अङ्कः_additional_fused_ops = 0;
                ctx->fused_ops_write_आवरणम् = 1;
                ctx->fused_topk_moe_बहुलकम् = TOPK_MOE_गणनम्;
                ctx->fused_topk_moe_scale = असत्य;
                ctx->fused_topk_qsa = असत्य;
                ctx->fused_hc_post_gate = असत्य;
                ctx->fused_rms_norm_बहुलकम् = RMS_NORM_गणनम्;
                fusion_string = शून्यसूचकः;
            }
        }

        // Signal the almost_ready fence when the graph is mostly complete (< 20% remaining)
        तार्किक almost_ready = (cgraph->n_nodes - i) < cgraph->n_nodes / 5;
        तार्किक submit = (submitted_nodes >= ctx->device->max_nodes_प्रत्येकम्_submit) ||
                      (flops_प्रत्येकम्_submit != 0 && batch_flops >= flops_प्रत्येकम्_submit) ||
                      (i + ctx->अङ्कः_additional_fused_ops >= चरमः_ग्रन्थिः) ||
                      (almost_ready && !ctx->almost_ready_fence_pending);

        तार्किक enqueued = ggml_vk_build_graph(ctx, cgraph, i, cgraph->nodes[submit_ग्रन्थिः_idx], submit_ग्रन्थिः_idx, i + ctx->अङ्कः_additional_fused_ops >= चरमः_ग्रन्थिः, almost_ready, submit);

        यदि (vk_perf_logger_enabled && enqueued) {
            compute_ctx = ggml_vk_get_compute_ctx(ctx);
            यदि (!vk_perf_logger_concurrent) {
                // track a single node/fusion for the current query
                ctx->query_nodes[ctx->query_idx] = cgraph->nodes[i];
                ctx->query_fusion_नामानि[ctx->query_idx] = fusion_string;
                compute_ctx->s->बफर->buf.writeTimestamp(vk::Pipeline_Stage_Flag_द्वयङ्काः::eAllCommands, ctx->query_pool, ctx->query_idx++);
                ggml_vk_sync_buffers(ctx, compute_ctx);
            } अन्यथा {
                // track a fusion string and number of fused ops for the current node_idx
                ctx->query_fusion_नामानि[i] = fusion_string;
                ctx->query_fusion_ग्रन्थिः_गणनम्[i] = ctx->अङ्कः_additional_fused_ops;
            }
        }

        यदि (enqueued) {
            ++submitted_nodes;

#यदि_अपरिभाषितम् GGML_VULKAN_CHECK_RESULTS
            यदि (first_ग्रन्थिः_अन्तर्_batch) {
                first_ग्रन्थिः_अन्तर्_batch = असत्य;
            }
#यदि_अन्तः
        }

        यदि (submit && enqueued) {
            submit_after(submit_ग्रन्थिः_idx, i + (पूर्णाङ्क)ctx->अङ्कः_additional_fused_ops);
        }
        i += ctx->अङ्कः_additional_fused_ops;
        ctx->अङ्कः_additional_fused_ops = 0;
        ctx->fused_ops_write_आवरणम् = 0;
    }

    ctx->चरमः_योग_flops = योग_flops;

    यदि (vk_perf_logger_enabled) {
        // End the command buffer and submit/wait
        GGML_ASSERT(!ctx->compute_ctx.expired());
        compute_ctx = ctx->compute_ctx.तालकम्();
        ggml_vk_ctx_अन्तः(compute_ctx);

        ggml_vk_submit(compute_ctx, ctx->device->fence);
        VK_CHECK(ctx->device->device.waitForFences({ ctx->device->fence }, सत्य, UINT64_MAX), "GGML_VULKAN_PERF waitForFences", ctx->device);
        ctx->device->device.resetFences({ ctx->device->fence });
        ctx->compute_ctx.reset();

        // Get the results and pass them to the logger
        std::vector<uint64_t> timestamps(cgraph->n_nodes + 1);
        VK_CHECK(ctx->device->device.getQueryPoolResults(ctx->query_pool, 0, ctx->query_idx, (cgraph->n_nodes + 1)*परिमाण(uint64_t), timestamps.दत्त(), परिमाण(uint64_t), vk::Query_परिणाम_Flag_द्वयङ्काः::e64 | vk::Query_परिणाम_Flag_द्वयङ्काः::eWait), "get timestamp results", ctx->device);
        यदि (!vk_perf_logger_concurrent) {
            // Log each op separately
            प्रति (पूर्णाङ्क i = 1; i < ctx->query_idx; i++) {
                स्वचालित ग्रन्थिः = ctx->query_nodes[i];
                स्वचालित name = ctx->query_fusion_नामानि[i];
                ctx->perf_logger->log_timing(ग्रन्थिः, name, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod));
            }
        } अन्यथा {
            // Log each group of nodes
            पूर्णाङ्क पूर्व_ग्रन्थिः_idx = 0;
            प्रति (पूर्णाङ्क i = 1; i < ctx->query_idx; i++) {
                स्वचालित cur_ग्रन्थिः_idx = ctx->query_ग्रन्थिः_idx[i];
                std::vector<ggml_tensor *> nodes;
                std::vector<ध्रुव वर्ण *> नामानि;
                प्रति (पूर्णाङ्क ग्रन्थिः_idx = पूर्व_ग्रन्थिः_idx; ग्रन्थिः_idx < cur_ग्रन्थिः_idx; ++ग्रन्थिः_idx) {
                    यदि (ggml_op_is_रिक्तानि(cgraph->nodes[ग्रन्थिः_idx]->op)) {
                        अनुवर्त;
                    }
                    nodes.push_back(cgraph->nodes[ग्रन्थिः_idx]);
                    नामानि.push_back(ctx->query_fusion_नामानि[ग्रन्थिः_idx]);
                    ग्रन्थिः_idx += ctx->query_fusion_ग्रन्थिः_गणनम्[ग्रन्थिः_idx];
                }
                पूर्व_ग्रन्थिः_idx = cur_ग्रन्थिः_idx;
                ctx->perf_logger->log_timing(nodes, नामानि, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod));
            }
        }
        ctx->perf_logger->print_timings();
    }

    यदि (!ctx->device->support_async) {
        ggml_vk_synchronize(ctx);
    }

    प्रत्यागम GGML_स्थितिः_SUCCESS;

    अप्रयुक्तम्(backend);
}

शून्य ggml_vk_graph_optimize(ggml_backend_t backend, संरचना ggml_cgraph * graph, संरचना ggml_backend_graph_optimize_params * params)
{
    VK_LOG_DEBUG("ggml_vk_graph_optimize(" << graph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    यदि (ctx->device->disable_graph_optimize) {
        प्रत्यागम;
    }

    स्वचालित ध्रुव &is_रिक्तानि = [](ध्रुव ggml_tensor * ग्रन्थिः) -> तार्किक {
        प्रत्यागम ग्रन्थिः->op == GGML_OP_NONE || ग्रन्थिः->op == GGML_OP_RESHAPE || ग्रन्थिः->op == GGML_OP_TRANSPOSE || ग्रन्थिः->op == GGML_OP_VIEW || ग्रन्थिः->op == GGML_OP_PERMUTE;
    };

    स्वचालित ध्रुव &is_src_सम्बन्धि = [&is_रिक्तानि](ध्रुव ggml_tensor *dst, ध्रुव ggml_tensor *src) -> तार्किक {
        स्वचालित ध्रुव &आधारः = [](ध्रुव ggml_tensor * tensor) {
            प्रत्यागम tensor->view_src ? tensor->view_src : tensor;
        };
        प्रति (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            यदि (dst->src[s] == src) {
                प्रत्यागम सत्य;
            }
            यदि (is_रिक्तानि(dst) || is_रिक्तानि(src)) {
                अनुवर्त;
            }
            // A source view of dst may read storage written through a different view by src.
            यदि (dst->src[s] && आधारः(dst->src[s]) == आधारः(src)) {
                प्रत्यागम सत्य;
            }
            // Moving dst forward may overwrite storage still read through a view by src.
            यदि (src->src[s] && आधारः(dst) == आधारः(src->src[s])) {
                प्रत्यागम सत्य;
            }
        }
        // implicit dependency if they view the same tensor
        यदि (आधारः(dst) == आधारः(src)) {
            प्रत्यागम सत्य;
        }
        प्रत्यागम असत्य;
    };

    std::vector<ggml_tensor *> new_क्रमः;
    std::vector<तार्किक> प्रयुक्तम्(graph->n_nodes, असत्य);
    std::set<ggml_tensor *> प्रयुक्तम्_ग्रन्थिः_समुच्चयः;

    पूर्णाङ्क first_अप्रयुक्तम् = 0;

    // scheduled or zero-compute nodes in [lo, hi)
    स्वचालित ध्रुव &रिक्तानि_or_scheduled_between = [&](पूर्णाङ्क lo, पूर्णाङ्क hi) -> तार्किक {
        प्रति (पूर्णाङ्क v = lo; v < hi; ++v) {
            यदि (!प्रयुक्तम्[v] && !is_रिक्तानि(graph->nodes[v])) {
                प्रत्यागम असत्य;
            }
        }
        प्रत्यागम सत्य;
    };
    यावत् (first_अप्रयुक्तम् < graph->n_nodes) {
        std::vector<पूर्णाङ्क> current_समुच्चयः;

        // Check for fusion patterns and avoid reordering them
        स्वचालित ध्रुव &मेलनम्_आदर्शः = [&](ध्रुव std::initializer_list<ggml_op> &pattern, पूर्णाङ्क start) -> तार्किक {
            यदि (start + (पूर्णाङ्क)pattern.size() <= graph->n_nodes) {
                तार्किक is_आदर्शः = सत्य;
                प्रति (size_t j = 0; j < pattern.size(); ++j) {
                    यदि (graph->nodes[start + j]->op != pattern.begin()[j] || प्रयुक्तम्[start + j]) {
                        is_आदर्शः = असत्य;
                    }
                }
                प्रत्यागम is_आदर्शः;
            }
            प्रत्यागम असत्य;
        };

        स्वचालित ध्रुव &keep_आदर्शः = [&](ध्रुव std::initializer_list<ggml_op> &pattern) -> तार्किक {
            यदि (मेलनम्_आदर्शः(pattern, first_अप्रयुक्तम्)) {
                प्रति (size_t j = 0; j < pattern.size(); ++j) {
                    new_क्रमः.push_back(graph->nodes[first_अप्रयुक्तम् + j]);
                    प्रयुक्तम्_ग्रन्थिः_समुच्चयः.insert(graph->nodes[first_अप्रयुक्तम् + j]);
                    प्रयुक्तम्[first_अप्रयुक्तम् + j] = सत्य;
                }
                यावत् (first_अप्रयुक्तम् < graph->n_nodes && प्रयुक्तम्[first_अप्रयुक्तम्]) {
                    first_अप्रयुक्तम्++;
                }
                प्रत्यागम सत्य;
            }
            प्रत्यागम असत्य;
        };

        स्वचालित ध्रुव &सङ्कलय_आदर्शः_alloc_deps = [&](ध्रुव std::initializer_list<ggml_op> &pattern, पूर्णाङ्क चरमः_ग्रन्थिः) {
            // Keep external inputs alive through the fused output.
            std::set<ggml_tensor *> आलोकितम्;
            प्रति (size_t j = 0; j < pattern.size(); ++j) {
                ggml_tensor * ग्रन्थिः = graph->nodes[first_अप्रयुक्तम् + j];
                प्रति (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
                    ggml_tensor * src = ग्रन्थिः->src[s];
                    यदि (src && आलोकितम्.insert(src).second) {
                        params->सङ्कलय_alloc_dep(params->user_दत्त, src, graph->nodes[चरमः_ग्रन्थिः]);
                    }
                }
                आलोकितम्.insert(ग्रन्थिः);
            }
        };

        स्वचालित ध्रुव &keep_topk_moe_आदर्शः = [&](ध्रुव std::initializer_list<ggml_op> &pattern) -> तार्किक {
            यदि (!मेलनम्_आदर्शः(pattern, first_अप्रयुक्तम्)) {
                प्रत्यागम असत्य;
            }

            पूर्णाङ्क चरमः_ग्रन्थिः = first_अप्रयुक्तम् + (पूर्णाङ्क) pattern.size() - 1;
            // Some TOPK_MOE variants fuse a trailing scale.
            यदि (चरमः_ग्रन्थिः + 1 < graph->n_nodes && graph->nodes[चरमः_ग्रन्थिः + 1]->op == GGML_OP_SCALE) {
                चरमः_ग्रन्थिः++;
            }

            सङ्कलय_आदर्शः_alloc_deps(pattern, चरमः_ग्रन्थिः);

            प्रत्यागम keep_आदर्शः(pattern);
        };

        यदि (keep_topk_moe_आदर्शः(topk_moe_early_softmax_norm)) {
            अनुवर्त;
        }
        यदि (keep_topk_moe_आदर्शः(topk_moe_sigmoid_norm_bias)) {
            अनुवर्त;
        }
        यदि (keep_topk_moe_आदर्शः(topk_moe_sqrt_softplus_norm_bias)) {
            अनुवर्त;
        }
        यदि (keep_topk_moe_आदर्शः(topk_moe_early_softmax)) {
            अनुवर्त;
        }
        यदि (keep_topk_moe_आदर्शः(topk_moe_late_softmax)) {
            अनुवर्त;
        }
        यदि (keep_आदर्शः(snake_आदर्शः)) {
            अनुवर्त;
        }
        यदि (keep_आदर्शः(topk_qsa_आदर्शः)) {
            अनुवर्त;
        }

        यदि (keep_आदर्शः(rms_norm_mul_सङ्कलय_mul_आदर्शः)) {
            अनुवर्त;
        }
        यदि (keep_आदर्शः(rms_norm_mul_सङ्कलय_आदर्शः)) {
            अनुवर्त;
        }
        यदि (keep_आदर्शः(rms_norm_mul_rope_view_समुच्चयः_पङ्क्तिकाः_आदर्शः)) {
            अनुवर्त;
        }
        यदि (keep_आदर्शः(rms_norm_view_समुच्चयः_पङ्क्तिकाः_आदर्शः)) {
            अनुवर्त;
        }
        यदि (keep_आदर्शः(rope_view_समुच्चयः_पङ्क्तिकाः_आदर्शः)) {
            अनुवर्त;
        }
        यदि (मेलनम्_आदर्शः(hc_post_gate_आदर्शः, first_अप्रयुक्तम्)) {
            सङ्कलय_आदर्शः_alloc_deps(hc_post_gate_आदर्शः, first_अप्रयुक्तम् + (पूर्णाङ्क) hc_post_gate_आदर्शः.size() - 1);
            keep_आदर्शः(hc_post_gate_आदर्शः);
            अनुवर्त;
        }

        // First, grab the next unused node.
        current_समुच्चयः.push_back(first_अप्रयुक्तम्);

        // Loop through the next N nodes. Grab any that don't depend on other nodes that
        // haven't already been run. Nodes that have already been run have used[i] set
        // to true. Allow nodes that depend on the previous node if it's a fusion pattern
        // that we support (e.g. RMS_NORM + MUL).
        // This first pass only grabs "real" (non-view nodes). Second pass grabs view nodes.
        // The goal is to not interleave real and view nodes in a way that breaks fusion.
        ध्रुव पूर्णाङ्क अङ्कः_गन्तव्यम्_CHECK = 20;
        प्रति (पूर्णाङ्क j = first_अप्रयुक्तम्+1; j < std::min(first_अप्रयुक्तम् + अङ्कः_गन्तव्यम्_CHECK, graph->n_nodes); ++j) {
            यदि (प्रयुक्तम्[j]) {
                अनुवर्त;
            }
            यदि (is_रिक्तानि(graph->nodes[j])) {
                अनुवर्त;
            }
            // Protect every interior QSA node (not just the start): the mask branch is
            // independent, so it gets pulled out and breaks keep_pattern otherwise.
            स्वचालित ध्रुव &अन्तर्_qsa_आदर्शः = [&](पूर्णाङ्क n) -> तार्किक {
                प्रति (पूर्णाङ्क o = 0; o < (पूर्णाङ्क) topk_qsa_आदर्शः.size(); ++o) {
                    यदि (n - o >= 0 && मेलनम्_आदर्शः(topk_qsa_आदर्शः, n - o)) {
                        प्रत्यागम सत्य;
                    }
                }
                प्रत्यागम असत्य;
            };
            यदि (मेलनम्_आदर्शः(topk_moe_early_softmax_norm, j) ||
                मेलनम्_आदर्शः(topk_moe_sigmoid_norm_bias, j) ||
                मेलनम्_आदर्शः(topk_moe_sqrt_softplus_norm_bias, j) ||
                मेलनम्_आदर्शः(topk_moe_early_softmax, j) ||
                मेलनम्_आदर्शः(topk_moe_late_softmax, j) ||
                मेलनम्_आदर्शः(snake_आदर्शः, j) ||
                अन्तर्_qsa_आदर्शः(j) ||
                मेलनम्_आदर्शः(rms_norm_mul_सङ्कलय_mul_आदर्शः, j) ||
                मेलनम्_आदर्शः(rms_norm_mul_सङ्कलय_आदर्शः, j) ||
                मेलनम्_आदर्शः(rms_norm_mul_rope_view_समुच्चयः_पङ्क्तिकाः_आदर्शः, j) ||
                मेलनम्_आदर्शः(rms_norm_view_समुच्चयः_पङ्क्तिकाः_आदर्शः, j) ||
                मेलनम्_आदर्शः(rope_view_समुच्चयः_पङ्क्तिकाः_आदर्शः, j) ||
                मेलनम्_आदर्शः(hc_post_gate_आदर्शः, j)) {
                अनुवर्त;
            }
            तार्किक ok = सत्य;
            प्रति (पूर्णाङ्क c = first_अप्रयुक्तम्; c < j; ++c) {
                यदि (!प्रयुक्तम्[c] &&
                    is_src_सम्बन्धि(graph->nodes[j], graph->nodes[c]) &&
                    !(c == current_समुच्चयः.back() && graph->nodes[c]->op == GGML_OP_RMS_NORM && graph->nodes[j]->op == GGML_OP_MUL && रिक्तानि_or_scheduled_between(c+1, j)) &&
                    !(c == current_समुच्चयः.back() && graph->nodes[c]->op == GGML_OP_UNARY && graph->nodes[j]->op == GGML_OP_MUL && रिक्तानि_or_scheduled_between(c+1, j)) &&
                    !(j == c+1 && c == current_समुच्चयः.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT && graph->nodes[j]->op == GGML_OP_सङ्कलय) &&
                    !(j == c+1 && c == current_समुच्चयः.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_सङ्कलय_ID) &&
                    !(j == c+1 && c == current_समुच्चयः.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_MUL) &&
                    !(j == c+1 && c == current_समुच्चयः.back() && graph->nodes[c]->op == GGML_OP_सङ्कलय && graph->nodes[j]->op == GGML_OP_सङ्कलय) &&
                    !(j == c+1 && c == current_समुच्चयः.back() && graph->nodes[c]->op == GGML_OP_SSM_CONV && graph->nodes[j]->op == GGML_OP_सङ्कलय) &&
                    !(j == c+1 && c == current_समुच्चयः.back() && graph->nodes[c]->op == GGML_OP_SSM_CONV && graph->nodes[j]->op == GGML_OP_UNARY)) {
                    ok = असत्य;
                    भङ्ग;
                }
            }
            यदि (ok) {
                current_समुच्चयः.push_back(j);

                पूर्णाङ्क rope_idx = j;

                // When we've found RMS_NORM + MUL, try to find a ROPE that uses it
                यदि (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_MUL &&
                    graph->nodes[j-1]->op == GGML_OP_RMS_NORM) {
                    प्रति (पूर्णाङ्क k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        यदि (graph->nodes[k]->op == GGML_OP_ROPE &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // Check that other srcs are already valid
                            graph->nodes[k]->src[1]->op == GGML_OP_NONE &&
                            (graph->nodes[k]->src[2] == शून्यसूचकः || graph->nodes[k]->src[2]->op == GGML_OP_NONE)) {
                            rope_idx = k;
                            current_समुच्चयः.push_back(rope_idx);
                            प्रयुक्तम्[rope_idx] = सत्य;
                            भङ्ग;
                        }
                    }
                }
                // Look for ROPE/RMS_NORM + VIEW + SET_ROWS and make them consecutive
                यदि (graph->nodes[rope_idx]->op == GGML_OP_ROPE || graph->nodes[rope_idx]->op == GGML_OP_RMS_NORM) {
                    पूर्णाङ्क view_idx = -1;
                    पूर्णाङ्क समुच्चयः_पङ्क्तिकाः_idx = -1;
                    प्रति (पूर्णाङ्क k = rope_idx + 1; k < std::min(rope_idx + 15, graph->n_nodes); ++k) {
                        यदि (प्रयुक्तम्[k]) {
                            अनुवर्त;
                        }
                        यदि (view_idx == -1 && graph->nodes[k]->op == GGML_OP_VIEW && graph->nodes[k]->src[0] == graph->nodes[rope_idx]) {
                            view_idx = k;
                            अनुवर्त;
                        }
                        यदि (view_idx != -1 && graph->nodes[k]->op == GGML_OP_समुच्चयः_पङ्क्तिकाः && graph->nodes[k]->src[0] == graph->nodes[view_idx]) {
                            समुच्चयः_पङ्क्तिकाः_idx = k;
                            भङ्ग;
                        }
                    }
                    यदि (समुच्चयः_पङ्क्तिकाः_idx != -1) {
                        ध्रुव पूर्णाङ्क ग्रन्थिः_idxs[] = { rope_idx, view_idx, समुच्चयः_पङ्क्तिकाः_idx };
                        ध्रुव ggml_op ops[] = { graph->nodes[rope_idx]->op, GGML_OP_VIEW, GGML_OP_समुच्चयः_पङ्क्तिकाः };
                        तार्किक can_pull = ggml_can_fuse_subgraph_ext(graph, ग्रन्थिः_idxs, 3, ops, &समुच्चयः_पङ्क्तिकाः_idx, 1);

                        प्रति (पूर्णाङ्क c = rope_idx + 1; can_pull && c < समुच्चयः_पङ्क्तिकाः_idx; ++c) {
                            यदि (!प्रयुक्तम्[c] && c != view_idx && !is_रिक्तानि(graph->nodes[c]) &&
                                is_src_सम्बन्धि(graph->nodes[समुच्चयः_पङ्क्तिकाः_idx], graph->nodes[c])) {
                                can_pull = असत्य;
                            }
                        }

                        यदि (can_pull) {
                            current_समुच्चयः.push_back(view_idx);
                            current_समुच्चयः.push_back(समुच्चयः_पङ्क्तिकाः_idx);
                            प्रयुक्तम्[view_idx] = सत्य;
                            प्रयुक्तम्[समुच्चयः_पङ्क्तिकाः_idx] = सत्य;
                        }
                    }
                }
                // Look for MUL_MAT_ID + ADD_ID + MUL
                यदि (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_सङ्कलय_ID &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT_ID) {
                    प्रति (पूर्णाङ्क k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        यदि (graph->nodes[k]->op == GGML_OP_MUL &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || प्रयुक्तम्_ग्रन्थिः_समुच्चयः.find(graph->nodes[k]->src[1]) != प्रयुक्तम्_ग्रन्थिः_समुच्चयः.end())) {
                            current_समुच्चयः.push_back(k);
                            प्रयुक्तम्[k] = सत्य;
                            भङ्ग;
                        }
                    }
                }
                // Look for MUL_MAT + ADD + ADD
                यदि (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_सङ्कलय &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT) {
                    प्रति (पूर्णाङ्क k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        यदि (graph->nodes[k]->op == GGML_OP_सङ्कलय &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || प्रयुक्तम्_ग्रन्थिः_समुच्चयः.find(graph->nodes[k]->src[1]) != प्रयुक्तम्_ग्रन्थिः_समुच्चयः.end())) {
                            current_समुच्चयः.push_back(k);
                            प्रयुक्तम्[k] = सत्य;
                            भङ्ग;
                        }
                    }
                }
                // SSM_CONV + ADD + UNARY: pull the consuming UNARY forward
                यदि (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_सङ्कलय &&
                    graph->nodes[j-1]->op == GGML_OP_SSM_CONV) {
                    प्रति (पूर्णाङ्क k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        यदि (graph->nodes[k]->op == GGML_OP_UNARY &&
                            graph->nodes[k]->src[0] == graph->nodes[j]) {
                            current_समुच्चयः.push_back(k);
                            प्रयुक्तम्[k] = सत्य;
                            भङ्ग;
                        }
                    }
                }
                // UNARY + MUL: pull the consuming MUL forward
                यदि (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_UNARY) {
                    प्रति (पूर्णाङ्क k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ggml_tensor * mul = graph->nodes[k];
                        यदि (mul->op != GGML_OP_MUL || (mul->src[0] != graph->nodes[j] && mul->src[1] != graph->nodes[j])) {
                            अनुवर्त;
                        }
                        ggml_tensor * other = (mul->src[0] == graph->nodes[j]) ? mul->src[1] : mul->src[0];
                        // the other src must either be weights or already processed
                        यदि (!(other->op == GGML_OP_NONE || प्रयुक्तम्_ग्रन्थिः_समुच्चयः.find(other) != प्रयुक्तम्_ग्रन्थिः_समुच्चयः.end())) {
                            अनुवर्त;
                        }
                        यदि (!ggml_vk_can_fuse_unary_mul(graph, j, k)) {
                            अनुवर्त;
                        }
                        current_समुच्चयः.push_back(k);
                        प्रयुक्तम्[k] = सत्य;
                        भङ्ग;
                    }
                }
            }
        }
        // Second pass grabs view nodes.
        // Skip this if it would break a fusion optimization (don't split up add->rms_norm or add->add).
        यदि (graph->nodes[current_समुच्चयः.back()]->op != GGML_OP_सङ्कलय) {
            प्रति (पूर्णाङ्क j = first_अप्रयुक्तम्+1; j < std::min(first_अप्रयुक्तम् + अङ्कः_गन्तव्यम्_CHECK, graph->n_nodes); ++j) {
                यदि (प्रयुक्तम्[j]) {
                    अनुवर्त;
                }
                यदि (!is_रिक्तानि(graph->nodes[j])) {
                    अनुवर्त;
                }
                तार्किक ok = सत्य;
                प्रति (पूर्णाङ्क c = first_अप्रयुक्तम्; c < j; ++c) {
                    तार्किक c_अन्तर्_current_समुच्चयः = std::find(current_समुच्चयः.begin(), current_समुच्चयः.end(), c) != current_समुच्चयः.end();
                    // skip views whose srcs haven't been processed.
                    यदि (!प्रयुक्तम्[c] &&
                        is_src_सम्बन्धि(graph->nodes[j], graph->nodes[c]) &&
                        !c_अन्तर्_current_समुच्चयः) {
                        ok = असत्य;
                        भङ्ग;
                    }
                }
                यदि (ok) {
                    current_समुच्चयः.push_back(j);
                }
            }
        }

        // Push the current set into new_order
        प्रति (स्वचालित c : current_समुच्चयः) {
            new_क्रमः.push_back(graph->nodes[c]);
            प्रयुक्तम्_ग्रन्थिः_समुच्चयः.insert(graph->nodes[c]);
            प्रयुक्तम्[c] = सत्य;
        }
        यावत् (first_अप्रयुक्तम् < graph->n_nodes && प्रयुक्तम्[first_अप्रयुक्तम्]) {
            first_अप्रयुक्तम्++;
        }
    }
    // Replace the graph with the new order.
    प्रति (पूर्णाङ्क i = 0; i < graph->n_nodes; ++i) {
        graph->nodes[i] = new_क्रमः[i];
    }
}

स्थिर शून्य ggml_backend_vk_event_लेखः(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_record(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    ggml_vk_submit_transfer_ctx(ctx);

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);
    स्वचालित* cmd_buf = compute_ctx->s->बफर; // retrieve pointer before it gets reset

    यदि (vkev->has_event) {
        // Move existing event into submitted
        vkev->events_submitted.push_back(vkev->event);
    }

    // Grab the next event and record it, create one if necessary
    यदि (vkev->events_free.रिक्तानि()) {
        vkev->event = ctx->device->device.createEvent({});
    } अन्यथा {
        vkev->event = vkev->events_free.back();
        vkev->events_free.pop_back();
    }

    vkev->has_event = सत्य;

    ggml_vk_समुच्चयः_event(compute_ctx, vkev->event);

    vkev->tl_semaphore.value++;
    compute_ctx->s->signal_semaphores.push_back(vkev->tl_semaphore);
    ggml_vk_ctx_अन्तः(compute_ctx);

    ggml_vk_submit(compute_ctx, {});
    ctx->submit_pending = सत्य;
    vkev->cmd_बफर = cmd_buf;
    vkev->cmd_बफर_use_गणक = cmd_buf->use_गणक;
    ctx->compute_ctx.reset();
}

स्थिर शून्य ggml_backend_vk_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_wait(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    यदि (vkev->has_event) {
        // Wait for latest event
        ggml_vk_wait_events(compute_ctx, { vkev->event });

        यदि (ctx->device->async_use_transfer_queue) {
            vk_context transfer_ctx = ggml_vk_get_transfer_ctx(ctx);
            transfer_ctx->s->wait_semaphores.push_back(vkev->tl_semaphore);
        }
    }
}

स्थिर ggml_backend_i ggml_backend_vk_interface = {
    /* .get_name                = */ ggml_backend_vk_name,
    /* .free                    = */ ggml_backend_vk_free,
    /* .set_tensor_async        = */ ggml_backend_vk_समुच्चयः_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_vk_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_vk_समुच्चयः_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_vk_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_vk_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_vk_synchronize,
    /* .graph_plan_create       = */ रिक्त,
    /* .graph_plan_free         = */ रिक्त,
    /* .graph_plan_update       = */ रिक्त,
    /* .graph_plan_compute      = */ रिक्त,
    /* .graph_compute           = */ ggml_backend_vk_graph_compute,
    /* .event_record            = */ ggml_backend_vk_event_लेखः,
    /* .event_wait              = */ ggml_backend_vk_event_wait,
    /* .graph_optimize          = */ ggml_vk_graph_optimize,
};

स्थिर ggml_guid_t ggml_backend_vk_guid() {
    स्थिर ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x40, 0x3c, 0xe1, 0x02, 0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };
    प्रत्यागम &guid;
}

ggml_backend_t ggml_backend_vk_init(size_t dev_अङ्कः) {
    VK_LOG_DEBUG("ggml_backend_vk_init(" << dev_अङ्कः << ")");

    ggml_backend_vk_context * ctx = नूतन ggml_backend_vk_context;
    ggml_vk_init(ctx, dev_अङ्कः);

    ggml_backend_t vk_backend = नूतन ggml_backend {
        /* .guid    = */ ggml_backend_vk_guid(),
        /* .iface   = */ ggml_backend_vk_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), dev_अङ्कः),
        /* .context = */ ctx,
    };

    यदि (!ctx->device->support_async) {
        vk_backend->iface.get_tensor_async = शून्यसूचकः;
    }

    प्रत्यागम vk_backend;
}

तार्किक ggml_backend_is_vk(ggml_backend_t backend) {
    प्रत्यागम backend != रिक्त && ggml_guid_matches(backend->guid, ggml_backend_vk_guid());
}

पूर्णाङ्क ggml_backend_vk_get_device_गणनम्() {
    प्रत्यागम ggml_vk_get_device_गणनम्();
}

शून्य ggml_backend_vk_get_device_description(पूर्णाङ्क device, वर्ण * description, size_t description_आकार) {
    GGML_ASSERT(device < (पूर्णाङ्क) vk_instance.device_indices.size());
    पूर्णाङ्क dev_idx = vk_instance.device_indices[device];
    ggml_vk_get_device_description(dev_idx, description, description_आकार);
}

शून्य ggml_backend_vk_get_device_स्मृतिः(पूर्णाङ्क device, size_t * मुक्त, size_t * योग) {
    GGML_ASSERT(device < (पूर्णाङ्क) vk_instance.device_indices.size());
    GGML_ASSERT(device < (पूर्णाङ्क) vk_instance.device_supports_membudget.size());

    vk::PhysicalDevice vkdev = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device]];
    vk::Physical_Device_स्मृतिः_Budget_Properties_EXT budgetprops;
    vk::Physical_Device_स्मृतिः_Properties2 memprops = {};
    ध्रुव तार्किक membudget_supported = vk_instance.device_supports_membudget[device];
    ध्रुव तार्किक is_integrated_gpu = vkdev.getProperties().deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

    यदि (membudget_supported) {
        memprops.p_अग्र = &budgetprops;
    }
    vkdev.get_स्मृतिः_Properties2(&memprops);

    *योग = 0;
    *मुक्त = 0;

    प्रति (uint32_t i = 0; i < memprops.स्मृतिः_Properties.स्मृतिः_Heap_गणनम्; ++i) {
        ध्रुव vk::स्मृतिः_Heap & heap = memprops.स्मृतिः_Properties.स्मृतिः_Heaps[i];

        यदि (is_integrated_gpu || (heap.flags & vk::स्मृतिः_Heap_Flag_द्वयङ्काः::eDeviceLocal)) {
            *योग += heap.size;

            यदि (membudget_supported && i < budgetprops.heapUsage.size()) {
                // Driver may report heapBudget > heap.size so we add a safeguard here
                // See https://github.com/ggml-org/llama.cpp/issues/29277
                ध्रुव vk::Device_आकार budget = std::min(budgetprops.heapBudget[i], heap.size);
                *मुक्त += budget > budgetprops.heapUsage[i] ? budget - budgetprops.heapUsage[i] : 0;
            } अन्यथा {
                *मुक्त += heap.size;
            }
        }
    }
}

स्थिर vk::PhysicalDeviceType ggml_backend_vk_get_device_type(पूर्णाङ्क device_idx) {
    GGML_ASSERT(device_idx >= 0 && device_idx < (पूर्णाङ्क) vk_instance.device_indices.size());

    vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

    vk::PhysicalDeviceProperties2 props = {};
    device.getProperties2(&props);

    प्रत्यागम props.properties.deviceType;
}

स्थिर std::string ggml_backend_vk_get_device_pci_id(पूर्णाङ्क device_idx) {
    GGML_ASSERT(device_idx >= 0 && device_idx < (पूर्णाङ्क) vk_instance.device_indices.size());

    vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

    ध्रुव std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

    तार्किक ext_support = असत्य;

    प्रति (ध्रुव स्वचालित& properties : ext_props) {
        यदि (वाक्यं_तुलय("VK_EXT_pci_bus_info", properties.extensionName) == 0) {
            ext_support = सत्य;
            भङ्ग;
        }
    }

    यदि (!ext_support) {
        प्रत्यागम "";
    }

    vk::PhysicalDeviceProperties2 props = {};
    vk::PhysicalDevicePCIBusInfoPropertiesEXT pci_bus_info = {};

    props.p_अग्र = &pci_bus_info;

    device.getProperties2(&props);

    ध्रुव uint32_t pci_domain = pci_bus_info.pciDomain;
    ध्रुव uint32_t pci_bus = pci_bus_info.pciBus;
    ध्रुव uint32_t pci_device = pci_bus_info.pciDevice;
    ध्रुव uint8_t pci_function = (uint8_t) pci_bus_info.pciFunction; // pci function is between 0 and 7, prevent printf overflow warning

    वर्ण pci_bus_id[16] = {};
    snprintf(pci_bus_id, परिमाण(pci_bus_id), "%04x:%02x:%02x.%x", pci_domain, pci_bus, pci_device, pci_function);

    प्रत्यागम std::string(pci_bus_id);
}

स्थिर ध्रुव वर्ण * ggml_backend_vk_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    प्रत्यागम ctx->name.c_str();
}

स्थिर ध्रुव वर्ण * ggml_backend_vk_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    प्रत्यागम ctx->description.c_str();
}

स्थिर शून्य ggml_backend_vk_device_get_स्मृतिः(ggml_backend_dev_t device, size_t * मुक्त, size_t * योग) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)device->context;
    ggml_backend_vk_get_device_स्मृतिः(ctx->device, मुक्त, योग);
}

स्थिर ggml_backend_बफर_type_t ggml_backend_vk_device_get_बफर_type(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    प्रत्यागम ggml_backend_vk_बफर_type(ctx->device);
}

स्थिर ggml_backend_बफर_type_t ggml_backend_vk_device_get_host_बफर_type(ggml_backend_dev_t dev) {
    अप्रयुक्तम्(dev);
    प्रत्यागम ggml_backend_vk_host_बफर_type();
}

स्थिर गणना ggml_backend_dev_type ggml_backend_vk_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;

    प्रत्यागम ctx->is_integrated_gpu ? GGML_BACKEND_DEVICE_TYPE_IGPU : GGML_BACKEND_DEVICE_TYPE_GPU;
}

स्थिर शून्य ggml_backend_vk_device_get_props(ggml_backend_dev_t dev, संरचना ggml_backend_dev_props * props) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;

    props->name        = ggml_backend_vk_device_get_name(dev);
    props->description = ggml_backend_vk_device_get_description(dev);
    props->type        = ggml_backend_vk_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.रिक्तानि() ? शून्यसूचकः : ctx->pci_bus_id.c_str();
    ggml_backend_vk_device_get_स्मृतिः(dev, &props->स्मृतिः_free, &props->स्मृतिः_योग);
    props->caps = {
        /* .async                 = */ सत्य,
        /* .host_buffer           = */ सत्य,
        /* .buffer_from_host_ptr  = */ असत्य,
        /* .events                = */ सत्य,
        /* .mmap_support          = */ !ctx->is_integrated_gpu,
    };
}

स्थिर ggml_backend_t ggml_backend_vk_device_init(ggml_backend_dev_t dev, ध्रुव वर्ण * params) {
    अप्रयुक्तम्(params);
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    प्रत्यागम ggml_backend_vk_init(ctx->device);
}

स्थिर तार्किक ggml_backend_vk_device_supports_op(ggml_backend_dev_t dev, ध्रुव ggml_tensor * op) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ध्रुव vk_device& device = ggml_vk_get_device(ctx->device);

    ध्रुव तार्किक उपयुङ्क्ते_bda = (op->op == GGML_OP_IM2COL || op->op == GGML_OP_IM2COL_3D) &&
                          device->shader_int64 && device->बफर_device_स्थानाङ्कः;

    स्वचालित ध्रुव & tensor_आकार_supported = [&](size_t tensor_आकार) {
        यदि (tensor_आकार > device->max_बफर_आकार) {
            प्रत्यागम असत्य;
        }
        // For im2col shaders using BDA, maxStorageBufferRange limit doesn't apply.
        // If shader64BitIndexing is enabled, maxStorageBufferRange limit doesn't apply.
        यदि (!उपयुङ्क्ते_bda && !device->shader_64b_indexing) {
            यदि (tensor_आकार > device->properties.limits.max_Storage_बफर_Range) {
                प्रत्यागम असत्य;
            }
        }
        प्रत्यागम सत्य;
    };
    // reject any tensors larger than the max buffer size
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
        यदि (op->src[i] && !tensor_आकार_supported(ggml_nbytes(op->src[i]))) {
            प्रत्यागम असत्य;
        }
    }
    यदि (!tensor_आकार_supported(ggml_nbytes(op))) {
        प्रत्यागम असत्य;
    }

    विकल्प (op->op) {
        पक्ष GGML_OP_UNARY:
            विकल्प (ggml_get_unary_op(op)) {
                पक्ष GGML_UNARY_OP_EXP:
                पक्ष GGML_UNARY_OP_EXPM1:
                पक्ष GGML_UNARY_OP_ELU:
                पक्ष GGML_UNARY_OP_GELU:
                पक्ष GGML_UNARY_OP_GELU_ERF:
                पक्ष GGML_UNARY_OP_GELU_शीघ्र:
                पक्ष GGML_UNARY_OP_SILU:
                पक्ष GGML_UNARY_OP_RELU:
                पक्ष GGML_UNARY_OP_XIELU:
                पक्ष GGML_UNARY_OP_NEG:
                पक्ष GGML_UNARY_OP_TANH:
                पक्ष GGML_UNARY_OP_SIGMOID:
                पक्ष GGML_UNARY_OP_HARDSIGMOID:
                पक्ष GGML_UNARY_OP_HARDSWISH:
                पक्ष GGML_UNARY_OP_ABS:
                पक्ष GGML_UNARY_OP_SOFTPLUS:
                पक्ष GGML_UNARY_OP_चरणः:
                पक्ष GGML_UNARY_OP_आवर्तनम्:
                पक्ष GGML_UNARY_OP_CEIL:
                पक्ष GGML_UNARY_OP_FLOOR:
                पक्ष GGML_UNARY_OP_TRUNC:
                पक्ष GGML_UNARY_OP_SGN:
                    प्रत्यागम (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type);
                स्वभाव:
                    प्रत्यागम असत्य;
            }
        पक्ष GGML_OP_GLU:
            विकल्प (ggml_get_glu_op(op)) {
                पक्ष GGML_GLU_OP_GEGLU:
                पक्ष GGML_GLU_OP_REGLU:
                पक्ष GGML_GLU_OP_SWIGLU:
                पक्ष GGML_GLU_OP_SWIGLU_OAI:
                पक्ष GGML_GLU_OP_GEGLU_ERF:
                पक्ष GGML_GLU_OP_GEGLU_शीघ्र:
                पक्ष GGML_GLU_OP_SWIGLU_CLAMP:
                    प्रत्यागम (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type) &&
                           (!op->src[1] || op->src[1]->type == op->src[0]->type);
                स्वभाव:
                    प्रत्यागम असत्य;
            }
        पक्ष GGML_OP_MUL_MAT:
        पक्ष GGML_OP_MUL_MAT_ID:
            {
                ggml_type src0_type = op->src[0]->type;
                यदि (op->op == GGML_OP_MUL_MAT_ID) {
                    यदि (!ggml_vk_use_mul_mat_vec_id(op)) {
                        ध्रुव ggml_tensor * ids = op->src[2];
                        // Shaders store row IDs as uint16_t slot and token indices.
                        यदि (ids->ne[0] > int64_t(UINT16_MAX) + 1 || ids->ne[1] > int64_t(UINT16_MAX) + 1) {
                            प्रत्यागम असत्य;
                        }
                    }
                    यदि (!device->mul_mat_id_s[src0_type] && !device->mul_mat_id_m[src0_type] && !device->mul_mat_id_l[src0_type]) {
                        // If there's not enough shared memory for row_ids and the result tile, fallback to CPU
                        प्रत्यागम असत्य;
                    }
                    यदि (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                        प्रत्यागम असत्य;
                    }
                }
                विकल्प (src0_type) {
                    पक्ष GGML_TYPE_F32:
                    पक्ष GGML_TYPE_F16:
                    पक्ष GGML_TYPE_BF16:
                    पक्ष GGML_TYPE_Q1_0:
                    पक्ष GGML_TYPE_Q2_0:
                    पक्ष GGML_TYPE_Q4_0:
                    पक्ष GGML_TYPE_Q4_1:
                    पक्ष GGML_TYPE_Q5_0:
                    पक्ष GGML_TYPE_Q5_1:
                    पक्ष GGML_TYPE_Q8_0:
                    पक्ष GGML_TYPE_Q2_K:
                    पक्ष GGML_TYPE_Q3_K:
                    पक्ष GGML_TYPE_Q4_K:
                    पक्ष GGML_TYPE_Q5_K:
                    पक्ष GGML_TYPE_Q6_K:
                    पक्ष GGML_TYPE_IQ1_S:
                    पक्ष GGML_TYPE_IQ1_M:
                    पक्ष GGML_TYPE_IQ2_XXS:
                    पक्ष GGML_TYPE_IQ2_XS:
                    पक्ष GGML_TYPE_IQ2_S:
                    पक्ष GGML_TYPE_IQ3_XXS:
                    पक्ष GGML_TYPE_IQ3_S:
                    पक्ष GGML_TYPE_IQ4_XS:
                    पक्ष GGML_TYPE_IQ4_NL:
                    पक्ष GGML_TYPE_MXFP4:
                    पक्ष GGML_TYPE_NVFP4:
                    पक्ष GGML_TYPE_TQ1_0:
                    पक्ष GGML_TYPE_TQ2_0:
                        भङ्ग;
                    स्वभाव:
                        प्रत्यागम असत्य;
                }
                संरचना ggml_tensor * a;
                संरचना ggml_tensor * b;
                यदि (op->op == GGML_OP_MUL_MAT) {
                    a = op->src[0];
                    b = op->src[1];
                } अन्यथा {
                    a = op->src[2];
                    b = op->src[1];
                }
                यदि (a->ne[3] != b->ne[3]) {
                    प्रत्यागम असत्य;
                }
                यदि (!(ggml_vk_dim01_contiguous(op->src[0]) || op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) ||
                    !(ggml_vk_dim01_contiguous(op->src[1]) || op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16)) {
                    प्रत्यागम असत्य;
                }
                यदि (op->src[0]->type == GGML_TYPE_BF16 && op->src[1]->type == GGML_TYPE_F16) {
                    // We currently don't have a bf16 x f16 shader, or an fp16->bf16 copy shader.
                    // So don't support this combination for now.
                    प्रत्यागम असत्य;
                }
                यदि (op->src[1]->type == GGML_TYPE_BF16 && op->src[0]->type != GGML_TYPE_BF16 && op->op != GGML_OP_MUL_MAT) {
                    // BF16 in src1 (widened to F32/F16) is only wired up for MUL_MAT
                    प्रत्यागम असत्य;
                }

                प्रत्यागम सत्य;
            }
        पक्ष GGML_OP_FLASH_ATTN_EXT:
            {
                तार्किक coopmat2 = device->coopmat2;
                uint32_t HSK = op->src[1]->ne[0];
                uint32_t HSV = op->src[2]->ne[0];
                यदि ((HSK % 8) != 0 || (HSV % 8) != 0) {
                    प्रत्यागम असत्य;
                }
                यदि (op->src[4] && op->src[4]->type != GGML_TYPE_F32) {
                    प्रत्यागम असत्य;
                }
                यदि (op->src[0]->type != GGML_TYPE_F32) {
                    प्रत्यागम असत्य;
                }
                यदि (op->type != GGML_TYPE_F32) {
                    प्रत्यागम असत्य;
                }
                यदि (op->src[3] && op->src[3]->type != GGML_TYPE_F16) {
                    प्रत्यागम असत्य;
                }
                स्वचालित fa_kv_ok = [](ggml_type t) {
                    विकल्प (t) {
                    पक्ष GGML_TYPE_F32:
                    पक्ष GGML_TYPE_F16:
                    पक्ष GGML_TYPE_BF16:
                    पक्ष GGML_TYPE_Q8_0:
                    पक्ष GGML_TYPE_Q5_1:
                    पक्ष GGML_TYPE_Q5_0:
                    पक्ष GGML_TYPE_Q4_1:
                    पक्ष GGML_TYPE_Q4_0:
                    पक्ष GGML_TYPE_IQ4_NL:
                        प्रत्यागम सत्य;
                    स्वभाव:
                        प्रत्यागम असत्य;
                    }
                };
                यदि (!fa_kv_ok(op->src[1]->type) || !fa_kv_ok(op->src[2]->type)) {
                    प्रत्यागम असत्य;
                }
                यदि ((op->src[1]->type == GGML_TYPE_BF16) != (op->src[2]->type == GGML_TYPE_BF16)) {
                    प्रत्यागम असत्य;
                }
                यदि (!coopmat2 && !(device->subgroup_shuffle && device->subgroup_vote)) {
                    // scalar/coopmat1 FA uses subgroupShuffle/subgroupAll
                    प्रत्यागम असत्य;
                }
                प्रत्यागम सत्य;
            }
        पक्ष GGML_OP_GET_पङ्क्तिकाः:
            {
                विकल्प (op->src[0]->type) {
                    पक्ष GGML_TYPE_F32:
                    पक्ष GGML_TYPE_F16:
                    पक्ष GGML_TYPE_BF16:
                    पक्ष GGML_TYPE_Q1_0:
                    पक्ष GGML_TYPE_Q2_0:
                    पक्ष GGML_TYPE_Q4_0:
                    पक्ष GGML_TYPE_Q4_1:
                    पक्ष GGML_TYPE_Q5_0:
                    पक्ष GGML_TYPE_Q5_1:
                    पक्ष GGML_TYPE_Q8_0:
                    पक्ष GGML_TYPE_Q2_K:
                    पक्ष GGML_TYPE_Q3_K:
                    पक्ष GGML_TYPE_Q4_K:
                    पक्ष GGML_TYPE_Q5_K:
                    पक्ष GGML_TYPE_Q6_K:
                    पक्ष GGML_TYPE_IQ1_S:
                    पक्ष GGML_TYPE_IQ1_M:
                    पक्ष GGML_TYPE_IQ2_XXS:
                    पक्ष GGML_TYPE_IQ2_XS:
                    पक्ष GGML_TYPE_IQ2_S:
                    पक्ष GGML_TYPE_IQ3_XXS:
                    पक्ष GGML_TYPE_IQ3_S:
                    पक्ष GGML_TYPE_IQ4_XS:
                    पक्ष GGML_TYPE_IQ4_NL:
                    पक्ष GGML_TYPE_MXFP4:
                    पक्ष GGML_TYPE_NVFP4:
                    पक्ष GGML_TYPE_TQ1_0:
                    पक्ष GGML_TYPE_TQ2_0:
                    पक्ष GGML_TYPE_I32:
                        प्रत्यागम सत्य;
                    स्वभाव:
                        प्रत्यागम असत्य;
                }
            }
        पक्ष GGML_OP_GET_पङ्क्तिकाः_पृष्ठभागः:
            प्रत्यागम op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        पक्ष GGML_OP_समुच्चयः_पङ्क्तिकाः:
            {
                यदि ((op->src[0]->type != GGML_TYPE_F32 && op->src[0]->type != GGML_TYPE_F16) ||
                    (op->src[1]->type != GGML_TYPE_I32 && op->src[1]->type != GGML_TYPE_I64)) {
                    प्रत्यागम असत्य;
                }
                विकल्प (op->type) {
                    पक्ष GGML_TYPE_F32:
                    पक्ष GGML_TYPE_F16:
                    पक्ष GGML_TYPE_BF16:
                    पक्ष GGML_TYPE_Q1_0:
                    पक्ष GGML_TYPE_Q2_0:
                    पक्ष GGML_TYPE_Q4_0:
                    पक्ष GGML_TYPE_Q4_1:
                    पक्ष GGML_TYPE_Q5_0:
                    पक्ष GGML_TYPE_Q5_1:
                    पक्ष GGML_TYPE_Q8_0:
                    पक्ष GGML_TYPE_IQ4_NL:
                        प्रत्यागम सत्य;
                    स्वभाव:
                        प्रत्यागम असत्य;
                }
            }
        पक्ष GGML_OP_CONT:
        पक्ष GGML_OP_CPY:
        पक्ष GGML_OP_DUP:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1] != शून्यसूचकः ? op->src[1]->type : src0_type;

                यदि (src0_type == GGML_TYPE_F32) {
                    विकल्प (src1_type) {
                    पक्ष GGML_TYPE_F32:
                    पक्ष GGML_TYPE_F16:
                    पक्ष GGML_TYPE_BF16:
                    पक्ष GGML_TYPE_Q1_0:
                    पक्ष GGML_TYPE_Q2_0:
                    पक्ष GGML_TYPE_Q4_0:
                    पक्ष GGML_TYPE_Q4_1:
                    पक्ष GGML_TYPE_Q5_0:
                    पक्ष GGML_TYPE_Q5_1:
                    पक्ष GGML_TYPE_Q8_0:
                    पक्ष GGML_TYPE_IQ4_NL:
                        प्रत्यागम सत्य;
                    स्वभाव:
                        भङ्ग;
                    }
                }
                यदि (src1_type == GGML_TYPE_F32) {
                    विकल्प (src0_type) {
                    पक्ष GGML_TYPE_F16:
                    पक्ष GGML_TYPE_BF16:
                    पक्ष GGML_TYPE_Q1_0:
                    पक्ष GGML_TYPE_Q2_0:
                    पक्ष GGML_TYPE_Q4_0:
                    पक्ष GGML_TYPE_Q4_1:
                    पक्ष GGML_TYPE_Q5_0:
                    पक्ष GGML_TYPE_Q5_1:
                    पक्ष GGML_TYPE_Q8_0:
                    पक्ष GGML_TYPE_IQ4_NL:
                        प्रत्यागम सत्य;
                    स्वभाव:
                        भङ्ग;
                    }
                }

                यदि (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    प्रत्यागम सत्य;
                }

                यदि (
                    (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) ||
                    (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32)
                ) {
                    प्रत्यागम सत्य;
                }

                // We can handle copying from a type to the same type if it's
                // either not quantized or is quantized and contiguous.
                // We use f16 or f32 shaders to do the copy,
                // so the type/block size must be a multiple of 4.
                यदि (src0_type == src1_type &&
                    (!ggml_is_quantized(src0_type) || (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op))) &&
                    (ggml_type_आकार(src0_type) % 2) == 0) {
                    प्रत्यागम सत्य;
                }
                प्रत्यागम असत्य;
            }
        पक्ष GGML_OP_REPEAT:
            प्रत्यागम ggml_type_आकार(op->type) == ggml_type_आकार(op->src[0]->type) &&
                  (ggml_type_आकार(op->type) == परिमाण(भिन्न) || ggml_type_आकार(op->type) == 2);
        पक्ष GGML_OP_REPEAT_पृष्ठभागः:
            प्रत्यागम op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        पक्ष GGML_OP_ROPE:
            प्रत्यागम ggml_is_contiguous_पङ्क्तिकाः(op) && ggml_is_contiguous_पङ्क्तिकाः(op->src[0]);
        पक्ष GGML_OP_ROPE_पृष्ठभागः:
        पक्ष GGML_OP_NONE:
        पक्ष GGML_OP_RESHAPE:
        पक्ष GGML_OP_VIEW:
        पक्ष GGML_OP_PERMUTE:
        पक्ष GGML_OP_TRANSPOSE:
        पक्ष GGML_OP_RMS_NORM:
            प्रत्यागम सत्य;
        पक्ष GGML_OP_गणः_NORM:
            प्रत्यागम ggml_is_contiguous(op->src[0]);
        पक्ष GGML_OP_NORM:
        पक्ष GGML_OP_L2_NORM:
            प्रत्यागम op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        पक्ष GGML_OP_सङ्कलय:
        पक्ष GGML_OP_SUB:
        पक्ष GGML_OP_MUL:
        पक्ष GGML_OP_DIV:
            प्रत्यागम (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        पक्ष GGML_OP_सङ्कलय_ID:
            प्रत्यागम op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->src[2]->type == GGML_TYPE_I32 &&
                   op->type == GGML_TYPE_F32;
        पक्ष GGML_OP_SILU_पृष्ठभागः:
        पक्ष GGML_OP_RMS_NORM_पृष्ठभागः:
            प्रत्यागम ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        पक्ष GGML_OP_SQR:
        पक्ष GGML_OP_SQRT:
        पक्ष GGML_OP_SIN:
        पक्ष GGML_OP_COS:
        पक्ष GGML_OP_CLAMP:
        पक्ष GGML_OP_LEAKY_RELU:
            प्रत्यागम (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->type == op->src[0]->type;
        पक्ष GGML_OP_OPT_चरणः_ADAMW:
        पक्ष GGML_OP_OPT_चरणः_SGD:
            प्रत्यागम ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        पक्ष GGML_OP_बहिर्गमनम्_PROD:
            प्रत्यागम ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && op->type == GGML_TYPE_F32;
        पक्ष GGML_OP_LOG:
        पक्ष GGML_OP_TRI:
        पक्ष GGML_OP_DIAG:
            प्रत्यागम (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->type == op->src[0]->type;
        पक्ष GGML_OP_ARGSORT:
            {
                यदि (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
                    प्रत्यागम असत्य;
                }
                // pipeline_argsort_large_f32 requires vulkan memory model.
                यदि (device->vulkan_स्मृतिः_model) {
                    प्रत्यागम सत्य;
                } अन्यथा {
                    प्रत्यागम op->ne[0] <= (1 << std::min(device->max_workgroup_आकार_log2, अङ्कः_argsort_pipelines - 1));
                }
            }
        पक्ष GGML_OP_शिखरम्_K:
            {
                यदि (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
                    प्रत्यागम असत्य;
                }
                // large k falls back to radix-select
                ध्रुव uint32_t min_pipeline =
                    std::max((uint32_t) log2f(भिन्न(op->ne[0])) + 1, device->subgroup_आकार_log2);
                यदि (min_pipeline < अङ्कः_topk_pipelines && device->pipeline_topk_f32[min_pipeline]) {
                    प्रत्यागम सत्य;
                }
                प्रत्यागम device->pipeline_topk_radix_f32 != शून्यसूचकः;
            }
        पक्ष GGML_OP_UPSCALE:
            यदि (op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS) {
                यदि ((op->op_params[0] & 0xFF) != GGML_SCALE_बहुलकम्_BILINEAR) {
                    प्रत्यागम असत्य;
                }
            }
            प्रत्यागम op->src[0]->type == GGML_TYPE_F32;
        पक्ष GGML_OP_ACC:
            प्रत्यागम op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        पक्ष GGML_OP_समुच्चयः:
            प्रत्यागम op->src[0]->type == op->src[1]->type && op->src[0]->type == op->type &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_I32);
        पक्ष GGML_OP_CONCAT: {
            प्रत्यागम ggml_vk_concat_supported(op->src[0], op->src[1], op);
        }
        पक्ष GGML_OP_ADD1:
            प्रत्यागम (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16);
        पक्ष GGML_OP_ARANGE:
            प्रत्यागम op->type == GGML_TYPE_F32;
        पक्ष GGML_OP_FILL:
            प्रत्यागम op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
        पक्ष GGML_OP_SCALE:
            प्रत्यागम ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        पक्ष GGML_OP_PAD:
        पक्ष GGML_OP_PAD_REFLECT_1D:
        पक्ष GGML_OP_ROLL:
            प्रत्यागम op->src[0]->type == GGML_TYPE_F32;
        पक्ष GGML_OP_DIAG_आवरणम्_INF:
            प्रत्यागम ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        पक्ष GGML_OP_SOFT_MAX:
            प्रत्यागम ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && (!op->src[1] || (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16));
        पक्ष GGML_OP_SOFT_MAX_पृष्ठभागः:
            प्रत्यागम ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32;
        पक्ष GGML_OP_योग:
            प्रत्यागम op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguously_allocated(op->src[0]);
        पक्ष GGML_OP_योग_पङ्क्तिकाः:
        पक्ष GGML_OP_माध्यम्:
            प्रत्यागम op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous_पङ्क्तिकाः(op->src[0]);
        पक्ष GGML_OP_CUMSUM:
            {
                यदि (device->subgroup_arithmetic && device->subgroup_require_पूर्णम्_support) {
                    प्रत्यागम op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous_पङ्क्तिकाः(op->src[0]);
                }
                प्रत्यागम असत्य;
            }
        पक्ष GGML_OP_DSV4_HC_COMB:
        पक्ष GGML_OP_DSV4_HC_PRE:
        पक्ष GGML_OP_DSV4_HC_POST:
            {
                यदि (op->type != GGML_TYPE_F32) {
                    प्रत्यागम असत्य;
                }
                प्रति (uint32_t i = 0; i < GGML_MAX_SRC; ++i) {
                    यदि (op->src[i] && op->src[i]->type != GGML_TYPE_F32) {
                        प्रत्यागम असत्य;
                    }
                }
                // hc is hardcoded to 4 in the shaders. ggml only constrains it
                // to 4 for COMB, so PRE/POST have to be checked here.
                यदि (op->op == GGML_OP_DSV4_HC_PRE && op->src[0]->ne[1] != 4) {
                    प्रत्यागम असत्य;
                }
                यदि (op->op == GGML_OP_DSV4_HC_POST && op->src[1]->ne[1] != 4) {
                    प्रत्यागम असत्य;
                }
                यदि (op->op == GGML_OP_DSV4_HC_COMB) {
                    प्रत्यागम device->pipeline_dsv4_hc_comb_f32 != शून्यसूचकः;
                }
                प्रत्यागम सत्य;
            }
        पक्ष GGML_OP_SOLVE_TRI:
            {
                यदि (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32) {
                    प्रत्यागम असत्य;
                }
                ध्रुव uint32_t N = op->src[0]->ne[0];
                ध्रुव uint32_t K = op->src[1]->ne[0];
                // K dimension limited to workgroup size
                यदि (K > 1u << device->max_workgroup_आकार_log2) {
                    प्रत्यागम असत्य;
                }
                ध्रुव uint32_t batch_N = device->properties.limits.max_Compute_Shared_स्मृतिः_आकार / ((N + K) * परिमाण(भिन्न));

                यदि (batch_N == 0) {
                    प्रत्यागम असत्य;
                }
                प्रत्यागम सत्य;
            }
        पक्ष GGML_OP_ARGMAX:
            प्रत्यागम ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        पक्ष GGML_OP_CROSS_ENTROPY_LOSS:
            प्रत्यागम ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && ggml_are_same_shape(op->src[0], op->src[1])
                && ggml_is_contiguous(op) && ggml_is_scalar(op) && op->type == GGML_TYPE_F32;
        पक्ष GGML_OP_CROSS_ENTROPY_LOSS_पृष्ठभागः:
            प्रत्यागम ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32 && ggml_is_scalar(op->src[0])
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[2]) && op->src[2]->type == GGML_TYPE_F32
                && ggml_are_same_shape(op->src[1], op->src[2])
                && ggml_are_same_shape(op->src[1], op)
                && ggml_is_contiguous(op) && op->type == GGML_TYPE_F32;
        पक्ष GGML_OP_गणनम्_EQUAL:
            प्रत्यागम ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_I32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_I32;
        पक्ष GGML_OP_IM2COL:
            प्रत्यागम ggml_is_contiguous(op->src[1])
                && op->src[1]->type == GGML_TYPE_F32
                && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        पक्ष GGML_OP_IM2COL_3D:
            प्रत्यागम op->src[1]->type == GGML_TYPE_F32
                && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        पक्ष GGML_OP_TIMESTEP_EMBEDDING:
            प्रत्यागम op->src[0]->type == GGML_TYPE_F32;
        पक्ष GGML_OP_CONV_2D_DW:
            प्रत्यागम (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16)
                && op->src[1]->type == GGML_TYPE_F32;
        पक्ष GGML_OP_POOL_1D:
            प्रत्यागम ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        पक्ष GGML_OP_POOL_2D:
            प्रत्यागम ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        पक्ष GGML_OP_RWKV_WKV6:
        पक्ष GGML_OP_RWKV_WKV7:
            प्रत्यागम सत्य; // all inputs are contiguous, see ggml.c
        पक्ष GGML_OP_GATED_रैखिकम्_ATTN:
            // the shader block size is hardcoded to head_size 64
            प्रत्यागम op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && op->src[0]->ne[0] == 64;
        पक्ष GGML_OP_LIGHTNING_INDEXER:
            {
                ध्रुव ggml_tensor * q = op->src[0];
                ध्रुव ggml_tensor * k = op->src[1];
                ध्रुव ggml_tensor * w = op->src[2];
                ध्रुव ggml_tensor * m = op->src[3];

                // the q/w/m types and the shape relationships between q, k, w, m and dst
                // are already asserted in ggml_lightning_indexer()
                यदि (!ggml_vk_lightning_indexer_k_type_supported(k->type) || !device->fp16) {
                    प्रत्यागम असत्य;
                }

                // the shader block size is hardcoded to head size 128
                यदि (q->ne[0] != 128) {
                    प्रत्यागम असत्य;
                }

                // the shader indexes the buffers by element stride, and is dispatched
                // without allow_misalign
                प्रति (ध्रुव ggml_tensor * t : {q, k, w, m, op}) {
                    यदि (t->nb[0] != ggml_type_आकार(t->type) ||
                        (vk_tensor_विस्थापनम्(t) + t->view_offs) % device->properties.limits.min_Storage_बफर_विस्थापनम्_Alignment != 0) {
                        प्रत्यागम असत्य;
                    }
                    // the strides get scaled down from bytes, so the division must be exact
                    प्रति (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; ++i) {
                        यदि (t->nb[i] % ggml_type_आकार(t->type) != 0) {
                            प्रत्यागम असत्य;
                        }
                    }
                }
                प्रत्यागम सत्य;
            }
        पक्ष GGML_OP_GATED_DELTA_NET:
            {
                ध्रुव uint32_t S_v = op->src[2]->ne[0];
                यदि (S_v != 16 && S_v != 32 && S_v != 64 && S_v != 128) {
                    प्रत्यागम असत्य;
                }
                प्रति (पूर्णाङ्क i = 0; i < 6; i++) {
                    यदि (op->src[i] == शून्यसूचकः || op->src[i]->type != GGML_TYPE_F32) {
                        प्रत्यागम असत्य;
                    }
                }
                प्रत्यागम op->type == GGML_TYPE_F32;
            }
        पक्ष GGML_OP_SSM_अवलोकनम्:
            {
                प्रति (पूर्णाङ्क i = 0; i < 6; i++) {
                    यदि (op->src[i] && ggml_is_quantized(op->src[i]->type)) {
                        प्रत्यागम असत्य;
                    }
                }
                यदि (op->src[6] && op->src[6]->type != GGML_TYPE_I32) {
                    प्रत्यागम असत्य;
                }
                यदि (op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    प्रत्यागम असत्य;
                }

                ध्रुव uint32_t d_state = op->src[0]->ne[0];
                ध्रुव uint32_t शीर्ष_dim = op->src[0]->ne[1];

                तार्किक is_mamba2 = (op->src[3] && op->src[3]->nb[1] == परिमाण(भिन्न));
                यदि (!is_mamba2) {
                    प्रत्यागम असत्य;
                }

                यदि ((d_state != 128 && d_state != 256) || शीर्ष_dim % 16 != 0) {
                    प्रत्यागम असत्य;
                }

                size_t shmem_आकार = d_state * परिमाण(भिन्न);

                यदि (shmem_आकार > device->properties.limits.max_Compute_Shared_स्मृतिः_आकार) {
                    प्रत्यागम असत्य;
                }

                यदि (!device->subgroup_basic) {
                    प्रत्यागम असत्य;
                }

                प्रत्यागम सत्य;
            }
        पक्ष GGML_OP_SSM_CONV:
            प्रत्यागम op->src[0]->type == GGML_TYPE_F32;
        पक्ष GGML_OP_CONV_TRANSPOSE_1D:
            प्रत्यागम op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        पक्ष GGML_OP_COL2IM_1D:
            प्रत्यागम (op->src[0]->type == GGML_TYPE_F32 ||
                    op->src[0]->type == GGML_TYPE_F16 ||
                    op->src[0]->type == GGML_TYPE_BF16) &&
                   op->type == op->src[0]->type &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op);
        पक्ष GGML_OP_CONV_2D:
        पक्ष GGML_OP_CONV_TRANSPOSE_2D:
            {
                ध्रुव तार्किक transpose = op->op == GGML_OP_CONV_TRANSPOSE_2D;
                ध्रुव int64_t निर्गमः = !transpose ? op->src[0]->ne[3] : op->src[0]->ne[2];
                ध्रुव int64_t आगमः  = !transpose ? op->src[0]->ne[2] : op->src[0]->ne[3];

                // Channel-contiguous format is not supported yet.
                प्रत्यागम ((op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                    (op->src[0]->nb[0] == परिमाण(भिन्न) || op->src[0]->nb[0] == परिमाण(ggml_fp16_t) ) &&
                    op->src[1]->type == GGML_TYPE_F32 &&
                    op->type == GGML_TYPE_F32 &&
                    निर्गमः == op->ne[2] &&
                    आगमः == op->src[1]->ne[2] &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op->src[1]) &&
                    ggml_is_contiguous(op));
            }
        पक्ष GGML_OP_CONV_3D:
            प्रत्यागम (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) &&
                ggml_is_contiguous(op);
        स्वभाव:
            प्रत्यागम असत्य;
    }

    अप्रयुक्तम्(dev);
}

स्थिर तार्किक ggml_backend_vk_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    यदि (buft->iface.get_name != ggml_backend_vk_बफर_type_name) {
        प्रत्यागम असत्य;
    }

    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ggml_backend_vk_बफर_type_context * buft_ctx = (ggml_backend_vk_बफर_type_context *)buft->context;

    प्रत्यागम buft_ctx->device->idx == ctx->device;
}

स्थिर तार्किक ggml_backend_vk_device_offload_op(ggml_backend_dev_t dev, ध्रुव ggml_tensor * op) {
    ggml_backend_vk_device_context * dev_ctx = (ggml_backend_vk_device_context *)dev->context;

    प्रत्यागम ggml_vk_get_op_batch_आकार(op) >= dev_ctx->op_offload_min_batch_आकार;
}

स्थिर ggml_backend_event_t ggml_backend_vk_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    स्वचालित device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = नूतन vk_event;
    यदि (!vkev) {
        प्रत्यागम शून्यसूचकः;
    }

    // No events initially, they get created on demand
    vkev->has_event = असत्य;

    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.समुच्चयः_PNext(&tci);
    vkev->tl_semaphore = { device->device.createSemaphore(ci), 0 };

    प्रत्यागम नूतन ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ vkev,
    };
}

स्थिर शून्य ggml_backend_vk_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    स्वचालित device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = (vk_event *)event->context;

    device->device.destroySemaphore(vkev->tl_semaphore.s);
    प्रति (स्वचालित& event : vkev->events_free) {
        device->device.destroyEvent(event);
    }
    प्रति (स्वचालित& event : vkev->events_submitted) {
        device->device.destroyEvent(event);
    }
    यदि (vkev->has_event) {
        device->device.destroyEvent(vkev->event);
    }
    विलोपय vkev;
    विलोपय event;
}

स्थिर शून्य ggml_backend_vk_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_device_event_synchronize(backend=" << dev << ", event=" << event << ")");
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    स्वचालित device = ggml_vk_get_device(ctx->device);
    vk_event *vkev = (vk_event *)event->context;

    // Only do something if the event has actually been used
    यदि (vkev->has_event) {
        vk::Semaphore sem = vkev->tl_semaphore.s;
        uint64_t val = vkev->tl_semaphore.value;
        vk::SemaphoreWaitInfo swi{vk::SemaphoreWaitFlags{}, sem, val};
        VK_CHECK(device->device.waitSemaphores(swi, UINT64_MAX), "event_synchronize", device);

        // Reset and move submitted events
        प्रति (स्वचालित& event : vkev->events_submitted) {
            device->device.resetEvent(event);
        }
        vkev->events_free.insert(vkev->events_free.end(), vkev->events_submitted.begin(), vkev->events_submitted.end());
        vkev->events_submitted.clear();

        // Finished using current command buffer so we flag for reuse
        यदि (vkev->cmd_बफर) {
            // Only flag for reuse if it hasn't been reused already
            यदि (vkev->cmd_बफर_use_गणक == vkev->cmd_बफर->use_गणक) {
                vkev->cmd_बफर->अन्तर्_use = असत्य;
                vkev->cmd_बफर->buf.reset();
            }
            vkev->cmd_बफर = शून्यसूचकः;
        }
    }
}

स्थिर ggml_backend_बफर_t ggml_backend_vk_device_बफर_from_host_ptr(ggml_backend_dev_t dev, शून्य * ptr, size_t size, size_t max_tensor_आकार) {
    VK_LOG_DEBUG("ggml_backend_vk_device_buffer_from_host_ptr(backend=" << dev << ", ptr=" << ptr << ", size=" << size << ")");
    GGML_अप्रयुक्तम्(max_tensor_आकार);

    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    स्वचालित device = ggml_vk_get_device(ctx->device);

    vk_बफर buf = ggml_vk_बफर_from_host_ptr(device, ptr, size);

    यदि (!buf) {
        प्रत्यागम {};
    }

    ggml_backend_vk_बफर_context * bufctx = नूतन ggml_backend_vk_बफर_context(device, std::move(buf), device->name);

    ggml_backend_बफर_t ret = ggml_backend_बफर_init(ggml_backend_vk_device_get_बफर_type(dev), ggml_backend_vk_बफर_interface, bufctx, size);

    प्रत्यागम ret;
}

स्थिर ध्रुव संरचना ggml_backend_device_i ggml_backend_vk_device_i = {
    /* .get_name             = */ ggml_backend_vk_device_get_name,
    /* .get_description      = */ ggml_backend_vk_device_get_description,
    /* .get_memory           = */ ggml_backend_vk_device_get_स्मृतिः,
    /* .get_type             = */ ggml_backend_vk_device_get_type,
    /* .get_props            = */ ggml_backend_vk_device_get_props,
    /* .init_backend         = */ ggml_backend_vk_device_init,
    /* .get_buffer_type      = */ ggml_backend_vk_device_get_बफर_type,
    /* .get_host_buffer_type = */ ggml_backend_vk_device_get_host_बफर_type,
    /* .buffer_from_host_ptr = */ ggml_backend_vk_device_बफर_from_host_ptr,
    /* .supports_op          = */ ggml_backend_vk_device_supports_op,
    /* .supports_buft        = */ ggml_backend_vk_device_supports_buft,
    /* .offload_op           = */ ggml_backend_vk_device_offload_op,
    /* .event_new            = */ ggml_backend_vk_device_event_new,
    /* .event_free           = */ ggml_backend_vk_device_event_free,
    /* .event_synchronize    = */ ggml_backend_vk_device_event_synchronize,
};

स्थिर ध्रुव वर्ण * ggml_backend_vk_reg_get_name(ggml_backend_reg_t reg) {
    अप्रयुक्तम्(reg);
    प्रत्यागम GGML_VK_NAME;
}

स्थिर size_t ggml_backend_vk_reg_get_device_गणनम्(ggml_backend_reg_t reg) {
    अप्रयुक्तम्(reg);
    प्रत्यागम ggml_backend_vk_get_device_गणनम्();
}

स्थिर ggml_backend_dev_t ggml_backend_vk_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    स्थिर std::vector<ggml_backend_dev_t> devices;

    स्थिर तार्किक initialized = असत्य;

    {
        स्थिर std::mutex mutex;
        std::lock_guard<std::mutex> तालकम्(mutex);
        यदि (!initialized) {
            ध्रुव पूर्णाङ्क min_batch_आकार = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;
            प्रति (पूर्णाङ्क i = 0; i < ggml_backend_vk_get_device_गणनम्(); i++) {
                ggml_backend_vk_device_context * ctx = नूतन ggml_backend_vk_device_context;
                वर्ण desc[256];
                ggml_backend_vk_get_device_description(i, desc, परिमाण(desc));
                ctx->device = i;
                ctx->name = GGML_VK_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->is_integrated_gpu = ggml_backend_vk_get_device_type(i) == vk::PhysicalDeviceType::eIntegratedGpu;
                ctx->pci_bus_id = ggml_backend_vk_get_device_pci_id(i);
                ctx->op_offload_min_batch_आकार = min_batch_आकार;
                devices.push_back(नूतन ggml_backend_device {
                    /* .iface   = */ ggml_backend_vk_device_i,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                });
            }
            initialized = सत्य;
        }
    }

    GGML_ASSERT(device < devices.size());
    प्रत्यागम devices[device];
}

स्थिर ध्रुव संरचना ggml_backend_reg_i ggml_backend_vk_reg_i = {
    /* .get_name         = */ ggml_backend_vk_reg_get_name,
    /* .get_device_count = */ ggml_backend_vk_reg_get_device_गणनम्,
    /* .get_device       = */ ggml_backend_vk_reg_get_device,
    /* .get_proc_address = */ रिक्त,
};

ggml_backend_reg_t ggml_backend_vk_reg() {
    स्थिर ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_संस्करणम्,
        /* .iface       = */ ggml_backend_vk_reg_i,
        /* .context     = */ शून्यसूचकः,
    };
    यत्न {
        ggml_vk_instance_init();
        प्रत्यागम &reg;
    } गृहाण (ध्रुव vk::SystemError& e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: System error: " << e.what());
        प्रत्यागम शून्यसूचकः;
    } गृहाण (ध्रुव std::exception &e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: " << e.what());
        प्रत्यागम शून्यसूचकः;
    } गृहाण (...) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: unknown exception during Vulkan init");
        प्रत्यागम शून्यसूचकः;
    }
}

तार्किक ggml_vk_instance_layer_settings_उपलब्धम्() {
#यदि_परिभाषितम् GGML_VULKAN_VALIDATE
    // Check if validation layer provides the extension
    ध्रुव std::string layer_name = "VK_LAYER_KHRONOS_validation";
    प्रति (ध्रुव स्वचालित& layer : vk::enumerateInstanceLayerProperties()) {
        यदि (layer_name == layer.layerName.दत्त()) {
            प्रति (ध्रुव स्वचालित& ext : vk::enumerateInstanceExtensionProperties(layer_name)) {
                यदि (वाक्यं_तुलय("VK_EXT_layer_settings", ext.extensionName.दत्त()) == 0) {
                    प्रत्यागम सत्य;
                }
            }
        }
    }

    std::cerr << "ggml_vulkan: WARNING: Validation layer or layer extension VK_EXT_layer_settings not found." << std::पङ्क्त्यन्तः;
#यदि_अन्तः
    प्रत्यागम असत्य;
}

तार्किक ggml_vk_instance_portability_enumeration_ext_उपलब्धम्(ध्रुव std::vector<vk::ExtensionProperties>& instance_extensions) {
#यदि_परिभाषितम् __APPLE__
    // Check for portability enumeration extension for MoltenVK support
    प्रति (ध्रुव स्वचालित& properties : instance_extensions) {
        यदि (वाक्यं_तुलय("VK_KHR_portability_enumeration", properties.extensionName) == 0) {
            प्रत्यागम सत्य;
        }
    }
    std::cerr << "ggml_vulkan: WARNING: Instance extension VK_KHR_portability_enumeration not found." << std::पङ्क्त्यन्तः;
#यदि_अन्तः
    प्रत्यागम असत्य;

    अप्रयुक्तम्(instance_extensions);
}

तार्किक ggml_vk_instance_debug_utils_ext_उपलब्धम्(
    ध्रुव std::vector<vk::ExtensionProperties> & instance_extensions) {
    // Check for portability enumeration extension for MoltenVK support
    प्रति (ध्रुव स्वचालित & properties : instance_extensions) {
        यदि (वाक्यं_तुलय("VK_EXT_debug_utils", properties.extensionName) == 0) {
            प्रत्यागम सत्य;
        }
    }

    std::cerr << "ggml_vulkan: WARNING: Instance extension VK_EXT_debug_utils not found." << std::पङ्क्त्यन्तः;
    प्रत्यागम असत्य;

    अप्रयुक्तम्(instance_extensions);
}

तार्किक ggml_vk_device_is_supported(ध्रुव vk::PhysicalDevice & vkdev) {
    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.p_अग्र = शून्यसूचकः;
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.p_अग्र = &vk11_features;

    vkGetPhysicalDeviceFeatures2(vkdev, &device_features2);

    प्रत्यागम vk11_features.storageBuffer16BitAccess;
}

तार्किक ggml_vk_khr_cooperative_matrix_support(ध्रुव vk::PhysicalDeviceProperties& props, ध्रुव vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch) {
    विकल्प (props.vendorID) {
    पक्ष VK_VENDOR_ID_INTEL:
        // Only allowing Xe2/Xe3 GPU and integrated Xe GPUs at the moment since older hardware (ex. Arc A770) has performance regressions.
        प्रत्यागम (arch == vk_device_architecture::INTEL_XE2) ||
            (arch == vk_device_architecture::INTEL_XE1 && props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu && driver_props.driverID == vk::DriverId::eIntelProprietaryWindows);
    पक्ष VK_VENDOR_ID_AMD:
        यदि (driver_props.driverID == vk::DriverId::eAmdProprietary || driver_props.driverID == vk::DriverId::eAmdOpenSource) {
            // Workaround for AMD proprietary driver reporting support on all GPUs
            प्रत्यागम arch == vk_device_architecture::AMD_RDNA3 || arch == vk_device_architecture::AMD_RDNA4;
        }
        प्रत्यागम सत्य;
    पक्ष VK_VENDOR_ID_QUALCOMM:
        // Only allow Adreno GPUs with hardware matrix cores (Gen 6+).
        प्रत्यागम arch == vk_device_architecture::QUALCOMM_ADRENO;
    स्वभाव:
        प्रत्यागम सत्य;
    }
}

uint32_t ggml_vk_intel_shader_core_गणनम्(ध्रुव vk::PhysicalDevice& vkdev) {
    VkPhysicalDeviceProperties2 props = vkdev.getProperties2();

    यदि (props.properties.vendorID != VK_VENDOR_ID_INTEL) {
        प्रत्यागम 0;
    }

    ध्रुव uint32_t device_id = props.properties.deviceID;

    विकल्प (device_id) {
    पक्ष 0x56A6:  // A310
        प्रत्यागम 6;
    पक्ष 0x5693:  // A370M
    पक्ष 0x56A5:  // A380
    पक्ष 0x56B1:  // Pro A40/A50
        प्रत्यागम 8;
    पक्ष 0x5697:  // A530M
        प्रत्यागम 12;
    पक्ष 0x5692:  // A550M
    पक्ष 0x56B3:  // Pro A60
        प्रत्यागम 16;
    पक्ष 0x56A2:  // A580
        प्रत्यागम 24;
    पक्ष 0x5691:  // A730M
    पक्ष 0x56A1:  // A750
        प्रत्यागम 28;
    पक्ष 0x56A0:  // A770
    पक्ष 0x5690:  // A770M
        प्रत्यागम 32;
    पक्ष 0xE212:  // Pro B50
        प्रत्यागम 16;
    पक्ष 0xE20C:  // B570
        प्रत्यागम 18;
    पक्ष 0xE20B:  // B580
    पक्ष 0xE211:  // Pro B60
    पक्ष 0xE222:  // Pro B65
        प्रत्यागम 20;
    पक्ष 0xE223:  // Pro B70
        प्रत्यागम 32;
    पक्ष 0xB080:  // PTL Xe3 LPG 2x6 (12 subslices)
        प्रत्यागम 12;
    स्वभाव:
        प्रत्यागम 0;
    }
}

तार्किक ggml_vk_intel_windows_driver_अन्तर्_range(uint32_t driver_संस्करणम्, uint32_t अधस्तनम्_major, uint32_t अधस्तनम्_minor, uint32_t ऊर्ध्वस्थम्_major, uint32_t ऊर्ध्वस्थम्_minor) {
#यदि defined(_WIN32)
    // Intel Windows encodes xxx.yyyy as [31:14].[13:0].
    ध्रुव uint32_t major = driver_संस्करणम् >> 14;
    ध्रुव uint32_t minor = driver_संस्करणम् & 0x3fff;

    ध्रुव तार्किक ge_अधस्तनम् = major > अधस्तनम्_major || (major == अधस्तनम्_major && minor >= अधस्तनम्_minor);
    ध्रुव तार्किक lt_ऊर्ध्वस्थम् = major < ऊर्ध्वस्थम्_major || (major == ऊर्ध्वस्थम्_major && minor < ऊर्ध्वस्थम्_minor);

    प्रत्यागम ge_अधस्तनम् && lt_ऊर्ध्वस्थम्;
#अन्यथा
    GGML_अप्रयुक्तम्(driver_संस्करणम्);
    GGML_अप्रयुक्तम्(अधस्तनम्_major);
    GGML_अप्रयुक्तम्(अधस्तनम्_minor);
    GGML_अप्रयुक्तम्(ऊर्ध्वस्थम्_major);
    GGML_अप्रयुक्तम्(ऊर्ध्वस्थम्_minor);
    प्रत्यागम सत्य;
#यदि_अन्तः
}

GGML_BACKEND_DL_IMPL(ggml_backend_vk_reg)


// out-of-lined header method definitions

शून्य vk_queue_handle_synchronized::submit(vk::ArrayProxy<ध्रुव vk::SubmitInfo> submits, vk::Fence fence) {
    // Workaround for NVIDIA driver bug
    std::unique_lock<std::mutex> device_guard;
    यदि (device_submit_mutex) {
        device_guard = std::unique_lock<std::mutex>(*device_submit_mutex);
    }
    std::lock_guard<std::mutex> guard(mutex);
    यत्न {
        queue.submit(submits, fence);
    } गृहाण (vk::Device_नष्टम्_Error &) {
        यदि (स्वचालित dev = device.तालकम्()) {
            ggml_vk_print_device_नष्टम्_info(dev);
        }
        क्षिप;
    }
}

शून्य vk_queue_handle_unsynchronized::submit(vk::ArrayProxy<ध्रुव vk::SubmitInfo> submits, vk::Fence fence) {
    // Workaround for NVIDIA driver bug
    std::unique_lock<std::mutex> device_guard;
    यदि (device_submit_mutex) {
        device_guard = std::unique_lock<std::mutex>(*device_submit_mutex);
    }
    यत्न {
        queue.submit(submits, fence);
    } गृहाण (vk::Device_नष्टम्_Error &) {
        यदि (स्वचालित dev = device.तालकम्()) {
            ggml_vk_print_device_नष्टम्_info(dev);
        }
        क्षिप;
    }
}

vk_device_struct::~vk_device_struct() {
    // the build failed before the device was created, nothing to destroy
    यदि (device == VK_NULL_HANDLE) {
        प्रत्यागम;
    }

    VK_LOG_DEBUG("destroy device " << name);

    device.destroyFence(fence);

    ggml_vk_destroy_बफर(sync_staging);

    यदि (compute_queue) compute_queue->cmd_pool.destroy(device);
    यदि (transfer_queue) transfer_queue->cmd_pool.destroy(device);

    // Explicitly clear to ensure queues drop their shared_ptrs to handles
    // before the Vulkan logical device instance is destroyed
    compute_queue.reset();
    transfer_queue.reset();

    प्रति (स्वचालित& pipeline : all_pipelines) {
        यदि (pipeline.expired()) {
            अनुवर्त;
        }

        vk_pipeline pl = pipeline.तालकम्();
        ggml_vk_destroy_pipeline(device, pl);
    }
    all_pipelines.clear();

    device.destroy_Descriptor_समुच्चयः_Layout(dsl);

    device.destroy();
}

शून्य vk_perf_logger::print_timings(तार्किक force) {
    यदि (timings.रिक्तानि()) {
        प्रत्यागम;
    }
    print_गणनम्++;
    यदि ((print_गणनम् % vk_perf_logger_frequency) != 0 && !force) {
        प्रत्यागम;
    }
    print_गणनम् = 0;
    uint64_t योग_all_op_times = 0;
    std::cerr << "----------------\nVulkan Timings:" << std::पङ्क्त्यन्तः;
    प्रति (ध्रुव स्वचालित & t : timings) {
        uint64_t योग_op_times = 0;
        प्रति (ध्रुव स्वचालित & time : t.second) {
            योग_op_times += time;
        }
        std::cerr << t.first << ": " << t.second.size() << " x " << (योग_op_times / t.second.size() / 1000.0)
                  << " us = " << (योग_op_times / 1000.0) << " us";

        // If we have as many flops entries as timing entries for the op, then compute and log the flops/S.
        स्वचालित it = flops.find(t.first);
        यदि (it != flops.end() && (it->second).size() == t.second.size()) {
            uint64_t योग_op_flops = 0;
            प्रति (ध्रुव स्वचालित & elem : it->second) {
                योग_op_flops += elem;
            }
            std::cerr << " ("
                      << (द्विगुण(योग_op_flops) / (1000.0 * 1000.0 * 1000.0)) /
                             (द्विगुण(योग_op_times) / (1000.0 * 1000.0 * 1000.0))
                      << " GFLOPS/s)";
        }

        योग_all_op_times += योग_op_times;

        std::cerr << std::पङ्क्त्यन्तः;
    }

    यदि (timings.size() > 0) {
        std::cerr << "Total time: " << योग_all_op_times / 1000.0 << " us." << std::पङ्क्त्यन्तः;
    }

    timings.clear();
    flops.clear();
}

std::string vk_perf_logger::get_ग्रन्थिः_fusion_name(ध्रुव ggml_tensor * ग्रन्थिः, ध्रुव वर्ण *fusion_name, uint64_t *n_flops) {
    *n_flops = ggml_vk_get_ग्रन्थिः_flops(ग्रन्थिः);
    std::string fusion_str;
    यदि (fusion_name) {
        fusion_str = fusion_name + std::string(" ");
    }
    यदि (ग्रन्थिः->op == GGML_OP_UNARY) {
        प्रत्यागम fusion_str + ggml_unary_op_name(ggml_get_unary_op(ग्रन्थिः));
    }
    यदि (ग्रन्थिः->op == GGML_OP_MUL_MAT || ग्रन्थिः->op == GGML_OP_MUL_MAT_ID) {
        ध्रुव uint64_t m     = ग्रन्थिः->ne[0];
        ध्रुव uint64_t n     = ग्रन्थिः->ne[1];
        ध्रुव uint64_t k     = ग्रन्थिः->src[1]->ne[0];
        ध्रुव uint64_t batch = ग्रन्थिः->ne[2] * ग्रन्थिः->ne[3];
        std::string    name  = ggml_op_name(ग्रन्थिः->op);
        यदि ((ग्रन्थिः->op == GGML_OP_MUL_MAT && n <= mul_mat_vec_max_स्तम्भसङ्ख्या) ||
            (ग्रन्थिः->op == GGML_OP_MUL_MAT_ID && ग्रन्थिः->src[2]->ne[1] == 1)) {
            name += "_VEC";
        }
        name += " ";
        name += ggml_type_name(ग्रन्थिः->src[0]->type);
        name += " m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=" + std::to_string(k);
        यदि (ग्रन्थिः->op == GGML_OP_MUL_MAT_ID) {
            name += " n_expert=" + std::to_string(ग्रन्थिः->src[0]->ne[2]);
        }
        यदि (batch > 1) {
            name += " batch=" + std::to_string(batch);
        }
        प्रत्यागम fusion_str + name;
    }
    यदि (ग्रन्थिः->op == GGML_OP_CONV_2D || ग्रन्थिः->op == GGML_OP_CONV_TRANSPOSE_2D) {
        std::string   name    = ggml_op_name(ग्रन्थिः->op);
        ध्रुव ggml_tensor * knl = ग्रन्थिः->src[0];
        uint64_t      Cout    = ग्रन्थिः->ne[2];
        uint64_t      आकार_K  = ग्रन्थिः->src[1]->ne[2] * knl->ne[0] * knl->ne[1];
        uint64_t      आकार_N_2  = ग्रन्थिः->ne[3] * ग्रन्थिः->ne[0] * ग्रन्थिः->ne[1];
        name += " M=Cout=" + std::to_string(Cout) + ", K=Cin*KW*KH=" + std::to_string(आकार_K) +
                ", N=N*OW*OH=" + std::to_string(आकार_N_2);
        प्रत्यागम fusion_str + name;
    }
    यदि (ग्रन्थिः->op == GGML_OP_RMS_NORM) {
        std::string   name    = ggml_op_name(ग्रन्थिः->op);
        name += "(" + std::to_string(ग्रन्थिः->ne[0]) + "," + std::to_string(ग्रन्थिः->ne[1]) + "," + std::to_string(ग्रन्थिः->ne[2]) + "," + std::to_string(ग्रन्थिः->ne[3]) + ")";
        प्रत्यागम fusion_str + name;
    }
    यदि (ग्रन्थिः->op == GGML_OP_FLASH_ATTN_EXT) {
        ध्रुव ggml_tensor * dst = ग्रन्थिः;
        ध्रुव ggml_tensor * q = ग्रन्थिः->src[0];
        ध्रुव ggml_tensor * k = ग्रन्थिः->src[1];
        ध्रुव ggml_tensor * v = ग्रन्थिः->src[2];
        ध्रुव ggml_tensor * m = ग्रन्थिः->src[3];
        std::stringstream name;
        name << fusion_str;
        name << ggml_op_name(ग्रन्थिः->op) <<
            " dst(" << dst->ne[0] << "," << dst->ne[1] << "," << dst->ne[2] << "," << dst->ne[3] << "), " <<
            " q(" << q->ne[0] << "," << q->ne[1] << "," << q->ne[2] << "," << q->ne[3] << "), " <<
            " k(" << k->ne[0] << "," << k->ne[1] << "," << k->ne[2] << "," << k->ne[3] << "), " <<
            " v(" << v->ne[0] << "," << v->ne[1] << "," << v->ne[2] << "," << v->ne[3] << "), " <<
            " m(" << (m?m->ne[0]:0) << "," << (m?m->ne[1]:0) << "," << (m?m->ne[2]:0) << "," << (m?m->ne[3]:0) << ")";
        प्रत्यागम name.str();
    }
    यदि (ग्रन्थिः->op == GGML_OP_शिखरम्_K) {
        std::stringstream name;
        name << fusion_str;
        name << ggml_op_name(ग्रन्थिः->op) <<
            " K=" << ग्रन्थिः->ne[0] <<
            " (" << ग्रन्थिः->src[0]->ne[0] << "," << ग्रन्थिः->src[0]->ne[1] << "," << ग्रन्थिः->src[0]->ne[2] << "," << ग्रन्थिः->src[0]->ne[3] << ")";
        प्रत्यागम name.str();
    }
    प्रत्यागम fusion_str + ggml_op_name(ग्रन्थिः->op);
}

ggml_backend_vk_बफर_context::~ggml_backend_vk_बफर_context() {
    ggml_vk_destroy_बफर(dev_बफर);
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_context & ctx, ध्रुव std::string & pipeline_name, uint32_t wg0, uint32_t wg1, uint32_t wg2) {
    यदि (!vk_instance.debug_utils_support || ctx->s == शून्यसूचकः) {
        प्रत्यागम;
    }
    begin(ctx, pipeline_name + " (" + std::to_string(wg0) + "," + std::to_string(wg1) + "," + std::to_string(wg2) + ")");
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_context & ctx, ध्रुव ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx, पूर्णाङ्क n_fused) {
    यदि (!vk_instance.debug_utils_support || ctx->s == शून्यसूचकः) {
        प्रत्यागम;
    }
    std::string name = ggml_op_name(cgraph->nodes[ग्रन्थिः_idx]->op);
    प्रति (पूर्णाङ्क i = 1; i <= n_fused; i++) {
        name += "+";
        name += ggml_op_name(cgraph->nodes[ग्रन्थिः_idx + i]->op);
    }
    name += " ";
    name += cgraph->nodes[ग्रन्थिः_idx]->name;
    begin(ctx, name);
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_queue_handle * handle, ध्रुव वर्ण * name) {
    यदि (!vk_instance.debug_utils_support || handle == शून्यसूचकः) {
        प्रत्यागम;
    }
    vk::DebugUtilsLabelEXT label = {};
    label.pLabelName = name;
    label.color = std::array<भिन्न, 4>{1.0f, 1.0f, 1.0f, 1.0f};

    qhandle = handle;
    std::lock_guard<vk_queue_handle> guard(*qhandle);
    vk_instance.pfn_vkQueueBeginDebugUtilsLabelEXT(qhandle->queue, reinterpret_cast<VkDebugUtilsLabelEXT *>(&label));
}

शून्य ggml_vk_debug_label::close() {
    यदि (subctx != शून्यसूचकः) {
        // close on the current command buffer, which may differ from the one begin used
        यदि (subctx->s != शून्यसूचकः) {
            vk_instance.pfn_vk_Cmd_अन्तः_Debug_Utils_Label_EXT(subctx->s->बफर->buf);
        }
        subctx->debug_labels.pop_back();
        subctx = शून्यसूचकः;
    }
    यदि (qhandle != शून्यसूचकः) {
        std::lock_guard<vk_queue_handle> guard(*qhandle);
        vk_instance.pfn_vk_Queue_अन्तः_Debug_Utils_Label_EXT(qhandle->queue);
        qhandle = शून्यसूचकः;
    }
}

शून्य ggml_vk_debug_label::begin(vk_context & ctx, ध्रुव std::string & name) {
    यदि (!vk_instance.debug_utils_support || ctx->s == शून्यसूचकः) {
        प्रत्यागम;
    }
    subctx = ctx.get();
    subctx->debug_labels.push_back(name);
    ggml_vk_cmd_label_begin(subctx->s->बफर->buf, subctx->debug_labels.back().c_str());
}


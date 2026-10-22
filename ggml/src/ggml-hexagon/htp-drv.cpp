#شامل <filesystem>
#شامل <set>
#شامل <sstream>
#شامل <string>

#اگر_تعریف_شدہ _WIN32
#   define WIN32_LEAN_AND_اوسط
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#ورنہ
#   include <dlfcn.h>
#   include <unistd.h>
#اگر_ختم

#ہدایت clang diagnostic ignored "-Wgnu-anonymous-struct"
#ہدایت clang diagnostic ignored "-Wmissing-prototypes"
#ہدایت clang diagnostic ignored "-Wsign-compare"
#ہدایت clang diagnostic ignored "-Wlanguage-extension-token"
#ہدایت clang diagnostic ignored "-Wmicrosoft-enum-value"
#ہدایت clang diagnostic ignored "-Wnested-anon-types"

#شامل "ggml-impl.h"
#شامل "htp-drv.h"
#شامل "libdl.h"

#شامل <domain.h>

//
// Driver API types
//

قسمتعریف خالی * (*rpcmem_alloc_pfn_t)(صحیح heapid, uint32_t flags, صحیح size);
قسمتعریف خالی * (*rpcmem_alloc2_pfn_t)(صحیح heapid, uint32_t flags, size_t size);
قسمتعریف خالی   (*rpcmem_free_pfn_t)(خالی * po);
قسمتعریف صحیح    (*rpcmem_مقصد_fd_pfn_t)(خالی * po);

قسمتعریف AEEResult (*dspqueue_create_pfn_t)(صحیح                 domain,
                                           uint32_t            flags,
                                           uint32_t            req_قطار_سائز,
                                           uint32_t            resp_قطار_سائز,
                                           dspqueue_callback_t packet_callback,
                                           dspqueue_callback_t error_callback,
                                           خالی *              callback_context,
                                           dspqueue_t *        queue);
قسمتعریف AEEResult (*dspqueue_close_pfn_t)(dspqueue_t queue);
قسمتعریف AEEResult (*dspqueue_export_pfn_t)(dspqueue_t queue, uint64_t *قطار_id);
قسمتعریف AEEResult (*dspqueue_write_pfn_t)(dspqueue_t queue, uint32_t flags,
                                          uint32_t عدد_buffers,
                                          ساخت dspqueue_بفر *buffers,
                                          uint32_t پیغام_لمبائی,
                                          ثابت uint8_t *پیغام,
                                          uint32_t timeout_us);
قسمتعریف AEEResult (*dspqueue_read_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                         uint32_t max_buffers, uint32_t *عدد_buffers,
                                         ساخت dspqueue_بفر *buffers,
                                         uint32_t max_پیغام_لمبائی,
                                         uint32_t *پیغام_لمبائی, uint8_t *پیغام,
                                         uint32_t timeout_us);
قسمتعریف AEEResult (*dspqueue_read_noblock_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                                 uint32_t max_buffers, uint32_t *عدد_buffers,
                                                 ساخت dspqueue_بفر *buffers,
                                                 uint32_t max_پیغام_لمبائی,
                                                 uint32_t *پیغام_لمبائی, uint8_t *پیغام);
قسمتعریف صحیح (*fastrpc_mmap_pfn_t)(صحیح domain, صحیح fd, خالی *addr, صحیح انحراف, size_t لمبائی, شمار fastrpc_map_flags flags);
قسمتعریف صحیح (*fastrpc_munmap_pfn_t)(صحیح domain, صحیح fd, خالی *addr, size_t لمبائی);

قسمتعریف صحیح (*remote_handle64_open_pfn_t)(ثابت حرف* name, remote_handle64 *ph);
قسمتعریف صحیح (*remote_handle64_invoke_pfn_t)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra);
قسمتعریف صحیح (*remote_handle64_close_pfn_t)(remote_handle h);
قسمتعریف صحیح (*remote_handle_control_pfn_t)(uint32_t req, خالی* ڈیٹا, uint32_t datalen);
قسمتعریف صحیح (*remote_handle64_control_pfn_t)(remote_handle64 h, uint32_t req, خالی* ڈیٹا, uint32_t datalen);
قسمتعریف صحیح (*remote_session_control_pfn_t)(uint32_t req, خالی *ڈیٹا, uint32_t datalen);
قسمتعریف صحیح (*remote_system_request_pfn_t)(system_req_payload * req);

//
// Driver API pfns
//

rpcmem_alloc_pfn_t  rpcmem_alloc_pfn  = خالی_اشاریہ;
rpcmem_alloc2_pfn_t rpcmem_alloc2_pfn = خالی_اشاریہ;
rpcmem_free_pfn_t   rpcmem_free_pfn   = خالی_اشاریہ;
rpcmem_مقصد_fd_pfn_t  rpcmem_مقصد_fd_pfn  = خالی_اشاریہ;

fastrpc_mmap_pfn_t   fastrpc_mmap_pfn   = خالی_اشاریہ;
fastrpc_munmap_pfn_t fastrpc_munmap_pfn = خالی_اشاریہ;

dspqueue_create_pfn_t       dspqueue_create_pfn       = خالی_اشاریہ;
dspqueue_close_pfn_t        dspqueue_close_pfn        = خالی_اشاریہ;
dspqueue_export_pfn_t       dspqueue_export_pfn       = خالی_اشاریہ;
dspqueue_write_pfn_t        dspqueue_write_pfn        = خالی_اشاریہ;
dspqueue_read_pfn_t         dspqueue_read_pfn         = خالی_اشاریہ;
dspqueue_read_noblock_pfn_t dspqueue_read_noblock_pfn = خالی_اشاریہ;

remote_handle64_open_pfn_t    remote_handle64_open_pfn    = خالی_اشاریہ;
remote_handle64_invoke_pfn_t  remote_handle64_invoke_pfn  = خالی_اشاریہ;
remote_handle64_close_pfn_t   remote_handle64_close_pfn   = خالی_اشاریہ;
remote_handle_control_pfn_t   remote_handle_control_pfn   = خالی_اشاریہ;
remote_handle64_control_pfn_t remote_handle64_control_pfn = خالی_اشاریہ;
remote_session_control_pfn_t  remote_session_control_pfn  = خالی_اشاریہ;
remote_system_request_pfn_t   remote_system_request_pfn   = خالی_اشاریہ;

//
// Driver API
//

خالی * rpcmem_alloc(صحیح heapid, uint32_t flags, صحیح size) {
    واپس rpcmem_alloc_pfn(heapid, flags, size);
}

خالی * rpcmem_alloc2(صحیح heapid, uint32_t flags, size_t size) {
    اگر (rpcmem_alloc2_pfn) {
        واپس rpcmem_alloc2_pfn(heapid, flags, size);
    } ورنہ {
        GGML_LOG_INFO("ggml-hex: rpcmem_alloc2 not found, falling back to rpcmem_alloc\n");
        واپس rpcmem_alloc_pfn(heapid, flags, size);
    }
}

خالی rpcmem_free(خالی * po) {
    واپس rpcmem_free_pfn(po);
}

صحیح rpcmem_مقصد_fd(خالی * po) {
    واپس rpcmem_مقصد_fd_pfn(po);
}

HTPDRV_API صحیح fastrpc_mmap(صحیح domain, صحیح fd, خالی * addr, صحیح انحراف, size_t لمبائی, شمار fastrpc_map_flags flags) {
    واپس fastrpc_mmap_pfn(domain, fd, addr, انحراف, لمبائی, flags);
}

HTPDRV_API صحیح fastrpc_munmap(صحیح domain, صحیح fd, خالی * addr, size_t لمبائی) {
    واپس fastrpc_munmap_pfn(domain, fd, addr, لمبائی);
}

AEEResult dspqueue_create(صحیح                 domain,
                          uint32_t            flags,
                          uint32_t            req_قطار_سائز,
                          uint32_t            resp_قطار_سائز,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          خالی *              callback_context,
                          dspqueue_t *        queue) {
    واپس dspqueue_create_pfn(domain, flags, req_قطار_سائز, resp_قطار_سائز, packet_callback, error_callback,
                               callback_context, queue);
}

AEEResult dspqueue_close(dspqueue_t queue) {
    واپس dspqueue_close_pfn(queue);
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * قطار_id) {
    واپس dspqueue_export_pfn(queue, قطار_id);
}

AEEResult dspqueue_write(dspqueue_t               queue,
                         uint32_t                 flags,
                         uint32_t                 عدد_buffers,
                         ساخت dspqueue_بفر * buffers,
                         uint32_t                 پیغام_لمبائی,
                         ثابت uint8_t *          پیغام,
                         uint32_t                 timeout_us) {
    واپس dspqueue_write_pfn(queue, flags, عدد_buffers, buffers, پیغام_لمبائی, پیغام, timeout_us);
}

AEEResult dspqueue_read(dspqueue_t               queue,
                        uint32_t *               flags,
                        uint32_t                 max_buffers,
                        uint32_t *               عدد_buffers,
                        ساخت dspqueue_بفر * buffers,
                        uint32_t                 max_پیغام_لمبائی,
                        uint32_t *               پیغام_لمبائی,
                        uint8_t *                پیغام,
                        uint32_t                 timeout_us) {
#اگر_تعریف_شدہ _WIN32
    اگر (timeout_us == 0) {
        واپس dspqueue_read_noblock_pfn(queue, flags, max_buffers, عدد_buffers, buffers, max_پیغام_لمبائی,
                                     پیغام_لمبائی, پیغام);
    }
#اگر_ختم
    واپس dspqueue_read_pfn(queue, flags, max_buffers, عدد_buffers, buffers, max_پیغام_لمبائی, پیغام_لمبائی,
                             پیغام, timeout_us);
}

HTPDRV_API صحیح remote_handle64_open(ثابت حرف * name, remote_handle64 * ph) {
    واپس remote_handle64_open_pfn(name, ph);
}

HTPDRV_API صحیح remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg * pra) {
    واپس remote_handle64_invoke_pfn(h, dwScalars, pra);
}

HTPDRV_API صحیح remote_handle64_close(remote_handle64 h) {
    واپس remote_handle64_close_pfn(h);
}

HTPDRV_API صحیح remote_handle_control(uint32_t req, خالی * ڈیٹا, uint32_t datalen) {
    واپس remote_handle_control_pfn(req, ڈیٹا, datalen);
}

HTPDRV_API صحیح remote_handle64_control(remote_handle64 h, uint32_t req, خالی * ڈیٹا, uint32_t datalen) {
    واپس remote_handle64_control_pfn(h, req, ڈیٹا, datalen);
}

HTPDRV_API صحیح remote_session_control(uint32_t req, خالی * ڈیٹا, uint32_t datalen) {
    واپس remote_session_control_pfn(req, ڈیٹا, datalen);
}

HTPDRV_API صحیح remote_system_request(system_req_payload * req) {
    اگر (!remote_system_request_pfn) {
        واپس AEE_EUNSUPPORTEDAPI;
    }
    واپس remote_system_request_pfn(req);
}

#اگر_تعریف_شدہ _WIN32

ساکن std::string wstr_مقصد_str(std::wstring_view wstr) {
    std::string نتیجہ;
    اگر (wstr.خالی_خانے()) {
        واپس نتیجہ;
    }
    خودکار بائٹیں_needed = Wide_Char_مقصد_Multi_Byte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.ڈیٹا(), (صحیح) wstr.size(),
                                            خالی_اشاریہ, 0, خالی_اشاریہ, خالی_اشاریہ);
    اگر (بائٹیں_needed == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", Get_آخری_Error());
        پھینکو std::runtime_error("Invalid wstring input");
    }

    نتیجہ.resize(بائٹیں_needed, '\0');
    صحیح بائٹیں_written = Wide_Char_مقصد_Multi_Byte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.ڈیٹا(), (صحیح) wstr.size(),
                                            نتیجہ.ڈیٹا(), بائٹیں_needed,
                                            خالی_اشاریہ, خالی_اشاریہ);
    اگر (بائٹیں_written == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", Get_آخری_Error());
        پھینکو std::runtime_error("Wstring conversion failed");
    }
    واپس نتیجہ;
}

ساکن std::string get_driver_راستہ() {
    std::wstring serviceName = L"qcnspmcdm";
    std::string نتیجہ;

    // Get a handle to the SCM database.
    SC_HANDLE schSCManager = OpenSCManagerW(تہی, تہی, STANDARD_RIGHTS_READ);
    اگر (خالی_اشاریہ == schSCManager) {
        GGML_LOG_ERROR("ggml-hex: Failed to open SCManager. Error: %lu\n", Get_آخری_Error());
        واپس نتیجہ;
    }

    // Get a handle to the service.
    SC_HANDLE schService = OpenServiceW(schSCManager,           // SCM database
                                        serviceName.c_str(),    // name of service
                                        SERVICE_QUERY_CONFIG);  // need query config access

    اگر (خالی_اشاریہ == schService) {
        GGML_LOG_ERROR("ggml-hex: Failed to open qcnspmcdm service. Error: %lu\n", Get_آخری_Error());
        CloseServiceHandle(schSCManager);
        واپس نتیجہ;
    }

    // Store the size of buffer used as an output.
    DWORD بفر_سائز_2;
    اگر (!QueryServiceConfigW(schService, تہی, 0, &بفر_سائز_2) &&
        (Get_آخری_Error() != ERROR_INSUFFICIENT_بفر)) {
        GGML_LOG_ERROR("ggml-hex: Failed to query service config. Error: %lu\n", Get_آخری_Error());
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        واپس نتیجہ;
    }
    // Get the configuration of the service.
    LPQUERY_SERVICE_CONFIGW serviceConfig =
        static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, بفر_سائز_2));
    اگر (!QueryServiceConfigW(schService, serviceConfig, بفر_سائز_2, &بفر_سائز_2)) {
        fprintf(stderr, "ggml-hex: Failed to query service config. Error: %lu\n", Get_آخری_Error());
        LocalFree(serviceConfig);
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        واپس نتیجہ;
    }

    // Read the driver file path get its parent directory
    std::wstring driver_راستہ = std::wstring(serviceConfig->lp_ثنائی_راستہ_Name);
    driver_راستہ = driver_راستہ.substr(0, driver_راستہ.find_last_of(L"\\"));

    // Clean up resources
    LocalFree(serviceConfig);
    CloseServiceHandle(schService);
    CloseServiceHandle(schSCManager);

    // Driver path would contain invalid path string, like:
    // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
    // "\SystemRoot" should be replace with a correct one (e.g. C:\Windows)
    ثابت std::wstring system_جڑ_Placeholder = L"\\SystemRoot";
    اگر (0 != driver_راستہ.compare(0, system_جڑ_Placeholder.لمبائی(), system_جڑ_Placeholder)) {
        GGML_LOG_ERROR("ggml-hex: String pattern not found in driver path.\n");
        واپس نتیجہ;
    }

    // Replace \SystemRoot with an absolute path from system ENV windir
    ثابت std::wstring system_جڑ_Env = L"windir";

    // Query the number of wide characters this variable requires
    DWORD عدد_الفاظ = GetEnvironmentVariableW(system_جڑ_Env.c_str(), تہی, 0);
    اگر (عدد_الفاظ == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed get systemRoot environment variable\n");
        واپس نتیجہ;
    }

    // Query the actual system root name from environment variable
    std::vector<wchar_t> system_جڑ(عدد_الفاظ + 1);
    عدد_الفاظ = GetEnvironmentVariableW(system_جڑ_Env.c_str(), system_جڑ.ڈیٹا(), عدد_الفاظ + 1);
    اگر (عدد_الفاظ == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed to read windir environment variable\n");
        واپس نتیجہ;
    }
    driver_راستہ.replace(0, system_جڑ_Placeholder.لمبائی(), std::wstring(system_جڑ.ڈیٹا()));

    واپس wstr_مقصد_str(driver_راستہ);
}

#اگر_ختم

استعمال dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

صحیح htpdrv_init() {
    ساکن dl_handle_ptr lib_cdsp_rpc_handle = خالی_اشاریہ;
    ساکن منطقی initialized = جھوٹ;
#اگر_تعریف_شدہ _WIN32
    std::string drv_راستہ = get_driver_راستہ() + "\\" + "libcdsprpc.dll";
#ورنہ
    std::string drv_راستہ = "libcdsprpc.so";
#اگر_ختم
    اگر (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        واپس AEE_SUCCESS;
    }
    GGML_LOG_INFO("ggml-hex: Loading driver %s\n", drv_راستہ.c_str());

    fs::path path{ drv_راستہ.c_str() };
    dl_handle_ptr handle { dl_بھرو_library(path) };
    اگر (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", path.u8string().c_str(), dl_error());
        واپس AEE_EUNABLETOLOAD;
    }

#تعریف_کرو dlsym(drv, type, pfn, علامت, ignore)                               \
    کرو {                                                                    \
        pfn = (type) dl_get_sym(drv, #symbol);                              \
        اگر (!ignore && خالی_اشاریہ == pfn) {                                    \
            GGML_LOG_ERROR("ggml-hex: failed to dlsym %s\n", #symbol);      \
            واپس AEE_EUNABLETOLOAD;                                       \
        }                                                                   \
    } جب (0)

    dlsym(handle.get(), rpcmem_alloc_pfn_t, rpcmem_alloc_pfn, rpcmem_alloc, جھوٹ);
    dlsym(handle.get(), rpcmem_alloc2_pfn_t, rpcmem_alloc2_pfn, rpcmem_alloc2, سچ);
    dlsym(handle.get(), rpcmem_free_pfn_t, rpcmem_free_pfn, rpcmem_free, جھوٹ);
    dlsym(handle.get(), rpcmem_مقصد_fd_pfn_t, rpcmem_مقصد_fd_pfn, rpcmem_مقصد_fd, جھوٹ);
    dlsym(handle.get(), fastrpc_mmap_pfn_t, fastrpc_mmap_pfn, fastrpc_mmap, جھوٹ);
    dlsym(handle.get(), fastrpc_munmap_pfn_t, fastrpc_munmap_pfn, fastrpc_munmap, جھوٹ);
    dlsym(handle.get(), dspqueue_create_pfn_t, dspqueue_create_pfn, dspqueue_create, جھوٹ);
    dlsym(handle.get(), dspqueue_close_pfn_t, dspqueue_close_pfn, dspqueue_close, جھوٹ);
    dlsym(handle.get(), dspqueue_export_pfn_t, dspqueue_export_pfn, dspqueue_export, جھوٹ);
    dlsym(handle.get(), dspqueue_write_pfn_t, dspqueue_write_pfn, dspqueue_write, جھوٹ);
    dlsym(handle.get(), dspqueue_read_pfn_t, dspqueue_read_pfn, dspqueue_read, جھوٹ);
    dlsym(handle.get(), dspqueue_read_noblock_pfn_t, dspqueue_read_noblock_pfn, dspqueue_read_noblock, جھوٹ);
    dlsym(handle.get(), remote_handle64_open_pfn_t, remote_handle64_open_pfn, remote_handle64_open, جھوٹ);
    dlsym(handle.get(), remote_handle64_invoke_pfn_t, remote_handle64_invoke_pfn, remote_handle64_invoke, جھوٹ);
    dlsym(handle.get(), remote_handle_control_pfn_t, remote_handle_control_pfn, remote_handle_control, جھوٹ);
    dlsym(handle.get(), remote_handle64_control_pfn_t, remote_handle64_control_pfn, remote_handle64_control, جھوٹ);
    dlsym(handle.get(), remote_session_control_pfn_t, remote_session_control_pfn, remote_session_control, جھوٹ);
    dlsym(handle.get(), remote_handle64_close_pfn_t, remote_handle64_close_pfn, remote_handle64_close, جھوٹ);
    dlsym(handle.get(), remote_system_request_pfn_t, remote_system_request_pfn, remote_system_request, سچ);

    lib_cdsp_rpc_handle = std::move(handle);
    initialized         = سچ;

    واپس AEE_SUCCESS;
}

domain * htpdrv_get_domain(صحیح domain_id) {
    صحیح i    = 0;
    صحیح size = حجم(supported_domains) / حجم(domain);

    برائے (i = 0; i < size; i++) {
        اگر (supported_domains[i].id == domain_id) {
            واپس &supported_domains[i];
        }
    }

    واپس تہی;
}

صحیح htpdrv_get_arch(صحیح domain, صحیح * arch) {
    اگر (!remote_handle_control_pfn) {
        GGML_LOG_ERROR("ggml-hex: remote_handle_control is not supported on this device\n");
        واپس AEE_EUNSUPPORTEDAPI;
    }

    ساخت remote_dsp_capability arch_ver;
    arch_ver.domain       = (uint32_t) domain;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability   = (uint32_t) 0;

    صحیح err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, حجم(arch_ver));
    اگر ((err & 0xff) == (AEE_EUNSUPPORTEDAPI & 0xff)) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability API is not supported on this device\n");
        واپس AEE_EUNSUPPORTEDAPI;
    }

    اگر (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability query failed (err %d)\n", err);
        واپس err;
    }

    uint32_t val = arch_ver.capability & 0xff;
    *arch = (صحیح) ((val >> 4) * 10 + (val & 0x0f));
    واپس 0;
}

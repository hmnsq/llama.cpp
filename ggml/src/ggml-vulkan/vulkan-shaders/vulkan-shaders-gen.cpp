#شامل <iostream>
#شامل <fstream>
#شامل <sstream>
#شامل <string>
#شامل <stdexcept>
#شامل <array>
#شامل <vector>
#شامل <map>
#شامل <thread>
#شامل <mutex>
#شامل <future>
#شامل <queue>
#شامل <condition_variable>
#شامل <atomic>
#شامل <cstdio>
#شامل <cstring>
#شامل <cstdlib>
#شامل <cassert>
#شامل <algorithm>
#شامل <sys/stat.h>
#شامل <sys/types.h>
#شامل <filesystem>

#اگر_تعریف_شدہ _WIN32
    #تعریف_کرو NOMINMAX
    #شامل <windows.h>
    #شامل <direct.h> // For _mkdir on Windows
#ورنہ
    #شامل <unistd.h>
    #شامل <sys/wait.h>
    #شامل <fcntl.h>
#اگر_ختم

#تعریف_کرو ASYNCIO_CONCURRENCY 64

std::mutex تالا;
std::vector<std::pair<std::string, std::string>> shader_fnames;
// Set when any shader subprocess fails (non-zero exit / stderr / launch failure) so the
// build is stopped instead of silently producing a broken libggml-vulkan. (issue #24393)
ساکن std::atomic<منطقی> compile_failed{جھوٹ};
std::locale c_locale("C");

std::string GLSLC = "glslc";
std::string داخلہ_filepath = "";
std::string output_dir = "/tmp";
std::string ہدف_hpp = "";
std::string ہدف_cpp = "";

ثابت std::vector<std::string> type_نام = {
    "f32",
    "f16",
    "q1_0",
    "q2_0",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "iq1_s",
    "iq1_m",
    "iq2_xxs",
    "iq2_xs",
    "iq2_s",
    "iq3_xxs",
    "iq3_s",
    "iq4_xs",
    "iq4_nl",
    "mxfp4",
    "nvfp4",
    "tq1_0",
    "tq2_0",
    "bf16",
};

شمار MatMulIdType {
    NONE,
    DEFAULT,
    SUBGROUP,
};

نامخلا {

صحیح execute_command(std::vector<std::string>& command, std::string& stdout_str, std::string& stderr_str) {
#اگر_تعریف_شدہ _WIN32
    HANDLE stdout_read, stdout_write;
    HANDLE stderr_read, stderr_write;
    SECURITY_اوصاف sa = { حجم(SECURITY_اوصاف), تہی, TRUE };

    اگر (!CreatePipe(&stdout_read, &stdout_write, &sa, 0) ||
        !سیٹ_Handle_Information(stdout_read, HANDLE_FLAG_INHERIT, 0)) {
        پھینکو std::runtime_error("Failed to create stdout pipe");
    }

    اگر (!CreatePipe(&stderr_read, &stderr_write, &sa, 0) ||
        !سیٹ_Handle_Information(stderr_read, HANDLE_FLAG_INHERIT, 0)) {
        پھینکو std::runtime_error("Failed to create stderr pipe");
    }

    PROCESS_INFORMATION pi;
    STARTUPINFOA si = {};
    si.cb = حجم(STARTUPINFOA);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = stdout_write;
    si.hStdError = stderr_write;

    std::string cmd;
    برائے (ثابت خودکار& part : command) {
        cmd += part + " ";
    }

    اگر (!CreateProcessA(تہی, cmd.ڈیٹا(), تہی, تہی, TRUE, 0, تہی, تہی, &si, &pi)) {
        پھینکو std::runtime_error("Failed to create process");
    }

    CloseHandle(stdout_write);
    CloseHandle(stderr_write);

    std::array<حرف, 128> بفر;
    DWORD بائٹیں_read;

    جب (ReadFile(stdout_read, بفر.ڈیٹا(), (DWORD)بفر.size(), &بائٹیں_read, تہی) && بائٹیں_read > 0) {
        stdout_str.append(بفر.ڈیٹا(), بائٹیں_read);
    }

    جب (ReadFile(stderr_read, بفر.ڈیٹا(), (DWORD)بفر.size(), &بائٹیں_read, تہی) && بائٹیں_read > 0) {
        stderr_str.append(بفر.ڈیٹا(), بائٹیں_read);
    }

    CloseHandle(stdout_read);
    CloseHandle(stderr_read);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_رمز = 1;
    Get_Exit_رمز_Process(pi.hProcess, &exit_رمز);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    واپس (صحیح)exit_رمز;
#ورنہ
    صحیح stdout_pipe[2];
    صحیح stderr_pipe[2];

    اگر (أنبوب(stdout_pipe) != 0 || أنبوب(stderr_pipe) != 0) {
        پھینکو std::runtime_error("Failed to create pipes");
    }

    pid_t pid = fork();
    اگر (pid < 0) {
        std::cerr << strerror(errno) << "\n";
        پھینکو std::runtime_error("Failed to fork process");
    }

    std::vector<حرف*> argv;
    برائے (std::string& part : command) {
        argv.push_back(part.ڈیٹا());
    }
    argv.push_back(خالی_اشاریہ);

    اگر (pid == 0) {
        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);
        execvp(argv[0], argv.ڈیٹا());
        _exit(EXIT_FAILURE);
    } ورنہ {
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        std::array<حرف, 128> بفر;
        ssize_t بائٹیں_read;

        جب ((بائٹیں_read = اقرأ(stdout_pipe[0], بفر.ڈیٹا(), بفر.size())) > 0) {
            stdout_str.append(بفر.ڈیٹا(), بائٹیں_read);
        }

        جب ((بائٹیں_read = اقرأ(stderr_pipe[0], بفر.ڈیٹا(), بفر.size())) > 0) {
            stderr_str.append(بفر.ڈیٹا(), بائٹیں_read);
        }

        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        صحیح حالت_2 = 0;
        waitpid(pid, &حالت_2, 0);
        واپس WIFEXITED(حالت_2) ? WEXITSTATUS(حالت_2) : -1;
    }
#اگر_ختم
}

منطقی directory_exists(ثابت std::string& path) {
    ساخت stat info;
    اگر (stat(path.c_str(), &info) != 0) {
        واپس جھوٹ; // Path doesn't exist or can't be accessed
    }
    واپس (info.st_mode & S_IFDIR) != 0; // Check if it is a directory
}

منطقی create_directory(ثابت std::string& path) {
#اگر_تعریف_شدہ _WIN32
    واپس _mkdir(path.c_str()) == 0 || errno == EEXIST; // EEXIST means the directory already exists
#ورنہ
    واپس mkdir(path.c_str(), 0755) == 0 || errno == EEXIST; // 0755 is the directory permissions
#اگر_ختم
}

std::string مقصد_uppercase(ثابت std::string& داخلہ) {
    std::string نتیجہ = داخلہ;
    برائے (حرف& c : نتیجہ) {
        c = std::بڑا_حرف(c);
    }
    واپس نتیجہ;
}

منطقی string_starts_with(ثابت std::string& str, ثابت std::string& prefix) {
    اگر (prefix.size() > str.size()) {
        واپس جھوٹ;
    }
    واپس std::equal(prefix.begin(), prefix.end(), str.begin());
}

منطقی string_سرے_with(ثابت std::string& str, ثابت std::string& suffix) {
    اگر (suffix.size() > str.size()) {
        واپس جھوٹ;
    }
    واپس std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

منطقی is_quantized_type(ثابت std::string& type_name) {
    واپس type_name != "f32" && type_name != "f16" && type_name != "bf16";
}

منطقی is_legacy_quant(ثابت std::string& type_name) {
    واپس type_name == "q2_0" || type_name == "q4_0" || type_name == "q4_1" || type_name == "q5_0" || type_name == "q5_1" || type_name == "q8_0";
}

منطقی is_k_quant(ثابت std::string& type_name) {
    واپس string_سرے_with(type_name, "_k");
}

منطقی is_iq_quant(ثابت std::string& type_name) {
    واپس string_starts_with(type_name, "iq");
}

منطقی is_lut_quant(ثابت std::string& type_name) {
    واپس is_iq_quant(type_name) || type_name == "mxfp4" || type_name == "nvfp4";
}

std::string lut_بھرو_vec_a(ثابت std::string& type_name) {
    اگر (type_name == "iq1_s" || type_name == "iq1_m" || type_name == "iq2_xxs" || type_name == "iq2_xs" || type_name == "iq2_s" || type_name == "iq4_xs") {
        واپس "8";
    }
    واپس "4";
}

ساکن ثابت حرف راستہ_separator = '/';

std::string join_paths(ثابت std::string& path1, ثابت std::string& path2) {
    واپس path1 + راستہ_separator + path2;
}

std::string basename(ثابت std::string &path) {
    واپس path.substr(path.find_last_of("/\\") + 1);
}

std::stringstream make_generic_stringstream() {
    std::stringstream ss;
    ss.imbue(c_locale);
    واپس ss;
}

std::string read_ثنائی_file(ثابت std::string& path, منطقی may_not_exist = جھوٹ) {
    FILE* f = fopen(path.c_str(), "rb");
    اگر (!f) {
        اگر (!may_not_exist) {
            std::cerr << "Error opening file: " << path << " (" << strerror(errno) << ")\n";
        }
        واپس {};
    }

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::string ڈیٹا(size, '\0');
    size_t read_سائز = fread(ڈیٹا.ڈیٹا(), 1, size, f);
    fclose(f);
    اگر (read_سائز != size) {
        std::cerr << "Error reading file: " << path << " (" << strerror(errno) << ")\n";
        واپس {};
    }

    واپس ڈیٹا;
}

خالی write_ثنائی_file(ثابت std::string& path, ثابت std::string& content) {
    FILE* f = fopen(path.c_str(), "wb");
    اگر (!f) {
        std::cerr << "Error opening file for writing: " << path << " (" << strerror(errno) << ")\n";
        واپس;
    }

    size_t write_سائز = fwrite(content.ڈیٹا(), 1, content.size(), f);
    fclose(f);
    اگر (write_سائز != content.size()) {
        std::cerr << "Error writing file: " << path << " (" << strerror(errno) << ")\n";
        واپس;
    }
}

خالی write_file_if_بدلا(ثابت std::string& path, ثابت std::string& content) {
    std::string existing = read_ثنائی_file(path, سچ);
    اگر (existing != content) {
        write_ثنائی_file(path, content);
    }
}


// variables to track number of compiles in progress
ساکن uint32_t compile_گنتی = 0;
ساکن std::mutex compile_گنتی_mutex;
ساکن std::condition_variable compile_گنتی_cond;
ساکن منطقی generate_dep_file = سچ;

خالی decrement_compile_گنتی(uint32_t * count) {
    اگر (count) {
        std::lock_guard<std::mutex> guard(compile_گنتی_mutex);
        أكد(compile_گنتی > 0);
        compile_گنتی--;
        compile_گنتی_cond.notify_all();
    }
}

استعمال compile_گنتی_guard = std::unique_ptr<uint32_t, decltype(&decrement_compile_گنتی)>;

compile_گنتی_guard acquire_compile_خانہ() {
    // wait until fewer than N compiles are in progress.
    // 16 is an arbitrary limit, the goal is to avoid "failed to create pipe" errors.
    uint32_t N = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::unique_lock<std::mutex> guard(compile_گنتی_mutex);
    compile_گنتی_cond.wait(guard, [N] { واپس compile_گنتی < N; });
    compile_گنتی++;
    واپس compile_گنتی_guard(&compile_گنتی, &decrement_compile_گنتی);
}

خالی string_مقصد_spv_func(std::string name, std::string اندر_راستہ, std::string برآمد_راستہ, std::map<std::string, std::string> متعین_کرتا, منطقی coopmat, منطقی dep_file, compile_گنتی_guard خانہ) {
    std::string ہدف_env = (name.find("_cm2") != std::string::npos) ? "--target-env=vulkan1.3" : "--target-env=vulkan1.2";

    #اگر_تعریف_شدہ _WIN32
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", ہدف_env, "\"" + اندر_راستہ + "\"", "-o", "\"" + برآمد_راستہ + "\""};
    #ورنہ
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", ہدف_env, اندر_راستہ, "-o", برآمد_راستہ};
    #اگر_ختم

    // disable spirv-opt for coopmat shaders for https://github.com/ggml-org/llama.cpp/issues/10734
    // disable spirv-opt for bf16 shaders for https://github.com/ggml-org/llama.cpp/issues/15344
    // disable spirv-opt for rope shaders for https://github.com/ggml-org/llama.cpp/issues/16860
    // disable spirv-opt for dot2 shaders (spirv-opt doesn't recognize SPV_VALVE_mixed_float_dot_product capability)
    اگر (!coopmat && name.find("bf16") == std::string::npos && name.find("rope") == std::string::npos && name.find("_dot2") == std::string::npos) {
        cmd.push_back("-O");
    }

    اگر (dep_file) {
        cmd.push_back("-MD");
        cmd.push_back("-MF");
#اگر_تعریف_شدہ _WIN32
        cmd.push_back("\"" + ہدف_cpp + ".d\"");
#ورنہ
        cmd.push_back(ہدف_cpp + ".d");
#اگر_ختم
    }

    #اگر_تعریف_شدہ GGML_VULKAN_SHADER_DEBUG_INFO
        cmd.push_back("-g");
    #اگر_ختم

    برائے (ثابت خودکار& define : متعین_کرتا) {
        cmd.push_back("-D" + define.پہلا + "=" + define.second);
    }

    std::string command;
    برائے (ثابت خودکار& part : cmd) {
        command += part + " ";
    }

    std::string stdout_str, stderr_str;
    کوشش {
        // std::cout << "Executing command: ";
        // for (const auto& part : cmd) {
        //     std::cout << part << " ";
        // }
        // std::cout << std::endl;

        صحیح exit_رمز = execute_command(cmd, stdout_str, stderr_str);
        اگر (exit_رمز != 0 || !stderr_str.خالی_خانے()) {
            std::cerr << "cannot compile " << name << " (exit code " << exit_رمز << ")\n\n";
            برائے (ثابت خودکار& part : cmd) {
                std::cerr << part << " ";
            }
            std::cerr << "\n\n" << stderr_str << std::سطر_ختم;
            compile_failed = سچ;
            واپس;
        }

        اگر (dep_file) {
            // replace .spv output path with the embed .cpp path which is used as output in CMakeLists.txt
            std::string dep = read_ثنائی_file(ہدف_cpp + ".d", سچ);
            اگر (!dep.خالی_خانے()) {
                size_t pos = dep.find(برآمد_راستہ);
                اگر (pos != std::string::npos) {
                    dep.replace(pos, برآمد_راستہ.لمبائی(), ہدف_cpp);
                }
                write_ثنائی_file(ہدف_cpp + ".d", dep);
            }
        }

        std::lock_guard<std::mutex> guard(تالا);
        shader_fnames.push_back(std::make_pair(name, برآمد_راستہ));
    } پکڑو (ثابت std::exception& e) {
        std::cerr << "Error executing command for " << name << ": " << e.what() << std::سطر_ختم;
        compile_failed = سچ;
    }
}

std::map<std::string, std::string> ملاؤ_maps(ثابت std::map<std::string, std::string>& a, ثابت std::map<std::string, std::string>& b) {
    std::map<std::string, std::string> نتیجہ = a;
    نتیجہ.insert(b.begin(), b.end());
    واپس نتیجہ;
}

ساکن std::deque<std::future<خالی>> compiles;
خالی string_مقصد_spv(std::string name, ثابت std::string& ماخذ, ثابت std::map<std::string, std::string>& متعین_کرتا, منطقی fp16 = سچ, منطقی coopmat = جھوٹ, منطقی coopmat2 = جھوٹ, منطقی f16acc = جھوٹ, ثابت std::string& suffix = "") {
    name = name + (f16acc ? "_f16acc" : "") + (coopmat ? "_cm1" : "") + (coopmat2 ? "_cm2" : (fp16 ? "" : "_fp32")) + suffix;
    std::string برآمد_راستہ = join_paths(output_dir, name + ".spv");

    اگر (داخلہ_filepath == "") {
        // No input source to compile, only generate header for all shaders
        shader_fnames.push_back(std::pair(name, برآمد_راستہ));
        واپس;
    } ورنہ اگر (basename(داخلہ_filepath) != ماخذ) {
        // Only compile shader variants matching the input filename
        واپس;
    }

    compile_گنتی_guard خانہ = acquire_compile_خانہ();
    compiles.push_back(std::async(
        string_مقصد_spv_func, name, داخلہ_filepath, برآمد_راستہ, متعین_کرتا, coopmat, generate_dep_file, std::move(خانہ)));
    // Don't write the same dep file from multiple processes
    generate_dep_file = جھوٹ;

    // Clean up completed futures - don't accumulate virtual memory for completed threads' stacks.
    جب (!compiles.خالی_خانے() && compiles.front().wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        compiles.pop_front();
    }
}

خالی matmul_shaders(منطقی fp16, MatMulIdType matmul_id_type, منطقی coopmat, منطقی coopmat2, منطقی f16acc, منطقی dot2 = جھوٹ) {
    std::string بھرو_vec = coopmat2 ? "1" : fp16 ? "8" : "4";
    std::string aligned_b_type_f32 = coopmat2 ? "float" : fp16 ? "mat2x4" : "vec4";
    std::string aligned_b_type_f16 = coopmat2 ? "float16_t" : fp16 ? "f16mat2x4" : "f16vec4";
    std::string dot2_sfx = dot2 ? "_dot2" : "";

    std::map<std::string, std::string> بنیاد_dict;
    std::string shader_name = "matmul";

    اگر (matmul_id_type == MatMulIdType::DEFAULT) {
        بنیاد_dict["MUL_MAT_ID"] = "1";
        shader_name = "matmul_id";
    } ورنہ اگر (matmul_id_type == MatMulIdType::SUBGROUP) {
        بنیاد_dict["MUL_MAT_ID"] = "1";
        بنیاد_dict["MUL_MAT_ID_USE_SUBGROUPS"] = "1";
        shader_name = "matmul_id_subgroup";
    }

    اگر (fp16) {
        بنیاد_dict["FLOAT16"] = "1";
    }

    بنیاد_dict["ACC_TYPE"     ] = f16acc ? "float16_t" : "float";
    بنیاد_dict["ACC_TYPEV2"   ] = f16acc ? "f16vec2"   : "vec2";
    بنیاد_dict["ACC_TYPE_VEC4"] = f16acc ? "f16vec4"   : "vec4";
    اگر (f16acc) {
        بنیاد_dict["ACC_TYPE_MAX"] = "float16_t(65504.0)";
        بنیاد_dict["ACC_F16"] = "1";
    }

    اگر (coopmat) {
        بنیاد_dict["COOPMAT"] = "1";
    }
#اگر defined(GGML_VULKAN_COOPMAT2_DECODE_سمتیہ_GLSLC_SUPPORT)
    اگر (coopmat2) {
        بنیاد_dict["GGML_VULKAN_COOPMAT2_DECODE_VECTOR"] = "1";
    }
#اگر_ختم

    اگر (dot2) {
        بنیاد_dict["DOT2_F16"] = "1";
    }

    ثابت std::string ماخذ_name = coopmat2 ? "mul_mm_cm2.comp" : "mul_mm.comp";

    خودکار ثابت &FLOAT_TYPE = [&](صحیح vec, ثابت std::string &t) -> std::string {
        انتخاب (vec) {
        حالت 1:
            اگر (t == "bf16") {
                // scalar path promotes to float
                اگر (!coopmat && !coopmat2) {
                    واپس "float";
                }
                واپس "bfloat16_t";
            }
            اگر (coopmat2 || fp16) {
                واپس "float16_t";
            }
            واپس "float";
        حالت 2:
            اگر (t == "bf16") {
                // scalar path promotes to float
                اگر (!coopmat && !coopmat2) {
                    واپس "vec2";
                }
                واپس "bf16vec2";
            }
            اگر (coopmat2 || fp16) {
                واپس "f16vec2";
            }
            واپس "vec2";
        حالت 4:
            اگر (t == "bf16") {
                // scalar path promotes to float
                اگر (!coopmat && !coopmat2) {
                    واپس "vec4";
                }
                واپس "bf16vec4";
            }
            اگر (coopmat2 || fp16) {
                واپس "f16vec4";
            }
            واپس "vec4";
        حالت 8:
            اگر (t == "bf16") {
                // scalar path promotes to float
                اگر (!coopmat && !coopmat2) {
                    واپس "mat2x4";
                }
                پھینکو std::runtime_error("bf16 vec8 not supported");
            }
            اگر (coopmat2 || fp16) {
                واپس "f16mat2x4";
            }
            واپس "mat2x4";
        طےشدہ:
            پھینکو std::runtime_error("invalid vector size");
        }
    };

    ثابت std::map<std::string, std::string> float_type_dict_f16 = {
        {"FLOAT_TYPE",   FLOAT_TYPE(1, "f16")},
        {"FLOAT_TYPEV2", FLOAT_TYPE(2, "f16")},
        {"FLOAT_TYPEV4", FLOAT_TYPE(4, "f16")},
        {"FLOAT_TYPEV8", FLOAT_TYPE(8, "f16")},
    };

    // Shaders with f16 B_TYPE
    string_مقصد_spv(shader_name + "_f32_f16" + dot2_sfx, ماخذ_name, ملاؤ_maps(ملاؤ_maps(بنیاد_dict, float_type_dict_f16), {{"DATA_A_F32", "1"}, {"LOAD_VEC_A", بھرو_vec}, {"LOAD_VEC_B", بھرو_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

    string_مقصد_spv(shader_name + "_f16" + dot2_sfx, ماخذ_name, ملاؤ_maps(ملاؤ_maps(بنیاد_dict, float_type_dict_f16), {{"DATA_A_F16", "1"}, {"LOAD_VEC_A", بھرو_vec}, {"LOAD_VEC_B", بھرو_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

    // bf16
    {
        // For aligned matmul loads
        std::string بھرو_vec_a = coopmat2 ? "1" : "4";

        // scalar path promotes to float
        std::string مقصد_float_type = (coopmat || coopmat2) ? "uintBitsToBFloat16EXT" : "bf16_to_fp32";

        ثابت std::map<std::string, std::string> float_type_dict_bf16 = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, "bf16")},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, "bf16")},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, "bf16")},
        };

        // If bfloat16 is not supported, then only compile the scalar (promote to fp32) shader
#اگر !defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        اگر (!(coopmat || coopmat2))
#اگر_ختم
        {
            اگر (!dot2) {
                string_مقصد_spv(shader_name + "_bf16", ماخذ_name, ملاؤ_maps(ملاؤ_maps(بنیاد_dict, float_type_dict_bf16), {{"TO_FLOAT_TYPE", مقصد_float_type}, {"DATA_A_BF16", "1"}, {"LOAD_VEC_A", بھرو_vec_a}, {"LOAD_VEC_B", "4"}, {"B_TYPE", coopmat2 ? "bfloat16_t" : "u16vec4"}, {"B_TYPE_SCALAR", coopmat2 ? "bfloat16_t" : "uint16_t"}, {"B_TYPEV4", "bf16vec4"}, {"D_TYPE", "float"}, {"B_IS_FLOAT", "1"}, {"DATA_B_BF16", "1"}}), fp16, coopmat, coopmat2, f16acc);
            }
        }
    }

    برائے (ثابت خودکار& tname : type_نام) {
        اگر (tname == "bf16") {
            جاری;
        }

        // Float types keep per-type compilation (different accumulation loop structure)
        اگر (tname == "f32" || tname == "f16") {
            std::string ڈیٹا_a_کلید = "DATA_A_" + مقصد_uppercase(tname);

            ثابت std::map<std::string, std::string> float_type_dict = {
                {"FLOAT_TYPE",   FLOAT_TYPE(1, tname)},
                {"FLOAT_TYPEV2", FLOAT_TYPE(2, tname)},
                {"FLOAT_TYPEV4", FLOAT_TYPE(4, tname)},
                {"FLOAT_TYPEV8", FLOAT_TYPE(8, tname)},
            };

            اگر (!coopmat2) {
                string_مقصد_spv(shader_name + "_" + tname + "_f32" + dot2_sfx, ماخذ_name, ملاؤ_maps(ملاؤ_maps(بنیاد_dict, float_type_dict), {{ڈیٹا_a_کلید, "1"}, {"LOAD_VEC_A", بھرو_vec}, {"LOAD_VEC_B", بھرو_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
            }
            جاری;
        }

        std::string ڈیٹا_a_کلید = "DATA_A_" + مقصد_uppercase(tname);
        ثابت std::map<std::string, std::string> float_type_dict = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, tname)},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, tname)},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, tname)},
            {"FLOAT_TYPEV8", FLOAT_TYPE(8, tname)},
        };

#اگر defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        اگر (!f16acc && !coopmat && !coopmat2 && !dot2 && (is_legacy_quant(tname) || is_k_quant(tname) || tname == "mxfp4" || tname == "iq3_s" || tname == "iq4_xs")) {
            string_مقصد_spv(shader_name + "_" + tname + "_q8_1", "mul_mmq.comp", ملاؤ_maps(ملاؤ_maps(بنیاد_dict, float_type_dict), {{ڈیٹا_a_کلید, "1"}, {"D_TYPE", "float"},}), fp16, coopmat, coopmat2, f16acc);
        }
#اگر_ختم

        اگر (!f16acc && coopmat && (tname == "q4_0" || tname == "q4_1" || tname == "q5_0" || tname == "q5_1" || tname == "q8_0" || tname == "iq4_nl" || tname == "iq4_xs" || tname == "mxfp4"
                     || tname == "q3_k" || tname == "q4_k" || tname == "q5_k" || tname == "q6_k" || tname == "nvfp4")) {
            string_مقصد_spv(shader_name + "_" + tname + "_q8_1", "mul_mmq_cm1.comp", ملاؤ_maps(ملاؤ_maps(بنیاد_dict, float_type_dict), {{ڈیٹا_a_کلید, "1"}, {"D_TYPE", "float"}, {"D_TYPE_VEC4", "vec4"}}), fp16, coopmat, coopmat2, f16acc);
        }

        اگر (is_lut_quant(tname)) {
            std::string lva = lut_بھرو_vec_a(tname);

            string_مقصد_spv(shader_name + "_" + tname + "_f16" + dot2_sfx, ماخذ_name, ملاؤ_maps(ملاؤ_maps(بنیاد_dict, float_type_dict), {{ڈیٹا_a_کلید, "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", بھرو_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

            اگر (!coopmat2) {
                string_مقصد_spv(shader_name + "_" + tname + "_f32" + dot2_sfx, ماخذ_name, ملاؤ_maps(ملاؤ_maps(بنیاد_dict, float_type_dict), {{ڈیٹا_a_کلید, "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", بھرو_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
            }

#اگر defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
            اگر ((tname == "mxfp4" || tname == "nvfp4") && (coopmat || coopmat2)) {
                string_مقصد_spv(shader_name + "_" + tname + "_f16_ocp" + dot2_sfx, ماخذ_name, ملاؤ_maps(ملاؤ_maps(بنیاد_dict, float_type_dict), {{ڈیٹا_a_کلید, "1"}, {"USE_OCP_FP4", "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", بھرو_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
                اگر (!coopmat2) {
                    string_مقصد_spv(shader_name + "_" + tname + "_f32_ocp" + dot2_sfx, ماخذ_name, ملاؤ_maps(ملاؤ_maps(بنیاد_dict, float_type_dict), {{ڈیٹا_a_کلید, "1"}, {"USE_OCP_FP4", "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", بھرو_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
                }
            }
#اگر_ختم
            جاری;
        }

        // dedicated shader needed due to regression on Ampere
        اگر (coopmat2 && (tname == "q4_k" || tname == "q5_k")) {
            string_مقصد_spv(shader_name + "_" + tname + "_f16" + dot2_sfx, ماخذ_name, ملاؤ_maps(ملاؤ_maps(بنیاد_dict, float_type_dict), {{ڈیٹا_a_کلید, "1"}, {"LOAD_VEC_A", بھرو_vec}, {"LOAD_VEC_B", بھرو_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
        }
    }

    // Quant shader: one SPIR-V for all quant types, selected via MmTypeA spec constant
    {
        ثابت std::map<std::string, std::string> quant_float_type_dict = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, "q4_0")},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, "q4_0")},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, "q4_0")},
            {"FLOAT_TYPEV8", FLOAT_TYPE(8, "q4_0")},
        };

        string_مقصد_spv(shader_name + "_quant_f16" + dot2_sfx, ماخذ_name, ملاؤ_maps(ملاؤ_maps(بنیاد_dict, quant_float_type_dict), {{"MULMAT_QUANT", "1"}, {"LOAD_VEC_B", بھرو_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

        اگر (!coopmat2) {
            string_مقصد_spv(shader_name + "_quant_f32" + dot2_sfx, ماخذ_name, ملاؤ_maps(ملاؤ_maps(بنیاد_dict, quant_float_type_dict), {{"MULMAT_QUANT", "1"}, {"LOAD_VEC_B", بھرو_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
        }
    }
}

خالی process_shaders() {
    // matmul
    برائے (ثابت MatMulIdType& matmul_id_type : {MatMulIdType::NONE, MatMulIdType::DEFAULT, MatMulIdType::SUBGROUP}) {
        // No coopmats
        // fp32
        matmul_shaders(جھوٹ, matmul_id_type, جھوٹ, جھوٹ, جھوٹ);

        // fp16, fp32acc and fp16acc
        matmul_shaders(سچ, matmul_id_type, جھوٹ, جھوٹ, جھوٹ);
        matmul_shaders(سچ, matmul_id_type, جھوٹ, جھوٹ, سچ);

        // dot2 variants (scalar fp16 only)
        matmul_shaders(سچ, matmul_id_type, جھوٹ, جھوٹ, جھوٹ, سچ);
        matmul_shaders(سچ, matmul_id_type, جھوٹ, جھوٹ, سچ,  سچ);

        اگر (matmul_id_type != MatMulIdType::DEFAULT) {
#اگر defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            // Coopmat, fp32acc and fp16acc
            matmul_shaders(سچ, matmul_id_type, سچ, جھوٹ, جھوٹ);
            matmul_shaders(سچ, matmul_id_type, سچ, جھوٹ, سچ);
#اگر_ختم

#اگر defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            // Coopmat2, fp32acc and fp16acc
            matmul_shaders(سچ, matmul_id_type, جھوٹ, سچ, جھوٹ);
            matmul_shaders(سچ, matmul_id_type, جھوٹ, سچ, سچ);
#اگر_ختم
        }
    }

    برائے (ثابت منطقی& fp16 : {جھوٹ, سچ}) {
        std::map<std::string, std::string> بنیاد_dict;
        اگر (fp16) {
            بنیاد_dict = {{"FLOAT_TYPE", "float16_t"}, {"FLOAT_TYPEV2", "f16vec2"}, {"FLOAT_TYPEV4", "f16vec4"}, {"FLOAT16", "1"}, {"FLOAT_TYPE_MAX", "float16_t(65504.0)"}};
        } ورنہ {
            بنیاد_dict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"FLOAT_TYPEV4", "vec4"}};
        }

        // flash attention
        برائے (ثابت منطقی& f16acc : {جھوٹ, سچ}) {
            std::map<std::string, std::string> fa_بنیاد_dict = بنیاد_dict;
            fa_بنیاد_dict["ACC_TYPE"] = fp16 && f16acc ? "float16_t" : "float";
            fa_بنیاد_dict["ACC_TYPEV2"] = fp16 && f16acc ? "f16vec2" : "vec2";
            fa_بنیاد_dict["ACC_TYPEV4"] = fp16 && f16acc ? "f16vec4" : "vec4";
            // Compile IQ4_NL support into all FA variants so its shared LUT is available when K or V uses it.
            fa_بنیاد_dict["DATA_A_IQ4_NL"] = "1";
            اگر (fp16 && f16acc) {
                fa_بنیاد_dict["ACC_TYPE_MAX"] = "float16_t(65504.0)";
            }

            اگر (fp16) {
#اگر defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
                string_مقصد_spv("flash_attn_f32_f16", "flash_attn_cm2.comp",
                    ملاؤ_maps(fa_بنیاد_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}), fp16, جھوٹ, سچ, f16acc);
#اگر_ختم

#اگر defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                string_مقصد_spv("flash_attn_f32_f16", "flash_attn_cm1.comp",
                    ملاؤ_maps(fa_بنیاد_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"COOPMAT", "1"}}), fp16, سچ, جھوٹ, f16acc);
#اگر_ختم
            }

            string_مقصد_spv("flash_attn_f32_f16", "flash_attn.comp",
                ملاؤ_maps(fa_بنیاد_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}), fp16, جھوٹ, جھوٹ, f16acc);

            اگر (fp16) {
                string_مقصد_spv("flash_attn_f32_f16_dot2", "flash_attn.comp",
                    ملاؤ_maps(fa_بنیاد_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"DOT2_F16", "1"}}), fp16, جھوٹ, جھوٹ, f16acc);
            }

#اگر defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            string_مقصد_spv("flash_attn_f32_f16", "flash_attn.comp",
                ملاؤ_maps(fa_بنیاد_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"MMQ", "1"}, {"FA_MMQ_MIXED", "1"}}), fp16, جھوٹ, جھوٹ, f16acc, "_int8");
#اگر_ختم
        }
    }

    ثابت std::map<std::string, std::string> fa_bf16_dict = {
        {"FLOAT_TYPE",   "bfloat16_t"},
        {"FLOAT_TYPEV2", "bf16vec2"},
        {"FLOAT_TYPEV4", "bf16vec4"},
        {"ACC_TYPE",     "float"},
        {"ACC_TYPEV2",   "vec2"},
        {"ACC_TYPEV4",   "vec4"},
        {"BFLOAT16",     "1"},
    };

#اگر defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    string_مقصد_spv("flash_attn_f32_f16_bf16", "flash_attn_cm1.comp",
        ملاؤ_maps(fa_bf16_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"COOPMAT", "1"}}),
        سچ, سچ, جھوٹ, جھوٹ);
#اگر_ختم

#اگر defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    string_مقصد_spv("flash_attn_f32_f16_bf16", "flash_attn_cm2.comp",
        ملاؤ_maps(fa_bf16_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}),
        سچ, جھوٹ, سچ, جھوٹ);
#اگر_ختم

    std::map<std::string, std::string> بنیاد_dict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}};

    برائے (ثابت خودکار& tname : type_نام) {
        // mul mat vec
        std::string ڈیٹا_a_کلید = "DATA_A_" + مقصد_uppercase(tname);
        std::string shader = (string_سرے_with(tname, "_k") || string_starts_with(tname, "iq1_") || string_starts_with(tname, "iq2_") || string_starts_with(tname, "iq3_") || tname == "iq4_xs" || tname == "tq2_0" || tname == "tq1_0") ? "mul_mat_vec_" + tname + ".comp" : "mul_mat_vec.comp";

        string_مقصد_spv("mul_mat_vec_" + tname + "_f32_f32", shader, ملاؤ_maps(بنیاد_dict, {{ڈیٹا_a_کلید, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
        string_مقصد_spv("mul_mat_vec_" + tname + "_f16_f32", shader, ملاؤ_maps(بنیاد_dict, {{ڈیٹا_a_کلید, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}));

        string_مقصد_spv("mul_mat_vec_" + tname + "_f32_f32_subgroup", shader, ملاؤ_maps(بنیاد_dict, {{ڈیٹا_a_کلید, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
        string_مقصد_spv("mul_mat_vec_" + tname + "_f16_f32_subgroup", shader, ملاؤ_maps(بنیاد_dict, {{ڈیٹا_a_کلید, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));

        string_مقصد_spv("mul_mat_vec_" + tname + "_f32_f32_subgroup_no_shmem", shader, ملاؤ_maps(بنیاد_dict, {{ڈیٹا_a_کلید, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        string_مقصد_spv("mul_mat_vec_" + tname + "_f16_f32_subgroup_no_shmem", shader, ملاؤ_maps(بنیاد_dict, {{ڈیٹا_a_کلید, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

#اگر defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        اگر (tname == "mxfp4" || tname == "nvfp4") {
            string_مقصد_spv("mul_mat_vec_" + tname + "_f32_f32_ocp", shader, ملاؤ_maps(بنیاد_dict, {{ڈیٹا_a_کلید, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
            string_مقصد_spv("mul_mat_vec_" + tname + "_f16_f32_ocp", shader, ملاؤ_maps(بنیاد_dict, {{ڈیٹا_a_کلید, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}));
            string_مقصد_spv("mul_mat_vec_" + tname + "_f32_f32_ocp_subgroup", shader, ملاؤ_maps(بنیاد_dict, {{ڈیٹا_a_کلید, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_مقصد_spv("mul_mat_vec_" + tname + "_f16_f32_ocp_subgroup", shader, ملاؤ_maps(بنیاد_dict, {{ڈیٹا_a_کلید, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_مقصد_spv("mul_mat_vec_" + tname + "_f32_f32_ocp_subgroup_no_shmem", shader, ملاؤ_maps(بنیاد_dict, {{ڈیٹا_a_کلید, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
            string_مقصد_spv("mul_mat_vec_" + tname + "_f16_f32_ocp_subgroup_no_shmem", shader, ملاؤ_maps(بنیاد_dict, {{ڈیٹا_a_کلید, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
            string_مقصد_spv("mul_mat_vec_id_" + tname + "_f32_f32_ocp", shader, ملاؤ_maps(بنیاد_dict, {{"MUL_MAT_ID", "1"}, {ڈیٹا_a_کلید, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
            string_مقصد_spv("mul_mat_vec_id_" + tname + "_f32_f32_ocp_subgroup", shader, ملاؤ_maps(بنیاد_dict, {{"MUL_MAT_ID", "1"}, {ڈیٹا_a_کلید, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_مقصد_spv("mul_mat_vec_id_" + tname + "_f32_f32_ocp_subgroup_no_shmem", shader, ملاؤ_maps(بنیاد_dict, {{"MUL_MAT_ID", "1"}, {ڈیٹا_a_کلید, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        }
#اگر_ختم

        string_مقصد_spv("mul_mat_vec_id_" + tname + "_f32_f32", shader, ملاؤ_maps(بنیاد_dict, {{"MUL_MAT_ID", "1"}, {ڈیٹا_a_کلید, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
        string_مقصد_spv("mul_mat_vec_id_" + tname + "_f32_f32_subgroup", shader, ملاؤ_maps(بنیاد_dict, {{"MUL_MAT_ID", "1"}, {ڈیٹا_a_کلید, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
        string_مقصد_spv("mul_mat_vec_id_" + tname + "_f32_f32_subgroup_no_shmem", shader, ملاؤ_maps(بنیاد_dict, {{"MUL_MAT_ID", "1"}, {ڈیٹا_a_کلید, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

        // mul mat vec with integer dot product
#اگر defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        اگر (is_legacy_quant(tname) || tname == "mxfp4" || is_k_quant(tname) || tname == "iq1_s" || tname == "iq1_m" || tname == "iq4_xs") {
            string_مقصد_spv("mul_mat_vec_" + tname + "_q8_1_f32", "mul_mat_vecq.comp", ملاؤ_maps(بنیاد_dict, {{ڈیٹا_a_کلید, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}}));
            string_مقصد_spv("mul_mat_vec_" + tname + "_q8_1_f32_subgroup", "mul_mat_vecq.comp", ملاؤ_maps(بنیاد_dict, {{ڈیٹا_a_کلید, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_مقصد_spv("mul_mat_vec_" + tname + "_q8_1_f32_subgroup_no_shmem", "mul_mat_vecq.comp", ملاؤ_maps(بنیاد_dict, {{ڈیٹا_a_کلید, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

            string_مقصد_spv("mul_mat_vec_id_" + tname + "_q8_1_f32", "mul_mat_vecq.comp", ملاؤ_maps(بنیاد_dict, {{"MUL_MAT_ID", "1"}, {ڈیٹا_a_کلید, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}}));
            string_مقصد_spv("mul_mat_vec_id_" + tname + "_q8_1_f32_subgroup", "mul_mat_vecq.comp", ملاؤ_maps(بنیاد_dict, {{"MUL_MAT_ID", "1"}, {ڈیٹا_a_کلید, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_مقصد_spv("mul_mat_vec_id_" + tname + "_q8_1_f32_subgroup_no_shmem", "mul_mat_vecq.comp", ملاؤ_maps(بنیاد_dict, {{"MUL_MAT_ID", "1"}, {ڈیٹا_a_کلید, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        }
#اگر_ختم

        // Dequant shaders
        اگر (tname != "f16" && tname != "bf16") {
            string_مقصد_spv("dequant_" + tname, "dequant_" + tname + ".comp", ملاؤ_maps(بنیاد_dict, {{ڈیٹا_a_کلید, "1"}, {"D_TYPE", "float16_t"}}));
        }
        // Fused dequant+transpose variant for FA quant-KV (per-head-contiguous f16 scratch).
        اگر (tname == "q8_0") {
            string_مقصد_spv("dequant_" + tname + "_transpose", "dequant_" + tname + ".comp", ملاؤ_maps(بنیاد_dict, {{ڈیٹا_a_کلید, "1"}, {"D_TYPE", "float16_t"}, {"DEQUANT_TRANSPOSE", "1"}}));
        }

        shader = (tname == "f32" || tname == "f16" || tname == "bf16") ? "get_rows.comp" : "get_rows_quant.comp";

        اگر (tname == "f16") {
            string_مقصد_spv("get_rows_" + tname, shader, ملاؤ_maps(بنیاد_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {ڈیٹا_a_کلید, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float16_t"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}}));
        } ورنہ {
            string_مقصد_spv("get_rows_" + tname, shader, ملاؤ_maps(بنیاد_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {ڈیٹا_a_کلید, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float16_t"}}));
        }
        string_مقصد_spv("get_rows_" + tname + "_f32", shader, ملاؤ_maps(بنیاد_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {ڈیٹا_a_کلید, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float"}}));
    }

    string_مقصد_spv("get_rows_i32", "get_rows.comp", {{"TEMP_TYPE", "uint"}, {"A_TYPE", "uint"}, {"B_TYPE", "int"}, {"D_TYPE", "uint"}});

    string_مقصد_spv("mul_mat_vec_p021_f16_f32_subgroup_add", "mul_mat_vec_p021.comp", {{"A_TYPE", "float16_t"}, {"A_TYPEV4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}});
    string_مقصد_spv("mul_mat_vec_p021_f16_f32",              "mul_mat_vec_p021.comp", {{"A_TYPE", "float16_t"}, {"A_TYPEV4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}});
    string_مقصد_spv("mul_mat_vec_nc_f16_f32", "mul_mat_vec_nc.comp", {{"A_TYPE", "float16_t"}, {"A_TYPEV4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}});

    // Norms
    string_مقصد_spv("norm_f32", "norm.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_مقصد_spv("group_norm_f32", "group_norm.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_مقصد_spv("rms_norm_f32", "rms_norm.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_مقصد_spv("rms_norm_mul_add_f32", "rms_norm.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"RMS_NORM_ADD_FUSION", "1"}}));
    string_مقصد_spv("rms_norm_mul_add_partials_f32", "rms_norm_partials.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"RMS_NORM_ADD_FUSION", "1"}}));
    string_مقصد_spv("rms_norm_set_rows_f32_f32", "rms_norm.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"RMS_NORM_SET_ROWS_FUSION", "1"}}));
    string_مقصد_spv("rms_norm_set_rows_f32_f16", "rms_norm.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"RMS_NORM_SET_ROWS_FUSION", "1"}}));
    string_مقصد_spv("rms_norm_partials_f32", "rms_norm_partials.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_مقصد_spv("rms_norm_mul_rope_f32_f32", "rms_norm.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"ROPE_D_TYPE", "float"}, {"RMS_NORM_ROPE_FUSION", "1"}}));
    string_مقصد_spv("rms_norm_mul_rope_f32_f16", "rms_norm.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}, {"RMS_NORM_ROPE_FUSION", "1"}}));
    string_مقصد_spv("rms_norm_back_f32", "rms_norm_back.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_مقصد_spv("l2_norm_f32", "l2_norm.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_مقصد_spv("cpy_32", "copy.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_مقصد_spv("cpy_f32_f16", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}});
    string_مقصد_spv("cpy_16", "copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_مقصد_spv("cpy_f16_f32", "copy.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_مقصد_spv("cpy_f32_bf16","copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "uint16_t"}, {"DATA_D_BF16", "1"}});
    string_مقصد_spv("cpy_bf16_f32","copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "float"}, {"DATA_A_BF16", "1"}});
    string_مقصد_spv("cpy_bf16_f16","copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "float16_t"}, {"DATA_A_BF16", "1"}});
    string_مقصد_spv("contig_cpy_32", "contig_copy.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_مقصد_spv("contig_cpy_f32_i32", "contig_copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "int"}});
    string_مقصد_spv("contig_cpy_i32_f32", "contig_copy.comp", {{"A_TYPE", "int"}, {"D_TYPE", "float"}});
    string_مقصد_spv("contig_cpy_f32_f16", "contig_copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}});
    string_مقصد_spv("contig_cpy_16", "contig_copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_مقصد_spv("contig_cpy_f16_f32", "contig_copy.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_مقصد_spv("contig_cpy_f32_bf16","contig_copy.comp",{{"A_TYPE", "float"}, {"D_TYPE", "uint16_t"}, {"DATA_D_BF16", "1"}});
    string_مقصد_spv("contig_cpy_bf16_f32","contig_copy.comp",{{"A_TYPE", "uint16_t"}, {"D_TYPE", "float"}, {"DATA_A_BF16", "1"}});
    string_مقصد_spv("contig_cpy_bf16_f16","contig_copy.comp",{{"A_TYPE", "uint16_t"}, {"D_TYPE", "float16_t"}, {"DATA_A_BF16", "1"}});
    string_مقصد_spv("cpy_f32_i32", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "int"}});
    string_مقصد_spv("cpy_i32_f32", "copy.comp", {{"A_TYPE", "int"}, {"D_TYPE", "float"}});

    string_مقصد_spv("cpy_transpose_16", "copy_transpose.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_مقصد_spv("cpy_transpose_32", "copy_transpose.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_مقصد_spv("cpy_transpose_02_16", "copy_transpose_02.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_مقصد_spv("cpy_transpose_02_32", "copy_transpose_02.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});

    برائے (std::string t : {"q1_0", "q2_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}) {
        string_مقصد_spv("cpy_f32_" + t, "copy_to_quant.comp", {{"DATA_A_" + مقصد_uppercase(t), "1"}, {"S_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
        string_مقصد_spv("cpy_" + t + "_f32", "copy_from_quant.comp", {{"DATA_A_" + مقصد_uppercase(t), "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    }

    برائے (خودکار src : {std::pair{"f32", "float"}, std::pair{"f16", "float16_t"}}) {
        برائے (std::string dst : {"f32", "f16", "bf16", "q1_0", "q2_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}) {
            string_مقصد_spv("set_rows_" + std::string(src.پہلا) + "_" + dst + "_i32", "copy_to_quant.comp", {{"SET_ROWS", "1"}, {"DATA_A_" + مقصد_uppercase(dst), "1"}, {"B_TYPE", "uint"}, {"B_SIZE", "32"}, {"S_TYPE", src.second}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
            string_مقصد_spv("set_rows_" + std::string(src.پہلا) + "_" + dst + "_i64", "copy_to_quant.comp", {{"SET_ROWS", "1"}, {"DATA_A_" + مقصد_uppercase(dst), "1"}, {"B_TYPE", "uvec2"}, {"B_SIZE", "64"}, {"S_TYPE", src.second}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
        }
    }

    خودکار get_type_str = [](منطقی f16) {
        واپس f16 ? "float16_t" : "float";
    };
    خودکار get_suffix = [](منطقی src0_f16, منطقی src1_f16, منطقی dst_f16) {
        std::string s;
        s += std::string(src0_f16 ? "_f16" : "_f32");
        s += std::string(src1_f16 ? "_f16" : "_f32");
        s += std::string(dst_f16 ? "_f16" : "_f32");
        واپس s;
    };
    برائے (std::string op : {"add", "sub", "mul", "div", "add_rms", }) {
    برائے (خودکار src0_f16 : {جھوٹ, سچ}) {
    برائے (خودکار src1_f16 : {جھوٹ, سچ}) {
    برائے (خودکار dst_f16  : {جھوٹ, سچ}) {
        خودکار ماخذ = op == "add_rms" ? std::string("add") : op;
        خودکار name = op + get_suffix(src0_f16, src1_f16, dst_f16);
        خودکار جوڑو_rms = op == "add_rms" ? "1" : "0";
        string_مقصد_spv(name.c_str(), ماخذ + ".comp", {{"A_TYPE", get_type_str(src0_f16)}, {"B_TYPE", get_type_str(src1_f16)}, {"D_TYPE", get_type_str(dst_f16)}, {"FLOAT_TYPE", "float"}, {"ADD_RMS" , جوڑو_rms}});
    }
    }
    }
    }

    string_مقصد_spv("sub_f32", "sub.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_مقصد_spv("acc_f32", "acc.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_مقصد_spv("split_k_reduce", "mul_mat_split_k_reduce.comp", {});
    string_مقصد_spv("fa_split_k_reduce", "flash_attn_split_k_reduce.comp", {});

    string_مقصد_spv("fa_mask_opt", "flash_attn_mask_opt.comp", {});

#اگر defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    string_مقصد_spv("fa_decode_ph1", "flash_attn_decode_phase_1.comp", {}, سچ, سچ, جھوٹ, جھوٹ);
    string_مقصد_spv("fa_decode_ph2", "flash_attn_decode_phase_2.comp", {}, سچ, سچ, جھوٹ, جھوٹ);
#اگر_ختم

    string_مقصد_spv("fa_sparse_compact", "flash_attn_sparse_compact.comp", {});
    string_مقصد_spv("fa_sparse_compact_subgroup", "flash_attn_sparse_compact.comp", {{"USE_SUBGROUPS", "1"}});

    string_مقصد_spv("quantize_q8_1", "quantize_q8_1.comp", {});
    string_مقصد_spv("quantize_q8_1_subgroup", "quantize_q8_1.comp", {{"USE_SUBGROUPS", "1"}});

    string_مقصد_spv("quantize_q8_1_x4", "quantize_q8_1.comp", {{"QBLOCK_X4", "1"}});
    string_مقصد_spv("quantize_q8_1_x4_subgroup", "quantize_q8_1.comp", {{"QBLOCK_X4", "1"}, {"USE_SUBGROUPS", "1"}});

    string_مقصد_spv("mul_f32", "mul.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_مقصد_spv("div_f32", "div.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_مقصد_spv("repeat_i32", "repeat.comp", {{"A_TYPE", "int32_t"}, {"D_TYPE", "int32_t"}});
    string_مقصد_spv("repeat_back_f32", "repeat_back.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_مقصد_spv("get_rows_back_f32", "get_rows_back.comp", {{"A_TYPE", "float"}, {"B_TYPE", "int"}, {"D_TYPE", "float"}});

    string_مقصد_spv("repeat_i16", "repeat.comp", {{"A_TYPE", "int16_t"}, {"D_TYPE", "int16_t"}});

    string_مقصد_spv("scale_f32", "scale.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_مقصد_spv("pad_f32", "pad.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_مقصد_spv("pad_reflect_1d_f32", "pad_reflect_1d.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

    string_مقصد_spv("concat_i8", "concat.comp", {{"A_TYPE", "uint8_t"}, {"B_TYPE", "uint8_t"}, {"D_TYPE", "uint8_t"}});
    string_مقصد_spv("concat_i16", "concat.comp", {{"A_TYPE", "uint16_t"}, {"B_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_مقصد_spv("concat_i32", "concat.comp", {{"A_TYPE", "uint"}, {"B_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_مقصد_spv("concat_i64", "concat.comp", {{"A_TYPE", "uvec2"}, {"B_TYPE", "uvec2"}, {"D_TYPE", "uvec2"}});

    string_مقصد_spv("upscale_f32", "upscale.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});

    string_مقصد_spv("exp_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_exp"}});
    string_مقصد_spv("exp_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_exp"}});
    string_مقصد_spv("expm1_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_expm1"}});
    string_مقصد_spv("expm1_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_expm1"}});

    string_مقصد_spv("log_f16",        "log.comp",         {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_مقصد_spv("log_f32",        "log.comp",         {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_مقصد_spv("gelu_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_gelu"}});
    string_مقصد_spv("gelu_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_gelu"}});
    string_مقصد_spv("gelu_erf_f16",   "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_gelu_erf"}});
    string_مقصد_spv("gelu_erf_f32",   "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_gelu_erf"}});
    string_مقصد_spv("gelu_quick_f16", "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_gelu_quick"}});
    string_مقصد_spv("gelu_quick_f32", "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_gelu_quick"}});
    string_مقصد_spv("silu_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_silu"}});
    string_مقصد_spv("silu_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_silu"}});
    string_مقصد_spv("relu_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_relu"}});
    string_مقصد_spv("relu_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_relu"}});
    string_مقصد_spv("sqr_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sqr"}});
    string_مقصد_spv("sqr_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sqr"}});
    string_مقصد_spv("sqrt_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sqrt"}});
    string_مقصد_spv("sqrt_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sqrt"}});
    string_مقصد_spv("sin_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sin"}});
    string_مقصد_spv("sin_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sin"}});
    string_مقصد_spv("cos_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_cos"}});
    string_مقصد_spv("cos_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_cos"}});
    string_مقصد_spv("clamp_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_clamp"}});
    string_مقصد_spv("clamp_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_clamp"}});
    string_مقصد_spv("leaky_relu_f16", "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_leaky_relu"}});
    string_مقصد_spv("leaky_relu_f32", "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_leaky_relu"}});
    string_مقصد_spv("neg_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_neg"}});
    string_مقصد_spv("neg_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_neg"}});
    string_مقصد_spv("tanh_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_tanh"}});
    string_مقصد_spv("tanh_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_tanh"}});
    string_مقصد_spv("sigmoid_f16",    "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sigmoid"}});
    string_مقصد_spv("sigmoid_f32",    "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sigmoid"}});
    string_مقصد_spv("hardsigmoid_f16","unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_hardsigmoid"}});
    string_مقصد_spv("hardsigmoid_f32","unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_hardsigmoid"}});
    string_مقصد_spv("hardswish_f16",  "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_hardswish"}});
    string_مقصد_spv("hardswish_f32",  "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_hardswish"}});
    string_مقصد_spv("abs_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_abs"}});
    string_مقصد_spv("abs_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_abs"}});
    string_مقصد_spv("elu_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_elu"}});
    string_مقصد_spv("elu_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_elu"}});
    string_مقصد_spv("xielu_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_xielu"}});
    string_مقصد_spv("xielu_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_xielu"}});
    string_مقصد_spv("sgn_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sgn"}});
    string_مقصد_spv("sgn_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sgn"}});

    string_مقصد_spv("tri_f16",        "tri.comp",         {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_مقصد_spv("tri_f32",        "tri.comp",         {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_مقصد_spv("diag_f16",       "diag.comp",        {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_مقصد_spv("diag_f32",       "diag.comp",        {{"A_TYPE", "float"},       {"D_TYPE", "float"}});

    string_مقصد_spv("softplus_f16",   "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_softplus"}});
    string_مقصد_spv("softplus_f32",   "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_softplus"}});

    string_مقصد_spv("gelu_mul_f32",    "unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_gelu"},     {"UNARY_MUL_FUSION", "1"}});
    string_مقصد_spv("gelu_mul_f16",    "unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_gelu"},     {"UNARY_MUL_FUSION", "1"}});
    string_مقصد_spv("sigmoid_mul_f32", "unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_sigmoid"},  {"UNARY_MUL_FUSION", "1"}});
    string_مقصد_spv("sigmoid_mul_f16", "unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_sigmoid"},  {"UNARY_MUL_FUSION", "1"}});
    string_مقصد_spv("silu_mul_f32",    "unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_silu"},     {"UNARY_MUL_FUSION", "1"}});
    string_مقصد_spv("silu_mul_f16",    "unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_silu"},     {"UNARY_MUL_FUSION", "1"}});
    string_مقصد_spv("softplus_mul_f32","unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_softplus"}, {"UNARY_MUL_FUSION", "1"}});
    string_مقصد_spv("softplus_mul_f16","unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_softplus"}, {"UNARY_MUL_FUSION", "1"}});

    string_مقصد_spv("add1_f16_f16",   "add1.comp",        {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}});
    string_مقصد_spv("add1_f16_f32",   "add1.comp",        {{"A_TYPE", "float16_t"},   {"B_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}});
    string_مقصد_spv("add1_f32_f32",   "add1.comp",        {{"A_TYPE", "float"},       {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    string_مقصد_spv("arange_f32",     "arange.comp",      {{"A_TYPE", "float"},       {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    string_مقصد_spv("fill_f32",       "fill.comp",        {{"D_TYPE", "float"},       {"FLOAT_TYPE", "float"}});
    string_مقصد_spv("fill_f16",       "fill.comp",        {{"D_TYPE", "float16_t"},   {"FLOAT_TYPE", "float"}});
    string_مقصد_spv("step_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_step"}});
    string_مقصد_spv("step_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_step"}});
    string_مقصد_spv("round_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_round"}});
    string_مقصد_spv("round_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_round"}});
    string_مقصد_spv("ceil_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_ceil"}});
    string_مقصد_spv("ceil_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_ceil"}});
    string_مقصد_spv("floor_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_floor"}});
    string_مقصد_spv("floor_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_floor"}});
    string_مقصد_spv("trunc_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_trunc"}});
    string_مقصد_spv("trunc_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_trunc"}});

    string_مقصد_spv("geglu_f16",      "geglu.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_مقصد_spv("geglu_f32",      "geglu.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_مقصد_spv("reglu_f16",      "reglu.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_مقصد_spv("reglu_f32",      "reglu.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_مقصد_spv("swiglu_f16",     "swiglu.comp",      {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_مقصد_spv("swiglu_f32",     "swiglu.comp",      {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_مقصد_spv("swiglu_oai_f16", "swiglu_oai.comp",  {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_مقصد_spv("swiglu_oai_f32", "swiglu_oai.comp",  {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_مقصد_spv("swiglu_clamp_f16", "swiglu_clamp.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_مقصد_spv("swiglu_clamp_f32", "swiglu_clamp.comp", {{"A_TYPE", "float"},     {"D_TYPE", "float"}});
    string_مقصد_spv("geglu_erf_f16",  "geglu_erf.comp",   {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_مقصد_spv("geglu_erf_f32",  "geglu_erf.comp",   {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_مقصد_spv("geglu_quick_f16","geglu_quick.comp", {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_مقصد_spv("geglu_quick_f32","geglu_quick.comp", {{"A_TYPE", "float"},       {"D_TYPE", "float"}});

    string_مقصد_spv("silu_back_f32",  "silu_back.comp",   {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});

    string_مقصد_spv("diag_mask_inf_f32", "diag_mask_inf.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

    string_مقصد_spv("soft_max_f32", "soft_max.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_مقصد_spv("soft_max_f32_f16", "soft_max.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_مقصد_spv("soft_max_back_f32", "soft_max_back.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_مقصد_spv("soft_max_large1_f32", "soft_max_large1.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_مقصد_spv("soft_max_large2_f32", "soft_max_large2.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_مقصد_spv("soft_max_large3_f32", "soft_max_large3.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_مقصد_spv("soft_max_large1_f32_f16", "soft_max_large1.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_مقصد_spv("soft_max_large2_f32_f16", "soft_max_large2.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_مقصد_spv("soft_max_large3_f32_f16", "soft_max_large3.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));

    string_مقصد_spv("rope_norm_f32", "rope_norm.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_مقصد_spv("rope_norm_f16", "rope_norm.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_مقصد_spv("rope_norm_f32_f16", "rope_norm.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});

    string_مقصد_spv("rope_neox_f32", "rope_neox.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_مقصد_spv("rope_neox_f16", "rope_neox.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_مقصد_spv("rope_neox_f32_f16", "rope_neox.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});

    string_مقصد_spv("rope_multi_f32", "rope_multi.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_مقصد_spv("rope_multi_f16", "rope_multi.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_مقصد_spv("rope_multi_f32_f16", "rope_multi.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});

    string_مقصد_spv("rope_vision_f32", "rope_vision.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_مقصد_spv("rope_vision_f16", "rope_vision.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});

    string_مقصد_spv("argsort_f32", "argsort.comp", {{"A_TYPE", "float"}});
    string_مقصد_spv("argsort_large_f32", "argsort_large.comp", {{"A_TYPE", "float"}});

    string_مقصد_spv("topk_argsort_f32", "topk_argsort.comp", {{"A_TYPE", "float"}});
    string_مقصد_spv("topk_nary_search_f32", "topk_nary_search.comp", {{"A_TYPE", "float"}});
    string_مقصد_spv("topk_radix_select_f32", "topk_radix_select.comp", {{"A_TYPE", "float"}});

    string_مقصد_spv("argmax_f32", "argmax.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"D_TYPE", "int"}}));
    string_مقصد_spv("sum_rows_f32", "sum_rows.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_مقصد_spv("cross_entropy_loss_f32", "cross_entropy_loss.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_مقصد_spv("cross_entropy_loss_back_f32", "cross_entropy_loss_back.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_مقصد_spv("fwht_f32", "fwht.comp", {});
    string_مقصد_spv("fwht_shmem_f32", "fwht.comp", {{"FWHT_SHMEM", "1"}});
    string_مقصد_spv("count_equal_i32", "count_equal.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "int"}, {"B_TYPE", "int"}, {"D_TYPE", "int"}}));
    string_مقصد_spv("dsv4_hc_comb_f32", "dsv4_hc_comb.comp", {});
    string_مقصد_spv("dsv4_hc_pre_f32",  "dsv4_hc_pre.comp",  {});
    string_مقصد_spv("dsv4_hc_post_f32", "dsv4_hc_post.comp", {});
    string_مقصد_spv("cumsum_f32", "cumsum.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_مقصد_spv("cumsum_multipass1_f32", "cumsum_multipass1.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_مقصد_spv("cumsum_multipass2_f32", "cumsum_multipass2.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_مقصد_spv("count_experts", "count_experts.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}}));
    string_مقصد_spv("count_experts_subgroup", "count_experts.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}, {"USE_SUBGROUPS", "1"}}));

    برائے (std::string dim_str : {"", "_3d"}) {
        برائے (منطقی bda : {جھوٹ, سچ}) {
            std::string bda_str = bda ? "_bda" : "";
            std::string bda_def = bda ? "1" : "0";
            string_مقصد_spv("im2col" + dim_str + "_f32" + bda_str, "im2col" + dim_str + ".comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"D_SIZE", "4"}, {"BDA", bda_def}}));
            string_مقصد_spv("im2col" + dim_str + "_f32_f16" + bda_str, "im2col" + dim_str + ".comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"D_SIZE", "2"}, {"BDA", bda_def}}));
        }
    }

    string_مقصد_spv("out_prod_f32", "out_prod.comp", {});

    string_مقصد_spv("timestep_embedding_f32", "timestep_embedding.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_مقصد_spv("conv_transpose_1d_f32", "conv_transpose_1d.comp", {{"A_TYPE", "float"},  {"B_TYPE", "float"}, {"D_TYPE", "float"}});
    string_مقصد_spv("col2im_1d_f32",  "col2im_1d.comp", {{"DATA_A_F32", "1"},  {"A_TYPE", "float"},     {"D_TYPE", "float"}});
    string_مقصد_spv("col2im_1d_f16",  "col2im_1d.comp", {{"DATA_A_F16", "1"},  {"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_مقصد_spv("col2im_1d_bf16", "col2im_1d.comp", {{"DATA_A_BF16", "1"}, {"A_TYPE", "uint16_t"},  {"D_TYPE", "uint16_t"}});

    string_مقصد_spv("snake_f32",  "snake.comp", {{"DATA_A_F32", "1"},  {"A_TYPE", "float"},     {"D_TYPE", "float"}});
    string_مقصد_spv("snake_f16",  "snake.comp", {{"DATA_A_F16", "1"},  {"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_مقصد_spv("snake_bf16", "snake.comp", {{"DATA_A_BF16", "1"}, {"DATA_D_BF16", "1"}, {"A_TYPE", "uint16_t"},  {"D_TYPE", "uint16_t"}});

    string_مقصد_spv("pool1d_f32", "pool1d.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_مقصد_spv("pool2d_f32", "pool2d.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_مقصد_spv("pool2d_subgroup_f32", "pool2d.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"USE_SUBGROUP", "1"}}));

    string_مقصد_spv("rwkv_wkv6_f32", "wkv6.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}}));

    string_مقصد_spv("gated_linear_attn_f32", "gla.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}}));

    // Compile IQ4_NL support in so its shared LUT is available when K uses it.
    // K quant type is selected at runtime via the FaTypeK spec constant.
    std::map<std::string, std::string> li_dict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV4", "vec4"}, {"DATA_A_IQ4_NL", "1"}};
    string_مقصد_spv("lightning_indexer_f32", "lightning_indexer.comp", li_dict);

    string_مقصد_spv("rwkv_wkv7_f32", "wkv7.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}}));

    string_مقصد_spv("gated_delta_net_f32", "gated_delta_net.comp", ملاؤ_maps(بنیاد_dict, {{"FLOAT_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}, {"USE_SUBGROUP_CLUSTERED", "1"}}));
    string_مقصد_spv("gated_delta_net_f32_nocluster", "gated_delta_net.comp", ملاؤ_maps(بنیاد_dict, {{"FLOAT_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}, {"USE_SUBGROUP_CLUSTERED", "0"}}));
    string_مقصد_spv("gated_delta_net_f32_shmem", "gated_delta_net.comp", ملاؤ_maps(بنیاد_dict, {{"FLOAT_TYPE", "float"}, {"USE_SUBGROUP_ADD", "0"}, {"USE_SUBGROUP_CLUSTERED", "0"}}));

    string_مقصد_spv("opt_step_adamw_f32", "opt_step_adamw.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}}));
    string_مقصد_spv("opt_step_sgd_f32", "opt_step_sgd.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}}));

    string_مقصد_spv("solve_tri_f32", "solve_tri.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    برائے (خودکار transpose : {جھوٹ, سچ}) {
        برائے (خودکار unroll : {جھوٹ, سچ}) {
            برائے (خودکار a_f16 : {جھوٹ, سچ}) {
                std::map<std::string, std::string> متعین_کرتا = {
                    {"A_TYPE", a_f16 ? "float16_t" : "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"},
                    {"USE_COLLECTIVES", "1"}, {"UNROLL", unroll ? "[[unroll]]" : ""},
                };
                اگر (transpose) متعین_کرتا["TRANSPOSE"] = "1";
                std::string name = std::string(transpose ? "conv_transpose_2d": "conv2d")
                    + (a_f16 ? "_f16" : "") + "_f32";
                string_مقصد_spv(name + (unroll ? "_unroll" : ""), "conv2d_mm.comp", متعین_کرتا);
#اگر defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
                اگر (unroll) {
                    خودکار cm2_متعین_کرتا = متعین_کرتا;
                    cm2_متعین_کرتا["COOPMAT2"] = "1";
                    string_مقصد_spv(name, "conv2d_mm.comp", cm2_متعین_کرتا, سچ, جھوٹ, سچ);
                }
#اگر_ختم
#اگر defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                اگر (unroll) {
                    خودکار cm1_متعین_کرتا = متعین_کرتا;
                    cm1_متعین_کرتا["COOPMAT"] = "1";
                    string_مقصد_spv(name, "conv2d_mm.comp", cm1_متعین_کرتا, سچ, سچ, جھوٹ);
                }
#اگر_ختم
            }
        }
    }

    برائے (خودکار unroll : {جھوٹ, سچ}) {
        برائے (خودکار a_f16 : {جھوٹ, سچ}) {
            std::map<std::string, std::string> متعین_کرتا = {
                {"A_TYPE", a_f16 ? "float16_t" : "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"},
                {"UNROLL", unroll ? "[[unroll]]" : ""},
            };
            std::string name = std::string("conv3d") + (a_f16 ? "_f16" : "") + "_f32";
            string_مقصد_spv(name + (unroll ? "_unroll" : ""), "conv3d_mm.comp", متعین_کرتا);
#اگر defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            اگر (unroll) {
                خودکار cm2_متعین_کرتا = متعین_کرتا;
                cm2_متعین_کرتا["COOPMAT2"] = "1";
                string_مقصد_spv(name, "conv3d_mm.comp", cm2_متعین_کرتا, سچ, جھوٹ, سچ);
            }
#اگر_ختم
#اگر defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            اگر (unroll) {
                خودکار cm1_متعین_کرتا = متعین_کرتا;
                cm1_متعین_کرتا["COOPMAT"] = "1";
                string_مقصد_spv(name, "conv3d_mm.comp", cm1_متعین_کرتا, سچ, سچ, جھوٹ);
            }
#اگر_ختم
        }
    }

    string_مقصد_spv("conv2d_dw_whcn_f32", "conv2d_dw.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"WHCN", "1"}}));
    string_مقصد_spv("conv2d_dw_cwhn_f32", "conv2d_dw.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"CWHN", "1"}}));
    string_مقصد_spv("conv2d_dw_whcn_f16_f32", "conv2d_dw.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"WHCN", "1"}}));
    string_مقصد_spv("conv2d_dw_cwhn_f16_f32", "conv2d_dw.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"CWHN", "1"}}));

    string_مقصد_spv("roll_f32", "roll.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_مقصد_spv("add_id_f32", "add_id.comp", ملاؤ_maps(بنیاد_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_مقصد_spv("multi_add_f32", "multi_add.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"ADD_RMS" , "0"}});
    string_مقصد_spv("multi_add_rms_f32", "multi_add.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"ADD_RMS" , "1"}});

    string_مقصد_spv("ssm_scan_f32",          "ssm_scan.comp", {{"A_TYPE", "float"}});
    string_مقصد_spv("ssm_scan_subgroup_f32", "ssm_scan.comp", {{"A_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}});

    string_مقصد_spv("ssm_conv_f32", "ssm_conv.comp", {{"A_TYPE", "float"}});

    string_مقصد_spv("topk_moe_f32", "topk_moe.comp", {});

    برائے (خودکار &c : compiles) {
        c.wait();
    }
}

خالی write_output_files() {
    std::stringstream hdr = make_generic_stringstream();
    std::stringstream src = make_generic_stringstream();

    hdr << "#include <cstdint>\n\n";
    src << "#include \"" << basename(ہدف_hpp) << "\"\n\n";

    std::sort(shader_fnames.begin(), shader_fnames.end());
    برائے (ثابت خودکار& pair : shader_fnames) {
        ثابت std::string& name = pair.پہلا;
        #اگر_تعریف_شدہ _WIN32
            std::string path = pair.second;
            std::replace(path.begin(), path.end(), '/', '\\' );
        #ورنہ
            ثابت std::string& path = pair.second;
        #اگر_ختم

        hdr << "extern const uint64_t " << name << "_len;\n";
        hdr << "extern const unsigned char " << name << "_data[];\n\n";

        اگر (داخلہ_filepath != "") {
            std::string ڈیٹا = read_ثنائی_file(path);
            اگر (ڈیٹا.خالی_خانے()) {
                جاری;
            }

            src << "const uint64_t " << name << "_len = " << ڈیٹا.size() << ";\n";
            src << "const unsigned char " << name << "_data[" << ڈیٹا.size() << "] = {\n" << std::hex;
            خودکار بائٹیں = reinterpret_cast<ثابت uint8_t*>(ڈیٹا.ڈیٹا());
            برائے (size_t i = 0; i < ڈیٹا.size(); ++i) {
                src << "0x" << static_cast<صحیح>(بائٹیں[i]) << ",";
                اگر ((i + 1) % 12 == 0) src << "\n";
            }
            src << std::dec << "\n};\n\n";
        }
    }

    std::string suffixes[2] = {"_f32", "_f16"};
    برائے (std::string op : {"add", "sub", "mul", "div", "add_rms"}) {
        hdr << "extern const void * " << op << "_data[2][2][2];\n";
        hdr << "extern const uint64_t " << op << "_len[2][2][2];\n";

        std::string op_file = op == "add_rms" ? "add.comp" : std::string(op) + ".comp";
        اگر (basename(داخلہ_filepath) != op_file) {
            جاری;
        }
        std::stringstream ڈیٹا = make_generic_stringstream();
        std::stringstream len  = make_generic_stringstream();
        ڈیٹا << "const void * " << op << "_data[2][2][2] = ";
        len  << "const uint64_t " << op << "_len[2][2][2] = ";
        برائے (uint32_t t0 = 0; t0 < 2; ++t0) {
            اگر (t0 == 0) {
                ڈیٹا << "{";
                len  << "{";
            }
            برائے (uint32_t t1 = 0; t1 < 2; ++t1) {
                اگر (t1 == 0) {
                    ڈیٹا << "{";
                    len  << "{";
                }
                برائے (uint32_t t2 = 0; t2 < 2; ++t2) {
                    اگر (t2 == 0) {
                        ڈیٹا << "{";
                        len  << "{";
                    }
                    ڈیٹا << op << suffixes[t0] << suffixes[t1] << suffixes[t2];
                    len  << op << suffixes[t0] << suffixes[t1] << suffixes[t2];
                    ڈیٹا << "_data,";
                    len  << "_len,";
                    اگر (t2 == 1) {
                        ڈیٹا << "}, ";
                        len  << "}, ";
                    }
                }
                اگر (t1 == 1) {
                    ڈیٹا << "}, ";
                    len  << "}, ";
                }
            }
            اگر (t0 == 1) {
                ڈیٹا << "};\n";
                len  << "};\n";
            }
        }
        src << ڈیٹا.str();
        src << len.str();
    }

    std::vector<std::string> btypes = {"f16", "f32"};

#اگر defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    btypes.push_back("q8_1");
#اگر_ختم

    برائے (ثابت std::string& btype : btypes) {
    برائے (ثابت خودکار& tname : type_نام) {
        اگر (btype == "q8_1" && !is_legacy_quant(tname) && tname != "mxfp4" && !is_k_quant(tname) && tname != "iq1_s" && tname != "iq1_m" && tname != "iq4_xs") {
            جاری;
        }
        hdr << "extern const void * arr_dmmv_"   << tname << "_" << btype << "_f32_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_len[3];\n";
        اگر (basename(داخلہ_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_"   << tname << "_" << btype << "_f32_data[3] = {mul_mat_vec_" << tname << "_" << btype << "_f32_data, mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_data, mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_len[3] =  {mul_mat_vec_" << tname << "_" << btype << "_f32_len,  mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_len, mul_mat_vec_"  << tname << "_" << btype << "_f32_subgroup_no_shmem_len};\n";
        }

        اگر (btype == "f16") {
            جاری;
        }
        hdr << "extern const void * arr_dmmv_id_"   << tname << "_" << btype << "_f32_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_id_" << tname << "_" << btype << "_f32_len[3];\n";
        اگر (basename(داخلہ_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_id_"   << tname << "_" << btype << "_f32_data[3] = {mul_mat_vec_id_" << tname << "_" << btype << "_f32_data, mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_data, mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_id_" << tname << "_" << btype << "_f32_len[3] =  {mul_mat_vec_id_" << tname << "_" << btype << "_f32_len,  mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_len, mul_mat_vec_id_"  << tname << "_" << btype << "_f32_subgroup_no_shmem_len};\n";
        }
    }
    }

#اگر defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    برائے (ثابت std::string& btype : {"f16", "f32"}) {
    برائے (ثابت std::string& tname : {"mxfp4", "nvfp4"}) {
        hdr << "extern const void * arr_dmmv_"   << tname << "_" << btype << "_f32_ocp_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_ocp_len[3];\n";
        اگر (basename(داخلہ_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_"   << tname << "_" << btype << "_f32_ocp_data[3] = {mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_data, mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_subgroup_data, mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_ocp_len[3] =  {mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_len,  mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_subgroup_len, mul_mat_vec_"  << tname << "_" << btype << "_f32_ocp_subgroup_no_shmem_len};\n";
        }
    }
    }
    برائے (ثابت std::string& tname : {"mxfp4", "nvfp4"}) {
        hdr << "extern const void * arr_dmmv_id_"   << tname << "_f32_f32_ocp_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_id_" << tname << "_f32_f32_ocp_len[3];\n";
        اگر (basename(داخلہ_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_id_"   << tname << "_f32_f32_ocp_data[3] = {mul_mat_vec_id_" << tname << "_f32_f32_ocp_data, mul_mat_vec_id_" << tname << "_f32_f32_ocp_subgroup_data, mul_mat_vec_id_" << tname << "_f32_f32_ocp_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_id_" << tname << "_f32_f32_ocp_len[3] =  {mul_mat_vec_id_" << tname << "_f32_f32_ocp_len,  mul_mat_vec_id_" << tname << "_f32_f32_ocp_subgroup_len, mul_mat_vec_id_"  << tname << "_f32_f32_ocp_subgroup_no_shmem_len};\n";
        }
    }
#اگر_ختم

    اگر (داخلہ_filepath == "") {
        write_file_if_بدلا(ہدف_hpp, hdr.str());
    }
    اگر (ہدف_cpp != "") {
        write_ثنائی_file(ہدف_cpp, src.str());
    }
}

} // namespace

صحیح مین(صحیح argc, حرف** argv) {
    std::map<std::string, std::string> دلائل_2;
    برائے (صحیح i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        اگر (arg.rfind("--", 0) == 0) {
            اگر (i + 1 < argc && argv[i + 1][0] != '-') {
                دلائل_2[arg] = argv[i + 1];
                ++i;
            } ورنہ {
                دلائل_2[arg] = "";
            }
        }
    }

    اگر (دلائل_2.find("--glslc") != دلائل_2.end()) {
        GLSLC = دلائل_2["--glslc"]; // Path to glslc
    }
    اگر (دلائل_2.find("--source") != دلائل_2.end()) {
        داخلہ_filepath = دلائل_2["--source"]; // The shader source file to compile
    }
    اگر (دلائل_2.find("--output-dir") != دلائل_2.end()) {
        output_dir = دلائل_2["--output-dir"]; // Directory for containing SPIR-V output
    }
    اگر (دلائل_2.find("--target-hpp") != دلائل_2.end()) {
        ہدف_hpp = دلائل_2["--target-hpp"]; // Path to generated header file
    }
    اگر (دلائل_2.find("--target-cpp") != دلائل_2.end()) {
        ہدف_cpp = دلائل_2["--target-cpp"]; // Path to generated cpp file
    }

    اگر (!directory_exists(output_dir)) {
        اگر (!create_directory(output_dir)) {
            std::cerr << "Error creating output directory: " << output_dir << "\n";
            واپس EXIT_FAILURE;
        }
    }

    process_shaders();

    اگر (compile_failed) {
        std::cerr << "vulkan-shaders-gen: one or more shaders failed to compile" << std::سطر_ختم;
        واپس EXIT_FAILURE;
    }

    write_output_files();

    واپس EXIT_SUCCESS;
}

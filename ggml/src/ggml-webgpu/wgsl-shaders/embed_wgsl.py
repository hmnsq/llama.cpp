درآمد os
درآمد re
درآمد argparse


تعریف expand_includes(shader, داخلہ_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_نمونہ = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    تعریف replacer(مطابقت):
        fname = مطابقت.گروہ(1)
        file_راستہ = os.راستہ.join(داخلہ_dir, fname)
        اگر نہیں os.راستہ.exists(file_راستہ):
            اٹھاؤ File_Not_ملا_Error(f"Included file not found: {file_path}")
        ساتھ open(file_راستہ, "r", encoding="utf-8") بطور f:
            included_رمز = f.اقرأ()
        # Recursively expand includes inside the included file
        واپس expand_includes(included_رمز, داخلہ_dir)

    واپس include_نمونہ.sub(replacer, shader)


تعریف chunk_shader(shader_رمز, max_chunk_len=60000):
    """Split shader_code into safe raw-string sized chunks."""
    واپس [shader_رمز[i : i + max_chunk_len] برائے i میں سلسلہ(0, لمبائی(shader_رمز), max_chunk_len)]


تعریف raw_delim(shader_رمز):
    """Pick a raw-string delimiter that does not appear in the shader."""
    delim = "wgsl"
    جب f"){delim}\"" میں shader_رمز:
        delim += "_x"
    واپس delim


تعریف write_shader(shader_name, shader_رمز, output_dir, outfile, داخلہ_dir):
    shader_رمز = expand_includes(shader_رمز, داخلہ_dir)

    اگر output_dir:
        wgsl_filename = os.راستہ.join(output_dir, f"{shader_name}.wgsl")
        ساتھ open(wgsl_filename, "w", encoding="utf-8") بطور f_برآمد:
            f_برآمد.اكتب(shader_رمز)

    delim = raw_delim(shader_رمز)
    chunks = chunk_shader(shader_رمز)

    اگر لمبائی(chunks) == 1:
        outfile.اكتب(f'const char* wgsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
    ورنہ:
        برائے idx, chunk میں enumerate(chunks):
            outfile.اكتب(f'static const char wgsl_{shader_name}_part{idx}[] = R"{delim}({chunk}){delim}";\n\n')
        outfile.اكتب(f'static const std::string& wgsl_{shader_name}_str() {{\n')
        outfile.اكتب('    static const std::string s = []{\n')
        outfile.اكتب('        std::string tmp;\n')
        outfile.اكتب(f'        tmp.reserve({len(shader_code)});\n')
        برائے idx میں سلسلہ(لمبائی(chunks)):
            outfile.اكتب(f'        tmp.append(wgsl_{shader_name}_part{idx});\n')
        outfile.اكتب('        return tmp;\n')
        outfile.اكتب('    }();\n')
        outfile.اكتب('    return s;\n')
        outfile.اكتب('}\n')
        outfile.اكتب(f'const char* wgsl_{shader_name} = wgsl_{shader_name}_str().c_str();\n\n')


تعریف مین():
    parser = argparse.دلیل_Parser()
    parser.جوڑو_دلیل("--input_dir", required=سچ)
    parser.جوڑو_دلیل("--output_file", required=سچ)
    دلائل_2 = parser.parse_دلائل()

    ساتھ open(دلائل_2.output_file, "w", encoding="utf-8") بطور برآمد:
        برآمد.اكتب("// Auto-generated shader embedding\n")
        برآمد.اكتب("#include <string>\n\n")
        برائے fname میں sorted(os.listdir(دلائل_2.داخلہ_dir)):
            اگر fname.endswith(".wgsl"):
                shader_راستہ = os.راستہ.join(دلائل_2.داخلہ_dir, fname)
                shader_name = fname.replace(".wgsl", "")

                ساتھ open(shader_راستہ, "r", encoding="utf-8") بطور f:
                    shader_رمز = f.اقرأ()

                write_shader(shader_name, shader_رمز, کچھ_نہیں, برآمد, دلائل_2.داخلہ_dir)


اگر __name__ == "__main__":
    مین()

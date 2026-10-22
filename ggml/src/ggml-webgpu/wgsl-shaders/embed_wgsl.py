استورد os
استورد re
استورد argparse


تعریف expand_includes(shader, ننوت_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_نمونه = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    تعریف replacer(سمون):
        fname = سمون.ډله(1)
        file_لار = os.لار.join(ننوت_dir, fname)
        إذا ليس os.لار.exists(file_لار):
            ارفع File_Not_وموندل_Error(f"Included file not found: {file_path}")
        سره open(file_لار, "r", encoding="utf-8") كـ f:
            included_کوډ = f.اقرأ()
        # Recursively expand includes inside the included file
        أرجع expand_includes(included_کوډ, ننوت_dir)

    أرجع include_نمونه.sub(replacer, shader)


تعریف chunk_shader(shader_کوډ, max_chunk_len=60000):
    """Split shader_code into safe raw-string sized chunks."""
    أرجع [shader_کوډ[i : i + max_chunk_len] لكل i في لړ(0, اوږدوالی(shader_کوډ), max_chunk_len)]


تعریف raw_delim(shader_کوډ):
    """Pick a raw-string delimiter that does not appear in the shader."""
    delim = "wgsl"
    طالما f"){delim}\"" في shader_کوډ:
        delim += "_x"
    أرجع delim


تعریف write_shader(shader_name, shader_کوډ, output_dir, outfile, ننوت_dir):
    shader_کوډ = expand_includes(shader_کوډ, ننوت_dir)

    إذا output_dir:
        wgsl_filename = os.لار.join(output_dir, f"{shader_name}.wgsl")
        سره open(wgsl_filename, "w", encoding="utf-8") كـ f_بهر:
            f_بهر.اكتب(shader_کوډ)

    delim = raw_delim(shader_کوډ)
    chunks = chunk_shader(shader_کوډ)

    إذا اوږدوالی(chunks) == 1:
        outfile.اكتب(f'const char* wgsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
    وإلا:
        لكل idx, chunk في enumerate(chunks):
            outfile.اكتب(f'static const char wgsl_{shader_name}_part{idx}[] = R"{delim}({chunk}){delim}";\n\n')
        outfile.اكتب(f'static const std::string& wgsl_{shader_name}_str() {{\n')
        outfile.اكتب('    static const std::string s = []{\n')
        outfile.اكتب('        std::string tmp;\n')
        outfile.اكتب(f'        tmp.reserve({len(shader_code)});\n')
        لكل idx في لړ(اوږدوالی(chunks)):
            outfile.اكتب(f'        tmp.append(wgsl_{shader_name}_part{idx});\n')
        outfile.اكتب('        return tmp;\n')
        outfile.اكتب('    }();\n')
        outfile.اكتب('    return s;\n')
        outfile.اكتب('}\n')
        outfile.اكتب(f'const char* wgsl_{shader_name} = wgsl_{shader_name}_str().c_str();\n\n')


تعریف اصلي():
    parser = argparse.دلیل_Parser()
    parser.جمع_کړه_دلیل("--input_dir", required=ریښتیا)
    parser.جمع_کړه_دلیل("--output_file", required=ریښتیا)
    دلیلونه_2 = parser.parse_دلیلونه()

    سره open(دلیلونه_2.output_file, "w", encoding="utf-8") كـ بهر:
        بهر.اكتب("// Auto-generated shader embedding\n")
        بهر.اكتب("#include <string>\n\n")
        لكل fname في sorted(os.listdir(دلیلونه_2.ننوت_dir)):
            إذا fname.endswith(".wgsl"):
                shader_لار = os.لار.join(دلیلونه_2.ننوت_dir, fname)
                shader_name = fname.replace(".wgsl", "")

                سره open(shader_لار, "r", encoding="utf-8") كـ f:
                    shader_کوډ = f.اقرأ()

                write_shader(shader_name, shader_کوډ, هیڅ, بهر, دلیلونه_2.ننوت_dir)


إذا __name__ == "__main__":
    اصلي()

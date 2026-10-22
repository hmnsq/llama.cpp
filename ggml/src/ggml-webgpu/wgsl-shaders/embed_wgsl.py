ਆਯਾਤ os
ਆਯਾਤ re
ਆਯਾਤ argparse


ਪਰਿਭਾਸ਼ਾ expand_includes(shader, ਨਿਵੇਸ਼_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_ਨਮੂਨਾ = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    ਪਰਿਭਾਸ਼ਾ replacer(ਮੇਲ):
        fname = ਮੇਲ.ਸਮੂਹ(1)
        file_ਰਸਤਾ = os.ਰਸਤਾ.join(ਨਿਵੇਸ਼_dir, fname)
        ਜੇ ਨਹੀਂ os.ਰਸਤਾ.exists(file_ਰਸਤਾ):
            ਉਠਾਓ File_Not_ਮਿਲਿਆ_Error(f"Included file not found: {file_path}")
        ਨਾਲ open(file_ਰਸਤਾ, "r", encoding="utf-8") ਇਵ f:
            included_ਕੋਡ = f.ਪਠ()
        # Recursively expand includes inside the included file
        ਵਾਪਸ expand_includes(included_ਕੋਡ, ਨਿਵੇਸ਼_dir)

    ਵਾਪਸ include_ਨਮੂਨਾ.sub(replacer, shader)


ਪਰਿਭਾਸ਼ਾ chunk_shader(shader_ਕੋਡ, max_chunk_len=60000):
    """Split shader_code into safe raw-string sized chunks."""
    ਵਾਪਸ [shader_ਕੋਡ[i : i + max_chunk_len] ਲਈ i ਵਿੱਚ ਸੀਮਾ(0, ਲੰਬਾਈ(shader_ਕੋਡ), max_chunk_len)]


ਪਰਿਭਾਸ਼ਾ raw_delim(shader_ਕੋਡ):
    """Pick a raw-string delimiter that does not appear in the shader."""
    delim = "wgsl"
    ਜਦੋਂ f"){delim}\"" ਵਿੱਚ shader_ਕੋਡ:
        delim += "_x"
    ਵਾਪਸ delim


ਪਰਿਭਾਸ਼ਾ write_shader(shader_name, shader_ਕੋਡ, output_dir, outfile, ਨਿਵੇਸ਼_dir):
    shader_ਕੋਡ = expand_includes(shader_ਕੋਡ, ਨਿਵੇਸ਼_dir)

    ਜੇ output_dir:
        wgsl_filename = os.ਰਸਤਾ.join(output_dir, f"{shader_name}.wgsl")
        ਨਾਲ open(wgsl_filename, "w", encoding="utf-8") ਇਵ f_ਬਾਹਰਲਾ:
            f_ਬਾਹਰਲਾ.ਲਿਖ(shader_ਕੋਡ)

    delim = raw_delim(shader_ਕੋਡ)
    chunks = chunk_shader(shader_ਕੋਡ)

    ਜੇ ਲੰਬਾਈ(chunks) == 1:
        outfile.ਲਿਖ(f'const char* wgsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
    ਨਹੀਂਤਾਂ:
        ਲਈ idx, chunk ਵਿੱਚ enumerate(chunks):
            outfile.ਲਿਖ(f'static const char wgsl_{shader_name}_part{idx}[] = R"{delim}({chunk}){delim}";\n\n')
        outfile.ਲਿਖ(f'static const std::string& wgsl_{shader_name}_str() {{\n')
        outfile.ਲਿਖ('    static const std::string s = []{\n')
        outfile.ਲਿਖ('        std::string tmp;\n')
        outfile.ਲਿਖ(f'        tmp.reserve({len(shader_code)});\n')
        ਲਈ idx ਵਿੱਚ ਸੀਮਾ(ਲੰਬਾਈ(chunks)):
            outfile.ਲਿਖ(f'        tmp.append(wgsl_{shader_name}_part{idx});\n')
        outfile.ਲਿਖ('        return tmp;\n')
        outfile.ਲਿਖ('    }();\n')
        outfile.ਲਿਖ('    return s;\n')
        outfile.ਲਿਖ('}\n')
        outfile.ਲਿਖ(f'const char* wgsl_{shader_name} = wgsl_{shader_name}_str().c_str();\n\n')


ਪਰਿਭਾਸ਼ਾ ਮੁੱਖ():
    parser = argparse.ਦਲੀਲ_Parser()
    parser.ਜੋੜੋ_ਦਲੀਲ("--input_dir", required=ਸੱਚ)
    parser.ਜੋੜੋ_ਦਲੀਲ("--output_file", required=ਸੱਚ)
    ਦਲੀਲਾਂ_2 = parser.parse_ਦਲੀਲਾਂ()

    ਨਾਲ open(ਦਲੀਲਾਂ_2.output_file, "w", encoding="utf-8") ਇਵ ਬਾਹਰਲਾ:
        ਬਾਹਰਲਾ.ਲਿਖ("// Auto-generated shader embedding\n")
        ਬਾਹਰਲਾ.ਲਿਖ("#include <string>\n\n")
        ਲਈ fname ਵਿੱਚ sorted(os.listdir(ਦਲੀਲਾਂ_2.ਨਿਵੇਸ਼_dir)):
            ਜੇ fname.endswith(".wgsl"):
                shader_ਰਸਤਾ = os.ਰਸਤਾ.join(ਦਲੀਲਾਂ_2.ਨਿਵੇਸ਼_dir, fname)
                shader_name = fname.replace(".wgsl", "")

                ਨਾਲ open(shader_ਰਸਤਾ, "r", encoding="utf-8") ਇਵ f:
                    shader_ਕੋਡ = f.ਪਠ()

                write_shader(shader_name, shader_ਕੋਡ, ਕੁਝ_ਨਹੀਂ, ਬਾਹਰਲਾ, ਦਲੀਲਾਂ_2.ਨਿਵੇਸ਼_dir)


ਜੇ __name__ == "__main__":
    ਮੁੱਖ()

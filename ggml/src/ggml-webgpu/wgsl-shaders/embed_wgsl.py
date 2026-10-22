இறக்குமதி os
இறக்குமதி re
இறக்குமதி argparse


def expand_includes(shader, உள்ளீடு_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_pattern = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    def replacer(match):
        fname = match.group(1)
        file_path = os.path.join(உள்ளீடு_dir, fname)
        எனில் இல்லை os.path.exists(file_path):
            எழுப்பு FileNotFoundError(f"Included file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") இவ f:
            included_code = f.பட()
        # Recursively expand includes inside the included file
        திரும்பு expand_includes(included_code, உள்ளீடு_dir)

    திரும்பு include_pattern.sub(replacer, shader)


def chunk_shader(shader_code, max_chunk_len=60000):
    """Split shader_code into safe raw-string sized chunks."""
    திரும்பு [shader_code[i : i + max_chunk_len] க்கு i உள் range(0, len(shader_code), max_chunk_len)]


def raw_delim(shader_code):
    """Pick a raw-string delimiter that does not appear in the shader."""
    delim = "wgsl"
    போது f"){delim}\"" உள் shader_code:
        delim += "_x"
    திரும்பு delim


def write_shader(shader_name, shader_code, output_dir, outfile, உள்ளீடு_dir):
    shader_code = expand_includes(shader_code, உள்ளீடு_dir)

    எனில் output_dir:
        wgsl_filename = os.path.join(output_dir, f"{shader_name}.wgsl")
        with open(wgsl_filename, "w", encoding="utf-8") இவ f_out:
            f_out.லிக(shader_code)

    delim = raw_delim(shader_code)
    chunks = chunk_shader(shader_code)

    எனில் len(chunks) == 1:
        outfile.லிக(f'const char* wgsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
    இல்லையெனில்:
        க்கு idx, chunk உள் enumerate(chunks):
            outfile.லிக(f'static const char wgsl_{shader_name}_part{idx}[] = R"{delim}({chunk}){delim}";\n\n')
        outfile.லிக(f'static const std::string& wgsl_{shader_name}_str() {{\n')
        outfile.லிக('    static const std::string s = []{\n')
        outfile.லிக('        std::string tmp;\n')
        outfile.லிக(f'        tmp.reserve({len(shader_code)});\n')
        க்கு idx உள் range(len(chunks)):
            outfile.லிக(f'        tmp.append(wgsl_{shader_name}_part{idx});\n')
        outfile.லிக('        return tmp;\n')
        outfile.லிக('    }();\n')
        outfile.லிக('    return s;\n')
        outfile.லிக('}\n')
        outfile.லிக(f'const char* wgsl_{shader_name} = wgsl_{shader_name}_str().c_str();\n\n')


def முதன்மை():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=உண்மை)
    parser.add_argument("--output_file", required=உண்மை)
    args = parser.parse_args()

    with open(args.output_file, "w", encoding="utf-8") இவ out:
        out.லிக("// Auto-generated shader embedding\n")
        out.லிக("#include <string>\n\n")
        க்கு fname உள் sorted(os.listdir(args.உள்ளீடு_dir)):
            எனில் fname.endswith(".wgsl"):
                shader_path = os.path.join(args.உள்ளீடு_dir, fname)
                shader_name = fname.replace(".wgsl", "")

                with open(shader_path, "r", encoding="utf-8") இவ f:
                    shader_code = f.பட()

                write_shader(shader_name, shader_code, None, out, args.உள்ளீடு_dir)


எனில் __name__ == "__main__":
    முதன்மை()

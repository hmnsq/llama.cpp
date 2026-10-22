ଆମଦାନୀ os
ଆମଦାନୀ re
ଆମଦାନୀ argparse


def expand_includes(shader, ନିବେଶ_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_pattern = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    def replacer(match):
        fname = match.group(1)
        file_path = os.path.join(ନିବେଶ_dir, fname)
        ଯଦି ନୁହେଁ os.path.exists(file_path):
            ଉଠାଅ FileNotFoundError(f"Included file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") ଇଵ f:
            included_code = f.ପଠ()
        # Recursively expand includes inside the included file
        ଫେରନ୍ତୁ expand_includes(included_code, ନିବେଶ_dir)

    ଫେରନ୍ତୁ include_pattern.sub(replacer, shader)


def chunk_shader(shader_code, max_chunk_len=60000):
    """Split shader_code into safe raw-string sized chunks."""
    ଫେରନ୍ତୁ [shader_code[i : i + max_chunk_len] ପାଇଁ i ଭିତରେ range(0, len(shader_code), max_chunk_len)]


def raw_delim(shader_code):
    """Pick a raw-string delimiter that does not appear in the shader."""
    delim = "wgsl"
    ଯେତେବେଳେ f"){delim}\"" ଭିତରେ shader_code:
        delim += "_x"
    ଫେରନ୍ତୁ delim


def write_shader(shader_name, shader_code, output_dir, outfile, ନିବେଶ_dir):
    shader_code = expand_includes(shader_code, ନିବେଶ_dir)

    ଯଦି output_dir:
        wgsl_filename = os.path.join(output_dir, f"{shader_name}.wgsl")
        with open(wgsl_filename, "w", encoding="utf-8") ଇଵ f_out:
            f_out.ଲିଖ(shader_code)

    delim = raw_delim(shader_code)
    chunks = chunk_shader(shader_code)

    ଯଦି len(chunks) == 1:
        outfile.ଲିଖ(f'const char* wgsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
    ଅନ୍ୟଥା:
        ପାଇଁ idx, chunk ଭିତରେ enumerate(chunks):
            outfile.ଲିଖ(f'static const char wgsl_{shader_name}_part{idx}[] = R"{delim}({chunk}){delim}";\n\n')
        outfile.ଲିଖ(f'static const std::string& wgsl_{shader_name}_str() {{\n')
        outfile.ଲିଖ('    static const std::string s = []{\n')
        outfile.ଲିଖ('        std::string tmp;\n')
        outfile.ଲିଖ(f'        tmp.reserve({len(shader_code)});\n')
        ପାଇଁ idx ଭିତରେ range(len(chunks)):
            outfile.ଲିଖ(f'        tmp.append(wgsl_{shader_name}_part{idx});\n')
        outfile.ଲିଖ('        return tmp;\n')
        outfile.ଲିଖ('    }();\n')
        outfile.ଲିଖ('    return s;\n')
        outfile.ଲିଖ('}\n')
        outfile.ଲିଖ(f'const char* wgsl_{shader_name} = wgsl_{shader_name}_str().c_str();\n\n')


def ମୁଖ୍ୟ():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=ସତ୍ୟ)
    parser.add_argument("--output_file", required=ସତ୍ୟ)
    args = parser.parse_args()

    with open(args.output_file, "w", encoding="utf-8") ଇଵ out:
        out.ଲିଖ("// Auto-generated shader embedding\n")
        out.ଲିଖ("#include <string>\n\n")
        ପାଇଁ fname ଭିତରେ sorted(os.listdir(args.ନିବେଶ_dir)):
            ଯଦି fname.endswith(".wgsl"):
                shader_path = os.path.join(args.ନିବେଶ_dir, fname)
                shader_name = fname.replace(".wgsl", "")

                with open(shader_path, "r", encoding="utf-8") ଇଵ f:
                    shader_code = f.ପଠ()

                write_shader(shader_name, shader_code, None, out, args.ନିବେଶ_dir)


ଯଦି __name__ == "__main__":
    ମୁଖ୍ୟ()

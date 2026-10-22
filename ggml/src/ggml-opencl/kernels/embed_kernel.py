#

আমদানি sys
আমদানি logging
logger = logging.getLogger("opencl-embed-kernel")


def প্রধান():
    logging.basicConfig(level=logging.INFO)

    যদি len(sys.argv) != 3:
        logger.info("Usage: python embed_kernel.py <input_file> <output_file>")
        sys.exit(1)

    ifile = open(sys.argv[1], "r")
    ofile = open(sys.argv[2], "w")

    জন্য i মধ্যে ifile:
        ofile.লিখ('R"({})"\n'.format(i))

    ifile.close()
    ofile.close()


যদি __name__ == "__main__":
    প্রধান()

#

આયાત sys
આયાત logging
logger = logging.getLogger("opencl-embed-kernel")


def મુખ્ય():
    logging.basicConfig(level=logging.INFO)

    જો len(sys.argv) != 3:
        logger.info("Usage: python embed_kernel.py <input_file> <output_file>")
        sys.exit(1)

    ifile = open(sys.argv[1], "r")
    ofile = open(sys.argv[2], "w")

    માટે i માં ifile:
        ofile.લિખ('R"({})"\n'.format(i))

    ifile.close()
    ofile.close()


જો __name__ == "__main__":
    મુખ્ય()

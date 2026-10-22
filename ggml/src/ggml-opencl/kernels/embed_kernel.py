#

आयात sys
आयात logging
logger = logging.getLogger("opencl-embed-kernel")


def मुख्य():
    logging.basicConfig(level=logging.INFO)

    अगर len(sys.argv) != 3:
        logger.info("Usage: python embed_kernel.py <input_file> <output_file>")
        sys.exit(1)

    ifile = open(sys.argv[1], "r")
    ofile = open(sys.argv[2], "w")

    क्रम i in ifile:
        ofile.म_लिखो('R"({})"\n'.format(i))

    ifile.बंद()
    ofile.बंद()


अगर __name__ == "__main__":
    मुख्य()

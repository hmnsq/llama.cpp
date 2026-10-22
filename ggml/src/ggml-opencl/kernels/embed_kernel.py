#

ഇറക്കുമതി sys
ഇറക്കുമതി logging
logger = logging.getLogger("opencl-embed-kernel")


def പ്രധാന():
    logging.basicConfig(level=logging.INFO)

    എങ്കിൽ len(sys.argv) != 3:
        logger.info("Usage: python embed_kernel.py <input_file> <output_file>")
        sys.exit(1)

    ifile = open(sys.argv[1], "r")
    ofile = open(sys.argv[2], "w")

    വേണ്ടി i ഇൽ ifile:
        ofile.ലിഖ('R"({})"\n'.format(i))

    ifile.close()
    ofile.close()


എങ്കിൽ __name__ == "__main__":
    പ്രധാന()

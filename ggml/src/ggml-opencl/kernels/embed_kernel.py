#

ଆମଦାନୀ sys
ଆମଦାନୀ logging
logger = logging.getLogger("opencl-embed-kernel")


def ମୁଖ୍ୟ():
    logging.basicConfig(level=logging.INFO)

    ଯଦି len(sys.argv) != 3:
        logger.info("Usage: python embed_kernel.py <input_file> <output_file>")
        sys.exit(1)

    ifile = open(sys.argv[1], "r")
    ofile = open(sys.argv[2], "w")

    ପାଇଁ i ଭିତରେ ifile:
        ofile.ଲିଖ('R"({})"\n'.format(i))

    ifile.close()
    ofile.close()


ଯଦି __name__ == "__main__":
    ମୁଖ୍ୟ()

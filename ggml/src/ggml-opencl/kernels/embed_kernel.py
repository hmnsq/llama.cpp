#

ಆಮದು sys
ಆಮದು logging
logger = logging.getLogger("opencl-embed-kernel")


ವ್ಯಾಖ್ಯಾನ ಮುಖ್ಯ():
    logging.basicConfig(level=logging.INFO)

    ಆದರೆ ಉದ್ದ(sys.argv) != 3:
        logger.info("Usage: python embed_kernel.py <input_file> <output_file>")
        sys.exit(1)

    ifile = open(sys.argv[1], "r")
    ofile = open(sys.argv[2], "w")

    ಗಾಗಿ i ಒಳಗೆ ifile:
        ofile.ಲಿಖ('R"({})"\n'.format(i))

    ifile.close()
    ofile.close()


ಆದರೆ __name__ == "__main__":
    ಮುಖ್ಯ()

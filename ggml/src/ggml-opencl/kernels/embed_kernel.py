#

import sys
import logging
logger = logging.getLogger("opencl-embed-kernel")


def 主要():
    logging.basicConfig(level=logging.INFO)

    如果 len(sys.argv) != 3:
        logger.info("Usage: python embed_kernel.py <input_file> <output_file>")
        sys.exit(1)

    ifile = open(sys.argv[1], "r")
    ofile = open(sys.argv[2], "w")

    为 i in ifile:
        ofile.write('R"({})"\n'.format(i))

    ifile.close()
    ofile.close()


如果 __name__ == "__main__":
    主要()

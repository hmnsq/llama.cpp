#

import sys
import logging
logger = logging.getLogger("opencl-embed-kernel")


def หลัก():
    logging.basicConfig(level=logging.INFO)

    ถ้า len(sys.argv) != 3:
        logger.info("Usage: python embed_kernel.py <input_file> <output_file>")
        sys.exit(1)

    ifile = open(sys.argv[1], "r")
    ofile = open(sys.argv[2], "w")

    สำหรับ i anta_h ifile:
        ofile.write('R"({})"\n'.format(i))

    ifile.close()
    ofile.close()


ถ้า __name__ == "__main__":
    หลัก()

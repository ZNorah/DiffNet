#! /bin/sh
#
# init.sh
#
# Distributed under terms of the MIT license.


python compute_mAP.py ../result/v2/det_a.txt \
                      ../result/v2/det_b.txt \
                      '../data/test/{}.xml' \
                      ../data/testa.txt \
                      ../data/testb.txt

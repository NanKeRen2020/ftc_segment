About
=====

Fine-to-Coarse histogram segmentation algorithm with contrario approach 

and 1D histogram local integral for character segmention.


This project implements a fast and auto character segment algorithm in C++.

The character segment algorithm is based on the FTC (Fine-to-Coarse) algorithm for histogram

segmentation, presented by Delon et al. in 2007. This algorithm also uses a non-parametric a

contrario approach to segment the 1D histogram into its meaningful modes. Besides, this implement

adopt a new tirck which is called local integral to refine to find the minimums of 1D histogram

of canny edge image project.


Environments
=============

Ubuntu1604  OpenCV3.4.x


Build & Usage
==============

cd ftc_segment

step1:  g++ -g -o test_segment_ftc test_segment_ftc.cpp std_utils.cpp libftc.cpp -I.  `pkg-config --cflags --libs opencv`


step2:  ./test_segment_ftc  test.png 
 
original image

![image](https://github.com/NanKeRen2020/ftc_segment/blob/main/test.png)

segment text rows image

![image](https://github.com/NanKeRen2020/ftc_segment/blob/main/segment_rows.png)

segment text first row chars image

![image](https://github.com/NanKeRen2020/ftc_segment/blob/main/segment_row_chars1.png)

segment text second row chars image

![image](https://github.com/NanKeRen2020/ftc_segment/blob/main/segment_row_chars2.png)

segment text third row chars image

![image](https://github.com/NanKeRen2020/ftc_segment/blob/main/segment_row_chars3.png)



References
==========

[1] Delon J, Desolneux A, Lisani J L, et al. A Nonparametric Approach for Histogram Segmentation[J]. 

IEEE Transactions on Image Processing A Publication of the IEEE Signal Processing Society, 2007, 16(1):253. 

[2] A. Desolneux, L. Moisan, and J-M. Morel, Meaningful alignments, International Journal

of Computer Vision, 40 (2000), pp. 7–23. https://doi.org/10.1023/A:1026593302236.

[3] A. Desolneux, L. Moisan, and J-M. Morel, From Gestalt theory to image analysis: a probabilistic approach, 

Interdisciplinary Applied Mathematics, Springer, 2008. ISBN 978-0-387-74378-3.

[4] Desolneux A ,  Moisan L ,  More J M . A grouping principle and four applications[J]. IEEE Computer Society, 2003.

# Resources

Every dataset, method and library the project relies on, numbered in order of first
appearance in the companion article. Sample scenes are used under the license noted
for each entry; any images derived from them are derivative works.

1. **NWPU-RESISC45** — G. Cheng, J. Han, X. Lu, *Remote Sensing Image Scene
   Classification: Benchmark and State of the Art*, Proceedings of the IEEE, 2017.
   45 classes × 700 images (256×256). Imagery extracted from Google Earth
   (© the respective imagery providers). **Distributed for research and educational
   use only — NOT cleared for commercial republication.** The scene tiles shown in the
   article (montage, retrieval strip) are reproduced at reduced size for research
   illustration; for a commercial deployment they must be replaced with a
   commercially-licensed scene source.
   https://arxiv.org/abs/1703.00121

2. **Haralick texture features** — R. M. Haralick, K. Shanmugam, I. Dinstein,
   *Textural Features for Image Classification*, IEEE Transactions on Systems, Man, and
   Cybernetics, 1973. The 13 grey-level co-occurrence descriptors used here.
   https://doi.org/10.1109/TSMC.1973.4309314

3. **Mahotas** — L. P. Coelho, *Mahotas: Open source software for scriptable computer
   vision*, Journal of Open Research Software, 2013. MIT License. Computes the Haralick
   descriptors from the co-occurrence matrices.
   https://github.com/luispedro/mahotas

4. **scikit-image** — S. van der Walt et al., *scikit-image: image processing in
   Python*, PeerJ, 2014. BSD-3-Clause. Its `graycomatrix` renders the GLCM figure.
   https://scikit-image.org

5. **scikit-learn** — F. Pedregosa et al., *Scikit-learn: Machine Learning in Python*,
   JMLR, 2011. BSD-3-Clause. Provides the SVM, MLP, K-means and standardization.
   https://scikit-learn.org

6. **OPTICS** — M. Ankerst, M. M. Breunig, H.-P. Kriegel, J. Sander, *OPTICS: Ordering
   Points To Identify the Clustering Structure*, ACM SIGMOD, 1999. The density-based
   clustering algorithm.
   https://doi.org/10.1145/304182.304187

7. **scikit-image GLCM gallery example** — Stéfan van der Walt & the scikit-image
   contributors, *Gray-Level Co-Occurrence Matrices* (gallery plot), derived work
   distributed under the BSD-3-Clause license of scikit-image. The GLCM demo image
   (Fig 5) is a derivative of this gallery example.
   https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_glcm.html

Supporting libraries: NumPy (BSD-3-Clause), SciPy (BSD-3-Clause), Matplotlib
(Matplotlib/PSF-based license), Pillow (MIT-CMU/HPND).

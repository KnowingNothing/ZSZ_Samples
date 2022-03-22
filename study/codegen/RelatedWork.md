# Deep learning compilers (kernel-level)
- Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines ([PLDI'13 paper by Jonathan Ragan-Kelley et al.](http://people.csail.mit.edu/jrk/halide-pldi13.pdf), [open source code](https://halide-lang.org/))

- The Tensor Algebra Compiler ([OOPSLA'17 paper by Fredrik Kjolstad et al.](https://dl.acm.org/doi/pdf/10.1145/3133901), [open source code](http://tensor-compiler.org/))

- TVM: An Automated End-to-End Optimizing Compiler for Deep Learning ([OSDI'18 paper by Tianqi Chen et al.](https://www.usenix.org/conference/osdi18/presentation/chen), [open source code](https://tvm.apache.org/))

- DLVM: A modern compiler infrastructure for deep learning systems ([ICLR'18 paper by Richard Wei et al.](https://arxiv.org/pdf/1711.03016.pdf), [open source code](https://dlvm-team.github.io/))

- Diesel: DSL for linear algebra and neural net computations on GPUs ([MAPL'18 paper by Venmugil Elango et al.](https://dl.acm.org/doi/abs/10.1145/3211346.3211354), open source code)

- Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions ([arXiv'18 paper by Nicolas Vasilache et al.](https://arxiv.org/abs/1802.04730), [open source code](https://facebookresearch.github.io/TensorComprehensions/))

- Tiramisu: A polyhedral compiler for expressing fast and portable code ([CGO'19 paper by Riyadh Baghdadi et al.](https://arxiv.org/abs/1804.10694), [open source code](http://tiramisu-compiler.org/))

- Triton: an intermediate language and compiler for tiled neural network computations ([MAPL'19 paper by Philippe Tillet et al.](https://dl.acm.org/doi/abs/10.1145/3315508.3329973?casa_token=w0MaltEBfKYAAAAA%3AX27ScRTBiDR3WfL1VKTuU34wXJhr0r4H32JEcFe-DkmkJogCDG9dG7Tvp45sR9aB5tUKwky_hE25xg), [open source code](https://github.com/ptillet/triton))

- Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks (nnfusion) ([OSDI'20 paper by Lingxiao Ma et al.](https://www.usenix.org/conference/osdi20/presentation/ma), [open source code](https://github.com/microsoft/nnfusion))

# Deep learning compilers (graph-level)
- Glow: Graph Lowering Compiler Techniques for Neural Networks ([arXiv'18 paper by Nadav Rotem et al.](https://arxiv.org/abs/1805.00907), [open source code](https://github.com/pytorch/glow))

- Intel nGraph: An Intermediate Representation, Compiler, and Executor for Deep Learning ([arXiv'18 paper by Scott Cyphers et al.](https://arxiv.org/abs/1801.08058), [open source code](https://www.ngraph.ai/))

- TASO: The Tensor Algebra SuperOptimizer for Deep Learning ([SOSP'19 paper by Zhihao Jia et al.](https://dl.acm.org/doi/pdf/10.1145/3341301.3359630?casa_token=dYBNBVyhmV0AAAAA:zD-feoFh6susJzp9mE6KKsffaV94Ec-LJxJL-GQoA_16mTjXtYL3q0Xqiuh5jdD5PAuhyHH1lPWkGQ), [open source code](https://github.com/jiazhihao/TASO))

- Relay: A High-Level Compiler for Deep Learning ([arXiv'19 paper by Jared Roesch et al.](https://arxiv.org/pdf/1904.08368.pdf), [open source code](https://tvm.apache.org/))

- A Tensor Compiler for Unified Machine Learning Prediction Serving (Hummingbird) ([OSDI'20 paper by Supun Nakandala et al.](https://www.usenix.org/conference/osdi20/presentation/nakandala), [open source code](https://github.com/microsoft/hummingbird))

- MLIR: A Compiler Infrastructure for the End of Moore's Law ([arXiv'20 paper by Chris Lattner et al.](https://arxiv.org/abs/2002.11054), [open source code](https://mlir.llvm.org/))


# Automatic tuning frameworks
- Automatically Scheduling Halide Image Processing Pipelines ([SIGGRAPH'16 paper by Ravi Teja Mullapudi et al.](http://graphics.cs.cmu.edu/projects/halidesched/), [open source code](https://halide-lang.org/))

- Learning to Optimize Tensor Programs ([NeurIPS'18 paper by Tianqi Chen et al.](https://arxiv.org/abs/1805.08166), [open source code](https://tvm.apache.org/))

- Learning to Optimize Halide with Tree Search and Random Programs ([SIGGRAPH'19 paper by Andrew Adams et al.](https://halide-lang.org/papers/autoscheduler2019.html), [open source code](https://halide-lang.org/))

- Chameleon: Adaptive Code Optimization for Expedited Deep Neural Network Compilation ([ICLR'20 paper by Byung Hoon Ahn et al.](https://openreview.net/forum?id=rygG4AVFvH), [open source code](https://github.com/anony-sub/chameleon))

- Optimizing the Memory Hierarchy by Compositing Automatic Transformations on Computations and Data ([MICRO'20 paper by Jie Zhao et al.](https://www.microarch.org/micro53/papers/738300a427.pdf), open source code)

- ProTuner: Tuning Programs with Monte Carlo Tree Search ([arXiv'20 paper by Ameer Haj-Ali et al.](https://arxiv.org/abs/2005.13685), open source code)

- FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System ([ASPLOS'20 paper by Size Zheng et al.](https://dl.acm.org/doi/abs/10.1145/3373376.3378508?casa_token=2mWk5Qp3Ll8AAAAA%3A67phDw6-xWqKmo9A2EMXhVwl8KhHOGU_MeYc0sGiORNtNQTP_IDYmTW1gFtapsPuV48i1U5FRmRNfg), [open source code](https://github.com/KnowingNothing/FlexTensor))

- Schedule Synthesis for Halide Pipelines on GPUs ([TACO'20 paper by Sioutas Savvas et al.](https://dl.acm.org/doi/fullHtml/10.1145/3406117), open source code)

- Ansor: Generating High-Performance Tensor Programs for Deep Learning ([OSDI'20 paper by Lianmin Zheng et al.](https://arxiv.org/abs/2006.06762), [open source code](https://tvm.apache.org/))

# Compiler techniques and optimizations (kernel-level)
- XXX ([XXX paper by XXX et al.](), [open source code]())

# Compiler techniques and optimizations (graph-level)

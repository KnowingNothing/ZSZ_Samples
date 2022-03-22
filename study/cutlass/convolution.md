读一个GEMM的实现感觉经常陷入大量相似的函数中不能找到方向，但是convolution就好多了，因为这个算子支持还特别弱，文件少，内容纵深好掌握。

## 一、convolution top header
从convolution/convolution.h入手，所有内容都定义在cutlass/convolution的namespace下。在这里，提供以下结构：
#### **ConvType : uint32_t的枚举类型，包含**
  - kConvolution
  - kBatchConvolution
  - kLocal
  - kLocalShare
#### **ConvParamBase : 一个结构体**
  1. **定义内部类型含有：**
  - typedef int Index
  - typedef Coord<2, Index> Vector
  2. **存储内容为：**
  - N, IC, OC, HW, NHW : Index
  - src_shape, filter_shape, dst_shape : Vector
  - stride, padding, dilate : Vector
  3. **提供构造方法包括：**
  - 空白构造，赋值0或0 Vector
  - 给值构造，HW, NHW都是指dst的shape计算出来的 (x2)
  4. **提供成员方法：**
  - 对内部存储内容进行查询/获取引用 n(), ci(), co(), hi(), wi(), fh(), fw(), ho(), wo(), hw(), nhw(), stride_h(), stride_w(), pad_h(), pad_w(), dilate_h(), dilate_w()
#### ConvParam：一个模板结构体
  1. public ConvParamBase
  2. template\<ConvType conv_type_\>
  3. **定义内部类型含有：**
  - using Base = ConvParamBase
  4. **存储内容为：**
  - conv_type : static ConvType const conv_type = conv_type_
  5. **提供构造方法包括：**
  - 空白构造
  - 给值构造 (x2)


## 二、device level header
在device目录下，有两个header file，一个是convolution.h，另一个是default_convolution_configuration.h
### convolution.h
在cutlass/convolution/device namespace下，提供一个类
#### Convolution : 一个类
  1. 
  ```c++
  template <
        /// Element type for Src Tensor operand
        typename ElementSrc_,
        /// Layout type for Src Tensor operand
        typename LayoutSrc_,
        /// Element type for Filter Tensor operand
        typename ElementFilter_,
        /// Layout type for Filter Tensor operand
        typename LayoutFilter_,
        /// Element type for Dst and Z Tensor operands
        typename ElementDst_,
        /// Layout type for Dst and Z Tensor operands
        typename LayoutDst_,
        /// Element type for Bias Tensor operands
        typename ElementBias_,
        /// Layout type for Bias Tensor operands
        typename LayoutBias_,
        /// Element type for internal accumulation
        typename ElementAccumulator_,
        /// Convolution Type
        ConvType ConvolutionType = ConvType::kConvolution,
        /// Operator class tag
        typename OperatorClass_ = arch::OpClassSimt,
        /// Tag indicating architecture to tune for
        typename ArchTag_ = arch::Sm61,
        /// Threadblock-level tile size (concept: GemmShape)
        typename ThreadblockShape_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::ThreadblockShape,
        /// Warp-level tile size (concept: GemmShape)
        typename WarpShape_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::WarpShape,
        /// Instruction-level tile size (concept: GemmShape)
        typename InstructionShape_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::InstructionShape,
        /// Epilogue output operator
        typename EpilogueOutputOp_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle_ = typename threadblock::
                ConvolutionCxRSKxThreadblockSwizzle<ConvolutionType>,
        /// Number of stages used in the pipelined mainloop
        int Stages = DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::kStages,
        /// Access granularity of Src Tensor in units of elements
        int AlignmentSrc = DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int AlignmentFilter = DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::kAlignmentFilter,
        /// whether use special optimization for convolution 1x1
        bool NeedLoadFromConstMem = true,
        /// Operation performed by Convolution
        typename Operator_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::Operator>
  ```
  2. **内部定义类型：**
  ```c++
    using ElementSrc = ElementSrc_;
    using LayoutSrc = LayoutSrc_;
    using TensorRefSrc = TensorRef<ElementSrc const, LayoutSrc>;
    using ElementFilter = ElementFilter_;
    using LayoutFilter = LayoutFilter_;
    using TensorRefFilter = TensorRef<ElementFilter const, LayoutFilter>;
    using ElementBias = ElementBias_;
    using LayoutBias = LayoutBias_;
    using TensorRefBias = TensorRef<ElementBias const, LayoutBias>;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using TensorRefDst = TensorRef<ElementDst const, LayoutDst>;
    using TensorRefZ = TensorRef<ElementDst, LayoutDst>;
    using ElementAccumulator = ElementAccumulator_;
    using OperatorClass = OperatorClass_;
    using ArchTag = ArchTag_;
    using ThreadblockShape = ThreadblockShape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using EpilogueOutputOp = EpilogueOutputOp_;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    using Operator = Operator_;
    using ConvolutionParameter = ConvParam<kConvolutionType>;
    using ConvolutionKernel = typename kernel::DefaultConvolution<
            ElementSrc, LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementDst, LayoutDst, ElementAccumulator,
            kConvolutionType, OperatorClass, ArchTag, ThreadblockShape,
            WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle,
            kStages, Operator, kNeedLoadFromConstMem>::ConvolutionKernel;
    using TransformSrc = typename ConvolutionKernel::Mma::TransformSrc;
    using TransformFilter = typename ConvolutionKernel::Mma::TransformFilter;

    struct Arguments {
        ConvolutionParameter conv_param;
        TensorRef<ElementSrc const, LayoutSrc> ref_src;
        TensorRef<ElementFilter const, LayoutFilter> ref_filter;
        TensorRef<ElementBias const, LayoutBias> ref_bias;
        TensorRef<ElementDst const, LayoutDst> ref_z;
        TensorRef<ElementDst, LayoutDst> ref_dst;
        typename EpilogueOutputOp::Params epilogue;
        typename TransformSrc::Params transform_src;
        typename TransformFilter::Params transform_filter;

        /// Default ctor
        CUTLASS_HOST_DEVICE
        Arguments() : conv_param(ConvolutionParameter()) {}

        /// Constructs an Arguments structure
        CUTLASS_HOST_DEVICE
        Arguments(ConvolutionParameter conv_param_,
                  TensorRef<ElementSrc const, LayoutSrc> ref_src_,
                  TensorRef<ElementFilter const, LayoutFilter> ref_filter_,
                  TensorRef<ElementBias const, LayoutBias> ref_bias_,
                  TensorRef<ElementDst const, LayoutDst> ref_z_,
                  TensorRef<ElementDst, LayoutDst> ref_dst_,
                  typename EpilogueOutputOp::Params epilogue_ =
                          typename EpilogueOutputOp::Params(),
                  typename TransformSrc::Params transform_src_ =
                          typename TransformSrc::Params(),
                  typename TransformFilter::Params transform_filter_ =
                          typename TransformFilter::Params())
                : conv_param(conv_param_),
                  ref_src(ref_src_),
                  ref_filter(ref_filter_),
                  ref_bias(ref_bias_),
                  ref_z(ref_z_),
                  ref_dst(ref_dst_),
                  epilogue(epilogue_),
                  transform_src(transform_src_),
                  transform_filter(transform_filter_) {}
    };
  ```
  3. **内部存储内容：**
  ```c++
  static const ConvType kConvolutionType = ConvolutionType;
  static int const kStages = Stages;
  static int const kAlignmentSrc = AlignmentSrc;
  static int const kAlignmentFilter = AlignmentFilter;
  static int const kAlignmentDst = EpilogueOutputOp::kCount;
  static bool const kNeedLoadFromConstMem = NeedLoadFromConstMem;

private:
  typename ConvolutionKernel::Params params_;
  ```
  4. **提供构造方法：**
  - 空白
  5. **提供成员方法：**
  - can_implement
  - get_workspace_size
  - initialize
  - update
  - run
  - operator() (x2)
  6. **解释：**
  - 两个operator()的重载是两种入口，一种提供Arguments作为参数的，是会调用initialize初始化params_然后调用run，另一个只会调用run()
  - initialize通过ThreadblockSwizzle获取grid_shape，然后初始化params_
  ```c++
  params_ = typename ConvolutionKernel::Params{
                args.conv_param,
                grid_shape,
                args.ref_src.non_const_ref(),
                args.ref_filter.non_const_ref(),
                args.ref_bias.non_const_ref(),
                args.ref_z.non_const_ref(),
                args.ref_dst,
                args.epilogue,
                args.transform_src,
                args.transform_filter,
                static_cast<int*>(workspace)};
  ```
  - run函数中使用<<< >>> launch kernel


### default_convolution_configuration.h
在cutlass/convolution/device namespace下，提供三个结构体模板，用来做特异化
#### DefaultConvolutionConfiguration： 空结构体
 1. 
 ```c++
template <typename OperatorClass, typename ArchTag, typename ElementSrc,
          typename ElementFilter, typename ElementDst,
          typename ElementAccumulator>
 ```
####  DefaultConvolutionConfiguration
  1. template <typename ArchTag, typename ElementDst>
  2. 特异化参数<arch::OpClassSimt, ArchTag, int8_t, int8_t, ElementDst, int32_t>
  3. **内部定义类型：**
  ```c++
  using ThreadblockShape = gemm::GemmShape<128, 128, 32>;
  using WarpShape = gemm::GemmShape<32, 64, 32>;
  using InstructionShape = gemm::GemmShape<1, 1, 4>;
  using EpilogueOutputOp = epilogue::thread::BiasAddLinearCombinationClamp<
            ElementDst, 4, int32_t, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
  ```
  4. **内部存储类型：**
  ```c++
  static int const kAlignmentSrc = 4;
  static int const kAlignmentFilter = 4;
  static int const kStages = 2;
  ```

#### DefaultConvolutionConfiguration
  1. template <typename ElementDst>
  2. 特异化参数<arch::OpClassTensorOp, arch::Sm75, int8_t, int8_t, ElementDst, int32_t>
  3. **内部定义类型：**
  ```c++
  using ThreadblockShape = gemm::GemmShape<128, 256, 64>;
  using WarpShape = gemm::GemmShape<64, 64, 64>;
  using InstructionShape = gemm::GemmShape<8, 8, 16>;
  using ArchTag = arch::Sm75;
  using EpilogueOutputOp = epilogue::thread::BiasAddLinearCombinationClamp<
            ElementDst, 128 / sizeof_bits<ElementDst>::value, int32_t, int32_t,
            float>;

  using Operator = arch::OpMultiplyAddSaturate;
  ```
  4. **内部存储内容：**
  ```c++
  static int const kAlignmentA = 128 / sizeof_bits<int8_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint8_t>::value;
  static int const kStages = 2;
  ```

## 三、kernel level header
在kernel目录下有三个文件convolution.h, default_convolution.h, convolution_precompute_offset.h
### convolution.h
在cutlass/convolution/kernel namespace下提供一个结构体模板
#### Convolution：结构体模板
  1. 
  ```c++
  template <typename Mma_,  ///! Threadblock-scoped matrix multiply-accumulate
          typename Epilogue_,            ///! Epilogue
          typename ThreadblockSwizzle_,  ///! Threadblock swizzling function
          ConvType ConvolutionType =
                  ConvType::kConvolution  ///! Convolution Type
          >
  ```
  2. **内部定义类型：**
  ```c++
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using ConvolutionParameter = ConvParam<kConvolutionType>;
  using WarpCount = typename Mma::WarpCount;

  struct Params {
        ConvolutionParameter conv_param;
        cutlass::gemm::GemmCoord grid_tiled_shape;
        typename Mma::IteratorSrc::Params params_src;
        typename Mma::IteratorSrc::TensorRef ref_src;
        typename Mma::IteratorFilter::Params params_filter;
        typename Mma::IteratorFilter::TensorRef ref_filter;
        typename Epilogue::BiasTileIterator::Params params_bias;
        typename Epilogue::BiasTileIterator::TensorRef ref_bias;
        typename Epilogue::OutputTileIterator::Params params_dst;
        typename Epilogue::OutputTileIterator::TensorRef ref_dst;
        typename Epilogue::OutputTileIterator::Params params_z;
        typename Epilogue::OutputTileIterator::TensorRef ref_z;
        typename OutputOp::Params output_op;
        typename Mma::TransformSrc::Params transform_src;
        typename Mma::TransformFilter::Params transform_filter;
        int* workspace;
        int conv_c_iterations;

        //
        // Methods
        //

        CUTLASS_HOST_DEVICE
        Params() : workspace(nullptr) {}

        CUTLASS_HOST_DEVICE
        Params(ConvolutionParameter const& conv_param,
               cutlass::gemm::GemmCoord const& grid_tiled_shape,
               typename Mma::IteratorSrc::TensorRef ref_src,
               typename Mma::IteratorFilter::TensorRef ref_filter,
               typename Epilogue::BiasTileIterator::TensorRef ref_bias,
               typename Epilogue::OutputTileIterator::TensorRef ref_z,
               typename Epilogue::OutputTileIterator::TensorRef ref_dst,
               typename OutputOp::Params output_op =
                       typename OutputOp::Params(),
               typename Mma::TransformSrc::Params transform_src =
                       typename Mma::TransformSrc::Params(),
               typename Mma::TransformFilter::Params transform_filter =
                       typename Mma::TransformFilter::Params(),
               int* workspace_ = nullptr)
                : conv_param(conv_param),
                  grid_tiled_shape(grid_tiled_shape),
                  params_src(ref_src.layout()),
                  ref_src(ref_src),
                  params_filter(ref_filter.layout()),
                  ref_filter(ref_filter),
                  params_bias(ref_bias.layout()),
                  ref_bias(ref_bias),
                  params_dst(ref_dst.layout()),
                  ref_dst(ref_dst),
                  params_z(ref_z.layout()),
                  ref_z(ref_z),
                  output_op(output_op),
                  transform_src(transform_src),
                  transform_filter(transform_filter),
                  workspace(workspace_) {
            conv_c_iterations =
                    (conv_param.ci() + Mma::Shape::kK - 1) / Mma::Shape::kK;
        }
    };

    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };
  ```
  3. **内部存储内容：**
  ```c++
  static const ConvType kConvolutionType = ConvolutionType;
  static int const kThreadCount = 32 * WarpCount::kCount;
  ```
  4. **构造方法：**
  - 空构造函数
  5. **内部方法：**
  - can_implement
  - get_workspace_size
  - operator()
  6. **解释**
  - 以operator()为入口 **这里留个锚点，之后再写**


## 四、thread_block level header
threadblock目录下有八个文件
- threadblock_swizzle.h
- mma_base.h
- mma_pipelined.h
- mma_precompute_offset.h
- default_mma.h
- default_mma_core.h
- default_mma_core_sm75.h
- default_mma_core_simt.h

### threadblock_swizzle.h
在cutlass/convolution/threadblock namespace下提供两个结构体模板
#### ConvolutionCxRSKxThreadblockSwizzle
  1. template\<ConvType ConvolutionType\>
  2. **内部定义类型：**
  - typedef ConvParam<ConvolutionType> ConvolutionParameter
  3. **构造方法：**
  - 空白构造方法
  4. **内部方法：**
  - get_tiled_shape
  - get_grid_shape
  - get_tile_offset (template<typename Shape>)
  5. **解释：**
  - get_tiled_shape把Output按N和K维分成gemm
  - get_grid_shape用来返回dim3
  - get_tile_offset需要Shape来计算n, h, w, c

#### ConvolutionNCxHWxThreadblockSwizzle
  1. template \<ConvType ConvolutionType\>
  2. **内部定义类型：**
  - typedef ConvParam<ConvolutionType> ConvolutionParameter
  3. **构造方法：**
  - 空白构造方法
  4. **内部方法：**
  - get_tiled_shape
  - get_grid_shape
  - get_tile_offset
  5. **解释：**
  - 用nhw / tile_n, c / tile_m, 1作为tiled_shape
  - get_tile_offset把block y乘以kM作为gemm coord第一个参数返回的


### mma_base.h
在cutlass/convolution/threadblock下提供一个类模板
#### MmaBase 类模板
  1. 
  ```c++
  template <
        /// Size of the Gemm problem - concept: gemm::GemmShape<>
        typename Shape_,
        /// Policy describing tuning details (concept: MmaPolicy)
        typename Policy_,
        /// Number of stages,
        int Stages,
        /// Used for partial specialization
        typename Enable = bool>
  ```
  2. **内部定义类型：**
  ```c++
  using Shape = Shape_;
  using Policy = Policy_;
  using Operator = typename Policy::Operator;
  using WarpGemm = typename Policy::Operator::Shape;
  using WarpCount =
            gemm::GemmShape<Shape::kM / WarpGemm::kM, Shape::kN / WarpGemm::kN,
                            Shape::kK / WarpGemm::kK>;
  using TensorRefSrc =
            TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;
  using TensorRefFilter =
            TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;
  class SharedStorage {
    public:
        //
        // Type definitions
        //

        /// Shape of the A matrix operand in shared memory
        using ShapeA = MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow,
                                   Shape::kK * kStages +
                                           Policy::SmemPaddingA::kColumn>;

        /// Shape of the B matrix operand in shared memory
        using ShapeB =
                MatrixShape<Shape::kK * kStages + Policy::SmemPaddingB::kRow,
                            Shape::kN + Policy::SmemPaddingB::kColumn>;

    public:
        //
        // Data members
        //

        /// Buffer for Src Tensor operand
        AlignedBuffer<typename Operator::ElementB, ShapeB::kCount> operand_src;

        /// Buffer for Filter Tensor operand
        AlignedBuffer<typename Operator::ElementA, ShapeA::kCount>
                operand_filter;

    public:
        //
        // Methods
        //

        /// Returns a layout object for the Src Tensor
        CUTLASS_DEVICE
        static typename Operator::LayoutB LayoutSrc() {
            return Operator::LayoutB::packed({ShapeB::kRow, ShapeB::kColumn});
        }

        /// Returns a layout object for the Filter Tensor
        CUTLASS_HOST_DEVICE
        static typename Operator::LayoutA LayoutFilter() {
            return Operator::LayoutA::packed({ShapeA::kRow, ShapeA::kColumn});
        }

        /// Returns a TensorRef to the Src Tensor operand
        CUTLASS_HOST_DEVICE
        TensorRefSrc operand_src_ref() {
            return TensorRefSrc{operand_src.data(), LayoutSrc()};
        }

        /// Returns a TensorRef to the Filter Tensor operand
        CUTLASS_HOST_DEVICE
        TensorRefFilter operand_filter_ref() {
            return TensorRefFilter{operand_filter.data(), LayoutFilter()};
        }
    };
  ```
  3. **内部存储内容：**
  ```c++
  static int const kWarpGemmIterations =
            (WarpGemm::kK / Operator::Policy::MmaShape::kK);
  /// Number of stages
  static int const kStages = Stages;
  protected:
  typename Operator::IteratorB warp_tile_iterator_src_;
  typename Operator::IteratorA warp_tile_iterator_filter_;
  ```
  4. **构造方法：**
  ```c++
  CUTLASS_DEVICE
    MmaBase(
            ///< Shared storage needed for internal use by threadblock-scoped
            ///< GEMM
            SharedStorage& shared_storage,
            ///< ID within the threadblock
            int thread_idx,
            ///< ID of warp
            int warp_idx,
            ///< ID of each thread within a warp
            int lane_idx)
            : warp_tile_iterator_src_(shared_storage.operand_src_ref(),
                                      lane_idx),
              warp_tile_iterator_filter_(shared_storage.operand_filter_ref(),
                                         lane_idx) {}
  ```

### mma_pipelined.h
在cutlass/convolution/threadblock namespace下提供一个类模板
####  MmaPipelined
  1. 
  ```c++
  template <
        /// Size of the Gemm problem - concept: gemm::GemmShape<>
        typename Shape_,
        /// Iterates over tiles of Src Tensor operand in global memory
        ///  (concept: ReadableTileIterator | ForwardTileIterator |
        ///  MaskedTileIterator | RandomAccessTileIterator)
        typename IteratorSrc_,
        /// Iterates over tiles of Src Tensor operand in shared memory
        /// (concept: WriteableTileIterator | RandomAccessTileIterator)
        typename SmemIteratorSrc_,
        /// Iterates over tiles of Filter Tensor operand in global memory
        ///  (concept: ReadableTileIterator | ForwardTileIterator |
        ///  MaskedTileIterator | RandomAccessTileIterator)
        typename IteratorFilter_,
        /// Iterates over tiles of Filter operand in shared memory
        /// (concept: WriteableTileIterator | RandomAccessTileIterator)
        typename SmemIteratorFilter_,
        /// Data type of accumulator Dst Tensor
        typename ElementDst_,
        /// Data type of accumulator Dst Tensor
        typename LayoutDst_,
        /// Policy describing tuning details (concept: MmaPolicy)
        typename Policy_,
        /// Transformation applied to A operand
        typename TransformSrc_ =
                NumericArrayConverter<typename SmemIteratorSrc_::Element,
                                      typename IteratorSrc_::Element,
                                      IteratorSrc_::Fragment::kElements>,
        ///
        /// Transformation applied to B operand
        typename TransformFilter_ =
                NumericArrayConverter<typename SmemIteratorFilter_::Element,
                                      typename IteratorFilter_::Element,
                                      IteratorFilter_::Fragment::kElements>,
        /// Used for partial specialization
        typename Enable = bool>
  ```
  2. public MmaBase<Shape_, Policy_, 2>
  3. **内部定义类型：**
  ```c++
  using Base = MmaBase<Shape_, Policy_, 2>;

  using Shape =
          Shape_;  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using IteratorSrc = IteratorSrc_;  ///< Iterates over tiles of Src Tensor
                                      ///< operand in global memory
  using IteratorFilter =
          IteratorFilter_;         ///< Iterates over tiles of Filter Tensor
                                    ///< operand in global memory
  using ElementDst = ElementDst_;  ///< Data type of accumulator matrix
  using LayoutDst = LayoutDst_;    ///< Layout of accumulator matrix
  using Policy = Policy_;          ///< Policy describing tuning details

  using SmemIteratorSrc = SmemIteratorSrc_;
  using SmemIteratorFilter = SmemIteratorFilter_;

  using TransformSrc = TransformSrc_;
  using TransformFilter = TransformFilter_;

  //
  // Dependent types
  //

  /// Fragment of operand Src Tensor loaded from global memory
  using FragmentSrc = typename IteratorSrc::Fragment;

  /// Fragment of operand B loaded from global memory
  using FragmentFilter = typename IteratorFilter::Fragment;

  /// Fragment of accumulator tile
  using FragmentDst = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Obtain the arch tag from the warp-level operator
  using ArchTag = typename Policy::Operator::ArchTag;

private:
  using WarpFragmentSrc = typename Operator::FragmentB;
  using WarpFragmentFilter = typename Operator::FragmentA;
  ```

  4. **内部存储内容：**
  ```c++
  static ComplexTransform const kTransformSrc = Operator::kTransformB;

  /// Complex transform on Tensor Filter (B operand)
  static ComplexTransform const kTransformFilter = Operator::kTransformA;

protected:
  SmemIteratorSrc smem_iterator_src_;

  /// Iterator to write threadblock-scoped tile of Filter Tensor operand to
  /// shared memory
  SmemIteratorFilter smem_iterator_filter_;
  ```
  4. **构造函数：**
  ```c++
  CUTLASS_DEVICE
    MmaPipelined(
            typename Base::SharedStorage&
                    shared_storage,  ///< Shared storage needed for internal
                                     ///< use by threadblock-scoped Convolution
            int thread_idx,          ///< ID within the threadblock
            int warp_idx,            ///< ID of warp
            int lane_idx             ///< ID of each thread within a warp
            )
            : Base(shared_storage, thread_idx, warp_idx, lane_idx),
              smem_iterator_src_(shared_storage.operand_src_ref(), thread_idx),
              smem_iterator_filter_(shared_storage.operand_filter_ref(),
                                    thread_idx) {
        // Compute warp location within threadblock tile by mapping the warp_id
        // to three coordinates:
        //   _m: the warp's position within the threadblock along the M
        //   dimension _n: the warp's position within the threadblock along the
        //   N dimension _k: the warp's position within the threadblock along
        //   the K dimension

        int warp_idx_mn =
                warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
        int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

        int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
        int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

        // Add per-warp offsets in units of warp-level tiles
        this->warp_tile_iterator_src_.add_tile_offset(
                {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
        this->warp_tile_iterator_filter_.add_tile_offset(
                {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    }
  ```
  5. **内部方法：**
  - operator()
  6. **解释：**

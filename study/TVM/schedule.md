src/te/*

先看schedule目录
=================================================================================
include/te/schedule.h
=================================================================================


AttachType enum : int
  kGroupRoot, kInline, kInlinedAlready, kScope, kScanUpdate

Stage : OjbectRef
  构造: () (ObjectPtr<Object>) (Operation)
  内部 -> 得到StageNode *, 不是const的
  方法：
    set_scope(std::string scope) 设置存储scope
    compute_at(Stage, IterVar)
    compute_inline()
    compute_root()
    bind(IterVar, IterVar)
    set_store_predicate(PrimExpr) 设置仅有满足条件的thread才存储
    env_threads(Array<IterVar>) 仅能用在group stage
    split(IterVar parent, PrimExpr factor, IterVar* p_outer, IterVar* p_inner)
    split_by_nparts(IterVar, PrimExpr, IterVar*, IterVar*)
    fuse(IterVar outer, IterVar inner, IterVar* p_target)
    fuse(const Array<IterVar>&, IterVar* p_target)
    reorder(const Array<IterVar>& order)
    tile(IterVar x_parent, IterVar y_parent, PrimExpr x_factor, PrimExpr y_factor, IterVar* p_x_outer, IterVar* p_y_outer, IterVar* p_x_inner, IterVar* p_y_inner)
    vectorize(IterVar)
    tensorize(IterVar, TensorIntrin f)
    unroll(IterVar var)
    parallel(IterVar var)
    pragma(IterVar, const std::string&, const PrimExpr& pragma_value)
    prefetch(const Tensor&, IterVar var, PrimExpr offset) 从tensor的var维度预取offset的数据 
    storage_align(IterVar, int, int)
    double_buffer()
    bool is_scheduled() const
    Stage GetAttachSpec() const


Schedule : ObjectRef
  构造：() (ObjectPtr<Object>) (Array<Operation>)
  方法：
    Schedule copy() const
    Stage operator[](const Operation&) 或 (const Tensor&)
    Stage create_group(const Array<Tensor>& outputs, const Array<Tensor>& inputs, bool include_inputs=false)
    Tensor cache_read(const Tensor& tensor, const std::string& scope, const Array<Operation>& readers)
    Array<Tensor> cache_write(const Array<Tensor>& tensor, const std::string& scope)
    Tensor cache_write(const Tensor&, const std::string&)
    Array<Tensor> rfactor(const Tensor& tensor, const InterVar& axis, int factor_axis=0)
    Schedule normalize()
    ->得到ScheduleNode*


IterVarRelation : ObjectRef

IterVarAttr : ObjectRef


StageNode : Object
  Operation op 空的时候，则是一个group stage
  Operation origin_op
  Array<IterVar> all_iter_vars
  Array<IterVar> leaf_iter_vars
  Array<IterVar> env_threads 仅对组合的ops有用，比如Scan
  PrimExpr store_predicate
  Array<IterVarRelation> relations
  Map<IterVar, IterVarAttr> iter_var_attrs
  AttachType attach_type{kGroupRoot}
  IterVar attach_ivar
  Stage attach_stage
  std::string scope
  bool is_output{false}
  bool double_buffer{false}
  Stage group
  int num_child_stages{0}


ScheduleNode : Object
  Array<Operation> outputs
  Array<Stage> stages
  Array<Stage> groups
  Map<Operation, Stage> stage_map
  std::unordered_map<cosnt Object*, Stage> op2stage_cache_

  void InitCache()
  void InvalidateCache()
  bool Contain(const Tensor& tensor) const 或者 (const Operation& op)


inline Schedule create_schedule(Array<Operation> ops)


IterVarAttrNode : Object
  IterVarType iter_type{kDataPar}
  IterVar bind_thread
  Array<Tensor> prefetch_data
  Array<PrimExpr> prefetch_offset
  TensorIntrin tensor_intrin
  int dim_align_factor{0}
  int dim_align_offset{0}
  Array<PrimExpr> pragma_keys
  Array<PrimExpr> pragma_values


IterVarRelationNode : Object


SplitNode : IterVarRelationNode
  IterVar parent
  IterVar outer
  IterVar inner
  PrimExpr factor
  PrimExpr nparts


Split : IterVarRelation


FuseNode : IterVarRelationNode
  IterVar outer
  IterVar inner
  IterVar fused


Fuse : IterVarRelation


RebaseNode : IterVarRelationNode
  IterVar parent
  IterVar rebased


Rebase : IterVarRelation


SingletonNode : IterVarRelationNode
  IterVar iter


Singleton : IterVarRelation


SpecializedConditionNode : Object
  Array<PrimExpr> clauses


SpecializedCondition : ObjectRef
  它有一些奇怪的方法
  static SpecializedCondition Current()
  friend class Internal
  friend class With<SpecializedCondition>
  void EnterWithScope()
  void ExitWithScope()



=================================================================================
include/te/schedule_pass.h
=================================================================================
void AutoInlineElemWise(Schedule sch)

void AutoInlineInjective(Schedule sch)

Map<IterVar, Range> InferBound(const Schedule& sch)

bool VerifyCompactBuffer(const Stmt&)

Stmt ScheduleOps(Schedule s, Map<IterVar, Range> dom_map, bool debug_keep_trivial_loop)

Stmt SchedulePostProcRewriteForTensorCore(Stmt stmt, Schedule schedule, Map<Tensor, Buffer> extern_buffer)

PrimFunc SchedulePostProcToPrimFunc(Array<ObjectRef> arg_list, Stmt body, Optional<Map<Tensor, Buffer>> bindings)




=================================================================================
src/te/auto_inline_elem_wise.cc
=================================================================================
ElemWiseDetector : tir::ExprVisitor
  Array<IterVar> axis_
  bool is_elem_wise_{true}
  靠数axis的长度或者比较axis元素来判断，比较对象是CallNode，只要遇到CallNode就判断一下，必须要求axis和call args完全一致
  并且是个短路逻辑，一旦是false了就不再改变

bool IsElemWise(const Operation& op)
  只对ComputeOpNode，判断的axis就是compute->axis

void AutoInlineElemWise(Schedule sch)
  对每个Stage，如果没is_scheduled()，并且op是IsElemWise，并且不是is_output，自动写个compute_inline()

bool IsBroadcast(const Operation& op)
  有reduce_axis就不是，其它都是

void AutoInlineBroadcast(Schedule sch) 和inline那个差不多

IsInjective(const Operation&) 以及 AutoInlineInjective和broadcast差不多


=================================================================================
src/te/graph.h
=================================================================================
ReadGraph = Map<Operation, Array<Tensor>>
AttachPath = Map<Operation, Array<IterVar>>
FeedGraph = std::unordered_map<Tensor, std::vector<Operation>>

ReadGraph CreateReadGraph(const Array<Operation>& roots)

Array<Operation> GetSubGraph(const Array<Tensor>& outputs, const Array<Tensor>& inputs, bool include_inputs)

Array<Operation> PostDFSOrder(const Array<Operation>& roots, const ReadGraph& g)

FeedGraph CreateFeedGraph(const ReadGraph& g)

AttachPath CreateAttachPath(Schedule sch)

Array<Operation> ScanGetBody(const Operation& scan_op)

Map<IterVar, PrimExpr> ScanFixPointAnalysis(const Operation& scan)


=================================================================================
src/te/graph.cc
=================================================================================
struct TensorDimKey
  Operation op
  int value_index
  int dim

ReadGraph CreateReadGraph(const Array<Operation>& roots)  就dfs把InputTensors全记下来

bool GetSubGraphByPostDFS_(const Operation& op, const std::unordered_set<const Object*>& boundary, bool include_boundary, std::unordered_map<const Object*, bool>* visited, Array<Operation>* result)
  这个函数顺着input向上找，如果能触碰到boundary，就算在subgraph内，还会把沿路的op记录在result里

Array<Operation> GetSubGraph(...) 这个函数用来找子图，但是不做真正的切割，只是把Op存一下返回

Array<Operation> PostDFSOrder(...) 有一对函数，都是dfs遍历图

FeedGraph CreateFeedGraph(const ReadGraph& g) 没有顺序

AttachPath CreateAttachPath(Schedule sch)
  首先要解释GetAttachSpec的实现：保持kGroupRoot且group不为空，上溯group，最后返回最头上的group
  返回的path是针对每个Op而言的，所以最外层是for每个stage
  内层从GetAttachSpec的结果开始上溯，如果是kScope和kScanUpdate以外的，直接break出去，path也记为空
  如果是kGroupRoot，那么就顺着compute_at的attach_stage继续找，并且把attach_ivar及以上的IterVar都记在path里，如果连续的compute_at出现，那每个被compute_at的位置的IterVar都会被过一遍
  如果是kScanUpdate，那么被compute_at的所有IterVar都会被记在path里

ReachGraph = std::unordered_map<TensorDimKey, std::vector<TensorDimKey>>

ReachGraph GetReachGraph(const Array<Operation>& ops)
  对每个op
    如果是ScanOpNode，针对每个output，找到对应的update（是Tensor），对这个update的shape维度1..n，记录在ReachGraph里，从output到update的对应维度，还有init（也是Tensor）的也记录
    如果是ComputeOpNode, 记录每个i位置的axis及output(0)的dim i的tensor key，同时ReachGraph记录dim i 的 tensor key，值为空数组，之后遍历body的AST，对于ProducerLoad node，根据index找输出Tensor每个维度都用了producer的Tensor的哪个维度，但是不重复记录，同样的输入Tensor遇到第二次就不再处理

Array<Operation> ScanGetBody(const Operation& scan_op)
  inputs是[state_placeholder, inputs]，然后用GetSubGraph获取从updates到inputs不包含inputs的子图

Map<IterVar, PrimExpr> ScanFixPointAnalysis(const Operation& scan_op)
  这个函数遇到再看


=================================================================================
src/te/message_passing.h
=================================================================================
void PassDownDomain(const Stage& stage, std::unordered_map<IterVar, Range>* p_state, arith::Analyzer* analyzer, bool allow_missing=false)
void PassUpIndex(const Stage& stage, const Map<IterVar, Range>&dom_map, std::unordered_map<IterVar, PrimExpr>* p_stage, bool allow_missing=false)
void PassDownIndex(const Stage& stagem const Map<IterVar, Range>& dom_map, std::unordered_map<IterVar, PrimExpr>* p_state, bool allow_missing=false)
void PassUpDomain(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map, std::unordered_map<IterVar, IntSet>* p_stage)
void PassUpBitMastOr(const Stage& stage, std::unordered_map<IterVar, int>* p_state, bool allow_missing=false)
void PassDownBitMaskOr(const Stage& stage, std::unordered_map<IterVar, int>* p_state, bool allow_missing = false);
std::vector<PrimExpr> MakeBoundCheck(const Stage& stage, const Map<IterVar, Range>& dom_map,
                                     const std::unordered_map<IterVar, PrimExpr>& value_map,
                                     bool skip_ivar_domain,
                                     const std::unordered_set<IterVar>& skip_iter)


=================================================================================
src/te/message_passing.cc
=================================================================================
void Update(std::unordered_map<IterVar, Range>* p_state, const IterVar& iv, Range r, arith::Analyzer* analyzer)
  会更新iv的range到r，但是如果之前iv就已经有range，还要做一次检测，看iv的range是不是从0开始且两次bind的range一样大

void PassUpThreadBinding(const Stage& stage, std::unordered_map<IterVar, bool>* p_state) 用于从下向上传递binding信息
  首先，如何确定一个IterVar被bind到thread了，需要查看stage里的iter_var_attr中的bind_thread，如果不为空（defined）就说明绑定了
  之后，顺着IterVar的relation向上传播bind信息

void PassDownDomain(const Stage& stage, std::unordered_map<IterVar, Range>* p_state, arith::Analyzer* actx, bool allow_missing)
  首先一个辅助函数是ceil_div(a, b)，如果b能整除a，那么就直接做indexdiv，否则还要加上b-1再除（相当于上取整）
  另一个是minimum_or_later是如果能证明a<b就取a，否则取b
  用PassUpThreadBinding记录哪些IterVar被bind了
  之后遍历所有stage内的IterVar relation
  如果说是split，根据factor或者nparts设置inner, outer的range，但是并非总是严格遵守factor, nparts，在允许的情况下也会tighten到parent的extent，创建range时候使用的是Range::FromMinExtent
  如果是fuse，就直接乘起来inner和outer的range
  如果是rebase，直接使用parent的range
  这之后会更新bind的thread们的range

void PassUpIndex(const Stage& stage, const Map<IterVar, Range>& dom_map, std::unordered_map<IterVar, PrimExpr>* p_state, bool allow_missing)
  遍历stage内部的所有relation（倒着遍历）
  如果是split，向上创建parent的index
  如果是fuse，就向上创建outer和inner
  如果是rebase，就向上创建parent

void PassDownIndex(const Stage& stage, const Map<IterVar, Range>& dom_map, std::unordered_map<IterVar, PrimExpr>* p_state, bool allow_missing)
  正向遍历所有relation，其它也都和PassUpIndex是互逆的

void PassUpDomain(const SplitNode* s, const std::unordered_map<IterVar, Range>& dom_map, const IntSet& outer, const IntSet& inner, IntSet* parent)
  PassUpDomain又分成了三个小的函数，这是对split的，主要目的是得到parent的IntSet，如果其它的都就绪了，直接用IntSet::FromRange创建
  否则用arith::EvalSet创建（这里第一次知道map可以用{{key, value}...}创建）

void PassUpDomain(const FuseNode* s, const std::unordered_map<IterVar, Range>& dom_map, const IntSet& fused, IntSet* outer, IntSet* inner)

void PassUpDomain(const RebaseNode* s, const std::unordered_map<IterVar, Range>& dom_map, const IntSet& rebased, IntSet* parent)

void PassUpDomain(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map, std::unordered_map<IterVar, IntSet>* p_state)
  倒着遍历relation，利用上述三个函数完成pass up

void PassUpBitMastOr(const Stage& stage, std::unordered_map<IterVar, int>* p_state, bool allow_missing=false)
void PassDownBitMaskOr(const Stage& stage, std::unordered_map<IterVar, int>* p_state, bool allow_missing = false)
  这两个先不看，遇到再说

void PassUpBoundCheck(const Stage& s, const Map<IterVar, Range>& dom_map, std::unordered_map<IterVar, bool>* p_state, arith::Analyzer* analyzer)
  用来传播是否需要bound check
  倒序遍历relation，主要看根据relation推导的bound都能不能对上，对不上的就要check一下

bool IsRangeSame(const Range input_1, const Range input_2)
  检测两个range是否一样

std::vector<PrimExpr> MakeBoundCheck(const Stage& stage, const Map<IterVar, Range>& dom_map,
                                     const std::unordered_map<IterVar, PrimExpr>& value_map,
                                     bool skip_ivar_domain,
                                     const std::unordered_set<IterVar>& skip_iter)
  把疑似bound有问题的都记录下来，以后加个check



=================================================================================
src/te/bound.cc
=================================================================================
struct GraphContext
  FeedGraph feed_graph
  AttachPath attach_path
  std::unordered_map<IterVar, IterVar> bind_map
  std::unordered_map<const Object*, Stage> op2stage_

bool NeedRelax(const IterVar& iv, bool found_attach, const std::unordered_map<IterVar, IterVar>& bind_map, const runtime::StorageScope& scope)
  这个函数的用处之后再说
  给人的感觉是判断stage的scope比某个axis的scope低的时候，需要仍然考虑axis的range，否则axis的就不用考虑，直接变成单点

StorageScope InferStorageScope(const Stage& stage, const GraphContext& ctx)
  推导storage scope的逻辑也和StorageScope这个结构的设计有关

void InferRootBound(const Stage& stage, const GraphContext& ctx, std::unordered_map<IterVar, Range>* rmap)
  不接受inline的stage，inline应该被normalize处理了
  如果是kInlinedAlready，直接返回
  如果stage是output，那么必须是所处group的attach_type必须是kGroupRoot
  对于placeholder和output，都直接把root_iter_vars的dom设置成range返回
  推导当前stage的storage scope，用的就是上面的InferStorageScope，这个推理过程是根据attach path中的itervar的ThreadScope确定的
  之后遍历这个stage的所有consumer op, 对每个consumer， 遍历所有的leaf_iter_vars（反向,为了区分是否attach的信息），收集其range，组装到一个up_state，这个是
  记录itervar到int set的map，在记录过程中要区分range，处理单点，不是单点的时候要判断是否需要relax（使用NeedRelax函数），不需要relax的会直接放一个单点（只不过是一个符号式的，放进去var），对于需要relax的，放进去的是from range的int set
  然后记录这个consumer的relax set，这里的逻辑不明确
  然后用up_state进行PassUpDomain
  后面的逻辑很奇怪，一个infer bound都写得这么晦涩...

Map<IterVar, Range> InferBound(const Schedule& sch)
  从输出开始向上逐个进行infer root range, pass down domain
  整个infer bound其实可以做得很简单，但是有大量边界情况要考虑的时候，让整个逻辑很乱


=================================================================================
src/te/operation_inline.h
=================================================================================
Stmt Inline(Stmt stmt, Operation op, Array<Var> args, PrimExpr body);


=================================================================================
src/te/operation_inline.cc
=================================================================================
OperationInliner : StmtExprMutator
  Operation operation_
  Array<var> args_
  PrimExpr body_
  专门看ProducerLoadNode，替换call body，但是要注意side effect
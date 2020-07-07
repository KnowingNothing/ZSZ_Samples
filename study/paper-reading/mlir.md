# MLIR: A Compiler Infrastructure for the End of Moore’s Law

> Google

## Abstract
1. reusable and extensible
2. software fragmentation, improve compilation for heterogeneous hardware, significantly reduce the cost of building domain specific compilers, aid in connecting existing compilers together.

## Introduction
1. algorithms in compiler design: code generation, static analysis, program transformation
2. one size fits all
3. make it very cheap to define and introduce new abstraction levels, provide "in the box" infrastructure to solve common compiler engineering problems.
4. standardizing the SSA
5. provide a declarative system for defining IR dialects
6. documentation, parsing, printing logic, location tracking, multithreaded compilation, pass management
7. common issues: poor error message, failure in edge cases, unpredictable performance, difficulty generalizing the stack to support new hardware.


## Design Principles
1. little builtin, everything customizable
2. customization criterion: express ML graph, AST, polyhedral, CFG, LLVM IR without hard coding concepts
3. nested regions (nested loops)
4. not fixed level of abstractions, interactions between passes
5. compiler pass:
  - optimizing
  - transforming
  - lowering
  - cleanup

6. 
```
   Op
   |- list of Region
                |- list of Block
                            |- list of Op
```

7. modeling program transformations as rewrite systems
8. source traceability

## IR Design Details
### Op
allow user-defined extensions, compiler passes treat unknown Ops conservatively
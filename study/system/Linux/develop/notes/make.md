## make 和 makefile
make命令用于构建代码项目，makefile作为make的指导文件。

### 1. makefile语法
依赖关系和规则。
依赖=<目标， 目标所依赖的源文件组>
规则=如何通过依赖创建目标

### 2. make命令选项和参数
-k: 检查出所有错误
-n: 输出将要执行的操作，但不真正执行
-f <filename>: 指定makefile
make <目标名称>: 指定创建特定目标，不指定名称，默认创建第一个目标。当有all时，会执行all对应的规则。

**书写依赖关系：**
```makefile
myapp: main.o 2.o 3.o
main.o: main.c a.h
2.o: 2.c a.h b.h
3.o: 3.c b.h c.h
```
目标不一定要是真实的，可以创建伪目标:
```makefile
all: myapp myapp.1
```
**书写规则：**
规则以tab开头，末尾不能有空格。
```makefile
myapp: main.o 2.o 3.o
    gcc -o myapp main.o 2.o 3.o
```

### 3. 注释
用#开头

### 4. 宏
定义：MACRONAME=value
引用：\$(MACRONAME)或${MACRONAME}

特殊宏定义：
```
$?  当前目标依赖的文件列表中比当前目标还要新的文件
$@  当前目标名字
$<  当前依赖文件名字
$*  不包括后缀名的当前依赖文件名字
```
特殊符号：
```
-  忽略所有错误
@  执行某条命令前不要将该命令显示在标准输出上
```

### 5. 内置规则
由于内置规则的存在，可以省去很多makefile的内容
```makefile
main.o: main.c a.h
```
直接make main.o，可以自动调动cc编译目标内容。

### 6. 后缀和模式规则
```makefile
.SUFFIXES:  .cpp
.cpp.o:
    ${CC} -xc++ ${CFLAGS} -I${INCLUDE} -c $<
```
这里会自动把.cpp结尾的文件转为.o结尾的文件
这是过去的make需要的语法，现在的make可以使用更高级的语法：
```makefile
%.cpp: %o
    ${CC} -xc++ ${CFLAGS} -I${INCLUDE} -c $<
```

### 7. 函数库
函数库就是一个.a文件
```makefile
MYLIB=mylib.a

myapp: main.o $(MYLIB)
    $(CC -o myapp main.o $(MYLIB)

$(MYLIB): $(MYLIB)(2.o) $(MYLIB)(3.0)

main.o: main.c a.h
2.o: 2.c a.h b.h
3.o: 3.c b.h c.h
```

以上内容表示，创建myapp需要函数库mylib.a，mylib.a需要包含2.o和3.o，后续规则用于创建2.o和3.o，由于内置规则的存在，创建函数库的时候会自动调用`ar rv mylib 3.o`, `ar rv mylib.a 2.o`

### 8. 子目录
处理子目录的方法有两个，这里只介绍能理解的一个
在某个子目录下有一些源文件，希望用它们构建一个目标函数库
```makefile
mylib.a:
    (cd mylibdir; ${MAKE})
```
这里要用括号扩起来是因为使得两句命令都在一个shell中执行。

### 9. 自动依赖处理
gcc -MM可以自动理清依赖关系，可以输出到一个临时文件
makedepend也可以做相似的事情，直接会加在makefile的末尾
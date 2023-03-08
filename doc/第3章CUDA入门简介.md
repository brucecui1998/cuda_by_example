---
title: 第三章CUDA入门简介
date: 2022年9月18日21点56分
categories:
 - 项目合集
 - CUDA by Example
---

## 3.1 本章目标

- 编写第一段CUDA代码
- 了解为主机（host）编写的代码与为设备（device）的代码之间的区别
- 如何从主机上运行设备代码
- 了解如何在支持CUDA的设备上使用设备内存
- 了解如何查询系统中支持CUDA的设别的信息

## 3.2 第一个程序

### 3.2.1 Hello World！

```c
#include <stdio.h>

__global__ void kernel(){

}

int main(){
        kernel<<<1,1>>>();
        printf("hello World!");
        return 0;

}
```

- 一个空的函数kernel()，并且带有修饰符__global__
- 对这个空函数的调用，并且带有修饰字符<<<1,1>>>

### 3.2.2 核函数调用

global修饰符将告诉编译器，函数应该编译为在设备而不是主机上运行。在这个例子中，函数kernel被交给**编译设备代码的编译器**，而main（）函数将被交给**主机编译器**。

那么，kernel（）的调用究竟代表着什么含义，并且为什么必须加上尖括号和两个数值？注意，这正是使用CUDA C的地方。

我们已经看到，CUDAC需要通过某种语法方法将一个函数标记为“设备代码(Device Code)。这并没有什么特别之处，而只是一种简单的表示方法，表示将主机代码发送到--个编译器．而将设备代码发送到另--个编译器。事实上，这里的关键在于如何在主机代码中调用设备代码。CUDA C的优势之一在于，它提供了与C在语言级别上的集成，因此这个设备函数调用看上去非常像主机函数调用。在后面将详细介绍在这个函数调用背后发生的动作，但就目前而言，只需知道CUDA编译器和运行时将负责实现从主机代码中调用设备代码。
因此，这个看上去有些奇怪的函数调用实际上表示调用设备代码，但为什么要使用尖括号和数字?尖括号表示要将-一些参数传递给运行时系统。这些参数并不是传递给设备代码的参数，而是告诉运行时如何启动设备代码。在第4章中，我们将了解这些参数对运行时的作用。传递给设备代码本身的参数是放在圆括号中传递的，就像标准的函数调用一样。

### 3.2.3 传递参数

前面提到可以将参数传递给和函数，现在就来看一个示例。考虑下面对“Hello World”程序的修改“(需要下载《CUDA for Example》的源码)

```c
#include "../common/book.h"

__global__ void add( int a, int b, int *c ) {
    *c = a + b;
}

int main( void ) {
    int c;
    int *dev_c; //设备指针
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c, sizeof(int) ) );

    add<<<1,1>>>( 2, 7, dev_c );

    HANDLE_ERROR( cudaMemcpy( &c, dev_c, sizeof(int),
                              cudaMemcpyDeviceToHost ) );
    printf( "2 + 7 = %d\n", c );
    HANDLE_ERROR( cudaFree( dev_c ) );

    return 0;
}

```

注意这里增加了多行代码，在这些代码中包含两个概念:

- 可以像调用C函数那样将参数传递给核函数。
- 当设备执行任何有用的操作时，都需要分配内存，例如将计算值返回给主机。

在将参数传递给核函数的过程中没有任何特别之处。除了尖括号语法之外，核函数的外表和行为看上去与标准C中的任何函数调用一样。**运行时系统负责处理将参数从主机传递给设备的过程中的所有复杂操作**。

更需要注意的地方在于通过cudaMalloc()来分配内存。这个函数调用的行为非常类似于标准的C函数malloc()，但该函数的作用是告诉CUDA运行时在设备上分配内存。**第一个参数是一个指针，指向用于保存新分配内存地址的变量**，**第二个参数是分配内存的大小**。除了分配内存的指针不是作为函数的返回值外，这个函数的行为与malloc()是相同的，并且**返回类型为void***。

函数调用外层的HANDLE_ERROR()是我们定义的一个宏，作为本书辅助代码的一部分。这个宏只是判断函数调用是否返回了一个错误值，如果是的话，那么将输出相应的错误消息，退出应用程序并将退出码设置为EXIT_FAILURE。虽然你也可以在自己的应用程序中使用这个错误处理码,但这种做法在产品级的代码中很可能是不够的。

这段代码引出了一-个微妙但却重要的问题。CUDA C的简单性及其强大功能在很大程度上都是来源于它淡化了主机代码和设备代码之间的差异。然而，程序员**一定不能**在主机代码中对cudaMalloc()返回的指针进行解引用(Dereference)。**主机代码可以将这个指针作为参数传递，对其执行算术运算，甚至可以将其转换为另一种不同的类型。但是，绝对不可以使用这个指针来读取或者写入内存。**
遗憾的是，编译器无法防止这种错误的发生。如果能够在主机代码中对设备指针进行解引用，那么CUDA C将非常完美，因为这看上去就与程序中其他的指针完全一样了。我们可以将设备指针的使用限制总结如下:

- 可以将cudaMalloc()分配的指针传递给在设备上执行的函数。
- 可以在设备代码中使用cudaMalloc()分配的指针进行内存读/写操作。
- 可以将cudaMalloc()分配的指针传递给在主机上执行的函数。
- 不能在主机代码中使用cudaMalloc()分配的指针进行内存读/写操作。

如果仔细阅读了前面的内容，那么可以得出以下推论:不能使用标准C的free()函数来释放cudaMalloc()分配的内存。**要释放cudaMalloc()分配的内存，需要调用cudaFree()**，这个函数的行为与free()的行为非常相似。

我们已经看到了如何在设备上分配内存和释放内存，同时也清楚地看到，在主机上不能对这块内存做任何修改。在示例程序中剩下来的两行代码给出了访问设备内存的两种最常见方法-—在设备代码中使用设备指针以及调用cudaMemcpy()。
设备指针的使用方式与标准C中指针的使用方式完全一样。语句*c = a + b的含义同样非常简单:将参数a和b相加，并将结果保存在c指向的内存中。这个计算过程非常简单，甚至吸引不了我们的兴趣。
在前面列出了在设备代码和主机代码中可以/不可以使用设备指针的各种情形。在主机指针的使用上有着类似的限制。虽然可以将主机指针传递给设备代码，但如果想通过主机指针来访问设备代码中的内存，那么同样会出现问题。总的来说，**主机指针只能访问主机代码中的内存**，而**设备指针也只能访问设备代码中的内存**。
前面曾提到过，在主机代码中可以通过调用cudaMemcpy()来访问设备上的内存。这个函数调用的行为类似于标准C中的memcpy()，只不过多了一个参数来指定设备内存指针究竟是源指针还是目标指针。在这个示例中，注意cudaMemcpy()的最后一个参数为cudaMemcpyDeviceToHost,这个参数将告诉运行时源指针是一个设备指针，而目标指针是一个主机指针。
显然，cudaMemcpyHostToDevice将告诉运行时相反的含义，即源指针位于主机上，而目标指针是位于设备上。此外还可以通过传递参数cudaMemcpyDeviceToDevice来告诉运行时这两个设备都是位于设备上。如果源指针和目标指针都位于主机上，那么可以直接调用memcpy（）函数

## 查询设备

由于我们希望在设备上分配内存和执行代码，因此如果在程序中能够知道设备拥有多少内存以及具备哪些功能，那么将非常有用。而且，在一台计算机上拥有多个支持CUDA的设备也是很常见的情形。在这些情况中，我们希望通过某种方式来确定使用的是哪个处理器。
例如，在许多主板中都集成了NVIDIA图形处理器。当计算机生产商或者用户将--块独立的图形处理器添加到计算机时，那么就有了两个支持CUDA的处理器。某些NVIDIA产品，例如GeForce GTX 295，在单块卡上包含了两个GPU，因此使用这类产品的计算机也就拥有了两个支持CUDA的处理器。
在深入研究如何编写设备代码之前，我们需要通过某种机制来判断计算机中当前有哪些设备，以及每个设备都支持哪些功能。幸运的是，可以通过一个非常简单的接口来获得这种信息。首先，我们希望知道在系统中有多少个设备是支持CUDA架构的，并且这些设备能够运行基于CUDA C编写的核函数。要获得CUDA设备的数量，可以调用cudaGetDeviceCount()。这个函数的作用从它的名字就可以看出来。

```
int count;
HANDLE_ERROR(cudaGetDeviceCount(&count));
```

在调用cudaGetDeviceCount()后，可以对每个设备进行迭代，并查询各个设备的相关信息。CUDA运行时将返回一个cudaDeviceProp类型的结构，其中包含了设备的相关属性。我们可以获得哪些属性?从CUDA 3.0开始，在cudaDeviceProp结构中包含了以下信息:

略（具体参考《NVIDIA CUDA Programming Guide》）

下面给出对设备进行查询的代码：

```c
#include "../common/book.h"

int main( void ) {
    cudaDeviceProp  prop;

    int count;
    HANDLE_ERROR( cudaGetDeviceCount( &count ) );
    for (int i=0; i< count; i++) {
        HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) );
        printf( "   --- General Information for device %d ---\n", i );
        printf( "Name:  %s\n", prop.name );
        printf( "Compute capability:  %d.%d\n", prop.major, prop.minor );
        printf( "Clock rate:  %d\n", prop.clockRate );
        printf( "Device copy overlap:  " );
        if (prop.deviceOverlap)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n");
        printf( "Kernel execution timeout :  " );
        if (prop.kernelExecTimeoutEnabled)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n" );

        printf( "   --- Memory Information for device %d ---\n", i );
        printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
        printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
        printf( "Max mem pitch:  %ld\n", prop.memPitch );
        printf( "Texture Alignment:  %ld\n", prop.textureAlignment );

        printf( "   --- MP Information for device %d ---\n", i );
        printf( "Multiprocessor count:  %d\n",
                    prop.multiProcessorCount );
        printf( "Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
        printf( "Registers per mp:  %d\n", prop.regsPerBlock );
        printf( "Threads in warp:  %d\n", prop.warpSize );
        printf( "Max threads per block:  %d\n",
                    prop.maxThreadsPerBlock );
        printf( "Max thread dimensions:  (%d, %d, %d)\n",
                    prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                    prop.maxThreadsDim[2] );
        printf( "Max grid dimensions:  (%d, %d, %d)\n",
                    prop.maxGridSize[0], prop.maxGridSize[1],
                    prop.maxGridSize[2] );
        printf( "\n" );
    }
}
```


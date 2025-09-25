# 1.linux内存共享

**linux共享内存是通过tmpfs这个文件系统来实现的，tmpfs文件系的目录为/dev/shm**



主要用到这几个函数

```cpp
int shm_open(const char *name, int oflag, mode_t mode);
void *mmap(void *addr, size_t length, int prot, int flags,int fd, off_t offset);
int munmap(void *addr, size_t length);
int shm_unlink(const char *name);
int ftruncate(int fd, off_t length);
```



## 1.shm_open

**用于创建或者打开共享内存文件**

**该文件会被储存在/dev/shm目录下**



**参数说明：**

**name**：要打开或创建的共享内存文件名

**oflag**：打开的文件操作属性：O_CREAT、O_RDWR、O_EXCL的按位或运算组合

| 标志       | 作用                                  | 典型用法/注意点                                              |
| ---------- | ------------------------------------- | ------------------------------------------------------------ |
| `O_RDONLY` | 只读打开共享内存对象                  | 只能 `mmap(PROT_READ, …)`；若你想可写映射或 `ftruncate` 改尺寸，**不要**用它。 |
| `O_RDWR`   | 读写打开                              | 要做写映射（`PROT_WRITE`/`PROT_READ                          |
| `O_CREAT`  | 若不存在则创建                        | 新建对象初始长度为 **0**，接着应 `ftruncate(fd, size)` 设定实际大小，否则 `mmap` 会因越界而失败。 |
| `O_EXCL`   | 与 `O_CREAT` 一起使用，要求“必须新建” | 如果已存在则返回 `-1` 且 `errno=EEXIST`。单独用 `O_EXCL` 没意义。 |
| `O_TRUNC`  | 截断为长度 0                          | 常与 `O_RDWR` 一起用来“清空并重设大小”；随后也要 `ftruncate` 到新尺寸。多数实现里，`O_TRUNC` 与只读打开不兼容。 |



**mode**：文件共享模式，例如 0777





**返回值**: 成功时返回大于等于 0 的 **文件描述符 fd**， 失败返回fd<0







## 2.**mmap**

**将打开的文件映射到内存**

**这个函数只是将文件映射到内存中，使得我们用操作内存指针的方式来操作文件数据。**





**参数说明:**

**addr**：要将文件映射到的内存地址，一般应该传递NULL来由Linux内核指定。

**length**：要映射的文件数据长度。**一般可以设置成共享数据的大小如sizeof(SharedData)**

**prot**：映射的内存区域的操作权限（保护属性）

**flags**：标志位参数

**fd**：  用来建立映射区的文件描述符，**用 shm_open打开或者open打开的文件**。

**offset**：映射文件相对于文件头的偏移位置，应该按4096字节对齐。一般为0



**返回值**:成功返回映射的内存地址指针，可以用这个地址指针对映射的文件内容进行读写操作，读写文件数据如同操作内存一样；如果 失败则返回MAP_FAILED。



**成功映射之后，这个返回值如data就可以当成一个指针来操作储存的数据**

如

```cpp
struct SharedData{
	int value;

};


.....

auto *data = (SharedData*)mmap(....);

//上面经过一系列操作后，映射成功后，就可以把他当成一个对应结构体的变量
data->value = 10;
```







## 3.munmap

**取消内存映射**

**munmap** 只是将映射的内存从进程的地址空间撤销，如果不调用这个函数，则在进程终止前，该片区域将得不到释放



**参数说明:**

​	addr是由mmap成功返回的地址

​	length是要取消的内存长度





## 4.shm_unlink

**删除/dev/shm目录的文件**



name就是shm_open创建的文件名





## 5.ftruncate

**重置文件大小。任何open打开的文件都可以用这个函数，不限于shm_open打开的文件。**、



**请注意每次打开文件之后，最好用这个来控制文件的大小。**





fd就是shm_open的返回值

length一般可以是sizeof(ShareData)












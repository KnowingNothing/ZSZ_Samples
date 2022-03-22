有幸可以负责一个centos机器的维护，以前没接触过，记录使用遇到的问题和解决，大部分内容可能都是网上google来的，记在这里方便查找

首先安装使用的yum
yum -y install htop

为了使用高版本gcc，使用devtoolset
yum -y install centos-release-scl
yum -y install devtoolset-8-gcc devtoolset-8-gcc-c++ devtoolset-8-binutils
scl enable devtoolset-8 bash
类似的可以安装版本7


添加用户
#首先登录 root 账号

#新建用户
useradd username
#更改用户密码，激活使用
passwd username

#首先登录 root 账户

visudo

#找到如下行数
root  ALL=(ALL)   ALL
添加
username ALL=(ALL) ALL


安装Python3
yum -y install zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel make libffi-devel
yum -y install epel-release python-pip
pip install --user wget
wget https://www.python.org/ftp/python/3.8.0/Python-3.8.0.tgz
tar xvf Python-3.8.0.tgz
./configure prefix=/...
make && make install
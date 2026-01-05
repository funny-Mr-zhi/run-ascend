# run-ascend
完整的ascend配置流程

## 配置流程

1. 检查NPU、OS状态：`npu-smi info` `cat /etc/release`
2. 检查CANN(可理解为NVIDIA的CUDA)是否安装好：`ls -l /usr/local/Ascend`
3. 尝试激活环境配置`source /usr/local/Ascend/ascend-toolkit/set_env.sh`。
4. 检查是否成功`atc --help`
5. 输出正常则成功，永久化配置`echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc` `source ~/.bashrc`
6. 检查是否可引用：`python -c "import acl; print('成功连接 NPU !')"`
7. 以上意味着硬件就绪，驱动/固件就绪，环境（CANN/ACL）就绪

接下来安装`torch-npu`或`Mindsopre`

1. 上网址`https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0002.html`
2. 参考安装即可(注意需要手动提前安装pyyaml库)
3. 验证安装成功



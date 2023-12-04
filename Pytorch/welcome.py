# 这是一个示例 Python 脚本。

# 按 ⌃R 执行或将其替换为您的代码。
# 按 双击 ⇧ 在所有地方搜索类、文件、工具窗口、操作和设置。

str = ("这是一个欢迎文件！"
       "本项目用于学习pytorch\n"
       "‘env’文件夹是一个本地安装包虚拟环境")
print(str)

content = ["torch tutorial", "mps tutorial"]
len = len(content)
nums = range(len)


def display():
    str = "数字键 0 -> 离开"
    print(str)
    for i in range(len):
        str = "数字键 {} -> 内容: {}".format(nums[i] + 1, content[i])
        print(str)


def choice1():
    import torch_practice


def choice2():
    import mps_practice


switch = {
    '1': choice1,
    '2': choice2
}

user_input = []
timer = 0
while True:
    display()
    user_input.append(input())
    if user_input[timer] == '0':
        break
    switch.get(user_input[timer], lambda: print("输入无效"))()
    timer += 1

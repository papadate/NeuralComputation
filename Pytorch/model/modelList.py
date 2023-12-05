models = []
names = []
user_input = []
timer = 0

switch

while True:
    print("请选择需要运行的模型")
    print("数字键：0 -> 离开模型列表(返回上级目录)")
    for i in range(len(list)):
        str = "数字键：{} -> 模型：{}".format(i + 1, names[i])
        print(str)
    user_input.append(input())
    # 断开程序
    if user_input[timer] == '0':
        break

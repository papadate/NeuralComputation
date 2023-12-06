def run():
    import model.FirstModel as FirstModel
    import model.NestedModel as NestedModel
    import model.ClassificationModel as ClassificationModel

    models = []
    user_input = []
    timer = 0

    models.append("First Model")
    models.append("Nested Model")
    models.append("Classification Model")

    def choice1():
        FirstModel.run()

    def choice2():
        NestedModel.run()

    def choice3():
        ClassificationModel.run()

    switch = {
        '1': choice1,
        '2': choice2,
        '3': choice3
    }

    while True:
        print("请选择需要运行的模型")
        print("数字键：0 -> 离开模型列表(返回上级目录)")
        for i in range(len(models)):
            str = "数字键：{} -> 模型：{}".format(i + 1, models[i])
            print(str)
        user_input.append(input())
        # 断开程序
        if user_input[timer] == '0':
            break
        switch.get(user_input[timer], lambda: print("输入无效"))()
        timer += 1

    print()

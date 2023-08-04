统一镜像：763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118

````
# length -> 12
PROMPT1="""我玩你们的游戏导致失恋了"""

# length -> 219
PROMPT2="""回答下列选择题：

Q: "g4dn的价格是多少"，这个问题的提问意图是啥？可选项[价格咨询, 知识问答]
A: 价格咨询

Q: "全局端点的费用是多少？"，这个问题的提问意图是啥？可选项[价格咨询, 知识问答]
A: 知识问答

Q: "Spot 实例的价格是多少？"，这个问题的提问意图是啥？可选项[价格咨询, 知识问答]
A: 知识问答

Q: "G5实例多少钱？", 这个问题的提问意图是啥？可选项[价格咨询, 知识问答]
A:"""

# length -> 588
PROMPT3="""你是扇贝单词app的智能客服机器人小贝，请严格根据反括号中的资料提取相关信息，回答用户的各种问题
```
问题: 忘记打卡了，可以补卡吗？ 
回答: 同学你好，可以通过以下方式补打卡哦～如果你已完成全部学习任务并上传数据，只是没有打卡，我们会在第二天通过客户端及网页端通知你补打卡，补打卡链接7天内有效。同学如果已经连续打卡21天，那么就可以在我的-打卡天数页面左上角启动时光机补全任意一次忘记打卡。 

问题: 如何补打卡 
回答: 同学你好，点击打卡日历左上角补打卡按钮，可以选择时光机（连续打卡21天可用）或大会员特权（每月1次）补打卡哦～ 

问题: 为什么打卡后看不到今日成就页面了？ 
回答: 同学你好，今日成就的入口调整了哦～完成打卡之后，可以在app【我的-学习数据】页面查看今日成就哦。 

问题: 今天打卡了，为什么没有计入队伍打卡天数内？ 
回答: 同学你好，报名活动后至少邀请1位好友加入队伍才能开启挑战哦！开启挑战后的打卡才会计入活动打卡天数内，挑战开启之前的打卡都不会计入活动中。举个例子：今天8:00加入队伍开启的挑战，但是7:00已经在扇贝打过卡了，那么，今天的打卡就不会计入活动打卡天数里哦，但是后面的打卡都会正常计入活动里哒～ 
```
用户: 打卡打不了。学完单词后点击打卡，结果页面一直没有跳转出来，一直是空白页面。 
小贝:"""
````
================

无Server-side-batch

GPU资源 = ml.g4.2xlarge*1

加速引擎: accelerate

Run time=5m, User Num: 20, Spawn rate:1, wait_time = between(1, 5)

| Prompt | max_new_tokens | Avg RPS | Request Count | P50   | P90   | Bytes | Fails |
| ------ | -------------- | ------- | ------------- | ----- | ----- | ----- | ----- |
| 1      | 512            | 0.016   | 84            | 60000 | 60000 | 945   | 79    |

Run time=5m, User Num: 10, Spawn rate:1, wait_time = between(1, 5)

| Prompt | max_new_tokens | Avg RPS | Request Count | P50   | P90   | Bytes | Fails |
| ------ | -------------- | ------- | ------------- | ----- | ----- | ----- | ----- |
| 1      | 512            | 0.016   | 43            | 60000 | 60000 | 945   | 38    |

Run time=5m, User Num: 4, Spawn rate:1, wait_time = between(1, 5)

| Prompt | max_new_tokens | Avg RPS | Request Count | P50   | P90   | Bytes | Fails |
| ------ | -------------- | ------- | ------------- | ----- | ----- | ----- | ----- |
| 1      | 512            | 0.08    | 24            | 44000 | 47000 | 1024  | 0     |
| 2      | 512            | 1.23    | 370           | 290   | 500   | 30    | 0     |
| 3      | 512            | 0.283   | 85            | 11000 | 13000 | 314   | 0     |
| 1      | 32             | 0.586   | 176           | 3800  | 5000  | 179   | 0     |
| 2      | 32             | 1.22    | 367           | 290   | 500   | 30    | 0     |
| 3      | 32             | 0.52    | 156           | 4600  | 6000  | 170   | 0     |

================

无Server-side-batch

GPU资源 = ml.g5.2xlarge*1

加速引擎: accelerate

Run time=5m, User Num: 10, Spawn rate:1, wait_time = between(1, 5)

| Prompt | max_new_tokens | Avg RPS | Request Count | P50   | P90   | Bytes | Fails |
| ------ | -------------- | ------- | ------------- | ----- | ----- | ----- | ----- |
| 1      | 512            | 0.033   | 47            | 60000 | 60000 | 1047  | 37    |

Run time=5m, User Num: 4, Spawn rate:1, wait_time = between(1, 5)

| Prompt | max_new_tokens | Avg RPS | Request Count | P50   | P90   | Bytes | Fails |
| ------ | -------------- | ------- | ------------- | ----- | ----- | ----- | ----- |
| 1      | 512            | 0.146   | 44            | 24000 | 26000 | 1047  | 0     |
| 2      | 512            | 1.27    | 381           | 210   | 310   | 30    | 0     |
| 3      | 512            | 0.506   | 152           | 5000  | 6400  | 314   | 0     |
| 1      | 32             | 0.87    | 262           | 1300  | 2200  | 179   | 0     |
| 2      | 32             | 1.25    | 375           | 210   | 290   | 30    | 0     |
| 3      | 32             | 0.85    | 255           | 1700  | 2600  | 170   | 0     |

================

Server-side-batch = 4， Delay = 1000

GPU资源 = ml.g4dn.2xlarge * 1

加速引擎: deepspeed

Run time=5m, User Num: 20, Spawn rate:1, wait_time = between(1, 5)

| Prompt | max_new_tokens | Avg RPS | Request Count | P50   | P90   | Bytes | Fails |
| ------ | -------------- | ------- | ------------- | ----- | ----- | ----- | ----- |
| 1      | 512            | 0.7     | 210           | 24000 | 26000 | 513   | 0     |
| 2      | 512            | 5.12    | 1537          | 790   | 1100  | 12    | 0     |
| 3      | 512            | 1.31    | 393           | 12000 | 13000 | 174   | 0     |
| 1      | 32             | 3.06    | 918           | 3300  | 4000  | 108   | 0     |
| 2      | 32             | 5.11    | 1533          | 790   | 1300  | 12    | 0     |
| 3      | 32             | 1.4     | 420           | 11000 | 12000 | 162   | 0     |

GPU利用率峰值99%

================

Server-side-batch = 4， Delay = 1000

GPU资源 = ml.g5.2xlarge * 1

加速引擎: deepspeed

Run time=5m, User Num: 20, Spawn rate:1, wait_time = between(1, 5)

| Prompt | max_new_tokens | Avg RPS | Request Count | P50   | P90   | Bytes | Fails |
| ------ | -------------- | ------- | ------------- | ----- | ----- | ----- | ----- |
| 1      | 512            | 1.23    | 370           | 13000 | 14000 | 513   | 0     |
| 2      | 512            | 5.52    | 1656          | 460   | 940   | 12    | 0     |
| 3      | 512            | 2.31    | 695           | 5500  | 6400  | 174   | 0     |
| 1      | 32             | 3.69    | 1109          | 2200  | 2900  | 161   | 0     |
| 2      | 32             | 5.51    | 1653          | 460   | 920   | 12    | 0     |
| 3      | 32             | 2.54    | 763           | 4700  | 5700  | 162   | 0     |

================

Server-side-batch = 4， Delay = 1000

GPU资源 = ml.g4dn.2xlarge * 4

加速引擎: deepspeed

Run time=5m, User Num: 20, Spawn rate:1, wait_time = between(1, 5)

| Prompt | max_new_tokens | Avg RPS | Request Count | P50  | P90   | Bytes | Fails |
| ------ | -------------- | ------- | ------------- | ---- | ----- | ----- | ----- |
| 1      | 512            | 1.59    | 477           | 9000 | 11000 | 517   | 0     |
| 2      | 512            | 4.77    | 1432          | 1300 | 1500  | 18    | 0     |
| 3      | 512            | 2.63    | 790           | 4100 | 5800  | 178   | 0     |
| 1      | 32             | 3.16    | 949           | 3000 | 4300  | 165   | 0     |
| 2      | 32             | 4.75    | 1426          | 1300 | 1600  | 17    | 0     |
| 3      | 32             | 2.81    | 844           | 3800 | 5200  | 166   | 0     |

GPU利用率在运行Prompt2的时候，并没有达到100%仅仅20%。

================

Server-side-batch = 4， Delay = 1000

GPU资源 = ml.g5.2xlarge * 4

加速引擎: deepspeed

Run time=5m, User Num: 20, Spawn rate:1, wait_time = between(1, 5)

| Prompt | max_new_tokens | Avg RPS | Request Count | P50  | P90  | Bytes | Fails |
| ------ | -------------- | ------- | ------------- | ---- | ---- | ----- | ----- |
| 1      | 512            | 2.39    | 719           | 4900 | 6800 | 517   | 0     |
| 2      | 512            | 4.96    | 1489          | 1100 | 1200 | 17    | 0     |
| 3      | 512            | 3.45    | 1035          | 2600 | 3600 | 179   | 0     |
| 1      | 32             | 3.92    | 1178          | 2000 | 2700 | 167   | 0     |
| 2      | 32             | 4.96    | 1488          | 1000 | 1200 | 17    | 0     |
| 3      | 32             | 3.63    | 1089          | 2300 | 3300 | 166   | 0     |

GPU利用率没用满

Run time=5m, User Num: 40, Spawn rate:1, wait_time = between(1, 5)

| Prompt | max_new_tokens | Avg RPS | Request Count | P50  | P90  | Bytes | Fails |
| ------ | -------------- | ------- | ------------- | ---- | ---- | ----- | ----- |
| 3      | 512            | 6.51    | 1954          | 2700 | 3800 | 175   | 0     |

Run time=5m, User Num: 60, Spawn rate:1, wait_time = between(1, 5)

| Prompt | max_new_tokens | Avg RPS | Request Count | P50  | P90  | Bytes | Fails |
| ------ | -------------- | ------- | ------------- | ---- | ---- | ----- | ----- |
| 3      | 512            | 8.22    | 2468          | 3400 | 5200 | 174   | 0     |

Run time=5m, User Num: 100, Spawn rate:5, wait_time = between(1, 5)

| Prompt | max_new_tokens | Avg RPS | Request Count | P50  | P90   | Bytes | Fails |
| ------ | -------------- | ------- | ------------- | ---- | ----- | ----- | ----- |
| 3      | 512            | 9.22    | 2768          | 7200 | 12000 | 174   | 0     |

![img](https://br5879sdns.feishu.cn/space/api/box/stream/download/asynccode/?code=MzQ5NzVlZWJlYjUzNTU4N2M0N2E0YTAyMjkzYTk0ZDFfc09aZkxxUWlwYkpWWnhJb1l1UjJOYzlsZUVYTDdjUkZfVG9rZW46V1djbWJucjRFb0s5YTR4dUFCZ2M3Q2pObjJmXzE2OTExMTM1MTI6MTY5MTExNzExMl9WNA)



# 总结:

1. SeverSide Batch Inference 可以大幅改善推理吞吐量，支持5-10倍以上用户
2. SeverSide Batch Inference开启的前提下，G5相对于G4能够提供175%的吞吐量
3. Completion token数决定了生成时间，而prompt token数影响较小
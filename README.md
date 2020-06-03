# 天才海神号最终代码说明
这是我们在天池大赛，2020数字中国创新大赛—算法赛：智慧海洋建设中复赛阶段B榜单获得最好成绩的代码。

比赛连接为，https://tianchi.aliyun.com/competition/entrance/231768/rankingList

作者: 天才海神号

时间: 2020年03月14日

## 最终结果

### 分数
```
score = 0.89937
```

### 运行结果日志
```
{
    "eval_score": 0.8993695740342255,
    "cost_time": 867,
    "info": "null",
    "deadline": 18000,
    "score_detail": {
        "success": "true",
        "score": 0.8993695740342255,
        "scoreJson": {
            "score": 0.8993695740342255
        }
    }
}
```


## 方案整体介绍
## 方案优点
- 简单高效，过拟合风险低。
- 代码逻辑清晰简单，约200+行代码，可读性高，易于扩展和使用。
- 在百万量级数据，只提取了100+有效特征，全程运行时间只有16分钟左右。
- 特征工程中包含，时间，空间，速度，位移，相对值等各个维度的特征，全面且精简。
- 符合现实世界中的使用要求。


## 包括5个模块
- 读数据
- 预处理
- 特征工程
- 模型训练和预测
- 保存结果


## 1 读数据
我们将第一阶段训练数据加到训练集合，达到数据增强的作用
```
# get_data('/tcdata/hy_round2_train_20200225', 'train')
# get_data('/tcdata/hy_round2_testA_20200225', 'testA')
# get_data('/tcdata/hy_round2_testB_20200312', 'testB')
# get_data('/tcdata/hy_round1_train_20200102', 'train_chusai')
```

## 2 预处理
为方便处理，我们将原始数据的中文映射到数字和英文
```
# label_dict1 = {'拖网': 0, '围网': 1, '刺网': 2}
# label_dict2 = {0: '拖网', 1: '围网', 2: '刺网'}
# name_dict = {'渔船ID': 'id', '速度': 'v', '方向': 'dir', 'type': 'label', 'lat': 'x', 'lon': 'y'}
```

对原始坐标做一个变换，得到近似的平面坐标。对时间进行格式化
```
df['x'] = df['x'] * 100000 - 5630000
df['y'] = df['y'] * 110000 + 2530000
df['time'] = pd.to_datetime(df['time'].apply(lambda x :'2019-'+ x[:2] + '-' + x[2:4] + ' ' + x[5:]))
```

## 3 特征工程
分为5大组特征

### 3.1
分箱特征，距离海岸线的近似值。
```
对v求分箱特征，等分为200份，求每一份的统计值
对x求分箱特征，1000份和10000份，求每一份的次数统计值，和每一个分箱对应不同id数目
对y求分箱特征，1000份和10000份，求每一份的次数统计值，和每一个分箱对应不同id数目
求x，y分箱后的组合特征做为分组，求对应的次数统计值，和对应的id的不同数目
根据x分组，求y距离最小y的距离 # 可以理解为距离海岸线距离
根据y分组，求x距离最小x的距离 # 可以理解为距离海岸线距离
```

### 3.2
间隔空间位移特征
```
根据id分组，对x求，上一个x，下一个x，间隔2个x的距离
根据id分组，对y求，上一个y，下一个y，间隔2个y的距离
根据上述距离，求上一时刻，下一时刻，间隔2个时刻的面积，相对值
```

### 3.3
空间位移的文本特征，提取Word2Vec，具有前后关系
```
根据id分组，以xy网格特征编号作为单词，求文本特征，Word2Vec，窗口大小为10，提取10维的特赠
```

### 3.4
常见统计特征，相对值
```
根据v_bin和dist_move_prev_bin分组，求其他列的常见统计特征
'id': ['count'], 'x_bin1': [mode], 'y_bin1': [mode], 'x_bin2': [mode], 'y_bin2': [mode], 'x_y_bin1': [mode],
'x': ['mean', 'max', 'min', 'std', np.ptp, start, end],
'y': ['mean', 'max', 'min', 'std', np.ptp, start, end],
'v': ['mean', 'max', 'min', 'std', np.ptp], 'dir': ['mean'],
'x_bin1_count': ['mean'], 'y_bin1_count': ['mean', 'max', 'min'],
'x_bin2_count': ['mean', 'max', 'min'], 'y_bin2_count': ['mean', 'max', 'min'],
'x_bin1_y_bin1_count': ['mean', 'max', 'min'],
'dist_move_prev': ['mean', 'max', 'std', 'min', 'sum'],
'x_y_min': ['mean', 'min'], 'y_x_min': ['mean', 'min'],
'x_y_max': ['mean', 'min'], 'y_x_max': ['mean', 'min'],
```

### 3.5
行程特征
```
总行程距离
每一步行程的占比
将'dist_move_prev_bin_sen', 'v_bin_sen'转化为onehot稀疏特征
```

## 4 模型训练和预测
我们使用全量数据进行训练，使用分层的K折作为验证，训练lgb多分类模型。

## 5 保存结果
将预测结果进行格式化，按要求保存的指定目录，得到最终结果，result.csv。


# enjoy : )

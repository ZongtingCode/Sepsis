import h2o
from h2o.automl import H2OAutoML
h2o.init()
# import seaborn as sns
# import pandas as pd
# import shap
# import matplotlib.colors as colors
# from h2o.model.model_base import ModelBase
# import matplotlib.colors as mcolors
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.calibration import calibration_curve
# from h2o.estimators.gbm import H2OGradientBoostingEstimator
# from statsmodels.genmod.generalized_linear_model import GLM
# from statsmodels.genmod.families import Binomial
# from statsmodels.genmod.families.links import logit
# from scipy.stats import chi2
# import statsmodels.api as sm
# from sklearn.metrics import log_loss
# from scipy import stats
# from sklearn.metrics import roc_curve, auc
# import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingClassifier
# from h2o.estimators import H2OGradientBoostingEstimator

#上传文件,文件链接来自亚马逊云盘
train = h2o.import_file("https://shujuchuli.s3.ap-northeast-1.amazonaws.com/PD%E5%8F%98%E9%87%8F%E7%AC%AC%E4%BA%8C%E7%89%88%EF%BC%8810%E5%8F%98%E9%87%8F%EF%BC%89.csv?response-content-disposition=attachment&X-Amz-Security-Token=IQoJb3JpZ2luX2VjENr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLXdlc3QtMiJHMEUCIQChYEa0%2FtFdfh5mVnLKqvZ5x93eheKt48vW2chetlU0TwIgFDbp9qC3cKqg7g97%2FrBBEBpBO7Dq%2F%2BGYzpTPRy4pY0Yq5AIIEhAAGgwwODcyMDMwNTgxNzYiDGNyBQYefs0lAMfuNyrBApMwp3X6NAYAxxcGIOKYz1PvB1CMGr19NCchsUm3IZn137Kf9LfGNA20XehLygnOVPD6nagR58wLHJKl8jg9%2FEqb0xXjGhLpjC8YWSsamnd2qip3I3KVtNa6Ph0XORMY%2B%2B3dscJyYGiumCJd9PfpsbsPaY%2Fddq3ecb8ATzc7u6MCjCI9chQkhtP5NZ5KyKDRJSWgySIsa3wRSXUJsza9iaXy0ufBuVfM7SOrxsc1JXvek3kVzZzENDcc4j1iwwDiOjA3Yobx2w6wF4Z4a%2BJ8FZADlau%2BrpgeQKnNmkYpQGW4XUrS1oEQATkVWCkSomWuCISkhVz9gywlJTvfRKgq5Bnpmo9n6Or%2BhsXItP7EE1CBMjlTbr3e0RTdqUipz26w3mVIyDjc7l51fZo73exukhm7v4AuLfDUt24pu1bLa1INfjDMsIiqBjqzAjJQINan6Zy9l7M2y3Lhd2OsVW9LSsDzoz4O691Wd3UE5y1LmG8W1WF35nx2ZrD7USnVa9RI8FywqCp2NiHDgjXW49kEukKdo2%2FrPHJz1UUyZ4ljOKR11vErYqghNnDJ3gAwQ7AFGr0b6PfaJw0pBITMg0XAIMKPveRZvo2v08t0lqrfoAXDuS0o2JNrGH%2BY9dlz3MnjrUGO%2BMcGWSHtGzEVydea89qvV%2Bca9MLiojqZmPe6th1jkaEo%2FvV91jmg49CarsMYzopRJHpKih3VDwCMMb%2FjOgYI%2BftjSvXJUOEHOXp%2FzZtgmHzIqyiaYkxxhnWw%2Fu30JSGkU85YOi2no73h3UjG3OvPaTl7Af9hzRhNdBvkMesMN%2B0NiB6MZkS1%2Fjv00E%2BElO1viG%2BZ%2F27pjjUU2oE%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20231101T092041Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIARITNVCIAEOX4SI7O%2F20231101%2Fap-northeast-1%2Fs3%2Faws4_request&X-Amz-Signature=65d7974ef5de3c9f0c61a325a856e0a14b4ffcd165799b4050fb0538694ea6d4")
valid = h2o.import_file("https://shujuchuli.s3.ap-northeast-1.amazonaws.com/valid_PD%E5%8F%98%E9%87%8F.csv?response-content-disposition=attachment&X-Amz-Security-Token=IQoJb3JpZ2luX2VjENr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLXdlc3QtMiJHMEUCIQChYEa0%2FtFdfh5mVnLKqvZ5x93eheKt48vW2chetlU0TwIgFDbp9qC3cKqg7g97%2FrBBEBpBO7Dq%2F%2BGYzpTPRy4pY0Yq5AIIEhAAGgwwODcyMDMwNTgxNzYiDGNyBQYefs0lAMfuNyrBApMwp3X6NAYAxxcGIOKYz1PvB1CMGr19NCchsUm3IZn137Kf9LfGNA20XehLygnOVPD6nagR58wLHJKl8jg9%2FEqb0xXjGhLpjC8YWSsamnd2qip3I3KVtNa6Ph0XORMY%2B%2B3dscJyYGiumCJd9PfpsbsPaY%2Fddq3ecb8ATzc7u6MCjCI9chQkhtP5NZ5KyKDRJSWgySIsa3wRSXUJsza9iaXy0ufBuVfM7SOrxsc1JXvek3kVzZzENDcc4j1iwwDiOjA3Yobx2w6wF4Z4a%2BJ8FZADlau%2BrpgeQKnNmkYpQGW4XUrS1oEQATkVWCkSomWuCISkhVz9gywlJTvfRKgq5Bnpmo9n6Or%2BhsXItP7EE1CBMjlTbr3e0RTdqUipz26w3mVIyDjc7l51fZo73exukhm7v4AuLfDUt24pu1bLa1INfjDMsIiqBjqzAjJQINan6Zy9l7M2y3Lhd2OsVW9LSsDzoz4O691Wd3UE5y1LmG8W1WF35nx2ZrD7USnVa9RI8FywqCp2NiHDgjXW49kEukKdo2%2FrPHJz1UUyZ4ljOKR11vErYqghNnDJ3gAwQ7AFGr0b6PfaJw0pBITMg0XAIMKPveRZvo2v08t0lqrfoAXDuS0o2JNrGH%2BY9dlz3MnjrUGO%2BMcGWSHtGzEVydea89qvV%2Bca9MLiojqZmPe6th1jkaEo%2FvV91jmg49CarsMYzopRJHpKih3VDwCMMb%2FjOgYI%2BftjSvXJUOEHOXp%2FzZtgmHzIqyiaYkxxhnWw%2Fu30JSGkU85YOi2no73h3UjG3OvPaTl7Af9hzRhNdBvkMesMN%2B0NiB6MZkS1%2Fjv00E%2BElO1viG%2BZ%2F27pjjUU2oE%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20231101T092105Z&X-Amz-SignedHeaders=host&X-Amz-Expires=299&X-Amz-Credential=ASIARITNVCIAEOX4SI7O%2F20231101%2Fap-northeast-1%2Fs3%2Faws4_request&X-Amz-Signature=75439268f52a879123d24023fe1b2fd7da2487a3f424b377e6f33b8ef47517f0")
#将结局事件和变量分出来
x = train.columns
y = "othbleed"
x.remove(y)
#将训练集和测试集的结局事件设为因子
train[y] = train[y].asfactor()
valid[y] = valid[y].asfactor()
#开始自动机器学习
aml = H2OAutoML(max_models=20,nfolds = 5,seed=1)
aml.train(x=x, y=y, training_frame=train, validation_frame = valid)
#保存模型
#查看自动机器学习的排行榜
lb = aml.leaderboard
lb.head(rows=lb.nrows)
print(lb)
#按型号获取对应模型
m = h2o.get_model("GBM_grid_1_AutoML_19_20231005_104801_model_2")
print(m)
#将模型转换为MOJO用来建网页
GBMmodelfile = m.download_mojo(path="/home/zz960908/PD运行python", get_genmodel_jar=True)
#使用验证集对模型的性能进行评估
perf = m.model_performance(valid)

#画出模型训练集ROC
selected_model_id = 'GBM_grid_1_AutoML_19_20231005_104801_model_2'
# 获取该模型的train的性能ROC
model = h2o.get_model(selected_model_id)
perf = model.model_performance(train)  # 获取验证集的性能度量
fpr = perf.fprs
tpr = perf.tprs
auc = perf.auc()
plt.figure(figsize=(15, 10))
# 绘制ROC曲线
plt.plot(fpr, tpr, color='#FD4B0D', label=f"AUC = {auc:.3f}")
ax = plt.gca()
ax.grid(False)
# 添加随机分类线
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#6C2A1A', label='Chance', alpha=.8)
plt.title("GBM Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("GBM_train_ROC.png", dpi=300)
plt.show()

#画出模型验证集ROC
selected_model_id = 'GBM_grid_1_AutoML_19_20231005_104801_model_2'
# 获取该模型的valid的性能ROC
model = h2o.get_model(selected_model_id)
perf = model.model_performance(valid)  # 获取验证集的性能度量
fpr = perf.fprs
tpr = perf.tprs
auc = perf.auc()
plt.figure(figsize=(15, 10))

# 绘制ROC曲线
plt.plot(fpr, tpr, color='#FD4B0D', label=f"AUC = {auc:.3f}")
ax = plt.gca()
ax.grid(False)
# 添加随机分类线
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#6C2A1A', label='Chance', alpha=.8)
plt.title("GBM Model Valid")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("GBM_valid_ROC.png", dpi=300)
plt.show()
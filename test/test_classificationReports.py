from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# [0, 1, 2] 标签转换为名称 ['setosa' 'versicolor' 'virginica']
y_labels = iris.target_names[y]

# 数据集拆分为训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2)

# 使用训练集训练模型
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# 使用测试集预测结果
y_pred = clf.predict(X_test)

# 生成文本型分类报告
print(classification_report(y_test, y_pred))
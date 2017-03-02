### 数据集

Paddle目前提供了很多demo，且各demo运行时需要从原生网站下载其数据，并进行复杂的预处理过程，整个过程会耗费大量时间。同时为了方便大家用Paddle做实验的时候，可以直接访问这些预处理好的数据，我们提供一套Python库。采用import数据源的方式(如：paddle.data.amazon_product_reviews)来简化获取训练所需数据的时间；但是如果你习惯自己处理原生数据，我们依然提供原生数据接口来满足你的需求。

## 接口设计
数据集的导入通过import paddle.data.amazon_product_reviews 来实现，你可以直接通过load_data(category=None,
directory=None)获取你所需的数据集。考虑到类似Amazon的数据类型不止一种，通过category你可以选择控制所需要的数据源;如果你不指定数据源，默认为"Electronics"。directory用来指定下载路径，如果你不指定下载路径，默认为"~/paddle_data/amazon"。通过load_data()导入的数据源data为object，他是我们预处理的numpy格式数据，直接通过data.train_data()获取训练数据或者通过data.test_data()获取测试数据。你还可以打印训练数据和测试数据的数据信息，

 ```python
  for each_train_data in data.train_data():
      print each_train_data
 ```
 即可。

 具体的demo使用情况如下：
 ```python
 import paddle.data.amazon_product_reviews  as raw

 raw.data(batch_size=10)
 ```
 你也可以打印出各数据集的数据信息：
 如果是测试集或者训练数据集,可以这么打印
 ```python
 import paddle.data.amazon_product_reviews  as raw

 raw.test_data(batch_size=10)
 raw.train_data(batch_size=10)

 ```

 打印出来的数据信息都是预处理之后的numpy格式的数据：
 ```python
 (array([1370072,  884914, 1658622, 1562803,    1579,  369164, 1129091,
        1073545, 1410234,  857854,  672274,  884920, 1078270, 1410234,
                777903, 1352600,  497103,  132906,  239745,   65294, 1502324,
                       1165610,  204273, 1610806,  942942,  709056,  452087,  118093,
                              1410234], dtype=int32), array([ True], dtype=bool))
 (array([ 777903,  713632,  452087, 1647686,  877980,  294749, 1575945,
         662947, 1431519,  462950,  452087,  902916,  479242,  294749,
                1278816,  672274,    1579,  394865, 1129091, 1352600,  294749,
                       1073545], dtype=int32), array([ True], dtype=bool))

 ```


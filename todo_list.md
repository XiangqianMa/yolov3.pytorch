## TODO List
- [x] 统一数据集接口（get_dataloader）
- [ ] 测试单独使用mosaic和混合mosaic、正常读取两种方案的效果
- [x] 搞清楚`.data`的作用
    - [x] 对检测结果无影响
    - [x] 对训练过程的影响
- [x] giou_loss未完成调试
- [ ] 在生成标注框时，是否对越界的框进行处理
- [ ] 数据集中的数据增强部分存在越界的情况，怎么回事？
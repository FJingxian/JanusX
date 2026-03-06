# Backend for JanusX ui

jx webui 启动

## 可视化相关

1. postgwas相关
前端输入: gwasfile（多个输入）, [可选: genotype files(bfile/vcf/tsv，单一输入), phenotype file (tsv，多个输入), genome annotation (gff/bed，单一输出)]
创建Job 将文件写入Job对应文件夹。前端可指定Job，进入对应任务 (db?)
后端解析数据：加载全部的gwasfile [phenotype file, genome annotation]; 不加载genotype files(后续流式加载减小内存占用). (解析策略采用python/janusx/script/postgwas.py中的策略)
进入Job后，chr/pos/pvalue解析gwasfile表头后提供选项；bimrange输入选项；manh/qq/gene/LDblock绘制选项
（布局是[[manh,qq],[gene,NA],[ldblock,NA]]）
点击一种图片类型，生成一种图片。目前最多可以生成四种图片。（缺少对应文件时不生成对应文件，例如：gene需要额外的gff文件，LDblock需要额外的genotype文件，同时需要指定bimrange才可以进行gene和ldblock的绘制）

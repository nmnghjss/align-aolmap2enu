# align-aolmap2enu

将colmap的稀疏重建结果，基于图像的gps或rtk信息，对齐到enu坐标系

# 输入

''' 
python align_colmap2enu.py -w colmapResultDir --gps_source_path gpsImageDir --gps_json gps.json --gps_csv gps.csv --aligned_dir outputDir

-w : colmap稀疏重建结果路径，其下为sparse/0/*.bin

--gps_source_path, 带gps信息的图像所在文件夹
--gps_json，gps信息json数据路径
--gps_csv, gps信息csv路径
--aligned_dir, 对齐结果输出路径

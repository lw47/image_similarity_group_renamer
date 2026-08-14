图片相似度分组与重命名工具

功能：
1. 扫描文件夹内 JPG/JPEG/PNG/WebP/BMP/GIF/TIF/TIFF。
2. 根据图片视觉特征计算相似度。
3. 将相似图片分组，并按组内相似程度排序。
4. 生成 Group-001-01.png、Group-001-02.png ... 这样的重命名方案。
5. 可先导出 CSV 预览，再执行重命名。

运行：
1. 安装 Python 3.10+。
2. 执行：pip install pillow numpy scipy
3. 执行：python image_similarity_renamer.py

建议：手机截图可先从 80%~88% 阈值开始测试。
阈值越高，要求越相似；阈值越低，越容易把“看起来相关”的图片放到同一组。

注意：执行重命名不会删除图片，但会修改文件名。建议首次使用先导出 CSV 检查结果。
